"""
Teste l'hypothese de la note : le facteur qui compte n'est peut-etre pas
l'activation elle-meme, mais le rapport entre l'echelle des pre-activations
et la largeur de la zone non lineaire de l'activation.

Protocole : on entraine un reseau normalement (beta=1), puis on multiplie
UNIQUEMENT la premiere couche (poids + biais) par beta avant de mesurer rho.
Cela change l'echelle des pre-activations qui entrent dans sigma, sans rien
changer d'autre. ReLU est positivement homogene : ReLU(beta*z) = beta*ReLU(z).
Combine a la couche lineaire suivante, rho ne doit quasiment pas bouger avec
beta pour ReLU -- c'est le controle negatif de l'experience.
"""
import numpy as np
from scipy.special import erf
import copy

SQ2, SQ2PI = np.sqrt(2.0), np.sqrt(2.0*np.pi)
def relu(x): return np.maximum(x,0.0)
def d_relu(x): return (x>0).astype(float)
def gelu(x): return 0.5*x*(1.0+erf(x/SQ2))
def d_gelu(x): return 0.5*(1.0+erf(x/SQ2)) + x*np.exp(-0.5*x**2)/SQ2PI
def _sig(x): return 0.5*(1.0+np.tanh(0.5*x))
def silu(x): return x*_sig(x)
def d_silu(x):
    s=_sig(x); return s*(1.0+x*(1.0-s))
def tanh_a(x): return np.tanh(x)
def d_tanh(x): return 1.0-np.tanh(x)**2

ACT={"relu":(relu,d_relu),"gelu":(gelu,d_gelu),"silu":(silu,d_silu),"tanh":(tanh_a,d_tanh)}

class MLP:
    def __init__(self, dims, act="relu", seed=0):
        self.dims,self.act=dims,act
        self.sigma,self.dsigma=ACT[act]
        rng=np.random.default_rng(seed)
        self.A,self.b=[],[]
        for i in range(len(dims)-1):
            self.A.append(rng.normal(0,np.sqrt(2.0/dims[i]),(dims[i+1],dims[i])))
            self.b.append(np.zeros(dims[i+1]))
        self.r=len(dims)-2
    def forward(self,X,upto=None):
        Z=np.atleast_2d(X); L=len(self.A) if upto is None else upto
        for i in range(L):
            Z=Z@self.A[i].T+self.b[i]
            if i<len(self.A)-1: Z=self.sigma(Z)
        return Z
    def label(self,X): return np.argmax(self.forward(X),axis=1)
    def fit(self,X,y,epochs=200,lr=0.05,bs=64,seed=0):
        rng=np.random.default_rng(seed); n,L=len(X),len(self.A)
        vA=[np.zeros_like(a) for a in self.A]; vb=[np.zeros_like(b) for b in self.b]
        for ep in range(epochs):
            idx=rng.permutation(n)
            for s in range(0,n,bs):
                xb,yb=X[idx[s:s+bs]],y[idx[s:s+bs]]
                pre,post=[],[xb]; Z=xb
                for i in range(L):
                    P=Z@self.A[i].T+self.b[i]; pre.append(P)
                    Z=self.sigma(P) if i<L-1 else P; post.append(Z)
                E=np.exp(Z-Z.max(1,keepdims=True)); S=E/E.sum(1,keepdims=True)
                G=S.copy(); G[np.arange(len(yb)),yb]-=1.0; G/=len(yb)
                for i in range(L-1,-1,-1):
                    gA=G.T@post[i]; gb=G.sum(0)
                    if i>0: G=(G@self.A[i])*self.dsigma(pre[i-1])
                    vA[i]=0.9*vA[i]-lr*gA; vb[i]=0.9*vb[i]-lr*gb
                    self.A[i]+=vA[i]; self.b[i]+=vb[i]
        return self
    def accuracy(self,X,y): return float((self.label(X)==y).mean())
    def preact(self,X,layer):
        Z=np.atleast_2d(X)
        for i in range(layer-1): Z=self.sigma(Z@self.A[i].T+self.b[i])
        return Z@self.A[layer-1].T+self.b[layer-1]

class HardLabelOracle:
    def __init__(self,net): self.net,self.n_queries=net,0
    def __call__(self,X):
        X=np.atleast_2d(X); self.n_queries+=len(X); return self.net.label(X)

def two_moons(n=1200,noise=0.18,seed=1):
    r=np.random.default_rng(seed); t=r.uniform(0,np.pi,n//2)
    A=np.c_[np.cos(t),np.sin(t)]; B=np.c_[1-np.cos(t),0.4-np.sin(t)]
    X=np.vstack([A,B])+r.normal(0,noise,(n,2))
    y=np.r_[np.zeros(n//2,int),np.ones(n//2,int)]
    return (X-X.mean(0))/X.std(0),y

def start_on_boundary(oracle,rng,dim=2,scale=1.2,tries=4000,iters=70):
    for _ in range(tries):
        x0,x1=rng.normal(0,scale,dim),rng.normal(0,scale,dim)
        if oracle(x0)[0]!=oracle(x1)[0]:
            a,b=x0.copy(),x1.copy(); la=oracle(a)[0]
            for _ in range(iters):
                m=0.5*(a+b)
                if oracle(m)[0]==la: a=m
                else: b=m
            return 0.5*(a+b)
    return None

def initial_tangent(oracle,p0,eps=0.02,n=120):
    for th in np.linspace(0,np.pi,n,endpoint=False):
        t=np.array([np.cos(th),np.sin(th)]); nv=np.array([-t[1],t[0]])
        if oracle(p0+eps*nv)[0]!=oracle(p0-eps*nv)[0]: return t
    return None

def trace_boundary(oracle,p0,t0,h=0.004,n_steps=500,halfwidth=0.08,iters=60):
    pts,tang=[np.array(p0,float)],[]
    t=np.array(t0,float); t/=np.linalg.norm(t)
    for _ in range(n_steps):
        p=pts[-1]; n=np.array([-t[1],t[0]]); q=p+h*t
        a,b=q-halfwidth*n,q+halfwidth*n
        la,lb=oracle(a)[0],oracle(b)[0]
        if la==lb: break
        for _ in range(iters):
            m=0.5*(a+b)
            if oracle(m)[0]==la: a=m
            else: b=m
        p_new=0.5*(a+b); d=p_new-p; nrm=np.linalg.norm(d)
        if nrm<1e-12: break
        t=d/nrm; pts.append(p_new); tang.append(np.arctan2(t[1],t[0]))
    return np.array(pts),np.array(tang)

def curvature(pts,tang):
    th=np.unwrap(tang); ds=np.linalg.norm(np.diff(pts[:-1],axis=0),axis=1)
    ds=np.maximum(ds,1e-12); k=np.abs(np.diff(th))/ds
    s=np.r_[0,np.cumsum(ds)]
    return s[:-1],k,ds

def crossings_arclength(net, pts, s):
    cross = []
    for layer in (1, 2):
        P = net.preact(pts[:len(s)], layer)
        sign_change = np.where(np.diff(np.sign(P), axis=0) != 0)[0]
        cross += [s[i] for i in np.unique(sign_change) if i < len(s)]
    return np.sort(np.array(cross))

def rho_misalignment(k, ds, s, cross):
    if len(cross) == 0 or len(k) == 0:
        return np.nan
    m = min(len(k), len(ds), len(s))
    k, ds, s = k[:m], ds[:m], s[:m]
    d = np.abs(s[:, None] - cross[None, :]).min(1)
    w_k = k * ds
    num = (w_k * d).sum() / max(w_k.sum(), 1e-15)
    den = (ds * d).sum() / max(ds.sum(), 1e-15)
    return float(num / max(den, 1e-15))

def measure_net(net, n_traces=5, h=0.004, n_steps=500, seed_offset=200):
    orc = HardLabelOracle(net)
    rhos, kmaxs = [], []
    for sd in range(n_traces):
        r = np.random.default_rng(seed_offset + sd)
        p0 = start_on_boundary(orc, r)
        if p0 is None: continue
        t0 = initial_tangent(orc, p0)
        if t0 is None: continue
        pts, tg = trace_boundary(orc, p0, t0, h=h, n_steps=n_steps)
        if len(pts) < 150: continue
        s_, k_, ds_ = curvature(pts, tg)
        cr = crossings_arclength(net, pts, s_)
        rh = rho_misalignment(k_, ds_, s_, cr)
        if np.isfinite(rh):
            rhos.append(rh); kmaxs.append(float(np.max(k_)))
    return dict(acc=net.accuracy(Xg, yg), n=len(rhos),
                rho=float(np.median(rhos)) if rhos else np.nan,
                kmax=float(np.median(kmaxs)) if kmaxs else np.nan)

def scaled_copy(net, beta):
    """Copie du reseau ou SEULE la couche 1 (poids+biais) est multipliee par beta.
    Change l'echelle des pre-activations qui entrent dans sigma, rien d'autre."""
    net2 = copy.deepcopy(net)
    net2.A[0] = net.A[0] * beta
    net2.b[0] = net.b[0] * beta
    return net2

if __name__ == "__main__":
    Xg, yg = two_moons()
    BETAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    SEEDS = [7, 17, 27]
    ACTS = ["relu", "gelu", "silu", "tanh"]

    print(f"{'activation':<10} {'graine':>7}" + "".join(f"{'b='+str(b):>9}" for b in BETAS))
    print("-" * (18 + 9*len(BETAS)))

    results = {a: {b: [] for b in BETAS} for a in ACTS}
    for a in ACTS:
        for sd in SEEDS:
            base = MLP([2, 8, 8, 2], act=a, seed=sd).fit(Xg, yg, epochs=400, lr=0.08)
            row = []
            for beta in BETAS:
                net_b = scaled_copy(base, beta)
                r = measure_net(net_b, n_traces=4)
                row.append(r["rho"])
                results[a][beta].append(r["rho"])
            print(f"{a:<10} {sd:7d}" + "".join(f"{v:9.3f}" if np.isfinite(v) else f"{'nan':>9}" for v in row))

    print()
    print("Rho median sur les 3 graines, par activation et par beta :")
    print(f"{'activation':<10}" + "".join(f"{'b='+str(b):>9}" for b in BETAS))
    for a in ACTS:
        meds = [np.nanmedian(results[a][b]) for b in BETAS]
        print(f"{a:<10}" + "".join(f"{v:9.3f}" for v in meds))
