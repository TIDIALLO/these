
"""
WP2 — La recuperation de la couche de sortie depend-elle de l'activation ?
 
Theoreme conjecture : NON. La preuve de la Section 3 de Canales-Martinez & Santos
(LATINCRYPT 2025) n'utilise que
   H1 : en un point de transition entre les classes i et j,  A_i y + b_i = A_j y + b_j
   H2 : y = f_{1..r}(x) est calculable  (couches precedentes deja extraites)
   H3 : softmax est invariante par translation et preserve l'ordre
Aucune de ces hypotheses ne mentionne ReLU.
 
Protocole (celui de l'article) :
  - reseau d0 - ... - d_r - d_{r+1}, entraine, quatre activations
  - on suppose les couches cachees connues (oracle partiel, hypothese assumee)
  - on collecte des points de transition en hard-label uniquement
  - on construit C, on fixe d_r + 2 variables, on resout
  - on verifie l'accord sur 5000 entrees tirees uniformement dans (-1,1)
"""
import numpy as np
from core import MLP, HardLabelOracle, binary_search_boundary
 
rng = np.random.default_rng(11)
 
D0, HID, NOUT = 40, [24, 16, 10], 4
DR = HID[-1]
NVAR = NOUT * (DR + 1)
RANK_MAX = NVAR - (DR + 2)
NFIX = DR + 2
 
print("=" * 78)
print(f"Architecture : {D0} - {' - '.join(map(str, HID))} - {NOUT}")
print(f"Inconnues de la couche de sortie : d_(r+1)(d_r+1) = {NOUT}x{DR+1} = {NVAR}")
print(f"Rang maximal theorique           : {NVAR} - ({DR}+2) = {RANK_MAX}")
print(f"Variables a fixer                : d_r + 2 = {NFIX}")
print("=" * 78)
 
 
def make_data(n=3000, seed=2):
    r = np.random.default_rng(seed)
    C = r.normal(0, 1.4, (NOUT, D0))
    y = r.integers(0, NOUT, n)
    X = C[y] + r.normal(0, 0.9, (n, D0))
    return X, y
 
 
def transition_points(oracle, need, scale=1.0, iters=100, cap=200000):
    """Collecte des points de transition + le couple de classes (i, j)."""
    out = []
    while len(out) < need and oracle.n_queries < cap:
        x0, x1 = rng.normal(0, scale, D0), rng.normal(0, scale, D0)
        l0, l1 = oracle(x0)[0], oracle(x1)[0]
        if l0 == l1:
            continue
        p = binary_search_boundary(oracle, x0, x1, iters)
        # les deux classes de part et d'autre
        eps = 1e-7 * (x1 - x0) / np.linalg.norm(x1 - x0)
        i, j = oracle(p - eps)[0], oracle(p + eps)[0]
        if i != j:
            out.append((p, int(i), int(j)))
    return out
 
 
def build_system(net, tps):
    """Une ligne par point de transition : bloc i = (y,1), bloc j = -(y,1)."""
    rows = []
    for p, i, j in tps:
        yv = net.hidden(p)[0]
        row = np.zeros(NVAR)
        row[i * (DR + 1):i * (DR + 1) + DR] = yv
        row[i * (DR + 1) + DR] = 1.0
        row[j * (DR + 1):j * (DR + 1) + DR] -= yv
        row[j * (DR + 1) + DR] -= 1.0
        rows.append(row)
    return np.array(rows)
 
 
def solve_output_layer(C):
    """
    On fixe d_r+2 variables : A_1 = 0 (d_r poids), b_1 = 0, et A_{2,1} = 1.
    Puis on resout le systeme reduit C_libre . theta_libre = -C_fixe . theta_fixe.
    """
    fixed_idx = list(range(0, DR + 1))          # bloc 1 entier : d_r + 1 variables
    fixed_val = np.zeros(DR + 1)
    fixed_idx.append(DR + 1)                    # A_{2,1}
    fixed_val = np.r_[fixed_val, 1.0]           # = 1        -> total d_r + 2
    assert len(fixed_idx) == NFIX
 
    free_idx = [k for k in range(NVAR) if k not in fixed_idx]
    rhs = -C[:, fixed_idx] @ fixed_val
    sol, *_ = np.linalg.lstsq(C[:, free_idx], rhs, rcond=None)
 
    theta = np.zeros(NVAR)
    theta[fixed_idx] = fixed_val
    theta[free_idx] = sol
    A = np.zeros((NOUT, DR)); b = np.zeros(NOUT)
    for k in range(NOUT):
        A[k] = theta[k * (DR + 1):k * (DR + 1) + DR]
        b[k] = theta[k * (DR + 1) + DR]
    return A, b, theta
 
 
X, ytr = make_data()
print(f"\n{'activation':<10} {'acc':>6} {'pts trans.':>11} {'requetes':>10} "
      f"{'rang':>6} {'/max':>5} {'residu':>10} {'inverse':>8} {'accord/5000':>12}")
print("-" * 88)
 
summary = {}
for act in ["relu", "gelu", "silu", "tanh"]:
    net = MLP([D0] + HID + [NOUT], act=act, seed=5).fit(X, ytr, epochs=220, lr=0.05)
    acc = net.accuracy(X, ytr)
 
    orc = HardLabelOracle(net)
    tps = transition_points(orc, need=3 * RANK_MAX)
    C = build_system(net, tps)
    rank = np.linalg.matrix_rank(C, tol=1e-8)
 
    A_hat, b_hat, theta = solve_output_layer(C)
    residu = float(np.abs(C @ theta).max())
 
    # --- levee de l'ambiguite de signe global ---------------------------
    # On a impose A_{2,1} = 1, ce qui revient a choisir c = 1/A_{2,1}.
    # Si le vrai A_{2,1} est negatif, alors c < 0 : l'argmax devient un argmin
    # et le reseau reconstruit classe systematiquement a l'envers.
    # L'article ne mentionne ce test que pour le cas a une sortie ; il est
    # necessaire aussi en multi-sorties. Cout : UNE requete.
    xprobe = rng.normal(0, 1.0, D0)
    y_probe = net.hidden(xprobe)[0]
    if np.argmax(A_hat @ y_probe + b_hat) != orc(xprobe)[0]:
        A_hat, b_hat = -A_hat, -b_hat
        flipped = "oui"
    else:
        flipped = "non"
 
    # reseau reconstruit : couches cachees vraies + couche de sortie recuperee
    rec = MLP([D0] + HID + [NOUT], act=act, seed=5)
    rec.A = [w.copy() for w in net.A]; rec.b = [v.copy() for v in net.b]
    rec.A[-1] = A_hat; rec.b[-1] = b_hat
 
    Xt = rng.uniform(-1, 1, (5000, D0))
    agree = int((net.label(Xt) == rec.label(Xt)).sum())
 
    summary[act] = dict(acc=acc, npts=len(tps), q=orc.n_queries, rank=int(rank),
                        residu=residu, agree=agree, flipped=flipped)
    print(f"{act:<10} {acc:6.3f} {len(tps):11d} {orc.n_queries:10d} "
          f"{rank:6d} {RANK_MAX:5d} {residu:10.2e} {flipped:>7} {agree:9d}/5000")
 
print("-" * 88)
ok = all(s["agree"] == 5000 and s["rank"] == RANK_MAX for s in summary.values())
print("VERDICT :", "theoreme valide sur les 4 activations" if ok
      else "ATTENTION - au moins un cas s'ecarte de la prediction")
np.save("/home/claude/these/wp2_results.npy", summary, allow_pickle=True)
 

