"""
TP6 — AU-DELA DE ReLU : identifier l'activation et exploiter la "fuite" (fiches 10-12).

Trois experiences au coeur de ta these :

 (A) IDENTIFICATION D'ACTIVATION : en scannant une ligne, la "signature
     geometrique" differe selon la famille (par morceaux vs lisse). On classe
     automatiquement chaque reseau -> primitive d'identification (idee B4).

 (B) FUITE DE SIGNE : pour Leaky/PReLU/ELU, les DEUX cotes du coude reagissent
     (pente negative non nulle) -> le signe est plus facile a lire que pour ReLU.

 (C) LE MUR DU LISSE : pour sigmoid/GELU, pas de coude net. On illustre pourquoi
     les outils ReLU echouent (idee B2 : le grand trou de recherche).

Lance :  python3 tp6_beyond_relu.py
"""

import numpy as np
from common import MLP, ACTIVATIONS, PIECEWISE_LINEAR, SMOOTH


def curvature_profile(net, x0, d, n=2000, t=3.0):
    """|derivee 2nde| de la sortie le long d'une ligne. Pic localise = coude
    (par morceaux) ; courbure etalee = activation lisse."""
    ts = np.linspace(-t, t, n)
    X = x0[None, :] + ts[:, None] * d[None, :]
    Y = net.forward(X, count=True)[:, 0]
    dt = ts[1] - ts[0]
    curv = np.abs(np.diff(Y, 2)) / dt**2
    return ts[1:-1], curv


def classify_activation_family(net, dim, seed=0, n_lines=5):
    """Heuristique : concentration de la courbure. Les coudes (par morceaux)
    donnent des pics tres concentres ; le lisse etale la courbure."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_lines):
        x0 = rng.standard_normal(dim)
        d = rng.standard_normal(dim); d /= np.linalg.norm(d)
        _, curv = curvature_profile(net, x0, d)
        if curv.max() <= 0:
            continue
        # fraction de la courbure totale portee par les 1% de points les plus courbes
        k = max(1, len(curv) // 100)
        top = np.sort(curv)[-k:].sum()
        ratios.append(top / (curv.sum() + 1e-300))
    r = np.mean(ratios) if ratios else 0.0
    # pics tres concentres => par morceaux ; sinon => lisse
    return ("par morceaux" if r > 0.5 else "lisse"), r


if __name__ == "__main__":
    dim = 3
    print("(A) IDENTIFICATION DE LA FAMILLE D'ACTIVATION (oracle raw-output)")
    print("-" * 64)
    print(f"{'activation':<12} {'vraie famille':<14} {'predite':<14} {'indice':>8}")
    for name in ["relu", "leaky_relu", "prelu", "hardtanh", "elu",
                 "sigmoid", "tanh", "gelu", "silu"]:
        net = MLP([dim, 6, 2], activation=name, seed=42)
        fam_pred, ratio = classify_activation_family(net, dim, seed=1)
        true_fam = "par morceaux" if name in PIECEWISE_LINEAR else "lisse"
        flag = "OK" if fam_pred == true_fam else "??"
        print(f"{name:<12} {true_fam:<14} {fam_pred:<14} {ratio:>8.3f}  {flag}")
    print()
    print("=> On distingue automatiquement les deux grandes familles. Affiner")
    print("   ce classifieur (et le passer en HARD-LABEL) = ta primitive B4.")
    print()

    print("(B) FUITE DE SIGNE : reaction des deux cotes du coude")
    print("-" * 64)
    # neurone isole f(x)=sigma(w.x+b), on regarde la reaction de part et d'autre
    w = np.array([1.0, -1.0]); b = 0.0
    x_crit = np.array([0.0, 0.0])              # w.x+b = 0
    d = w / np.linalg.norm(w)
    eps = 1e-2
    for name in ["relu", "leaky_relu", "prelu", "elu"]:
        sigma = ACTIVATIONS[name]
        f = lambda x: float(sigma(np.array([w @ x + b]))[0])
        plus = f(x_crit + eps * d) - f(x_crit)
        minus = f(x_crit - eps * d) - f(x_crit)
        leak = "les 2 cotes" if abs(minus) > 1e-9 else "1 seul cote"
        print(f"{name:<12} delta+ = {plus:+.4f}   delta- = {minus:+.4f}   -> {leak}")
    print()
    print("=> ReLU ne fuit que d'un cote (signe DUR). Leaky/PReLU/ELU fuient des")
    print("   DEUX cotes => le signe se lit plus facilement (idee B1, fiche 10).")
    print()

    print("(C) LE MUR DU LISSE : pas de coude exploitable")
    print("-" * 64)
    for name in ["relu", "sigmoid", "gelu"]:
        net = MLP([dim, 6, 2], activation=name, seed=42)
        rng = np.random.default_rng(0)
        x0 = rng.standard_normal(dim); d = rng.standard_normal(dim); d/=np.linalg.norm(d)
        ts, curv = curvature_profile(net, x0, d)
        # nombre de "pics" nets de courbure
        peaks = int(np.sum(curv > 0.3 * curv.max())) if curv.max() > 0 else 0
        print(f"{name:<10} courbure max = {curv.max():.3e}  points 'pics' = {peaks}")
    print()
    print("=> ReLU : pics nets et rares (coudes localisables). Lisse : courbure")
    print("   etalee, pas de coude a localiser => les attaques f.03-09 echouent.")
    print("   Inventer un substitut en HARD-LABEL = le coeur de ta these (B2).")
