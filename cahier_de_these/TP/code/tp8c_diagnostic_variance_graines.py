"""
TP8c — Pourquoi Leaky ReLU et PReLU varient-ils autant d'une graine a l'autre ?

TP8b a mesure le fait (variance forte, 0/4 a 4/4 selon la graine). Ce script
cherche une explication geometrique simple : les 4 neurones caches de chaque
graine ont-ils des hyperplans "faciles" (bien separes) ou "difficiles" (deux
hyperplans presque paralleles, donc faciles a confondre pour l'ajustement
robuste de TP8) ? Si les mauvaises graines ont systematiquement des
hyperplans plus proches entre eux, la variance observee n'est pas un defaut
aleatoire de l'attaque mais une consequence mesurable de la geometrie du
reseau cible -- une explication, pas juste un constat.

Lance :  python3 tp8c_diagnostic_variance_graines.py
"""

import numpy as np
from common import MLP, cosine

NETWORK_SEEDS = [7, 11, 23, 42, 100]
DIM, HIDDEN, OUT = 3, 4, 2

# resultats bruts de TP8b (nombre de neurones retrouves / 4), pour comparaison
FOUND_TP8B = {
    "leaky_relu": {7: 1, 11: 4, 23: 0, 42: 4, 100: 0},
    "prelu":      {7: 3, 11: 2, 23: 4, 42: 2, 100: 0},
}


def min_pairwise_angle_deg(W):
    """Plus petit angle (en degres) entre deux lignes de W, normales aux
    hyperplans des neurones caches. Petit angle = deux neurones proches,
    donc hyperplans faciles a confondre pour un ajustement robuste."""
    n = W.shape[0]
    best = 180.0
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(cosine(W[i], W[j]))
            c = min(1.0, c)
            angle = np.degrees(np.arccos(c))
            best = min(best, angle)
    return best


def min_bias_gap(net):
    """Plus petit ecart entre deux neurones caches sur la distance de leur
    hyperplan a l'origine (b / ||w||) -- un indicateur grossier supplementaire
    de proximite geometrique."""
    W, b = net.W[0], net.b[0]
    d = b / np.linalg.norm(W, axis=1)
    n = len(d)
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, abs(d[i] - d[j]))
    return best


if __name__ == "__main__":
    print("=" * 78)
    print("Angle minimal entre deux hyperplans caches, par graine et par activation")
    print("=" * 78)
    header = f"{'activation':<12}{'graine':>8}{'retrouves/4':>13}{'angle min (deg)':>18}{'ecart biais min':>17}"
    print(header)
    print("-" * len(header))

    for name in ["leaky_relu", "prelu"]:
        rows = []
        for seed in NETWORK_SEEDS:
            net = MLP([DIM, HIDDEN, OUT], activation=name, seed=seed)
            angle = min_pairwise_angle_deg(net.W[0])
            gap = min_bias_gap(net)
            found = FOUND_TP8B[name][seed]
            rows.append((seed, found, angle, gap))
            print(f"{name:<12}{seed:>8}{found:>10}/4{angle:>18.1f}{gap:>17.3f}")
        print("-" * len(header))
        # correlation grossiere : les graines a 0 ou faible "trouves" ont-elles
        # un angle plus petit que les graines a 4 ?
        found_vals = np.array([r[1] for r in rows], dtype=float)
        angle_vals = np.array([r[2] for r in rows], dtype=float)
        if np.std(found_vals) > 0 and np.std(angle_vals) > 0:
            corr = np.corrcoef(found_vals, angle_vals)[0, 1]
            print(f"  Correlation (retrouves vs angle minimal) = {corr:+.3f}")
        print()

    print("Lecture : une correlation positive nette voudrait dire que les graines")
    print("ou deux hyperplans sont presque paralleles sont aussi celles ou l'attaque")
    print("retrouve le moins de neurones -- la variance serait alors expliquee par")
    print("la geometrie du reseau cible, pas par un defaut aleatoire de l'attaque.")
    print("Une correlation faible ou nulle voudrait dire que la cause est ailleurs")
    print("(par exemple dans le choix du germe initial, qui est lui-meme aleatoire).")
