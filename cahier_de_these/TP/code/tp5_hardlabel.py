"""
TP5 — Le cadre HARD-LABEL : trouver des POINTS DE TRANSITION (fiches 06-08).

Ici l'attaquant ne voit QUE le label (argmax), pas les logits. Il ne peut donc
plus mesurer de "cassure de pente". Mais il peut detecter les endroits ou la
CLASSE PREDITE change : les points de transition, sur la FRONTIERE DE DECISION.

Ce TP : recherche binaire d'un point de transition le long d'un segment.
C'est la primitive de base des attaques hard-label (Chen 2024, Carlini 2025) et
de ton article fondateur (Canales-Martinez & Santos 2025).

Lance :  python3 tp5_hardlabel.py
"""

import numpy as np
from common import MLP


def find_transition(net, x_a, x_b, tol=1e-12, max_iter=200):
    """Recherche binaire d'un changement de label entre x_a et x_b.
    N'utilise QUE l'oracle hard-label (aucun acces aux logits)."""
    la = net.oracle_label(x_a)
    lb = net.oracle_label(x_b)
    if la == lb:
        return None  # pas de transition sur ce segment
    for _ in range(max_iter):
        x_m = 0.5 * (x_a + x_b)
        lm = net.oracle_label(x_m)
        if np.linalg.norm(x_b - x_a) < tol:
            break
        if lm == la:
            x_a = x_m
        else:
            x_b = x_m
    return 0.5 * (x_a + x_b)


def collect_boundary_points(net, dim, n_points=20, seed=0):
    """Collecte des points de la frontiere de decision par segments aleatoires."""
    rng = np.random.default_rng(seed)
    pts = []
    tries = 0
    while len(pts) < n_points and tries < 2000:
        tries += 1
        x_a = rng.standard_normal(dim) * 2
        x_b = rng.standard_normal(dim) * 2
        p = find_transition(net, x_a, x_b)
        if p is not None:
            pts.append(p)
    return np.array(pts)


if __name__ == "__main__":
    dim = 3
    net = MLP([dim, 6, 4, 3], activation="relu", seed=5)
    print(f"Cible : ReLU [{dim},6,4,3], oracle HARD-LABEL uniquement.")
    print("-" * 64)

    pts = collect_boundary_points(net, dim, n_points=25, seed=2)
    print(f"Points de transition (frontiere de decision) trouves : {len(pts)}")
    print(f"Requetes hard-label utilisees                        : {net.n_queries}")
    print()
    # verification : sur la frontiere, deux classes sont a egalite (logits ~ croisent)
    logits = net.forward(pts, count=False)
    top2 = np.sort(logits, axis=1)[:, -2:]
    gaps = np.abs(top2[:, 1] - top2[:, 0])
    print(f"Ecart moyen entre les 2 meilleurs logits sur ces points : {gaps.mean():.2e}")
    print("(proche de 0 => on est bien SUR la frontiere de decision)")
    print()
    print("=> En hard-label, ces points de transition REMPLACENT les points")
    print("   critiques. Les attaques f.06-08 reconstruisent la geometrie des")
    print("   hyperplans caches a partir de milliers de tels points.")
    print()
    print("PISTE THESE (idee B2/B4) : pour une activation LISSE (sigmoid/GELU),")
    print("la frontiere n'est plus polygonale mais COURBE. Mesurer sa courbure")
    print("locale pourrait remplacer les derivees d'ordre superieur. -> TP6.")
