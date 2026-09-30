"""
TP3 — Recuperer la SIGNATURE des neurones de la 1ere couche (Carlini 2020, fiche 03).

Idee : pour un reseau ReLU, le gradient de la sortie par rapport a l'entree est
CONSTANT par morceaux. Quand on traverse un point critique du neurone i (lui seul
bascule on/off), le gradient fait un SAUT proportionnel a la ligne de poids W1[i].
Donc :  direction(saut de gradient)  =  direction(W1[i])  =  la SIGNATURE.

L'attaquant n'a qu'un acces ORACLE (raw-output) : le gradient est estime par
differences finies (requetes). On NE recupere pas l'echelle ni le signe (c'est
le role de TP4) — seulement la direction du vecteur de poids.

Lance :  python3 tp3_signature.py
"""

import numpy as np
from common import MLP, cosine


def numerical_gradient(net, x, h=1e-5):
    """Gradient du logit 0 par rapport a x, par differences finies centrees.
    (purement oracle : on n'utilise que net.forward)"""
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for j in range(len(x)):
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        g[j] = (net.forward(xp, count=True)[0, 0] -
                net.forward(xm, count=True)[0, 0]) / (2 * h)
    return g


def find_one_critical_point(net, x0, d, t_lo, t_hi, tol=1e-10):
    """Recherche binaire d'un point critique entre t_lo et t_hi le long de x0+t d.
    On suit la pente locale (derivee selon d) : elle est constante par morceaux,
    le coude est la ou elle change. On encadre puis dichotomie."""
    def slope(t):
        x = x0 + t * d
        hh = 1e-6
        return (net.forward(x + hh * d, count=True)[0, 0] -
                net.forward(x - hh * d, count=True)[0, 0]) / (2 * hh)
    s_lo = slope(t_lo)
    # dichotomie sur le changement de pente
    for _ in range(100):
        t_mid = 0.5 * (t_lo + t_hi)
        if abs(t_hi - t_lo) < tol:
            break
        if abs(slope(t_mid) - s_lo) < 1e-9:
            t_lo = t_mid
        else:
            t_hi = t_mid
    return 0.5 * (t_lo + t_hi)


def recover_signature_at(net, x_crit, d, eps=1e-3):
    """Saut de gradient de part et d'autre du point critique -> direction de W1[i]."""
    g_before = numerical_gradient(net, x_crit - eps * d)
    g_after = numerical_gradient(net, x_crit + eps * d)
    jump = g_after - g_before
    return jump / (np.linalg.norm(jump) + 1e-300)


if __name__ == "__main__":
    # --- cible : un reseau ReLU a UNE couche cachee (cas le plus clair) ---
    dim, hidden = 4, 5
    net = MLP([dim, hidden, 2], activation="relu", seed=3)
    rng = np.random.default_rng(11)

    print(f"Cible : ReLU [{dim},{hidden},2]. On veut recuperer les {hidden} lignes de W1.")
    print("-" * 64)

    recovered = []
    # on lance plusieurs scans aleatoires pour trouver des points critiques
    found_dirs = []
    attempts = 0
    while len(found_dirs) < hidden and attempts < 60:
        attempts += 1
        x0 = rng.standard_normal(dim)
        d = rng.standard_normal(dim); d /= np.linalg.norm(d)
        # scan grossier pour reperer un changement de pente
        ts = np.linspace(-3, 3, 400)
        slopes = []
        for t in ts:
            x = x0 + t * d
            hh = 1e-6
            slopes.append((net.forward(x + hh*d, count=True)[0,0] -
                           net.forward(x - hh*d, count=True)[0,0])/(2*hh))
        slopes = np.array(slopes)
        change_idx = np.where(np.abs(np.diff(slopes)) > 1e-4)[0]
        for ci in change_idx:
            t_star = find_one_critical_point(net, x0, d, ts[ci], ts[ci+1])
            x_crit = x0 + t_star * d
            sig = recover_signature_at(net, x_crit, d)
            # est-ce une nouvelle direction ?
            if all(abs(cosine(sig, f)) < 0.999 for f in found_dirs):
                found_dirs.append(sig)
            if len(found_dirs) >= hidden:
                break

    # --- verification (boite blanche : on a le droit de regarder pour EVALUER) ---
    print(f"Directions distinctes recuperees : {len(found_dirs)} / {hidden}")
    print(f"Nombre total de requetes oracle   : {net.n_queries}")
    print()
    print("Appariement signature recuperee <-> vraie ligne de W1 (|cosinus|, ~1 = parfait) :")
    W1 = net.W[0]
    used = set()
    for k, row in enumerate(W1):
        best, bj = 0, -1
        for j, sig in enumerate(found_dirs):
            c = abs(cosine(sig, row))
            if c > best and j not in used:
                best, bj = c, j
        used.add(bj)
        print(f"  neurone {k}: |cos| = {best:.6f}")
    print()
    print("=> |cos| ~ 1 : la DIRECTION des poids est recuperee (la 'signature').")
    print("   Ce qui reste : l'ECHELLE (facteur multiplicatif) et le SIGNE -> TP4.")
