"""
TP2 — Trouver des POINTS CRITIQUES (cassures de pente) d'un reseau ReLU.

Objectif pedagogique : voir de tes yeux qu'un reseau ReLU est affine PAR MORCEAUX.
On parcourt une ligne dans l'espace d'entree et on detecte les endroits ou la
pente de la sortie change brutalement = un neurone passe par zero.

C'est la brique de base de l'attaque de Carlini 2020 (fiche 03).
Lance :  python3 tp2_points_critiques.py
"""

import numpy as np
from common import MLP


def scan_line(net, x0, d, t_min=-4, t_max=4, n=4000, out_index=0):
    """Evalue la sortie le long de x(t) = x0 + t*d et renvoie (t, y)."""
    ts = np.linspace(t_min, t_max, n)
    X = x0[None, :] + ts[:, None] * d[None, :]
    Y = net.forward(X, count=True)[:, out_index]
    return ts, Y


def detect_kinks(ts, Y, threshold=1e-3):
    """Detecte les cassures de pente : la derivee seconde discrete depasse un seuil."""
    dt = ts[1] - ts[0]
    slope = np.diff(Y) / dt                 # pente locale (derivee 1ere)
    curv = np.abs(np.diff(slope))           # variation de pente (|derivee 2nde|)
    idx = np.where(curv > threshold)[0] + 1
    # regrouper les indices voisins (un meme coude s'etale sur qq points)
    kinks = []
    for i in idx:
        if not kinks or i - kinks[-1] > 3:
            kinks.append(i)
    return [ts[i] for i in kinks]


if __name__ == "__main__":
    net = MLP([2, 4, 2], activation="relu", seed=7)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(2)
    d = rng.standard_normal(2); d /= np.linalg.norm(d)

    ts, Y = scan_line(net, x0, d)
    kinks = detect_kinks(ts, Y)

    print(f"Reseau ReLU [2,4,2]. Direction de scan d = {d.round(3)}")
    print(f"Nombre de cassures de pente detectees : {len(kinks)}")
    print(f"Positions (t) des points critiques     : {[round(k,3) for k in kinks]}")
    print(f"Nombre de requetes utilisees           : {net.n_queries}")
    print()
    print("=> Chaque cassure = un neurone de couche 1 qui passe par zero.")
    print("   C'est la signature geometrique du ReLU (affine par morceaux).")
    print()
    print("EXERCICE : relance avec activation='sigmoid' dans le MLP ci-dessus.")
    print("           Tu ne trouveras PLUS de cassure nette : c'est tout le")
    print("           probleme du 'au-dela de ReLU' (chapitre 02, idee B2).")
