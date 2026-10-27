"""
TP7c — Meme test que TP7b, avec une defense MOINS extreme.

TP7b utilisait similarity=0.97 sur un reseau a seulement 3 entrees, ce qui a
rendu les 4 neurones caches parfaitement colineaires (|cos|=1.000) : un cas
degenere, plus fort que ce que vise la vraie defense (qui garde <1% de perte
de precision, donc une similarite partielle, pas totale). Ce script balaie
plusieurs niveaux de similarite pour voir a partir d'ou l'attaque hard-label
commence vraiment a decrocher, plutot que de sauter directement au cas extreme.

Lance :  python3 tp7c_defense_moderee.py
"""

import time
import numpy as np
from common import MLP, cosine
from tp7_defense import make_defended
from tp8_hardlabel_beyond_relu import recover_hidden_layer_hardlabel, match_against_truth

DIM, HIDDEN, OUT = 3, 4, 2
NET_SEED = 7
ATTACK_SEEDS = [1, 2, 3]  # 3 graines seulement, pour tenir dans un temps raisonnable
SIMILARITIES = [0.0, 0.3, 0.6, 0.9]


def worst_pair_cos(W):
    n = W.shape[0]
    worst = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            worst = max(worst, abs(cosine(W[i], W[j])))
    return worst


if __name__ == "__main__":
    t0 = time.time()
    for name in ["leaky_relu", "prelu"]:
        print("=" * 78)
        print(f"Activation : {name}  (reseau cible, seed={NET_SEED})")
        print("=" * 78)
        header = f"{'similarite':>11}{'cos max entre neurones':>25}{'retrouves moyen /4':>20}{'cos moyen':>12}"
        print(header)
        print("-" * len(header))
        for sim in SIMILARITIES:
            net = MLP([DIM, HIDDEN, OUT], activation=name, seed=NET_SEED)
            if sim > 0:
                net = make_defended(net, similarity=sim)
            sep = worst_pair_cos(net.W[0])
            founds, coss_all = [], []
            for a_seed in ATTACK_SEEDS:
                found = recover_hidden_layer_hardlabel(net, DIM, target_count=HIDDEN, seed=a_seed)
                rows = match_against_truth(found, net.W[0])
                founds.append(len(found))
                coss_all.extend([c for _, c, _ in rows])
            mean_found = np.mean(founds)
            mean_cos = np.mean(coss_all) if coss_all else float("nan")
            print(f"{sim:>11.2f}{sep:>25.4f}{mean_found:>20.2f}{mean_cos:>12.4f}   (detail {founds})")
        print()
    print(f"Temps total : {time.time() - t0:.1f} s")
