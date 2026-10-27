"""
TP7b — Idee B6 : la defense "Train to Defend" tient-elle hors ReLU, en hard-label ?

TP7 teste le mecanisme de la defense (rapprocher les lignes de W1) contre
l'attaque RAW-OUTPUT de TP3, sur ReLU. C'est utile mais ce n'est pas le cadre
de la these : le cadre reel est le HARD-LABEL (TP8), et la question ouverte
B6 porte precisement sur les activations non-ReLU dans CE cadre.

Ce script reutilise make_defended() de TP7 et recover_hidden_layer_hardlabel()
de TP8, sur Leaky ReLU et PReLU, reseau normal contre reseau "defendu". Vu la
variance deja mesuree dans TP8b/TP8d (qui vient de l'attaque, pas du reseau),
on compare des MOYENNES sur plusieurs graines d'attaque, pas un seul essai,
pour ne pas confondre l'effet de la defense avec le bruit deja caracterise.

Lance :  python3 tp7b_defense_hardlabel_nonrelu.py
"""

import time
import numpy as np
from common import MLP, cosine
from tp7_defense import make_defended
from tp8_hardlabel_beyond_relu import recover_hidden_layer_hardlabel, match_against_truth

DIM, HIDDEN, OUT = 3, 4, 2
NET_SEED = 7
ATTACK_SEEDS = [1, 2, 3, 4, 5]


def run_condition(name, net, label):
    founds, coss_all = [], []
    for a_seed in ATTACK_SEEDS:
        found = recover_hidden_layer_hardlabel(net, DIM, target_count=HIDDEN, seed=a_seed)
        rows = match_against_truth(found, net.W[0])
        coss = [c for _, c, _ in rows]
        founds.append(len(found))
        coss_all.extend(coss)
        print(f"  {label:<10} graine attaque {a_seed} : {len(found)}/{HIDDEN} trouves, "
              f"requetes={net.n_queries}")
    mean_found = np.mean(founds)
    mean_cos = np.mean(coss_all) if coss_all else float("nan")
    return mean_found, mean_cos, founds


if __name__ == "__main__":
    t0 = time.time()
    for name in ["leaky_relu", "prelu"]:
        print("=" * 78)
        print(f"Activation : {name}  (reseau cible, seed={NET_SEED})")
        print("=" * 78)

        net_normal = MLP([DIM, HIDDEN, OUT], activation=name, seed=NET_SEED)
        print(" -- reseau NORMAL --")
        mf_n, mc_n, founds_n = run_condition(name, net_normal, "normal")

        net_def = make_defended(MLP([DIM, HIDDEN, OUT], activation=name, seed=NET_SEED),
                                 similarity=0.97)
        worst_sep = 0.0
        W1 = net_def.W[0]
        for i in range(HIDDEN):
            for j in range(i + 1, HIDDEN):
                worst_sep = max(worst_sep, abs(cosine(W1[i], W1[j])))
        print(f" -- reseau DEFENDU (similarite=0.97, |cos| max entre vraies lignes = {worst_sep:.3f}) --")
        mf_d, mc_d, founds_d = run_condition(name, net_def, "defendu")

        print()
        print(f"  RESUME {name} :")
        print(f"    normal  : {mf_n:.2f}/{HIDDEN} trouves en moyenne, cos moyen = {mc_n:.4f}  (detail {founds_n})")
        print(f"    defendu : {mf_d:.2f}/{HIDDEN} trouves en moyenne, cos moyen = {mc_d:.4f}  (detail {founds_d})")
        print()

    print(f"Temps total : {time.time() - t0:.1f} s")
