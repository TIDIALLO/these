"""
TP8d — La variance vient-elle du reseau cible, ou du hasard propre a l'attaque ?

TP8c a infirme l'hypothese geometrique (l'angle entre hyperplans ne correle
pas, ou correle a l'envers, avec le taux de reussite). Ce script teste l'autre
source possible : sur UN SEUL reseau cible fixe, re-lancer l'attaque avec
plusieurs graines d'attaque differentes (qui changent uniquement les lignes
et germes aleatoires explores, pas le reseau lui-meme). Si le taux de reussite
varie autant ici qu'entre reseaux dans TP8b, la variance vient principalement
du hasard de l'attaque -- pas de la geometrie de la cible.

Lance :  python3 tp8d_variance_attaque_vs_reseau.py
"""

import numpy as np
from common import MLP, cosine
from tp8_hardlabel_beyond_relu import recover_hidden_layer_hardlabel, match_against_truth

DIM, HIDDEN, OUT = 3, 4, 2
ATTACK_SEEDS = [1, 2, 3, 4, 5]


if __name__ == "__main__":
    for name, net_seed in [("leaky_relu", 7), ("prelu", 7)]:
        net = MLP([DIM, HIDDEN, OUT], activation=name, seed=net_seed)
        print("=" * 78)
        print(f"Reseau cible FIXE ({name}, seed reseau={net_seed}) -- on ne fait varier que la graine d'attaque")
        print("=" * 78)
        header = f"{'graine attaque':>15}{'retrouves/4':>13}{'cos moyen':>12}"
        print(header)
        print("-" * len(header))
        founds = []
        for a_seed in ATTACK_SEEDS:
            found = recover_hidden_layer_hardlabel(net, DIM, target_count=HIDDEN, seed=a_seed)
            rows = match_against_truth(found, net.W[0])
            coss = [c for _, c, _ in rows]
            mean_cos = float(np.mean(coss)) if coss else float("nan")
            founds.append(len(found))
            print(f"{a_seed:>15}{len(found):>10}/4{mean_cos:>12.4f}")
        print(f"\nVariation sur ce seul reseau : {min(founds)}/4 a {max(founds)}/4 selon la seule graine d'attaque.")
        print()
