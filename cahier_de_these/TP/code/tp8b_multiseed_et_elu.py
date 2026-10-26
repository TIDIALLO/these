"""
TP8b — Suite du TP8 : robustesse multi-graines + diagnostic de l'echec ELU.

Deux questions laissees ouvertes par TP8 (voir sa note de complexite) :

  1. Les resultats de recover_hidden_layer_hardlabel ne sont mesures que sur
     UNE graine reseau. Varient-ils beaucoup d'une graine a l'autre, en
     particulier pour PReLU dont la variance etait deja suspectee grande ?
  2. Pourquoi ELU echoue-t-il TOUJOURS (0/4) ? Est-ce que find_seed_kink
     ne trouve meme pas de point de frontiere, ou bien trouve-t-il des
     points mais jamais de pli net dessus ? Les deux diagnostics appellent
     une correction tres differente.

Lance :  python3 tp8b_multiseed_et_elu.py
"""

import time
import numpy as np
from common import MLP, cosine
from tp5_hardlabel import find_transition
from tp8_hardlabel_beyond_relu import (
    recover_hidden_layer_hardlabel, match_against_truth,
    sweep_kink, _random_unit, _random_orthogonal,
)

NETWORK_SEEDS = [7, 11, 23, 42, 100]
DIM, HIDDEN, OUT = 3, 4, 2


def run_multiseed(dim=DIM, hidden=HIDDEN, out=OUT):
    print("=" * 78)
    print("1. ROBUSTESSE MULTI-GRAINES (5 graines reseau, meme graine d'attaque)")
    print("=" * 78)
    header = f"{'activation':<12}{'graine':>8}{'trouves':>9}{'cos moyen':>11}{'requetes':>11}{'temps(s)':>10}"
    print(header)
    print("-" * len(header))

    summary = {}
    for name in ["relu", "leaky_relu", "prelu", "elu"]:
        founds, costs = [], []
        for seed in NETWORK_SEEDS:
            net = MLP([dim, hidden, out], activation=name, seed=seed)
            t0 = time.time()
            found = recover_hidden_layer_hardlabel(net, dim, target_count=hidden, seed=3)
            dt = time.time() - t0
            rows = match_against_truth(found, net.W[0])
            coss = [c for _, c, _ in rows]
            mean_cos = float(np.mean(coss)) if coss else float("nan")
            founds.append(len(found))
            costs.append(mean_cos)
            print(f"{name:<12}{seed:>8}{len(found):>6}/{hidden:<2}{mean_cos:>11.4f}{net.n_queries:>11}{dt:>10.1f}")
        summary[name] = (founds, costs)
        print("-" * len(header))

    print()
    print("RESUME (moyenne +/- ecart-type sur les 5 graines)")
    print("-" * len(header))
    for name, (founds, costs) in summary.items():
        valid_cos = [c for c in costs if not np.isnan(c)]
        f_mean, f_std = np.mean(founds), np.std(founds)
        c_mean = np.mean(valid_cos) if valid_cos else float("nan")
        c_std = np.std(valid_cos) if valid_cos else float("nan")
        print(f"{name:<12}  trouves = {f_mean:.1f} +/- {f_std:.1f} / {HIDDEN}   "
              f"cos = {c_mean:.4f} +/- {c_std:.4f}")
    print()


def diagnose_elu(dim=DIM, hidden=HIDDEN, out=OUT, n_network_seeds=5, max_tries=40):
    """Reproduit la logique interne de find_seed_kink mais compte separement
    les echecs 'pas de point de frontiere' et 'point trouve mais pas de pli'."""
    print("=" * 78)
    print("2. DIAGNOSTIC ELU : ou est-ce que ca casse exactement ?")
    print("=" * 78)
    header = f"{'graine':>8}{'essais':>8}{'frontiere trouvee':>20}{'pli detecte':>14}"
    print(header)
    print("-" * len(header))

    for seed in NETWORK_SEEDS[:n_network_seeds]:
        net = MLP([dim, hidden, out], activation="elu", seed=seed)
        rng = np.random.default_rng(3)
        n_boundary, n_kink = 0, 0
        for _ in range(max_tries):
            x_a, x_b = rng.standard_normal(dim) * 2, rng.standard_normal(dim) * 2
            p0 = find_transition(net, x_a, x_b)
            if p0 is None:
                continue
            n_boundary += 1
            d_line = _random_unit(rng, dim)
            d_offset = _random_orthogonal(rng, dim, d_line)
            if d_offset is None:
                continue
            res = sweep_kink(net, p0, d_line, d_offset, radius=0.5, s_range=2.5)
            if res is not None:
                n_kink += 1
        print(f"{seed:>8}{max_tries:>8}{n_boundary:>20}{n_kink:>14}")

    print()
    print("Lecture : si 'frontiere trouvee' est deja bas, le probleme est que la")
    print("plupart des segments aleatoires ne croisent pas la frontiere de decision")
    print("(improbable ici, ReLU/Leaky/PReLU n'ont pas ce souci avec les memes reglages).")
    print("Si 'frontiere trouvee' est haut mais 'pli detecte' est bas, le probleme est")
    print("que la frontiere ELU, meme quand on la trouve, n'a presque jamais le coude")
    print("net que sweep_kink cherche -- coherent avec le fait que la partie negative")
    print("d'ELU est courbe, pas plate, donc le pli y est etale au lieu d'etre ponctuel.")
    print()


if __name__ == "__main__":
    t0 = time.time()
    run_multiseed()
    diagnose_elu()
    print(f"Temps total : {time.time() - t0:.1f} s")
