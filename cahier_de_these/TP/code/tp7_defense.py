"""
TP7 — DEFENSE "Train to Defend" : casser l'unicite des neurones (fiche 13).

Insight de Kurian & Aysu (NeurIPS 2025) : l'attaque recupere d'autant mieux les
poids que les neurones d'une couche sont DISTINCTS. Defense = regulariser pour
RAPPROCHER les poids des neurones d'une meme couche (similarite intra-couche).

Ce TP illustre le MECANISME sans entrainement complet : on compare l'attaque de
signature (TP3) sur (1) un reseau normal vs (2) un reseau "defendu" dont les
lignes de W1 sont rendues quasi-identiques. On observe que la defense empeche de
SEPARER les neurones (leurs signatures deviennent colineaires).

Lance :  python3 tp7_defense.py
"""

import numpy as np
from common import MLP, cosine
from tp3_signature import find_one_critical_point, recover_signature_at


def make_defended(net, similarity=0.95):
    """Rapproche les lignes de W1 d'une direction commune (proxy de la
    regularisation de similarite intra-couche de la fiche 13)."""
    W1 = net.W[0].copy()
    common = W1.mean(axis=0)
    common /= np.linalg.norm(common)
    for i in range(W1.shape[0]):
        norm = np.linalg.norm(W1[i])
        dir_i = W1[i] / norm
        # melange : 'similarity' vers la direction commune
        mixed = similarity * common + (1 - similarity) * dir_i
        net.W[0][i] = mixed / np.linalg.norm(mixed) * norm
    return net


def attack_count_distinct(net, dim, hidden, seed=11, sep_threshold=0.999):
    """Lance la recuperation de signatures (TP3) et compte les directions
    DISTINCTES trouvees. Renvoie aussi la separation min entre vraies lignes."""
    rng = np.random.default_rng(seed)
    found = []
    attempts = 0
    while len(found) < hidden and attempts < 80:
        attempts += 1
        x0 = rng.standard_normal(dim)
        d = rng.standard_normal(dim); d /= np.linalg.norm(d)
        ts = np.linspace(-3, 3, 400)
        slopes = []
        for t in ts:
            x = x0 + t * d; hh = 1e-6
            slopes.append((net.forward(x+hh*d, count=True)[0,0] -
                           net.forward(x-hh*d, count=True)[0,0])/(2*hh))
        slopes = np.array(slopes)
        for ci in np.where(np.abs(np.diff(slopes)) > 1e-4)[0]:
            t_star = find_one_critical_point(net, x0, d, ts[ci], ts[ci+1])
            sig = recover_signature_at(net, x0 + t_star*d, d)
            if all(abs(cosine(sig, f)) < sep_threshold for f in found):
                found.append(sig)
    # separation entre vraies lignes (|cos| max entre paires : 1 = identiques)
    W1 = net.W[0]
    worst = 0.0
    for i in range(hidden):
        for j in range(i+1, hidden):
            worst = max(worst, abs(cosine(W1[i], W1[j])))
    return len(found), worst


if __name__ == "__main__":
    dim, hidden = 4, 5
    print("Comparaison ATTAQUE (TP3) : reseau normal vs reseau defendu")
    print("-" * 64)

    net_normal = MLP([dim, hidden, 2], activation="relu", seed=3)
    nd, sep_n = attack_count_distinct(net_normal, dim, hidden)
    print(f"NORMAL   : neurones separes recuperes = {nd}/{hidden} ; "
          f"|cos| max entre vraies lignes = {sep_n:.3f}")

    net_def = make_defended(MLP([dim, hidden, 2], activation="relu", seed=3),
                            similarity=0.97)
    dd, sep_d = attack_count_distinct(net_def, dim, hidden)
    print(f"DEFENDU  : neurones separes recuperes = {dd}/{hidden} ; "
          f"|cos| max entre vraies lignes = {sep_d:.3f}")
    print()
    print("=> Quand les neurones se ressemblent (|cos| -> 1), l'attaque ne peut")
    print("   plus les SEPARER : elle recupere moins de signatures distinctes.")
    print("   C'est exactement le levier de la defense 'Train to Defend' (f.13).")
    print()
    print("PISTE THESE (idee B6/B7) : tester cette defense avec activation non-ReLU")
    print("(remplace 'relu' ci-dessus par 'prelu', 'gelu'...). Garde-t-elle son")
    print("efficacite ? Sinon, concevoir une defense propre au cas LISSE (B7).")
    print()
    print("NOTE : ceci illustre le MECANISME. La vraie defense REGULARISE a")
    print("l'ENTRAINEMENT (terme de perte) pour obtenir cette similarite SANS")
    print("perdre en precision (<1%). Reproduire cela en PyTorch = TP7-bis.")
