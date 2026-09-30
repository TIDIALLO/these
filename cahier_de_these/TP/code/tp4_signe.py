"""
TP4 — Le probleme du SIGNE (Carlini 2020 -> Canales-Martinez 2024, fiches 03-04).

TP3 recupere la DIRECTION des poids du neurone, mais pas son SIGNE : le ReLU
verifie  max(0, w.x+b)  et la signature est definie a un facteur pres. Or
ReLU(w.x+b) != ReLU(-w.x-b) : il faut donc trancher le signe de chaque neurone.

Pour ReLU, distinguer le signe demande de l'information SUPPLEMENTAIRE (le cote
"actif" du neurone). Ce TP montre POURQUOI le signe est ambigu et illustre une
idee de recuperation : observer de quel cote du coude la sortie "reagit".

Pour ta these : avec PReLU/ELU (fiches 10,12) la partie negative n'est PAS nulle,
donc le neurone "fuit" de l'information des DEUX cotes -> le signe est souvent
PLUS FACILE a recuperer que pour ReLU. TP6 illustre ce point.

Lance :  python3 tp4_signe.py
"""

import numpy as np
from common import MLP


def side_activity(net, x_crit, direction, eps=1e-3):
    """Mesure combien la sortie 'bouge' de chaque cote du point critique.
    Pour ReLU, le cote ACTIF (w.x+b>0) fait varier la sortie ; le cote inactif
    la laisse ~constante. Le cote actif revele le signe de (w,b)."""
    g0 = net.forward(x_crit, count=True)[0, 0]
    g_plus = net.forward(x_crit + eps * direction, count=True)[0, 0]
    g_minus = net.forward(x_crit - eps * direction, count=True)[0, 0]
    var_plus = abs(g_plus - g0)
    var_minus = abs(g_minus - g0)
    return var_plus, var_minus


if __name__ == "__main__":
    print("Demonstration de l'ambiguite de signe sur un neurone ReLU isole.")
    print("-" * 64)

    # un seul neurone : f(x) = ReLU(w.x + b), w connu en direction seulement
    w = np.array([1.5, -2.0]); b = 0.3

    def f(x):
        return max(0.0, w @ x + b)

    # point critique : w.x + b = 0
    # on prend un x sur l'hyperplan
    x_crit = np.array([-b / w[0], 0.0])  # w0*x0 + b = 0
    print(f"Vrai neurone : w = {w}, b = {b}")
    print(f"Signature (direction) recuperee en TP3 : +-{(w/np.linalg.norm(w)).round(3)}")
    print()

    d = w / np.linalg.norm(w)            # direction normale a l'hyperplan
    eps = 1e-2
    plus = f(x_crit + eps * d) - f(x_crit)
    minus = f(x_crit - eps * d) - f(x_crit)
    print(f"Variation de sortie cote +d : {plus:+.5f}")
    print(f"Variation de sortie cote -d : {minus:+.5f}")
    print()
    print("=> Un seul cote fait reagir le ReLU (le cote actif w.x+b>0).")
    print("   Ce cote ACTIF donne le SIGNE de w. Mais dans un reseau profond,")
    print("   le neurone est cache derriere d'autres ReLU : lire ce signe")
    print("   devient difficile -> Carlini 2020 le fait en temps EXPONENTIEL,")
    print("   Canales-Martinez 2024 (fiche 04) le rend POLYNOMIAL.")
    print()
    print("PISTE THESE (idee B1) : avec une activation a fuite (Leaky/PReLU/ELU),")
    print("les DEUX cotes reagissent -> compare avec TP6.")
