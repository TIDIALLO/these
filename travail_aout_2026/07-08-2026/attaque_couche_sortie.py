#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Attaque cryptanalytique de la couche de sortie, en hard-label
 Reproduction pedagogique de :
   Canales-Martinez & Santos, LATINCRYPT 2025
   "Extracting Some Layers of Deep Neural Networks in the Hard-Label Setting"
   ePrint 2025/1118, section 3.3
================================================================================

UTILISATION
-----------
  python attaque_couche_sortie.py                 # execution standard
  python attaque_couche_sortie.py --activation gelu
  python attaque_couche_sortie.py --etape 6       # une seule etape
  python attaque_couche_sortie.py --log sortie.txt

DEPENDANCE
----------
  numpy uniquement.

CE QUE FAIT CE SCRIPT
---------------------
  Il recupere la derniere couche d'un reseau de neurones (celle qui produit les
  logits) en n'ayant acces qu'a la CLASSE PREDITE, jamais aux logits.

MODELE DE MENACE
----------------
  L'attaquant voit  : argmax f(x), un entier, pour tout x qu'il choisit.
  L'attaquant sait  : l'architecture, et les couches cachees (hypothese H2).
  L'attaquant cherche : A^(r+1) et b^(r+1).
  L'attaquant obtient : une CLASSE D'EQUIVALENCE, pas les vraies valeurs.

  H2 est une hypothese forte et assumee : ce n'est pas une attaque de bout en
  bout, c'est le dernier maillon d'une chaine dont EUROCRYPT 2025 fournit les
  maillons precedents.
"""

import argparse
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("numpy est requis :  pip install numpy")

np.set_printoptions(precision=4, suppress=True, linewidth=150)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DIM = 24                 # d_0
HIDDEN = [20, 16, 8]           # couches cachees
N_CLASSES = 4                  # d_{r+1}

D_R = HIDDEN[-1]                        # d_r : taille de la derniere couche cachee
N_UNKNOWNS = N_CLASSES * (D_R + 1)      # coefficients de la couche de sortie
N_FIXED = D_R + 2                       # degres de liberte irreductibles
MAX_RANK = N_UNKNOWNS - N_FIXED         # rang maximal theorique

BISECTION_ITERS = 100          # iterations de dichotomie
PROBE_EPS = 1e-7               # ecart des sondes pour identifier le couple
POINTS_FACTOR = 6              # on collecte POINTS_FACTOR x MAX_RANK points
RANK_TOL = 1e-8                # seuil de troncature pour le calcul de rang
N_VALIDATION = 5000            # taille du jeu de validation


# ==============================================================================
# SORTIE : affichage console + fichier de log optionnel
# ==============================================================================

_LOG = None


def say(*args):
    """print() qui ecrit aussi dans le fichier de log si demande."""
    text = " ".join(str(a) for a in args)
    print(text)
    if _LOG is not None:
        _LOG.write(text + "\n")
        _LOG.flush()


def banner(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


# ==============================================================================
# ACTIVATIONS
# ==============================================================================

def relu(x):
    return np.maximum(x, 0.0)


def relu_prime(x):
    return (x > 0).astype(float)


def gelu(x):
    """GELU exacte : x * Phi(x). erf est reimplemente pour eviter scipy."""
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def gelu_prime(x):
    return (0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
            + x * np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi))


def _erf(x):
    """Approximation d'Abramowitz-Stegun 7.1.26, erreur < 1.5e-7.
    Evite la dependance a scipy : le script tourne avec numpy seul."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def tanh_act(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def silu(x):
    return x * _sigmoid(x)


def silu_prime(x):
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


ACTIVATIONS = {
    "relu": (relu, relu_prime),
    "gelu": (gelu, gelu_prime),
    "tanh": (tanh_act, tanh_prime),
    "silu": (silu, silu_prime),
}


# ==============================================================================
# LE RESEAU CIBLE
# ==============================================================================

class MLP:
    """
    Reseau entierement connecte.

        f = f_{r+1} o sigma o f_r o ... o sigma o f_1
        f_i(z) = A^(i) z + b^(i)

    Pas d'activation apres la derniere couche : la sortie est un vecteur de
    logits, qui part ensuite au softmax.
    """

    def __init__(self, dims, activation="relu", seed=0):
        self.dims = dims
        self.activation_name = activation
        self.sigma, self.sigma_prime = ACTIVATIONS[activation]
        self.n_hidden = len(dims) - 2

        rng = np.random.default_rng(seed)
        self.A, self.b = [], []
        for i in range(len(dims) - 1):
            std = np.sqrt(2.0 / dims[i])                      # initialisation He
            self.A.append(rng.normal(0.0, std, (dims[i + 1], dims[i])))
            self.b.append(np.zeros(dims[i + 1]))

    def forward(self, X, upto=None):
        """upto=None -> logits ; upto=k -> sortie post-activation de la couche k."""
        Z = np.atleast_2d(X)
        n_layers = len(self.A) if upto is None else upto
        for i in range(n_layers):
            Z = Z @ self.A[i].T + self.b[i]
            if i < len(self.A) - 1:
                Z = self.sigma(Z)
        return Z

    def hidden_output(self, X):
        """y = f_{1..r}(x). C'est l'hypothese H2 qui rend ceci accessible."""
        return self.forward(X, upto=self.n_hidden)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    def fit(self, X, y, epochs=200, lr=0.05, batch=64, momentum=0.9, seed=0):
        """SGD avec momentum, perte softmax + entropie croisee."""
        rng = np.random.default_rng(seed)
        n_layers = len(self.A)
        vA = [np.zeros_like(a) for a in self.A]
        vb = [np.zeros_like(b) for b in self.b]

        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch):
                idx = order[start:start + batch]
                xb, yb = X[idx], y[idx]

                preacts, acts = [], [xb]
                Z = xb
                for i in range(n_layers):
                    P = Z @ self.A[i].T + self.b[i]
                    preacts.append(P)
                    Z = self.sigma(P) if i < n_layers - 1 else P
                    acts.append(Z)

                shifted = Z - Z.max(axis=1, keepdims=True)
                probs = np.exp(shifted)
                probs /= probs.sum(axis=1, keepdims=True)
                grad = probs
                grad[np.arange(len(yb)), yb] -= 1.0
                grad /= len(yb)

                for i in range(n_layers - 1, -1, -1):
                    gA = grad.T @ acts[i]
                    gb = grad.sum(axis=0)
                    if i > 0:
                        grad = (grad @ self.A[i]) * self.sigma_prime(preacts[i - 1])
                    vA[i] = momentum * vA[i] - lr * gA
                    vb[i] = momentum * vb[i] - lr * gb
                    self.A[i] += vA[i]
                    self.b[i] += vb[i]
        return self


# ==============================================================================
# L'ORACLE : la seule interface de l'attaquant
# ==============================================================================

class HardLabelOracle:
    """
    Renvoie UNIQUEMENT l'argmax des logits.

    Compte les requetes par phase : c'est la metrique qui rend une attaque
    comparable a la litterature, et qui revele ou part reellement le cout.
    """

    def __init__(self, net):
        self.net = net
        self.counts = {}
        self.phase = "init"

    def set_phase(self, name):
        self.phase = name
        self.counts.setdefault(name, 0)

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.counts[self.phase] = self.counts.get(self.phase, 0) + len(X)
        return self.net.predict(X)

    def total(self):
        return sum(self.counts.values())


# ==============================================================================
# DONNEES
# ==============================================================================

def make_data(n=4000, dim=INPUT_DIM, n_classes=N_CLASSES, seed=2):
    """Melange de gaussiennes bien separees."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.4, (n_classes, dim))
    y = rng.integers(0, n_classes, n)
    return centers[y] + rng.normal(0, 0.9, (n, dim)), y


# ==============================================================================
# PRIMITIVES DE L'ATTAQUE
# ==============================================================================

def bisect(oracle, x_a, x_b, iters=BISECTION_ITERS):
    """
    Dichotomie vers un point de transition.

    Invariant maintenu a chaque iteration :
        label(lo) == label(x_a)     et     label(hi) != label(x_a)
    Tant qu'il tient, la frontiere est strictement entre lo et hi.

    Chaque iteration divise ||hi - lo|| par exactement 2.
    Cout : 1 requete par iteration.
    """
    lo, hi = np.array(x_a, float), np.array(x_b, float)
    label_lo = oracle(lo)[0]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if oracle(mid)[0] == label_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def identify_pair(oracle, point, direction, eps=PROBE_EPS):
    """
    Identifie le couple (i, j) de classes qui se rencontrent au point.

    On sonde de part et d'autre le long de la direction de recherche.
    Cout : 2 requetes.

    eps trop petit  -> les deux sondes tombent du meme cote (arrondi).
    eps trop grand  -> la sonde franchit une AUTRE frontiere.
    """
    d = direction / np.linalg.norm(direction)
    return int(oracle(point - eps * d)[0]), int(oracle(point + eps * d)[0])


def build_row(y, class_i, class_j, d_r=D_R, n_unknowns=N_UNKNOWNS):
    """
    Une ligne du systeme, a partir d'un point de transition entre i et j.

        A_i y + b_i = A_j y + b_j
     => (A_i - A_j) y + (b_i - b_j) = 0

    theta est range bloc par bloc :
        theta = [ A_0 (d_r coeffs), b_0, A_1, b_1, ..., A_{k-1}, b_{k-1} ]
    Le bloc de la classe k commence a l'indice k*(d_r + 1).

    Le bloc i recoit (y, 1), le bloc j recoit -(y, 1), le reste est nul.
    """
    row = np.zeros(n_unknowns)
    si = class_i * (d_r + 1)
    sj = class_j * (d_r + 1)
    row[si:si + d_r] += y
    row[si + d_r] += 1.0
    row[sj:sj + d_r] -= y
    row[sj + d_r] -= 1.0
    return row


def sample_point(rng, X_train):
    """
    Echantillonnage diversifie : moitie autour des donnees, moitie a des
    echelles variees. Objectif : explorer des motifs d'activation differents,
    donc obtenir des y qui balaient tout l'espace de dimension d_r.
    """
    if rng.random() < 0.5:
        return X_train[rng.integers(len(X_train))] + rng.normal(0, 1.0, INPUT_DIM)
    return rng.normal(0, rng.choice([0.5, 2.0, 6.0]), INPUT_DIM)


def collect_points(oracle, net, X_train, n_needed, rng, max_attempts=200_000):
    """Collecte des points de transition et assemble le systeme."""
    rows, pairs, attempts, rejected = [], [], 0, 0
    while len(rows) < n_needed and attempts < max_attempts:
        attempts += 1
        x_a = sample_point(rng, X_train)
        x_b = sample_point(rng, X_train)
        if oracle(x_a)[0] == oracle(x_b)[0]:
            continue
        p = bisect(oracle, x_a, x_b)
        ci, cj = identify_pair(oracle, p, x_b - x_a)
        if ci == cj:
            rejected += 1
            continue
        rows.append(build_row(net.hidden_output(p)[0], ci, cj))
        pairs.append((min(ci, cj), max(ci, cj)))
    return np.array(rows), pairs, attempts, rejected


def solve_output_layer(system):
    """
    On fixe exactement d_r + 2 variables, puis on resout le reste.

      - tout le bloc de la classe 0 -> 0   (d_r + 1 variables)
        consomme les d_r translations de colonnes + celle du biais
      - A_{1,1} -> 1                       (1 variable)
        consomme la mise a l'echelle

    Le choix est arbitraire : n'importe quel jeu non degenere de d_r + 2
    variables conviendrait.
    """
    fixed_idx = list(range(0, D_R + 1)) + [D_R + 1]
    fixed_val = np.r_[np.zeros(D_R + 1), 1.0]
    assert len(fixed_idx) == N_FIXED

    free_idx = [k for k in range(N_UNKNOWNS) if k not in fixed_idx]
    rhs = -system[:, fixed_idx] @ fixed_val
    solution, _, rank_reduced, _ = np.linalg.lstsq(system[:, free_idx], rhs,
                                                   rcond=None)

    theta = np.zeros(N_UNKNOWNS)
    theta[fixed_idx] = fixed_val
    theta[free_idx] = solution

    A = np.zeros((N_CLASSES, D_R))
    b = np.zeros(N_CLASSES)
    for k in range(N_CLASSES):
        A[k] = theta[k * (D_R + 1):k * (D_R + 1) + D_R]
        b[k] = theta[k * (D_R + 1) + D_R]
    return A, b, theta, rank_reduced, len(free_idx)


def build_recovered(target, A, b, activation):
    """Reseau reconstruit : couches cachees vraies (H2) + couche de sortie extraite."""
    net = MLP([INPUT_DIM] + HIDDEN + [N_CLASSES], activation=activation, seed=5)
    net.A = [w.copy() for w in target.A]
    net.b = [v.copy() for v in target.b]
    net.A[-1], net.b[-1] = A, b
    return net


# ==============================================================================
# LES ONZE ETAPES
# ==============================================================================

def etape_0(activation, seed):
    banner("ETAPE 0 — Construction du reseau cible")
    X_train, y_train = make_data()
    target = MLP([INPUT_DIM] + HIDDEN + [N_CLASSES],
                 activation=activation, seed=seed).fit(X_train, y_train)

    say(f"Architecture      : {INPUT_DIM} - {' - '.join(map(str, HIDDEN))} - {N_CLASSES}")
    say(f"Activation        : {activation}")
    say(f"Accuracy          : {target.accuracy(X_train, y_train):.3f}")
    say(f"d_r               : {D_R}")
    say(f"d_(r+1)           : {N_CLASSES}")
    say(f"Inconnues         : {N_CLASSES} x {D_R + 1} = {N_UNKNOWNS}")
    say(f"Degres de liberte : d_r + 2 = {N_FIXED}")
    say(f"Rang maximal      : {N_UNKNOWNS} - {N_FIXED} = {MAX_RANK}")
    say("")
    say("La VRAIE couche de sortie (que l'attaquant ne voit pas) :")
    say("A^(r+1) =")
    say(target.A[-1])
    say("b^(r+1) =", target.b[-1])
    return target, X_train, y_train


def etape_1(target):
    banner("ETAPE 1 — L'oracle hard-label")
    oracle = HardLabelOracle(target)
    oracle.set_phase("exploration")
    x_demo = np.full(INPUT_DIM, 0.5)
    say(f"oracle(x) pour x = (0.5, ..., 0.5) : classe {oracle(x_demo)[0]}")
    say("")
    say("L'attaquant NE VOIT PAS les logits. Pour information seulement :")
    say("   ", target.forward(x_demo)[0])
    say("")
    say("Toute la difficulte est la : reconstruire A et b a partir de la seule")
    say("information 'quel indice est le plus grand'.")
    return oracle


def etape_2(target, oracle, rng, X_train):
    banner("ETAPE 2 — Trouver un point de transition")
    oracle.set_phase("recherche_transitions")

    while True:
        x_a = sample_point(rng, X_train)
        x_b = sample_point(rng, X_train)
        if oracle(x_a)[0] != oracle(x_b)[0]:
            break

    say(f"x_a -> classe {oracle(x_a)[0]}")
    say(f"x_b -> classe {oracle(x_b)[0]}")
    say(f"distance initiale ||x_b - x_a|| = {np.linalg.norm(x_b - x_a):.4f}")
    say("")
    say("Largeur de l'intervalle apres k iterations de dichotomie :")

    lo, hi = x_a.copy(), x_b.copy()
    label_lo = oracle(lo)[0]
    for k in range(1, BISECTION_ITERS + 1):
        mid = 0.5 * (lo + hi)
        if oracle(mid)[0] == label_lo:
            lo = mid
        else:
            hi = mid
        if k in (1, 5, 10, 20, 30, 40, 50, 55, 60, 80, 100):
            say(f"   k = {k:3d}   ||hi - lo|| = {np.linalg.norm(hi - lo):.3e}")

    p = 0.5 * (lo + hi)
    logits = target.forward(p)[0]
    order = np.argsort(logits)[::-1]
    say("")
    say("Au point trouve, les logits (invisibles pour l'attaquant) valent :")
    say("   ", logits)
    say(f"Les deux plus grands sont les classes {order[0]} et {order[1]}.")
    say(f"Ecart entre eux : {abs(logits[order[0]] - logits[order[1]]):.3e}")
    say("")
    say("Le plafond a ~2.2e-16 est l'epsilon machine du flottant double (2^-52).")
    say("Au-dela d'environ 55 iterations, chaque requete est gaspillee.")
    return p, x_a, x_b, order


def etape_3(oracle, p, x_a, x_b, order):
    banner("ETAPE 3 — Identifier le couple de classes")
    ci, cj = identify_pair(oracle, p, x_b - x_a)
    say(f"oracle(p - eps*d) = {ci}")
    say(f"oracle(p + eps*d) = {cj}")
    say(f"-> couple identifie : (i, j) = ({ci}, {cj})")
    say(f"   coherent avec les logits : ({order[0]}, {order[1]})")
    say("")
    say("Si les deux sondes renvoient la MEME classe, le point est rejete :")
    say("la dichotomie a converge pres d'un coin ou trois classes se rencontrent,")
    say("ou la precision ne permet plus de separer.")
    return ci, cj


def etape_4(target, p, ci, cj):
    banner("ETAPE 4 — Construire une ligne du systeme")
    y_vec = target.hidden_output(p)[0]
    say("y = f_(1..r)(p) — connu par hypothese H2 :")
    say("   ", y_vec)
    say("")
    say("     A_i y + b_i = A_j y + b_j")
    say("  => (A_i - A_j) y + (b_i - b_j) = 0")
    say("")
    say(f"theta est range bloc par bloc ; le bloc de la classe k commence")
    say(f"a l'indice k*(d_r+1) = k*{D_R + 1}. Total : {N_UNKNOWNS} inconnues.")
    say("")
    row = build_row(y_vec, ci, cj)
    say(f"Ligne obtenue (classe {ci} en +, classe {cj} en -) :")
    for k in range(N_CLASSES):
        say(f"   bloc classe {k} : {row[k * (D_R + 1):(k + 1) * (D_R + 1)]}")

    theta_true = np.concatenate([np.r_[target.A[-1][k], target.b[-1][k]]
                                 for k in range(N_CLASSES)])
    say("")
    say(f"Verification : row . theta_vrai = {row @ theta_true:.3e}")
    say("(doit etre ~0 : la vraie couche de sortie satisfait bien l'equation)")
    say("")
    say("FAIS TOUJOURS CETTE VERIFICATION. Elle attrape les erreurs d'indice")
    say("et de signe en une ligne, avant qu'elles ne coutent une journee.")
    return theta_true


def etape_5(target, oracle, X_train, rng):
    banner("ETAPE 5 — Collecter les points et assembler le systeme")
    oracle.set_phase("recherche_transitions")
    n_points = POINTS_FACTOR * MAX_RANK
    t0 = time.time()
    system, pairs, attempts, rejected = collect_points(
        oracle, target, X_train, n_points, rng)
    dt = time.time() - t0

    from collections import Counter
    say(f"Points collectes  : {len(system)}")
    say(f"Tirages tentes    : {attempts}")
    say(f"Points rejetes    : {rejected}")
    say(f"Systeme           : matrice {system.shape[0]} x {system.shape[1]}")
    say(f"Duree             : {dt:.1f} s")
    say(f"Couples rencontres: {dict(Counter(pairs))}")
    say("")
    say("Si un couple de classes n'apparait jamais, les equations qui le")
    say("concernent manquent, et le systeme perd du rang.")
    return system


def etape_6(system, theta_true, activation, seed):
    banner("ETAPE 6 — Les d_r + 2 degres de liberte")
    rank = np.linalg.matrix_rank(system, tol=RANK_TOL)
    say(f"rang(systeme)          = {rank}")
    say(f"rang maximal theorique = {MAX_RANK}")
    say(f"dimension du noyau     = {N_UNKNOWNS} - {rank} = {N_UNKNOWNS - rank}"
        f"   (attendu : {N_FIXED})")
    say("")
    say("Les vecteurs du noyau sont exactement les transformations qui laissent")
    say("le reseau equivalent. Verification sur les generateurs de l'article :")
    say("")

    generators = []
    for c in range(D_R):
        v = np.zeros(N_UNKNOWNS)
        for k in range(N_CLASSES):
            v[k * (D_R + 1) + c] = 1.0
        generators.append((f"translation colonne {c} de A", v))
    v = np.zeros(N_UNKNOWNS)
    for k in range(N_CLASSES):
        v[k * (D_R + 1) + D_R] = 1.0
    generators.append(("translation du biais b", v))
    generators.append(("mise a l'echelle de (A, b)", theta_true.copy()))

    for name, v in generators:
        say(f"   {name:<32} ||systeme . v|| = {np.linalg.norm(system @ v):.2e}")

    say("")
    say(f"{len(generators)} generateurs, tous dans le noyau. d_r + 2 = {N_FIXED}. CQFD.")
    say("")
    say("Le dernier generateur n'est pas exactement nul : c'est theta_vrai lui-meme,")
    say("et les lignes ne sont satisfaites qu'a la precision des points de")
    say("transition. Les autres sont des identites structurelles, exactes.")
    return rank


def diagnostic_rang(activation, seed):
    """Compare le rang atteint sur trois architectures. Reentraine 2 reseaux."""
    say("")
    say("-" * 78)
    say("DIAGNOSTIC — quand le rang maximal n'est PAS atteint")
    say("-" * 78)

    def run(input_dim, hidden, n_classes, n_points):
        d_r = hidden[-1]
        n_unknowns = n_classes * (d_r + 1)
        max_rank = n_unknowns - (d_r + 2)
        Xtr, ytr = make_data(n=4000, dim=input_dim, n_classes=n_classes)
        net = MLP([input_dim] + hidden + [n_classes],
                  activation=activation, seed=seed).fit(Xtr, ytr)
        o = HardLabelOracle(net)
        o.set_phase("diag")
        r = np.random.default_rng(11)
        rows, Ys, pairs, tries = [], [], set(), 0
        while len(rows) < n_points and tries < 120_000:
            tries += 1
            a = Xtr[r.integers(len(Xtr))] + r.normal(0, 1.0, input_dim)
            b = r.normal(0, r.choice([0.5, 2.0, 6.0]), input_dim)
            if o(a)[0] == o(b)[0]:
                continue
            p = bisect(o, a, b)
            d = (b - a) / np.linalg.norm(b - a)
            ci, cj = int(o(p - PROBE_EPS * d)[0]), int(o(p + PROBE_EPS * d)[0])
            if ci == cj:
                continue
            yv = net.hidden_output(p)[0]
            Ys.append(yv)
            pairs.add((min(ci, cj), max(ci, cj)))
            rows.append(build_row(yv, ci, cj, d_r=d_r, n_unknowns=n_unknowns))
        S, Y = np.array(rows), np.array(Ys)
        return (np.linalg.matrix_rank(S, tol=RANK_TOL), max_rank,
                np.linalg.matrix_rank(Y, tol=RANK_TOL), d_r,
                len(pairs), n_classes * (n_classes - 1) // 2)

    say(f"{'architecture':<26} {'rang':>9} {'rang(Y)':>9} {'paires vues':>13}")
    say("-" * 62)
    configs = {
        "10 - 8 - 6 - 5 - 3":   (10, [8, 6, 5], 3),
        "12 - 10 - 8 - 5 - 4":  (12, [10, 8, 5], 4),
        f"{INPUT_DIM} - {' - '.join(map(str, HIDDEN))} - {N_CLASSES}":
            (INPUT_DIM, HIDDEN, N_CLASSES),
    }
    for label, (idim, hid, ncl) in configs.items():
        mr = ncl * (hid[-1] + 1) - hid[-1] - 2
        rk, mrk, rky, dr, np_, mp = run(idim, hid, ncl, POINTS_FACTOR * mr)
        say(f"{label:<26} {f'{rk}/{mrk}':>9} {f'{rky}/{dr}':>9} {f'{np_}/{mp}':>13}")

    say("")
    say("Sur les reseaux trop petits, le rang maximal n'est PAS atteint, et ce")
    say("n'est pas une question de nombre de points : c'est structurel.")
    say("")
    say("  1. PARCIMONIE. Aux points de transition, une partie des neurones de la")
    say("     derniere couche cachee est systematiquement inactive. y ne balaie")
    say("     qu'un sous-espace de dimension < d_r, et les lignes sont liees.")
    say("     Symptome : rang(Y) < d_r.")
    say("  2. ADJACENCE DES CLASSES. Si deux classes ne se touchent nulle part,")
    say("     aucun point de transition ne porte ce couple.")
    say("     Symptome : paires vues < d_(r+1)(d_(r+1)-1)/2.")
    say("")
    say("CONSEQUENCE : d_(r+1)(d_r + 1) - (d_r + 2) est une BORNE SUPERIEURE.")
    say("Toujours verifier le rang atteint AVANT de resoudre. lstsq ne previent")
    say("pas : il renvoie une solution de norme minimale qui n'est pas la bonne.")


def etape_7(target, system):
    banner("ETAPE 7 — Fixer d_r + 2 variables, puis resoudre")
    say(f"On fixe {N_FIXED} variables :")
    say(f"   indices 0..{D_R}  -> 0   (bloc de la classe 0 : A_0 = 0, b_0 = 0)")
    say(f"   indice  {D_R + 1}      -> 1   (A_(1,1) = 1, fixe l'echelle)")
    say(f"   soit (d_r + 1) + 1 = {N_FIXED}. Exactement le nombre de degres de liberte.")

    t0 = time.time()
    A_hat, b_hat, theta_hat, rank_reduced, n_free = solve_output_layer(system)
    dt = time.time() - t0

    say("")
    say(f"rang du systeme reduit : {rank_reduced} / {n_free}   (doit etre plein)")
    say(f"residu max |systeme . theta_hat| = {np.abs(system @ theta_hat).max():.3e}")
    say(f"duree de la resolution : {dt * 1000:.1f} ms")
    say("")
    say("A_hat =")
    say(A_hat)
    say("b_hat =", b_hat)

    A_norm = target.A[-1] - target.A[-1][0]
    b_norm = target.b[-1] - target.b[-1][0]
    scale = A_norm[1, 0]
    A_norm, b_norm = A_norm / scale, b_norm / scale
    say("")
    say("Comparaison avec la verite terrain, APRES normalisation identique")
    say("(memes conventions : bloc 0 a zero, A_(1,1) = 1) :")
    say(f"ecart max |A_hat - A_vrai_normalise| = {np.abs(A_hat - A_norm).max():.3e}")
    say("")
    say("NE COMPARE JAMAIS A_hat directement a A : ils vivent dans la meme")
    say("classe d'equivalence, mais pas au meme representant.")
    return A_hat, b_hat, scale


def etape_8(target, oracle, A_hat, b_hat, scale, rng, activation, X_test, labels_true):
    banner("ETAPE 8 — L'ambiguite de signe global")
    say(f"On a impose A_(1,1) = 1, donc implicitement c = 1 / A_(1,1)_vrai.")
    say(f"A_(1,1) vrai (apres translation du bloc 0) = {scale:.6f}")
    say(f"-> c = {1 / scale:.6f}   ({'POSITIF' if scale > 0 else 'NEGATIF'})")
    say("")
    if scale < 0:
        say("c < 0 : argmax(c*u) = argmin(u). Le reseau reconstruit classe A L'ENVERS.")
    else:
        say("c > 0 : pas d'inversion dans ce cas precis.")

    before = int((build_recovered(target, A_hat, b_hat, activation).predict(X_test)
                  == labels_true).sum())
    say("")
    say(f"Accord AVANT correction : {before} / {N_VALIDATION}"
        f"   ({100 * before / N_VALIDATION:.1f} %)")
    say(f"Hasard sur {N_CLASSES} classes : {100 / N_CLASSES:.1f} %")

    if before < N_VALIDATION * 0.5:
        say("")
        say("Un score TRES INFERIEUR au hasard n'est jamais un echec de la methode.")
        say("C'est une convention de signe inversee. Correction : UNE requete.")

    oracle.set_phase("correction_signe")
    x_probe = rng.normal(0, 1.0, INPUT_DIM)
    y_probe = target.hidden_output(x_probe)[0]
    if np.argmax(A_hat @ y_probe + b_hat) != oracle(x_probe)[0]:
        A_hat, b_hat = -A_hat, -b_hat
        say("")
        say("Une requete a suffi : signes inverses.")
    else:
        say("")
        say("Une requete a suffi : aucune inversion necessaire.")
    return A_hat, b_hat


def etape_9(target, A_hat, b_hat, activation, rng, X_test, labels_true):
    banner("ETAPE 9 — Validation de l'equivalence fonctionnelle")
    recovered = build_recovered(target, A_hat, b_hat, activation)
    a1 = int((recovered.predict(X_test) == labels_true).sum())
    say(f"Accord sur {N_VALIDATION} entrees uniformes   : {a1} / {N_VALIDATION}")

    X2 = rng.normal(0, 2.0, (N_VALIDATION, INPUT_DIM))
    a2 = int((recovered.predict(X2) == target.predict(X2)).sum())
    say(f"Accord sur {N_VALIDATION} entrees gaussiennes : {a2} / {N_VALIDATION}")
    say("")
    say("Tester sur DEUX distributions n'est pas redondant : une equivalence qui")
    say("ne tiendrait que sur la distribution d'entrainement serait locale, pas")
    say("fonctionnelle.")
    say("")
    say("Les matrices A_hat et A different — on a recupere une CLASSE")
    say("D'EQUIVALENCE, pas theta. Mais les argmax coincident partout.")
    return a1, a2


def etape_10(oracle):
    banner("ETAPE 10 — Repartition du cout en requetes")
    total = oracle.total()
    for phase, n in sorted(oracle.counts.items(), key=lambda kv: -kv[1]):
        say(f"   {phase:<26} {n:8d}   {100 * n / total:5.1f} %")
    say(f"   {'TOTAL':<26} {total:8d}")
    say("")
    say("Resolution du systeme lineaire : 0 requete, quelques millisecondes.")
    say("")
    say("-> Presque 100 % du cout est dans la RECHERCHE DES POINTS DE TRANSITION.")
    say("   L'article rapporte 6 h 30 de recherche contre une fraction de seconde")
    say("   de resolution sur CIFAR-10.")
    say("")
    say("   Cote attaque : c'est trivialement parallelisable.")
    say("   Cote defense : c'est LA cible. L'attaque a besoin de 10^-13, une")
    say("   contrainte extraordinairement fragile.")


# ==============================================================================
# PROGRAMME PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Attaque de la couche de sortie en hard-label.")
    parser.add_argument("--activation", default="relu",
                        choices=sorted(ACTIVATIONS),
                        help="activation des couches cachees (defaut : relu)")
    parser.add_argument("--seed", type=int, default=5,
                        help="graine du reseau cible (defaut : 5)")
    parser.add_argument("--etape", type=int, default=None,
                        help="n'executer que jusqu'a cette etape (0-10)")
    parser.add_argument("--sans-diagnostic", action="store_true",
                        help="sauter le diagnostic de l'etape 6 (~4 min)")
    parser.add_argument("--log", default=None,
                        help="ecrire aussi la sortie dans ce fichier")
    args = parser.parse_args()

    global _LOG
    if args.log:
        _LOG = open(args.log, "w", encoding="utf-8")

    stop = args.etape if args.etape is not None else 10
    t_start = time.time()
    try:
        run_attack(args, stop, t_start)
    finally:
        if _LOG is not None:
            if args.log:
                print(f"Sortie ecrite dans : {args.log}")
            _LOG.close()


def run_attack(args, stop, t_start):

    target, X_train, y_train = etape_0(args.activation, args.seed)
    if stop == 0:
        return

    oracle = etape_1(target)
    if stop == 1:
        return

    rng = np.random.default_rng(11)
    p, x_a, x_b, order = etape_2(target, oracle, rng, X_train)
    if stop == 2:
        return

    ci, cj = etape_3(oracle, p, x_a, x_b, order)
    if stop == 3:
        return

    theta_true = etape_4(target, p, ci, cj)
    if stop == 4:
        return

    system = etape_5(target, oracle, X_train, rng)
    if stop == 5:
        return

    etape_6(system, theta_true, args.activation, args.seed)
    if not args.sans_diagnostic:
        diagnostic_rang(args.activation, args.seed)
    if stop == 6:
        return

    A_hat, b_hat, scale = etape_7(target, system)
    if stop == 7:
        return

    X_test = rng.uniform(-1, 1, (N_VALIDATION, INPUT_DIM))
    labels_true = target.predict(X_test)
    A_hat, b_hat = etape_8(target, oracle, A_hat, b_hat, scale, rng,
                           args.activation, X_test, labels_true)
    if stop == 8:
        return

    etape_9(target, A_hat, b_hat, args.activation, rng, X_test, labels_true)
    if stop == 9:
        return

    etape_10(oracle)

    banner("FIN")
    say(f"Duree totale : {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
