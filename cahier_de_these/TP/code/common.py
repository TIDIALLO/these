"""
common.py — Boîte à outils partagée pour les TP d'extraction de DNN.

Un petit MLP en NumPy (float64) avec activation paramétrable, et deux oracles :
  - raw-output (logits)         -> ce que voient les attaques f.03-05, f.10-11
  - hard-label  (argmax)        -> ce que voient les attaques f.06-09 (cadre de ta thèse)

Tout est volontairement simple et lisible : c'est un support pédagogique, pas
une implémentation optimisée. On travaille en float64 pour des mesures propres.

Auteur : cahier de thèse — 2026.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Fonctions d'activation (et leurs dérivées quand utile)
# ----------------------------------------------------------------------------

def relu(z):
    return np.maximum(0.0, z)

def leaky_relu(z, alpha=0.1):
    return np.where(z >= 0, z, alpha * z)

def prelu(z, alpha=0.25):
    return np.where(z >= 0, z, alpha * z)

def elu(z, alpha=1.0):
    return np.where(z >= 0, z, alpha * (np.exp(np.minimum(z, 0)) - 1.0))

def hardtanh(z):
    return np.clip(z, -1.0, 1.0)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def gelu(z):
    # approximation tanh de GELU (suffit pour la pédagogie)
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))

def silu(z):
    return z * sigmoid(z)

ACTIVATIONS = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "prelu": prelu,
    "elu": elu,
    "hardtanh": hardtanh,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "gelu": gelu,
    "silu": silu,
}

# Familles (utile pour ta taxonomie, idée B3)
PIECEWISE_LINEAR = {"relu", "leaky_relu", "prelu", "hardtanh"}
SMOOTH = {"sigmoid", "tanh", "gelu", "silu"}


# ----------------------------------------------------------------------------
# Le MLP cible (la "boîte noire" à attaquer)
# ----------------------------------------------------------------------------

class MLP:
    """MLP entièrement connecté. La couche de sortie est SANS activation
    (comme dans la réalité et dans les articles)."""

    def __init__(self, layer_sizes, activation="relu", seed=0, scale=1.0):
        """
        layer_sizes : ex. [2, 3, 3, 2]  (entree=2, deux couches cachees, sortie=2)
        activation  : nom dans ACTIVATIONS
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.sigma = ACTIVATIONS[activation]
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # init type Xavier, en float64
            w = rng.standard_normal((n_out, n_in)) * (scale / np.sqrt(n_in))
            self.W.append(w.astype(np.float64))
            self.b.append(rng.standard_normal(n_out).astype(np.float64) * 0.1)
        self.n_queries = 0  # compteur de requetes (metrique cle)

    # --- propagation avant ---
    def forward(self, x, count=True):
        """Retourne les logits (raw-output). x : vecteur (n_in,) ou batch (N,n_in)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if count:
            self.n_queries += x.shape[0]
        a = x
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i].T + self.b[i]
            if i < L - 1:            # pas d'activation sur la couche de sortie
                a = self.sigma(z)
            else:
                a = z
        return a

    # --- les deux oracles ---
    def oracle_raw(self, x):
        """Oracle RAW-OUTPUT : renvoie les logits."""
        out = self.forward(x)
        return out[0] if out.shape[0] == 1 else out

    def oracle_label(self, x):
        """Oracle HARD-LABEL : renvoie seulement l'argmax (la classe)."""
        out = self.forward(x)
        lab = np.argmax(out, axis=1)
        return int(lab[0]) if lab.shape[0] == 1 else lab

    # --- pre-activations d'une couche (pour la VERIFICATION, pas pour l'attaque) ---
    def preactivations(self, x, layer):
        """z de la couche `layer` (0-indexe). Sert uniquement a verifier les
        attaques en "boite blanche" — un attaquant n'y a pas acces."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        a = x
        for i in range(layer):
            z = a @ self.W[i].T + self.b[i]
            a = self.sigma(z)
        return a @ self.W[layer].T + self.b[layer]


# ----------------------------------------------------------------------------
# Utilitaires de mesure (les 3 metriques a TOUJOURS rapporter)
# ----------------------------------------------------------------------------

def fidelity_error(model_a, model_b, n=2000, dim=None, seed=123):
    """Erreur de fidelite : ecart max des logits sur des entrees aleatoires."""
    dim = dim or model_a.layer_sizes[0]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    A = model_a.forward(X, count=False)
    B = model_b.forward(X, count=False)
    return float(np.max(np.abs(A - B)))


def cosine(u, v):
    """Cosinus entre deux vecteurs (compare des signatures a un facteur pres)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


if __name__ == "__main__":
    # demonstration rapide
    net = MLP([2, 3, 2], activation="relu", seed=1)
    x = np.array([0.3, -0.7])
    print("logits (raw-output) :", net.oracle_raw(x))
    print("label  (hard-label) :", net.oracle_label(x))
    print("nb de requetes       :", net.n_queries)
