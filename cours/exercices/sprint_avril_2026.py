"""
Sprint Avril 2026 — Exercices Pratiques
Tidiane DIALLO — Thèse EPT

INSTRUCTIONS:
- Exécuter avec: python sprint_avril_2026.py
- Compléter les sections marquées # TODO
- Chaque exercice est indépendant (peut être testé séparément)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EXERCICE 1 — JOUR 1: Matrices (ne pas modifier)
# ============================================================
print("=" * 60)
print("EXERCICE 1: Matrices et produit matriciel")
print("=" * 60)

# Données
x = np.array([1, 2, 3])           # vecteur d'entrée
W = np.array([[1, 0, 2],          # matrice de poids
              [3, 1, 0]])

# Calcul
resultat = np.dot(W, x)
print(f"x = {x}")
print(f"W =\n{W}")
print(f"W·x = {resultat}")        # Attendu: [7, 5]
print()

# Simuler une couche de réseau
x_image = np.random.randn(784)    # 784 pixels
W_couche = np.random.randn(100, 784)
sortie = np.dot(W_couche, x_image)
print(f"Simulation couche DNN: {x_image.shape} → {sortie.shape}")
print()

# ============================================================
# EXERCICE 2 — JOUR 2: Descente de gradient
# ============================================================
print("=" * 60)
print("EXERCICE 2: Descente de gradient")
print("=" * 60)

def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

x_init = 5.0
lr = 0.1
x_courant = x_init
historique_x = [x_courant]
historique_f = [f(x_courant)]

for i in range(25):
    gradient = df(x_courant)
    x_courant = x_courant - lr * gradient
    historique_x.append(x_courant)
    historique_f.append(f(x_courant))

print(f"Point de départ: x={x_init:.2f}, f(x)={f(x_init):.2f}")
print(f"Point final: x={x_courant:.4f}, f(x)={f(x_courant):.6f}")
print(f"Minimum théorique: x=-1.0, f(x)=0.0")
print()

# ============================================================
# EXERCICE 3 — JOUR 3: Softmax et Cross-Entropy
# ============================================================
print("=" * 60)
print("EXERCICE 3: Softmax et fonction de perte")
print("=" * 60)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()

def cross_entropy(y_vrai, y_pred):
    """y_vrai: one-hot, y_pred: probabilités"""
    return -np.sum(y_vrai * np.log(y_pred + 1e-10))

# Exemple: classification 3 classes
scores_bon = np.array([3.0, 1.0, 0.5])   # bonne prédiction
scores_mauvais = np.array([0.5, 1.0, 3.0])  # mauvaise prédiction
y_vrai = np.array([1, 0, 0])              # vraie classe: 0

p_bon = softmax(scores_bon)
p_mauvais = softmax(scores_mauvais)

print(f"Bonne prédiction → softmax: {p_bon.round(3)}")
print(f"  Loss: {cross_entropy(y_vrai, p_bon):.4f} (faible = bien)")
print(f"Mauvaise prédiction → softmax: {p_mauvais.round(3)}")
print(f"  Loss: {cross_entropy(y_vrai, p_mauvais):.4f} (élevé = mal)")
print()

# ============================================================
# EXERCICE 4 — JOUR 4: Perceptron
# ============================================================
print("=" * 60)
print("EXERCICE 4: Perceptron — Apprendre AND")
print("=" * 60)

class Perceptron:
    def __init__(self, n_entrees):
        self.poids = np.random.randn(n_entrees) * 0.1
        self.biais = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        z = np.dot(self.poids, x) + self.biais
        return self.sigmoid(z)

    def train(self, x, y, lr=0.1):
        pred = self.forward(x)
        erreur = y - pred
        self.poids += lr * erreur * x
        self.biais += lr * erreur
        return abs(erreur)

# Données AND
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0, 0, 0, 1])

p = Perceptron(2)
print("Entraînement sur AND:")
for epoch in range(200):
    err = sum(p.train(xi, yi) for xi, yi in zip(X_and, y_and))
    if epoch % 50 == 0:
        print(f"  Epoch {epoch}: erreur={err:.4f}")

print("\nTest final:")
for xi, yi in zip(X_and, y_and):
    pred = p.forward(xi)
    correct = "✓" if (pred > 0.5) == yi else "✗"
    print(f"  {xi} → {pred:.3f} (vrai:{yi}) {correct}")
print()

# ============================================================
# EXERCICE 5 — JOUR 6: Comparer les activations (THÈSE)
# ============================================================
print("=" * 60)
print("EXERCICE 5: Fonctions d'activation (cœur de la thèse)")
print("=" * 60)

x_vals = np.linspace(-4, 4, 1000)

# Définitions
relu  = np.maximum(0, x_vals)
gelu  = x_vals * (1 + np.vectorize(lambda z:
        np.math.erf(z/np.sqrt(2)))(x_vals)) / 2
silu  = x_vals / (1 + np.exp(-x_vals))
tanh  = np.tanh(x_vals)

print("Valeurs en quelques points:")
print(f"{'x':>6} | {'ReLU':>8} | {'GELU':>8} | {'SiLU':>8}")
print("-" * 40)
for xv in [-2, -1, 0, 1, 2]:
    idx = np.argmin(np.abs(x_vals - xv))
    print(f"{xv:>6} | {relu[idx]:>8.4f} | {gelu[idx]:>8.4f} | {silu[idx]:>8.4f}")

print("\nObservation: ReLU = 0 exactement pour x≤0")
print("GELU et SiLU: transitions douces, pas de point de coude")
print("\n→ C'est pourquoi les méthodes ReLU ne marchent pas sur GELU/SiLU!")
print()

# ============================================================
# EXERCICE 6 — JOUR 9: Simulation boîte noire
# ============================================================
print("=" * 60)
print("EXERCICE 6: Simulation interrogation boîte noire")
print("=" * 60)

class ModeleSimple:
    """Modèle victime simplifié (sans PyTorch)"""
    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(4, 2)
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(1, 4)
        self.b2 = np.zeros(1)

    def gelu(self, z):
        return z * (1 + np.vectorize(lambda v: np.math.erf(v/np.sqrt(2)))(z)) / 2

    def forward(self, x):
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.gelu(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        return z2[0]

class API:
    def __init__(self, modele):
        self.modele = modele
        self.n_requetes = 0

    def query(self, x):
        self.n_requetes += 1
        return self.modele.forward(x)

# Créer le modèle et l'API
modele_secret = ModeleSimple()
api = API(modele_secret)

# Collecter des observations
print("Collecte de données sur le modèle boîte noire:")
observations = []
for i in range(10):
    x = np.random.randn(2)
    y = api.query(x)
    observations.append((x, y))
    if i < 5:
        print(f"  Requête {i+1}: f({x.round(3)}) = {y:.4f}")

print(f"  ... (total: {api.n_requetes} requêtes)")
print(f"\nDéfi: reconstruire le modèle à partir de ces {api.n_requetes} observations")
print("C'est l'essence de l'attaque d'extraction!")

print("\n" + "=" * 60)
print("TOUS LES EXERCICES COMPLÉTÉS AVEC SUCCÈS!")
print("Tu as les bases pour la réunion avec le Pr. Ciss.")
print("=" * 60)
