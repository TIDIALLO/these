#!/usr/bin/env python3
"""
=============================================================================
EXERCICES PRATIQUES - SEMAINES 1 & 2
Thèse: Attaques d'extraction et défenses pour les DNN au-delà de ReLU
Doctorant: Tidiane DIALLO
=============================================================================

Ce fichier contient tous les exercices des semaines 1 et 2 avec leurs solutions.
Exécutez section par section pour comprendre chaque concept.

Prérequis:
    pip install numpy matplotlib scipy torch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("="*70)
print("COURS DEEP LEARNING - EXERCICES SEMAINES 1 & 2")
print("Thèse: Attaques d'extraction pour DNN au-delà de ReLU")
print("="*70)


# =============================================================================
# SEMAINE 1 - EXERCICE 1.1 : PERCEPTRON POUR AND
# =============================================================================
print("\n" + "="*70)
print("EXERCICE 1.1 : PERCEPTRON POUR LA FONCTION AND")
print("="*70)

class Perceptron:
    """
    Implémentation d'un perceptron simple.

    Le perceptron est le modèle le plus simple de neurone artificiel.
    Il calcule: y = f(w·x + b) où f est une fonction de seuil.
    """

    def __init__(self, n_inputs, learning_rate=0.1):
        """
        Initialise le perceptron.

        Args:
            n_inputs: Nombre d'entrées (features)
            learning_rate: Taux d'apprentissage η
        """
        # Initialisation aléatoire des poids (petites valeurs)
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.lr = learning_rate

        print(f"Perceptron créé avec {n_inputs} entrées")
        print(f"  - Poids initiaux: {self.weights}")
        print(f"  - Biais initial: {self.bias}")
        print(f"  - Learning rate: {self.lr}")

    def activation(self, z):
        """
        Fonction de seuil (step function).

        f(z) = 1 si z >= 0
        f(z) = 0 sinon
        """
        return 1 if z >= 0 else 0

    def forward(self, x):
        """
        Calcule la sortie du perceptron.

        y = f(w·x + b)

        Args:
            x: Vecteur d'entrée
        Returns:
            Sortie (0 ou 1)
        """
        # Somme pondérée
        z = np.dot(self.weights, x) + self.bias
        # Application de l'activation
        return self.activation(z)

    def train(self, X, y, epochs=100):
        """
        Entraîne le perceptron avec la règle d'apprentissage du perceptron.

        Règle: w = w + η * erreur * x
               b = b + η * erreur

        Args:
            X: Matrice d'entrées (n_samples, n_features)
            y: Vecteur de labels
            epochs: Nombre d'époques

        Returns:
            Historique des erreurs par époque
        """
        history = []

        for epoch in range(epochs):
            total_error = 0

            for xi, yi in zip(X, y):
                # 1. Prédiction
                y_pred = self.forward(xi)

                # 2. Calcul de l'erreur
                error = yi - y_pred
                total_error += abs(error)

                # 3. Mise à jour des poids (règle du perceptron)
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

            history.append(total_error)

            # Arrêt si convergence
            if total_error == 0:
                print(f"\n✓ Convergence atteinte à l'époque {epoch + 1}")
                break

        return history

# Test sur la fonction AND
print("\n--- Test sur la fonction AND ---")
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_and = np.array([0, 0, 0, 1])  # AND

print("\nTable de vérité AND:")
print("  x1  x2  |  y")
print("  --------|----")
for xi, yi in zip(X_and, y_and):
    print(f"   {xi[0]}   {xi[1]}  |  {yi}")

# Entraînement
np.random.seed(42)
perceptron_and = Perceptron(n_inputs=2, learning_rate=0.1)
history = perceptron_and.train(X_and, y_and, epochs=100)

# Résultats
print(f"\nPoids finaux: {perceptron_and.weights}")
print(f"Biais final: {perceptron_and.bias}")

print("\nVérification:")
for xi, yi in zip(X_and, y_and):
    pred = perceptron_and.forward(xi)
    status = "✓" if pred == yi else "✗"
    print(f"  {xi} -> {pred} (attendu: {yi}) {status}")


# =============================================================================
# SEMAINE 1 - EXERCICE 1.2 : MLP POUR XOR
# =============================================================================
print("\n" + "="*70)
print("EXERCICE 1.2 : MLP POUR LA FONCTION XOR")
print("="*70)

def sigmoid(x):
    """
    Fonction sigmoid: σ(x) = 1 / (1 + e^(-x))

    Propriétés:
    - Sortie dans (0, 1)
    - Utilisée historiquement pour les couches cachées
    - Dérivée: σ'(x) = σ(x) * (1 - σ(x))
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Dérivée de sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

class MLP:
    """
    Perceptron Multicouches (MLP) avec une couche cachée.

    Architecture: input_size -> hidden_size -> output_size

    Ce réseau peut résoudre des problèmes non linéairement séparables
    comme XOR grâce à sa couche cachée.
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        """
        Initialise le MLP avec Xavier initialization.

        Xavier init: W ~ N(0, sqrt(2/n_in))
        Aide à maintenir la variance des activations à travers les couches.
        """
        # Couche 1 (entrée -> caché)
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))

        # Couche 2 (caché -> sortie)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

        # Cache pour backpropagation
        self.cache = {}

        print(f"MLP créé: {input_size} -> {hidden_size} -> {output_size}")
        print(f"  Paramètres: {input_size*hidden_size + hidden_size + hidden_size*output_size + output_size}")

    def forward(self, X):
        """
        Propagation avant.

        Calcule:
            z1 = W1 @ X + b1
            a1 = sigmoid(z1)
            z2 = W2 @ a1 + b2
            a2 = sigmoid(z2)

        Args:
            X: Entrées (n_features, n_samples)
        Returns:
            Sortie du réseau
        """
        # Couche 1
        self.cache['Z1'] = self.W1 @ X + self.b1
        self.cache['A1'] = sigmoid(self.cache['Z1'])

        # Couche 2
        self.cache['Z2'] = self.W2 @ self.cache['A1'] + self.b2
        self.cache['A2'] = sigmoid(self.cache['Z2'])

        return self.cache['A2']

    def backward(self, X, Y, learning_rate=0.5):
        """
        Rétropropagation du gradient.

        Utilise la règle de la chaîne pour calculer les gradients:
            dL/dW2 = dL/dA2 * dA2/dZ2 * dZ2/dW2
            dL/dW1 = dL/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1

        Args:
            X: Entrées
            Y: Sorties attendues
            learning_rate: Taux d'apprentissage
        """
        m = X.shape[1]  # Nombre d'exemples

        # Récupérer les valeurs du cache
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        Z1 = self.cache['Z1']
        Z2 = self.cache['Z2']

        # ===== BACKWARD PASS =====

        # Couche de sortie
        # dL/dZ2 = (A2 - Y) * sigmoid'(Z2)
        dZ2 = (A2 - Y) * sigmoid_derivative(Z2)
        dW2 = (1/m) * dZ2 @ A1.T
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Couche cachée
        # dL/dZ1 = (W2.T @ dZ2) * sigmoid'(Z1)
        dZ1 = (self.W2.T @ dZ2) * sigmoid_derivative(Z1)
        dW1 = (1/m) * dZ1 @ X.T
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # ===== MISE À JOUR DES POIDS =====
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, Y, epochs=10000, learning_rate=1.0, print_every=2000):
        """Entraîne le MLP"""
        losses = []

        for epoch in range(epochs):
            # Forward
            output = self.forward(X)

            # Calcul de la perte (MSE)
            loss = np.mean((output - Y) ** 2)
            losses.append(loss)

            # Backward
            self.backward(X, Y, learning_rate)

            if epoch % print_every == 0:
                print(f"  Époque {epoch:5d} | Loss: {loss:.6f}")

        return losses

    def predict(self, X):
        """Prédiction avec seuil à 0.5"""
        return (self.forward(X) > 0.5).astype(int)

# Test sur XOR
print("\n--- Test sur la fonction XOR ---")

# Données XOR (format: n_features x n_samples)
X_xor = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])
Y_xor = np.array([[0, 1, 1, 0]])

print("\nTable de vérité XOR:")
print("  x1  x2  |  y")
print("  --------|----")
for i in range(4):
    print(f"   {X_xor[0,i]}   {X_xor[1,i]}  |  {Y_xor[0,i]}")

print("\nXOR n'est pas linéairement séparable -> nécessite MLP !")

# Entraînement
np.random.seed(42)
mlp = MLP(input_size=2, hidden_size=4, output_size=1)
print("\nEntraînement...")
losses = mlp.train(X_xor, Y_xor, epochs=10000, learning_rate=2.0, print_every=2000)

# Résultats
print("\n--- Résultats ---")
output = mlp.forward(X_xor)
predictions = mlp.predict(X_xor)

print("\nPrédictions:")
for i in range(4):
    status = "✓" if predictions[0,i] == Y_xor[0,i] else "✗"
    print(f"  [{X_xor[0,i]}, {X_xor[1,i]}] -> {output[0,i]:.4f} ≈ {predictions[0,i]} (attendu: {Y_xor[0,i]}) {status}")


# =============================================================================
# SEMAINE 2 - EXERCICE 2.1 : FONCTIONS D'ACTIVATION
# =============================================================================
print("\n" + "="*70)
print("EXERCICE 2.1 : FONCTIONS D'ACTIVATION ET LEURS DÉRIVÉES")
print("="*70)

# Implémentation des fonctions d'activation
def relu(x):
    """ReLU: max(0, x) - Linéaire par morceaux"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Dérivée de ReLU: 1 si x > 0, 0 sinon"""
    return (x > 0).astype(float)

def gelu_exact(x):
    """
    GELU exact: x * Φ(x)
    où Φ est la CDF de la loi normale standard.

    Utilisé dans BERT, GPT, ViT...
    """
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_derivative(x):
    """
    GELU'(x) = Φ(x) + x * φ(x)
    où φ est la PDF normale.
    """
    phi_cdf = 0.5 * (1 + erf(x / np.sqrt(2)))
    phi_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return phi_cdf + x * phi_pdf

def silu(x):
    """
    SiLU/Swish: x * σ(x)

    Utilisé dans EfficientNet, YOLOv5...
    """
    return x * sigmoid(x)

def silu_derivative(x):
    """SiLU'(x) = σ(x) * (1 + x * (1 - σ(x)))"""
    s = sigmoid(x)
    return s * (1 + x * (1 - s))

def tanh_activation(x):
    """Tanh: (e^x - e^-x) / (e^x + e^-x)"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh'(x) = 1 - tanh²(x)"""
    return 1 - np.tanh(x)**2

# Vérification des dérivées par différences finies
print("\n--- Vérification des dérivées (différences finies) ---")

def verify_derivative(f, f_prime, x, epsilon=1e-7):
    """Compare dérivée analytique vs numérique"""
    numerical = (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
    analytical = f_prime(x)
    error = np.max(np.abs(numerical - analytical))
    return error

x_test = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])

print(f"\nPoints de test: {x_test}")
print("\nRésultats:")

for name, f, f_prime in [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("Tanh", tanh_activation, tanh_derivative),
    ("GELU", gelu_exact, gelu_derivative),
    ("SiLU", silu, silu_derivative),
]:
    error = verify_derivative(f, f_prime, x_test)
    status = "✓" if error < 1e-5 else "✗"
    print(f"  {name:10s}: erreur max = {error:.2e} {status}")

print("\nNote: ReLU n'est pas dérivable en x=0")


# =============================================================================
# VISUALISATION COMPARATIVE
# =============================================================================
print("\n" + "="*70)
print("VISUALISATION : COMPARAISON DES ACTIVATIONS")
print("="*70)

x = np.linspace(-4, 4, 1000)

# Créer la figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Liste des activations à visualiser
activations = [
    ('ReLU', relu, relu_derivative, 'blue'),
    ('Sigmoid', sigmoid, sigmoid_derivative, 'orange'),
    ('Tanh', tanh_activation, tanh_derivative, 'green'),
    ('GELU', gelu_exact, gelu_derivative, 'red'),
    ('SiLU/Swish', silu, silu_derivative, 'purple'),
]

for idx, (name, func, deriv, color) in enumerate(activations):
    ax = axes.flatten()[idx]

    # Fonction
    ax.plot(x, func(x), color=color, linewidth=2.5, label=f'{name}(x)')
    # Dérivée
    ax.plot(x, deriv(x), color=color, linewidth=1.5, linestyle='--',
            label=f"{name}'(x)", alpha=0.7)

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.5, 4)

# Comparaison dans le dernier subplot
ax = axes.flatten()[5]
ax.plot(x, relu(x), 'b-', linewidth=2, label='ReLU')
ax.plot(x, gelu_exact(x), 'r-', linewidth=2, label='GELU')
ax.plot(x, silu(x), 'purple', linewidth=2, label='SiLU')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_title('Comparaison ReLU vs GELU vs SiLU', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)
ax.set_ylim(-1, 4)

plt.tight_layout()
plt.savefig('/root/these-tidiane-diallo/cours/exercices/activations_comparison.png', dpi=150)
print("\n✓ Figure sauvegardée: activations_comparison.png")
plt.close()


# =============================================================================
# EXERCICE 2.2 : RÉGIONS LINÉAIRES RELU
# =============================================================================
print("\n" + "="*70)
print("EXERCICE 2.2 : RÉGIONS LINÉAIRES CRÉÉES PAR ReLU")
print("="*70)

print("""
CONCEPT CLÉ POUR LA THÈSE:
ReLU crée des régions LINÉAIRES PAR MORCEAUX dans l'espace d'entrée.
C'est cette propriété qui rend les réseaux ReLU VULNÉRABLES aux attaques d'extraction!

Chaque neurone ReLU définit un hyperplan: w·x + b = 0
- D'un côté: le neurone est actif (sortie = x)
- De l'autre: le neurone est inactif (sortie = 0)

Avec n neurones, on peut avoir jusqu'à 2^n régions différentes!
""")

# Réseau simple: 2 entrées, 3 neurones ReLU, 1 sortie
np.random.seed(123)

W1 = np.array([[1.0, 0.5],
               [-0.5, 1.0],
               [0.8, -0.8]])
b1 = np.array([0.2, -0.3, 0.1])
W2 = np.array([[1.0, 0.5, 0.8]])
b2 = np.array([0.0])

def relu_network(x):
    """Réseau ReLU simple"""
    z1 = x @ W1.T + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2.T + b2
    return z2

# Créer une grille
resolution = 200
x1 = np.linspace(-2, 2, resolution)
x2 = np.linspace(-2, 2, resolution)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.column_stack([X1.ravel(), X2.ravel()])

# Évaluer le réseau
Z = relu_network(X_grid).reshape(X1.shape)

# Patterns d'activation (quels neurones sont actifs)
activation_patterns = (X_grid @ W1.T + b1 > 0).astype(int)
pattern_ids = activation_patterns[:, 0] * 4 + activation_patterns[:, 1] * 2 + activation_patterns[:, 2]
pattern_ids = pattern_ids.reshape(X1.shape)

# Visualiser
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sortie du réseau
im1 = axes[0].contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(im1, ax=axes[0], label='Sortie du réseau')

# Tracer les hyperplans critiques
colors = ['red', 'blue', 'green']
for i in range(3):
    if abs(W1[i, 1]) > 1e-6:
        x2_line = -(W1[i, 0] * x1 + b1[i]) / W1[i, 1]
        mask = (x2_line >= -2) & (x2_line <= 2)
        axes[0].plot(x1[mask], x2_line[mask], colors[i], linewidth=2.5,
                    label=f'Hyperplan neurone {i+1}')

axes[0].set_title('Sortie du réseau ReLU\n(notez les régions linéaires)', fontsize=12)
axes[0].set_xlabel('x₁')
axes[0].set_ylabel('x₂')
axes[0].legend(loc='upper right')
axes[0].set_xlim(-2, 2)
axes[0].set_ylim(-2, 2)

# Régions d'activation
im2 = axes[1].contourf(X1, X2, pattern_ids, levels=8, cmap='Set3')
axes[1].set_title('Régions d\'activation\n(chaque couleur = pattern différent)', fontsize=12)
axes[1].set_xlabel('x₁')
axes[1].set_ylabel('x₂')

# Tracer les hyperplans
for i in range(3):
    if abs(W1[i, 1]) > 1e-6:
        x2_line = -(W1[i, 0] * x1 + b1[i]) / W1[i, 1]
        mask = (x2_line >= -2) & (x2_line <= 2)
        axes[1].plot(x1[mask], x2_line[mask], 'k-', linewidth=2)

unique_regions = len(np.unique(pattern_ids))
axes[1].text(0.02, 0.98, f'Régions: {unique_regions}', transform=axes[1].transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('/root/these-tidiane-diallo/cours/exercices/relu_linear_regions.png', dpi=150)
print(f"\n✓ Figure sauvegardée: relu_linear_regions.png")
print(f"  Nombre de régions linéaires distinctes: {unique_regions}")
print(f"  Maximum théorique avec 3 neurones: 2³ = 8")
plt.close()


# =============================================================================
# EXERCICE 2.3 : COMPARAISON ReLU vs GELU POUR L'EXTRACTION
# =============================================================================
print("\n" + "="*70)
print("EXERCICE 2.3 : POURQUOI GELU EST PLUS RÉSISTANT AUX ATTAQUES")
print("="*70)

print("""
CONCEPT CLÉ:
L'attaque CRYPTO 2020 détecte les "points critiques" de ReLU
en cherchant où la DÉRIVÉE change brutalement.

Pour ReLU: dérivée = 0 ou 1 (changement DISCRET)
Pour GELU: dérivée varie CONTINÛMENT de 0 à ~1.1

Comparons les dérivées secondes:
- ReLU''(x) = 0 partout (sauf en x=0 où c'est une distribution de Dirac)
- GELU''(x) = φ(x)(2 - x²) ≠ 0 pour tout x fini
""")

x = np.linspace(-3, 3, 1000)

# Dérivées premières
relu_d1 = relu_derivative(x)
gelu_d1 = gelu_derivative(x)
silu_d1 = silu_derivative(x)

# Dérivées secondes (numériques)
eps = 1e-5
relu_d2 = (relu_derivative(x + eps) - relu_derivative(x - eps)) / (2 * eps)
gelu_d2 = (gelu_derivative(x + eps) - gelu_derivative(x - eps)) / (2 * eps)
silu_d2 = (silu_derivative(x + eps) - silu_derivative(x - eps)) / (2 * eps)

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Fonctions
axes[0].plot(x, relu(x), 'b-', linewidth=2, label='ReLU')
axes[0].plot(x, gelu_exact(x), 'r-', linewidth=2, label='GELU')
axes[0].plot(x, silu(x), 'g-', linewidth=2, label='SiLU')
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].axvline(0, color='k', linewidth=0.5)
axes[0].set_title('Fonctions', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dérivées premières
axes[1].plot(x, relu_d1, 'b-', linewidth=2, label="ReLU'")
axes[1].plot(x, gelu_d1, 'r-', linewidth=2, label="GELU'")
axes[1].plot(x, silu_d1, 'g-', linewidth=2, label="SiLU'")
axes[1].axhline(0, color='k', linewidth=0.5)
axes[1].axhline(1, color='k', linewidth=0.5, linestyle='--')
axes[1].axvline(0, color='k', linewidth=0.5)
axes[1].set_title("Dérivées premières\n(ReLU: changement discret!)", fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.5, 1.5)

# Dérivées secondes
axes[2].plot(x, relu_d2, 'b-', linewidth=2, label="ReLU''")
axes[2].plot(x, gelu_d2, 'r-', linewidth=2, label="GELU''")
axes[2].plot(x, silu_d2, 'g-', linewidth=2, label="SiLU''")
axes[2].axhline(0, color='k', linewidth=0.5)
axes[2].axvline(0, color='k', linewidth=0.5)
axes[2].set_title("Dérivées secondes\n(ReLU: pic en x=0 révèle la structure!)", fontsize=14)
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-1, 1)

plt.tight_layout()
plt.savefig('/root/these-tidiane-diallo/cours/exercices/derivatives_comparison.png', dpi=150)
print("\n✓ Figure sauvegardée: derivatives_comparison.png")
plt.close()


# =============================================================================
# DÉMONSTRATION : DÉTECTION DE POINTS CRITIQUES
# =============================================================================
print("\n" + "="*70)
print("DÉMONSTRATION : DÉTECTION DE POINTS CRITIQUES (Attaque simplifiée)")
print("="*70)

try:
    import torch
    import torch.nn as nn

    print("\nCréation de deux réseaux identiques (mêmes poids):")
    print("  - Un avec ReLU")
    print("  - Un avec GELU")

    # Réseaux
    relu_net = nn.Sequential(
        nn.Linear(1, 3, bias=True),
        nn.ReLU(),
        nn.Linear(3, 1, bias=True)
    )

    gelu_net = nn.Sequential(
        nn.Linear(1, 3, bias=True),
        nn.GELU(),
        nn.Linear(3, 1, bias=True)
    )

    # Copier les mêmes poids
    with torch.no_grad():
        gelu_net[0].weight.copy_(relu_net[0].weight)
        gelu_net[0].bias.copy_(relu_net[0].bias)
        gelu_net[2].weight.copy_(relu_net[2].weight)
        gelu_net[2].bias.copy_(relu_net[2].bias)

    def detect_critical_points(oracle_fn, x_range=(-3, 3), resolution=1000, threshold=0.5):
        """Détecte les points où la dérivée change brutalement"""
        x = np.linspace(x_range[0], x_range[1], resolution)
        epsilon = (x_range[1] - x_range[0]) / resolution

        # Calculer les dérivées numériques
        derivatives = []
        for xi in x:
            df = (oracle_fn(xi + epsilon) - oracle_fn(xi - epsilon)) / (2 * epsilon)
            derivatives.append(df)

        derivatives = np.array(derivatives)

        # Détecter les changements brusques
        derivative_changes = np.abs(np.diff(derivatives))
        critical_indices = np.where(derivative_changes > threshold)[0]

        return x[critical_indices], derivative_changes

    def relu_oracle(x_val):
        with torch.no_grad():
            return relu_net(torch.tensor([[x_val]], dtype=torch.float32)).item()

    def gelu_oracle(x_val):
        with torch.no_grad():
            return gelu_net(torch.tensor([[x_val]], dtype=torch.float32)).item()

    # Détecter les points critiques
    relu_critical, relu_changes = detect_critical_points(relu_oracle, threshold=0.3)
    gelu_critical, gelu_changes = detect_critical_points(gelu_oracle, threshold=0.3)

    print(f"\nPoints critiques détectés:")
    print(f"  ReLU: {len(relu_critical)} points")
    print(f"  GELU: {len(gelu_critical)} points")
    print(f"\n→ ReLU révèle sa structure interne!")
    print(f"→ GELU cache mieux ses paramètres!")

    # Visualisation
    x = np.linspace(-3, 3, 500)
    relu_out = [relu_oracle(xi) for xi in x]
    gelu_out = [gelu_oracle(xi) for xi in x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, relu_out, 'b-', linewidth=2, label='Sortie ReLU')
    for cp in relu_critical:
        axes[0].axvline(cp, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title(f'Réseau ReLU\n({len(relu_critical)} points critiques détectés)', fontsize=14)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, gelu_out, 'r-', linewidth=2, label='Sortie GELU')
    for cp in gelu_critical:
        axes[1].axvline(cp, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title(f'Réseau GELU\n({len(gelu_critical)} points critiques détectés)', fontsize=14)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('f(x)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/these-tidiane-diallo/cours/exercices/critical_points_demo.png', dpi=150)
    print("\n✓ Figure sauvegardée: critical_points_demo.png")
    plt.close()

except ImportError:
    print("\nPyTorch non disponible. Installez avec: pip install torch")


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "="*70)
print("RÉSUMÉ DES EXERCICES")
print("="*70)

print("""
SEMAINE 1 - CE QUE VOUS AVEZ APPRIS:
─────────────────────────────────────
✓ Perceptron: neurone simple, séparation linéaire
✓ MLP: réseau multicouches pour problèmes non-linéaires (XOR)
✓ Forward pass: calcul de la sortie couche par couche
✓ Backpropagation: calcul des gradients par la règle de la chaîne

SEMAINE 2 - CE QUE VOUS AVEZ APPRIS:
─────────────────────────────────────
✓ Sigmoid/Tanh: activations classiques, problème de saturation
✓ ReLU: simple, efficace, mais LINÉAIRE PAR MORCEAUX
✓ GELU: lisse, utilisé dans Transformers
✓ SiLU: lisse, utilisé dans modèles de vision

POINT CLÉ POUR VOTRE THÈSE:
────────────────────────────
La propriété "piecewise linear" de ReLU crée des POINTS CRITIQUES
détectables qui révèlent la structure du réseau.

GELU et SiLU, étant LISSES, ne présentent pas ces points critiques,
ce qui les rend plus résistants aux attaques d'extraction.

VOTRE DÉFI: Trouver comment extraire des modèles malgré cela!
""")

print("\nFichiers générés:")
print("  - activations_comparison.png")
print("  - relu_linear_regions.png")
print("  - derivatives_comparison.png")
print("  - critical_points_demo.png")

print("\n" + "="*70)
print("FIN DES EXERCICES")
print("="*70)
