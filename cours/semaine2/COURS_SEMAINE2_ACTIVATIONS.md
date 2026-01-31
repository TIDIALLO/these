# Cours Semaine 2 : Fonctions d'Activation

**Thèse : Attaques d'extraction et défenses pour les DNN au-delà de ReLU**
**Doctorant : Tidiane DIALLO**
**Date : Janvier 2026**

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Fonctions d'Activation Classiques](#2-fonctions-dactivation-classiques)
3. [ReLU et ses Variantes](#3-relu-et-ses-variantes)
4. [GELU - Gaussian Error Linear Unit](#4-gelu---gaussian-error-linear-unit)
5. [SiLU/Swish](#5-siluswish)
6. [Analyse Comparative](#6-analyse-comparative)
7. [Implications pour l'Extraction de Modèles](#7-implications-pour-lextraction-de-modèles)
8. [Exercices Pratiques](#8-exercices-pratiques)
9. [Ressources](#9-ressources)

---

## 1. Introduction

### 1.1 Rôle des Fonctions d'Activation

Les fonctions d'activation introduisent la **non-linéarité** dans les réseaux de neurones. Sans elles, un réseau profond serait équivalent à une simple transformation linéaire :

$$\text{Si } f(x) = x \text{ (identité), alors:}$$
$$W_2(W_1 x) = (W_2 W_1) x = W' x$$

→ Peu importe le nombre de couches, c'est juste une matrice !

### 1.2 Propriétés Souhaitables

Une bonne fonction d'activation devrait avoir :

| Propriété | Description | Pourquoi ? |
|-----------|-------------|------------|
| **Non-linéarité** | $f(ax + by) \neq af(x) + bf(y)$ | Permet d'apprendre des fonctions complexes |
| **Dérivable** | $f'(x)$ existe partout (ou presque) | Pour la backpropagation |
| **Monotonie** | $f'(x) \geq 0$ ou $\leq 0$ | Facilite l'optimisation |
| **Bornée ou non** | Dépend de l'application | Bornée = stabilité, Non-bornée = expressivité |

### 1.3 Importance Critique pour ta Thèse

**Le choix de l'activation détermine si le réseau est vulnérable aux attaques d'extraction !**

- **ReLU** → Crée des régions **linéaires par morceaux** → Attaques efficaces (CRYPTO 2020)
- **GELU/SiLU** → Fonctions **lisses** → Attaques beaucoup plus difficiles

C'est le **cœur de ton sujet de thèse** !

---

## 2. Fonctions d'Activation Classiques

### 2.1 Sigmoid (Logistique)

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Dérivée :**
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Graphe :**
```
     1 ─────────────────●●●●●●●●
       │               ●
       │              ●
     0.5 ─────────────●
       │            ●
       │          ●●
     0 ●●●●●●●●●●●───────────────
      -5          0          5
```

**Propriétés :**
- Sortie dans $(0, 1)$ → interprétation probabiliste
- **Problème** : Gradient s'annule aux extrêmes (saturation)
- Maximum de $\sigma'(x) = 0.25$ à $x = 0$

**Historique :** Utilisé dans les premiers réseaux, puis abandonné pour les couches cachées.

### 2.2 Tanh (Tangente Hyperbolique)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

**Dérivée :**
$$\tanh'(x) = 1 - \tanh^2(x)$$

**Graphe :**
```
     1 ─────────────────●●●●●●●●
       │               ●
       │              ●
     0 ───────────────●────────
       │            ●
       │          ●●
    -1 ●●●●●●●●●●●───────────────
      -5          0          5
```

**Propriétés :**
- Sortie dans $(-1, 1)$ → centrée autour de 0
- Gradient max = 1 à $x = 0$
- Souffre aussi de saturation aux extrêmes

**Avantage sur sigmoid :** Sorties centrées → meilleure convergence

---

## 3. ReLU et ses Variantes

### 3.1 ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{si } x > 0 \\ 0 & \text{si } x \leq 0 \end{cases}$$

**Dérivée :**
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{si } x > 0 \\ 0 & \text{si } x < 0 \end{cases}$$

(Non définie en $x = 0$, mais on utilise généralement 0 ou 1)

**Graphe :**
```
     y │
       │            ●
     3 ─          ●
       │        ●
     2 ─      ●
       │    ●
     1 ─  ●
       │●
     0 ●───────────────────
      -3 -2 -1  0  1  2  3  x
```

**Propriétés :**

| Avantage | Inconvénient |
|----------|--------------|
| Calcul très rapide | "Dying ReLU" : neurones bloqués à 0 |
| Pas de saturation pour $x > 0$ | Non centrée (sortie $\geq 0$) |
| Sparse activation | Gradient 0 pour $x < 0$ |

### 3.2 Propriété Cruciale : Linéaire par Morceaux (Piecewise Linear)

**C'est LA propriété qui rend ReLU vulnérable aux attaques !**

Un réseau avec ReLU divise l'espace d'entrée en **régions linéaires** :

```
Espace d'entrée 2D avec 3 neurones ReLU :

        ╱│╲
       ╱ │ ╲
      ╱  │  ╲     Chaque région =
     ╱ R3│R4 ╲    fonction LINÉAIRE
    ╱────┼────╲   différente
   ╱  R1 │ R2  ╲
  ╱──────┴──────╲
```

**Mathématiquement :**
- Chaque neurone ReLU crée un hyperplan : $w^T x + b = 0$
- De part et d'autre de l'hyperplan, le neurone est soit actif ($x$) soit inactif ($0$)
- Avec $n$ neurones, on peut avoir jusqu'à $2^n$ régions !

**Implication pour l'extraction :**
> Les points où le comportement change (hyperplans) révèlent les poids du réseau !

### 3.3 Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha x & \text{si } x \leq 0 \end{cases}$$

Typiquement $\alpha = 0.01$

**Avantage :** Évite les "dying ReLU" en permettant un petit gradient négatif.

### 3.4 PReLU (Parametric ReLU)

Comme Leaky ReLU, mais $\alpha$ est **appris** pendant l'entraînement.

### 3.5 ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha(e^x - 1) & \text{si } x \leq 0 \end{cases}$$

**Propriété :** Lisse partout sauf en $x = 0$.

---

## 4. GELU - Gaussian Error Linear Unit

### 4.1 Définition

**Paper original :** Hendrycks & Gimpel (2016) - "Gaussian Error Linear Units (GELUs)"
- https://arxiv.org/abs/1606.08415

$$\text{GELU}(x) = x \cdot \Phi(x)$$

où $\Phi(x)$ est la **fonction de répartition** (CDF) de la loi normale standard :

$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Forme explicite :**
$$\text{GELU}(x) = \frac{x}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

### 4.2 Interprétation Probabiliste

GELU peut être vu comme une **porte stochastique** :

$$\text{GELU}(x) = x \cdot P(X \leq x) \text{ où } X \sim \mathcal{N}(0, 1)$$

L'entrée est **pondérée par la probabilité qu'elle soit "grande"** selon une distribution gaussienne.

**Intuition :**
- Si $x \gg 0$ : $\Phi(x) \approx 1$ → $\text{GELU}(x) \approx x$ (comme ReLU)
- Si $x \ll 0$ : $\Phi(x) \approx 0$ → $\text{GELU}(x) \approx 0$ (comme ReLU)
- Autour de $0$ : transition **douce** (contrairement à ReLU)

### 4.3 Approximation Pratique

Dans les implémentations (GPT, BERT), on utilise souvent :

$$\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right]$$

Cette approximation est très précise et plus rapide à calculer.

### 4.4 Dérivée de GELU

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

où $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$ est la **densité** (PDF) de la loi normale.

**Calcul détaillé :**

$$\frac{d}{dx}[x \cdot \Phi(x)] = \Phi(x) + x \cdot \frac{d\Phi}{dx}$$

Or $\frac{d\Phi}{dx} = \phi(x)$, donc :

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

### 4.5 Graphe

```
GELU vs ReLU

     y │
       │         GELU ●●●
     3 ─        ●●   ReLU ──
       │      ●●   ──
     2 ─    ●●   ──
       │  ●●  ──
     1 ─ ●● ──
       ●●──
     0 ●─────────────────
       ●
   -0.2●  ← GELU a une petite zone négative !
      -3 -2 -1  0  1  2  3  x
```

**Observation importante :** GELU a un **minimum local** autour de $x \approx -0.17$

### 4.6 Utilisation

GELU est utilisé dans :
- **BERT** (Bidirectional Encoder Representations from Transformers)
- **GPT** (Generative Pre-trained Transformer)
- **Vision Transformers (ViT)**
- La plupart des architectures Transformer modernes

---

## 5. SiLU/Swish

### 5.1 Définition

**Paper original :** Ramachandran et al. (2017) - "Searching for Activation Functions"
- https://arxiv.org/abs/1710.05941

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

où $\sigma(x)$ est la fonction **sigmoid**.

**Autre nom :** Swish (avec $\beta = 1$)

La forme générale de Swish est : $\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$

### 5.2 Interprétation

Comme GELU, SiLU est une **auto-pondération** :
- L'entrée $x$ est multipliée par une "porte" $\sigma(x) \in (0, 1)$
- Pour $x \gg 0$ : $\sigma(x) \approx 1$ → $\text{SiLU}(x) \approx x$
- Pour $x \ll 0$ : $\sigma(x) \approx 0$ → $\text{SiLU}(x) \approx 0$

### 5.3 Dérivée de SiLU

$$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))$$
$$= \sigma(x)[1 + x(1 - \sigma(x))]$$

**Calcul :**
$$\frac{d}{dx}[x \cdot \sigma(x)] = \sigma(x) + x \cdot \sigma'(x)$$
$$= \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))$$

### 5.4 Graphe

```
SiLU vs ReLU

     y │
       │         SiLU ●●●
     3 ─        ●●   ReLU ──
       │      ●●   ──
     2 ─    ●●   ──
       │  ●●  ──
     1 ─ ●● ──
       ●●──
     0 ●─────────────────
       ●
   -0.3●  ← SiLU aussi négatif !
      -3 -2 -1  0  1  2  3  x
```

### 5.5 Utilisation

SiLU est populaire dans :
- **EfficientNet** (modèles de vision)
- **YOLOv5/v8** (détection d'objets)
- Modèles optimisés pour le déploiement edge

---

## 6. Analyse Comparative

### 6.1 Tableau Récapitulatif

| Activation | Formule | Dérivée | Lisse ? | Bornée ? |
|------------|---------|---------|---------|----------|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $\sigma(1-\sigma)$ | Oui | Oui $(0,1)$ |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $1-\tanh^2$ | Oui | Oui $(-1,1)$ |
| **ReLU** | $\max(0,x)$ | $\mathbb{1}_{x>0}$ | **Non** | Non |
| **GELU** | $x\Phi(x)$ | $\Phi(x)+x\phi(x)$ | Oui | Non |
| **SiLU** | $x\sigma(x)$ | $\sigma(1+x(1-\sigma))$ | Oui | Non |

### 6.2 Dérivées Secondes

Pour comprendre les attaques d'extraction, la **dérivée seconde** est cruciale :

| Activation | Dérivée seconde | Zéros |
|------------|-----------------|-------|
| **ReLU** | $0$ partout (sauf en 0) | Partout ! |
| **GELU** | Non nulle partout | Aucun (sauf $\pm\infty$) |
| **SiLU** | Non nulle partout | Aucun (sauf $\pm\infty$) |

**Implication :**
- ReLU : dérivée constante par morceaux → changements **discrets**
- GELU/SiLU : dérivée varie **continuellement** → pas de "points critiques" nets

### 6.3 Visualisation Comparative

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Définition des fonctions
x = np.linspace(-4, 4, 1000)

relu = np.maximum(0, x)
gelu = 0.5 * x * (1 + erf(x / np.sqrt(2)))
silu = x / (1 + np.exp(-x))

# Dérivées
relu_d = (x > 0).astype(float)
phi_cdf = 0.5 * (1 + erf(x / np.sqrt(2)))
phi_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
gelu_d = phi_cdf + x * phi_pdf
sigmoid = 1 / (1 + np.exp(-x))
silu_d = sigmoid * (1 + x * (1 - sigmoid))

# Graphique
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fonctions
axes[0].plot(x, relu, 'b-', label='ReLU', linewidth=2)
axes[0].plot(x, gelu, 'r-', label='GELU', linewidth=2)
axes[0].plot(x, silu, 'g-', label='SiLU', linewidth=2)
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].axvline(0, color='k', linewidth=0.5)
axes[0].set_title('Fonctions d\'activation', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dérivées
axes[1].plot(x, relu_d, 'b-', label="ReLU'", linewidth=2)
axes[1].plot(x, gelu_d, 'r-', label="GELU'", linewidth=2)
axes[1].plot(x, silu_d, 'g-', label="SiLU'", linewidth=2)
axes[1].axhline(0, color='k', linewidth=0.5)
axes[1].axhline(1, color='k', linewidth=0.5, linestyle='--')
axes[1].axvline(0, color='k', linewidth=0.5)
axes[1].set_title('Dérivées', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=150)
```

---

## 7. Implications pour l'Extraction de Modèles

### 7.1 Pourquoi ReLU est Vulnérable

**L'attaque CRYPTO 2020** (Carlini et al.) exploite la structure piecewise-linear de ReLU :

1. **Points critiques** : Aux hyperplans $w^Tx + b = 0$, le comportement du réseau **change brutalement**

2. **Détection des changements** : En calculant les dérivées numériques, on peut détecter ces points :
```python
# Différences finies
df = (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
# Si df change brutalement → point critique !
```

3. **Extraction des poids** : La différence de gradients de part et d'autre d'un point critique révèle les poids :
$$\nabla f(x^+) - \nabla f(x^-) \propto w_i$$

### 7.2 Pourquoi GELU/SiLU sont Plus Résistants

**Pas de points critiques "nets" :**

| ReLU | GELU/SiLU |
|------|-----------|
| Changement **discret** à $x=0$ | Transition **continue** |
| Dérivée = 0 ou 1 | Dérivée varie de 0 à ~1.1 |
| Points critiques **détectables** | Pas de signature claire |

**Mathématiquement :**

Pour ReLU, la dérivée seconde est :
$$\text{ReLU}''(x) = \delta(x) \text{ (distribution de Dirac)}$$

Pour GELU :
$$\text{GELU}''(x) = 2\phi(x) + x\phi'(x) \neq 0 \text{ pour tout } x \text{ fini}$$

### 7.3 Défis pour ta Thèse

**Question de recherche centrale :**

> Comment adapter les attaques cryptanalytiques pour extraire des modèles utilisant GELU/SiLU ?

**Pistes possibles :**

1. **Approximation piecewise-linear** : Approximer GELU par segments linéaires
2. **Points d'inflexion** : Utiliser la dérivée seconde au lieu des points critiques
3. **Méthodes d'optimisation** : Formuler l'extraction comme un problème d'optimisation
4. **Attaques différentes** : Peut-être que les attaques cryptanalytiques ne sont pas adaptées ?

### 7.4 Code de Démonstration

```python
"""
Démonstration : Détection de points critiques ReLU vs GELU
"""
import numpy as np
import torch
import torch.nn as nn

def detect_critical_points(oracle_fn, x_range=(-3, 3), resolution=1000, threshold=0.1):
    """
    Détecte les points où la dérivée change brutalement
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    epsilon = (x_range[1] - x_range[0]) / resolution

    # Calcul des dérivées numériques
    derivatives = []
    for xi in x:
        df = (oracle_fn(xi + epsilon) - oracle_fn(xi - epsilon)) / (2 * epsilon)
        derivatives.append(df)

    derivatives = np.array(derivatives)

    # Détecter les changements brusques
    derivative_changes = np.abs(np.diff(derivatives))
    critical_indices = np.where(derivative_changes > threshold)[0]

    return x[critical_indices], derivative_changes

# Réseau ReLU
relu_net = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

# Réseau GELU
gelu_net = nn.Sequential(
    nn.Linear(1, 3),
    nn.GELU(),
    nn.Linear(3, 1)
)

# Copier les mêmes poids
with torch.no_grad():
    gelu_net[0].weight.copy_(relu_net[0].weight)
    gelu_net[0].bias.copy_(relu_net[0].bias)
    gelu_net[2].weight.copy_(relu_net[2].weight)
    gelu_net[2].bias.copy_(relu_net[2].bias)

def relu_oracle(x):
    with torch.no_grad():
        return relu_net(torch.tensor([[x]], dtype=torch.float32)).item()

def gelu_oracle(x):
    with torch.no_grad():
        return gelu_net(torch.tensor([[x]], dtype=torch.float32)).item()

# Détecter les points critiques
relu_critical, relu_changes = detect_critical_points(relu_oracle)
gelu_critical, gelu_changes = detect_critical_points(gelu_oracle)

print(f"Points critiques ReLU : {len(relu_critical)}")
print(f"Points critiques GELU : {len(gelu_critical)}")
print(f"\n→ ReLU révèle plus d'information structurelle !")
```

---

## 8. Exercices Pratiques

### Exercice 2.1 : Implémentation des Activations

```python
"""
Exercice 2.1 : Implémenter toutes les activations et leurs dérivées
Vérifier avec des différences finies
"""
import numpy as np
from scipy.special import erf

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def gelu_exact(x):
    """GELU avec la fonction d'erreur"""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_derivative(x):
    """
    GELU'(x) = Φ(x) + x·φ(x)
    """
    phi_cdf = 0.5 * (1 + erf(x / np.sqrt(2)))  # CDF normale
    phi_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)  # PDF normale
    return phi_cdf + x * phi_pdf

def silu(x):
    """SiLU = x * sigmoid(x)"""
    return x * sigmoid(x)

def silu_derivative(x):
    """SiLU'(x) = σ(x) + x·σ(x)·(1-σ(x))"""
    s = sigmoid(x)
    return s * (1 + x * (1 - s))

def verify_derivative(f, f_prime, x, epsilon=1e-7):
    """Vérifie la dérivée par différences finies"""
    numerical = (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
    analytical = f_prime(x)
    error = np.abs(numerical - analytical)
    return error

# Test
x_test = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

print("Vérification des dérivées :")
print("="*50)

for name, f, f_prime in [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("GELU", gelu_exact, gelu_derivative),
    ("SiLU", silu, silu_derivative),
]:
    errors = verify_derivative(f, f_prime, x_test)
    max_error = np.max(errors)
    print(f"{name}: erreur max = {max_error:.2e} {'✓' if max_error < 1e-5 else '✗'}")
```

### Exercice 2.2 : Visualisation des Régions Linéaires ReLU

```python
"""
Exercice 2.2 : Visualiser les régions linéaires créées par ReLU
"""
import numpy as np
import matplotlib.pyplot as plt

# Réseau simple : 2 entrées, 3 neurones ReLU, 1 sortie
np.random.seed(42)

W1 = np.array([[1.0, 0.5],
               [-0.5, 1.0],
               [0.8, -0.8]])
b1 = np.array([0.2, -0.3, 0.1])
W2 = np.array([[1.0, 0.5, 0.8]])
b2 = np.array([0.0])

def relu_network(x):
    """Évalue le réseau ReLU"""
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

# Visualiser
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sortie du réseau
im = axes[0].contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(im, ax=axes[0], label='Sortie')
axes[0].set_title('Sortie du réseau ReLU')
axes[0].set_xlabel('x₁')
axes[0].set_ylabel('x₂')

# Tracer les hyperplans critiques (W1·x + b1 = 0)
colors = ['red', 'blue', 'green']
for i in range(3):
    # W1[i,0]*x1 + W1[i,1]*x2 + b1[i] = 0
    # x2 = -(W1[i,0]*x1 + b1[i]) / W1[i,1]
    if abs(W1[i, 1]) > 1e-6:
        x2_line = -(W1[i, 0] * x1 + b1[i]) / W1[i, 1]
        mask = (x2_line >= -2) & (x2_line <= 2)
        axes[0].plot(x1[mask], x2_line[mask], colors[i], linewidth=2,
                    label=f'Hyperplan {i+1}')

axes[0].legend()
axes[0].set_xlim(-2, 2)
axes[0].set_ylim(-2, 2)

# Régions d'activation
# Pour chaque point, déterminer quels neurones sont actifs
activation_patterns = (X_grid @ W1.T + b1 > 0).astype(int)
# Convertir en nombre unique (pattern)
pattern_ids = activation_patterns[:, 0] * 4 + activation_patterns[:, 1] * 2 + activation_patterns[:, 2]
pattern_ids = pattern_ids.reshape(X1.shape)

axes[1].contourf(X1, X2, pattern_ids, levels=8, cmap='Set3')
axes[1].set_title('Régions d\'activation (patterns)')
axes[1].set_xlabel('x₁')
axes[1].set_ylabel('x₂')

# Tracer les hyperplans
for i in range(3):
    if abs(W1[i, 1]) > 1e-6:
        x2_line = -(W1[i, 0] * x1 + b1[i]) / W1[i, 1]
        mask = (x2_line >= -2) & (x2_line <= 2)
        axes[1].plot(x1[mask], x2_line[mask], 'k-', linewidth=2)

plt.tight_layout()
plt.savefig('relu_linear_regions.png', dpi=150)
print("Figure sauvegardée: relu_linear_regions.png")

# Compter les régions uniques
unique_regions = len(np.unique(pattern_ids))
print(f"\nNombre de régions linéaires distinctes: {unique_regions}")
print(f"Maximum théorique avec 3 neurones: 2³ = 8")
```

### Exercice 2.3 : Dérivation Formelle de GELU (Sur papier)

**Objectif :** Calculer GELU'(x) et GELU''(x)

**Rappel :**
$$\text{GELU}(x) = x \cdot \Phi(x) = \frac{x}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Questions :**

1. Calculer $\frac{d\Phi}{dx}$ (rappel : $\Phi$ est la CDF normale)

2. Calculer $\text{GELU}'(x)$ en utilisant la règle du produit

3. Calculer $\text{GELU}''(x)$

4. Montrer que $\text{GELU}''(x) \neq 0$ pour tout $x$ fini

**Solution :**

1. $\frac{d\Phi}{dx} = \phi(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$ (PDF normale)

2. $\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$

3. $\text{GELU}''(x) = \phi(x) + \phi(x) + x \cdot \phi'(x)$
   $= 2\phi(x) + x \cdot (-x)\phi(x)$
   $= \phi(x)(2 - x^2)$

4. $\text{GELU}''(x) = 0 \Leftrightarrow x^2 = 2 \Leftrightarrow x = \pm\sqrt{2}$

   Donc GELU a exactement **deux points d'inflexion** à $x = \pm\sqrt{2} \approx \pm 1.414$

   Mais la dérivée PREMIÈRE reste continue et non-constante partout !

---

## 9. Ressources

### 9.1 Papers Originaux

1. **GELU** (2016)
   - "Gaussian Error Linear Units (GELUs)"
   - Hendrycks & Gimpel
   - https://arxiv.org/abs/1606.08415

2. **SiLU/Swish** (2017)
   - "Searching for Activation Functions"
   - Ramachandran, Zoph, Le (Google Brain)
   - https://arxiv.org/abs/1710.05941

### 9.2 Tutoriels et Articles

- [Ultralytics - GELU Explained](https://www.ultralytics.com/glossary/gelu-gaussian-error-linear-unit)
- [Ultralytics - SiLU Deep Learning Guide](https://www.ultralytics.com/glossary/silu-sigmoid-linear-unit)
- [Medium - Activation Functions in Focus: ReLU, GELU, SiLU](https://medium.com/@varun_mishra/activation-functions-in-focus-understanding-relu-gelu-and-silu-841ed1c6df0c)

### 9.3 Livres

- **Deep Learning** - Goodfellow, Bengio, Courville
  - Section 6.3: Hidden Units
  - https://www.deeplearningbook.org/

### 9.4 Code de Référence

Le code d'activation est déjà dans ton projet :
`/root/these-tidiane-diallo/code/utils/activations.py`

---

## Résumé

| Concept | Point clé | Importance pour la thèse |
|---------|-----------|--------------------------|
| **ReLU** | Piecewise linear, points critiques | Vulnérable aux attaques |
| **GELU** | Lisse, probabiliste | Plus résistant, utilisé dans Transformers |
| **SiLU** | Lisse, auto-gating | Plus résistant, utilisé dans vision |
| **Dérivée seconde** | ReLU = 0, GELU/SiLU ≠ 0 | Explique la différence de vulnérabilité |

**Message central :**

> La nature **piecewise-linear** de ReLU crée des "signatures" détectables par les attaques cryptanalytiques. Les activations lisses comme GELU et SiLU masquent ces signatures, rendant l'extraction beaucoup plus difficile.

**C'est le défi que ta thèse doit résoudre : comment extraire des modèles malgré cette difficulté ?**

---

*Cours créé le 31 Janvier 2026 - Thèse Tidiane DIALLO*
