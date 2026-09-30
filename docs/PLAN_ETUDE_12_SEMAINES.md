# Plan d'Étude Intensif - 12 Semaines

## Remise à Niveau: Deep Learning + Cryptographie + Sécurité ML

**Doctorant**: Tidiane DIALLO
**Objectif**: Maîtrise complète du sujet de thèse
**Durée**: 12 semaines (environ 20-25h/semaine recommandées)

---

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 (S1-S4)     │  PHASE 2 (S5-S8)    │  PHASE 3 (S9-S12)    │
│  ─────────────────   │  ─────────────────  │  ─────────────────   │
│  Deep Learning       │  Cryptographie      │  Spécialisation      │
│  Fondamentaux        │  & Sécurité ML      │  Thèse               │
└─────────────────────────────────────────────────────────────────────┘
```

---

# PHASE 1: FONDATIONS DEEP LEARNING

## Semaine 1: Perceptron et MLP

### Objectifs
- [ ] Comprendre le perceptron simple
- [ ] Maîtriser la propagation avant (forward pass)
- [ ] Comprendre la rétropropagation (backpropagation)
- [ ] Calculer les gradients à la main

### Ressources

#### Vidéos (5h)
1. **3Blue1Brown - Neural Networks** (4 vidéos, ~1h total)
   - https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
   - Excellente intuition visuelle

2. **Andrej Karpathy - Micrograd** (2h30)
   - https://www.youtube.com/watch?v=VMj-3S1tku0
   - Implémentation from scratch

#### Lecture (10h)
- **"Deep Learning"** - Goodfellow, Bengio, Courville
  - Chapitre 6: Deep Feedforward Networks
  - Disponible gratuitement: https://www.deeplearningbook.org/

#### Code de référence
```python 
# micrograd de Karpathy
# https://github.com/karpathy/micrograd
```

### Exercices pratiques

#### Exercice 1.1: Perceptron simple (2h)
```python
"""
Implémenter un perceptron simple pour la fonction AND
Sans utiliser de bibliothèque (NumPy autorisé pour matrices)
"""
import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        # TODO: Initialiser poids et biais
        pass

    def forward(self, x):
        # TODO: Calculer sortie
        pass

    def train(self, X, y, epochs=100, lr=0.1):
        # TODO: Implémenter l'apprentissage
        pass

# Test sur AND
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

# Entraîner et vérifier
```

#### Exercice 1.2: MLP pour XOR (4h)
```python
"""
Le XOR n'est pas linéairement séparable
Implémenter un MLP 2-2-1 (2 entrées, 2 cachés, 1 sortie)
Avec backpropagation manuelle
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self):
        # Couche 1: 2 -> 2
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.zeros((1, 2))
        # Couche 2: 2 -> 1
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        # TODO: Implémenter forward pass
        # Sauvegarder les activations pour backprop
        pass

    def backward(self, X, y, output):
        # TODO: Implémenter backpropagation
        # Calculer gradients et mettre à jour poids
        pass

    def train(self, X, y, epochs=10000, lr=0.5):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

# Test
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

mlp = MLP()
mlp.train(X, y)
print(mlp.forward(X))  # Devrait approcher [0, 1, 1, 0]
```

#### Exercice 1.3: Dérivation manuelle (2h)
```
Sur papier, pour un MLP 2-3-1:
1. Écrire les équations du forward pass
2. Calculer ∂L/∂W2 (couche de sortie)
3. Calculer ∂L/∂W1 (couche cachée) par chain rule
4. Vérifier numériquement avec différences finies
```

### Validation
- [ ] MLP XOR converge vers erreur < 0.01
- [ ] Capable d'expliquer backpropagation sans notes
- [ ] Dérivation manuelle correcte

---

## Semaine 2: Fonctions d'activation

### Objectifs
- [ ] Comprendre mathématiquement ReLU, Sigmoid, Tanh, GELU, SiLU
- [ ] Calculer les dérivées de chaque fonction
- [ ] Visualiser les comportements et comprendre les différences
- [ ] Comprendre pourquoi ReLU est "piecewise linear"

### Ressources

#### Papers originaux (obligatoires)
1. **GELU**: Hendrycks & Gimpel (2016)
   - "Gaussian Error Linear Units (GELUs)"
   - https://arxiv.org/abs/1606.08415

2. **Swish/SiLU**: Ramachandran et al. (2017)
   - "Searching for Activation Functions"
   - https://arxiv.org/abs/1710.05941

#### Lecture complémentaire
- Deep Learning Book, Section 6.3: Hidden Units

### Exercices pratiques

#### Exercice 2.1: Implémentation et visualisation (3h)
```python
"""
Implémenter toutes les activations et leurs dérivées
Visualiser sur [-5, 5]
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def gelu(x):
    """GELU exact"""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_derivative(x):
    """Dérivée de GELU"""
    # TODO: Implémenter
    pass

def gelu_approx(x):
    """Approximation GELU utilisée en pratique"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    """SiLU = Swish avec beta=1"""
    return x * sigmoid(x)

def silu_derivative(x):
    """Dérivée de SiLU"""
    s = sigmoid(x)
    return s + x * s * (1 - s)

# Visualisation
x = np.linspace(-5, 5, 1000)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

activations = [
    ('ReLU', relu, relu_derivative),
    ('Sigmoid', sigmoid, sigmoid_derivative),
    ('Tanh', tanh, tanh_derivative),
    ('GELU', gelu, gelu_derivative),
    ('SiLU/Swish', silu, silu_derivative),
]

for ax, (name, func, deriv) in zip(axes.flat, activations):
    ax.plot(x, func(x), 'b-', label=f'{name}(x)', linewidth=2)
    if deriv is not None:
        ax.plot(x, deriv(x), 'r--', label=f"{name}'(x)", linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activations_comparison.png', dpi=150)
plt.show()
```

#### Exercice 2.2: Comprendre piecewise linear (2h)
```python
"""
Démontrer visuellement que ReLU crée des régions linéaires
Pour un MLP simple: 2 entrées, 3 neurones cachés ReLU, 1 sortie
"""
import numpy as np
import matplotlib.pyplot as plt

# Réseau fixe (poids prédéfinis pour illustration)
W1 = np.array([[1, -1], [-1, 1], [0.5, 0.5]])  # 3x2
b1 = np.array([0.5, 0.5, -0.5])                 # 3
W2 = np.array([[1], [1], [1]])                  # 3x1
b2 = np.array([0])                              # 1

def network(x):
    """x: (N, 2)"""
    h = np.maximum(0, x @ W1.T + b1)  # ReLU
    return h @ W2 + b2

# Grille d'entrée
xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
X_grid = np.column_stack([xx.ravel(), yy.ravel()])
Z = network(X_grid).reshape(xx.shape)

# Visualisation
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=50, cmap='RdBu')
plt.colorbar(label='Output')

# Tracer les hyperplans critiques (W1·x + b1 = 0)
for i in range(3):
    # W1[i,0]*x + W1[i,1]*y + b1[i] = 0
    # y = -(W1[i,0]*x + b1[i]) / W1[i,1]
    if abs(W1[i,1]) > 1e-6:
        x_line = np.linspace(-2, 2, 100)
        y_line = -(W1[i,0]*x_line + b1[i]) / W1[i,1]
        mask = (y_line >= -2) & (y_line <= 2)
        plt.plot(x_line[mask], y_line[mask], 'k--', linewidth=2,
                 label=f'Hyperplan {i+1}')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Régions linéaires créées par ReLU')
plt.legend()
plt.savefig('relu_piecewise_linear.png', dpi=150)
plt.show()

# TODO: Compter le nombre de régions distinctes
```

#### Exercice 2.3: Dérivation formelle GELU (2h)
```
Sur papier:
1. Écrire GELU(x) = x · Φ(x) où Φ est la CDF normale standard
2. Calculer GELU'(x) en utilisant la règle du produit
3. Montrer que GELU'(x) = Φ(x) + x·φ(x) où φ est la PDF normale
4. Vérifier numériquement avec différences finies
```

#### Exercice 2.4: Impact sur l'entraînement (3h)
```python
"""
Comparer l'entraînement d'un MLP sur MNIST avec différentes activations
Mesurer: vitesse de convergence, accuracy finale
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# TODO: Implémenter comparaison ReLU vs GELU vs SiLU
# Tracer les courbes de loss et accuracy
```

### Validation
- [ ] Capable de calculer dérivées à la main
- [ ] Visualisation claire des différences
- [ ] Compréhension de "piecewise linear"

---

## Semaine 3: Réseaux Convolutionnels (CNN)

### Objectifs
- [ ] Comprendre l'opération de convolution 2D
- [ ] Maîtriser: stride, padding, pooling
- [ ] Comprendre le partage de poids
- [ ] Implémenter un CNN simple

### Ressources

#### Cours
- **Stanford CS231n**: Module sur CNN
  - http://cs231n.stanford.edu/
  - Vidéos disponibles sur YouTube

#### Lecture
- Deep Learning Book, Chapitre 9: Convolutional Networks

### Exercices pratiques

#### Exercice 3.1: Convolution from scratch (3h)
```python
"""
Implémenter la convolution 2D sans utiliser de bibliothèque
"""
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    Convolution 2D

    Args:
        image: (H, W) ou (H, W, C)
        kernel: (kH, kW) ou (kH, kW, C)
        stride: pas de déplacement
        padding: ajout de zéros

    Returns:
        output: image convoluée
    """
    # Ajouter padding
    if padding > 0:
        if len(image.shape) == 2:
            image = np.pad(image, padding, mode='constant')
        else:
            image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)),
                          mode='constant')

    H, W = image.shape[:2]
    kH, kW = kernel.shape[:2]

    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(region * kernel)

    return output

# Test
image = np.random.randn(28, 28)
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])  # Sobel vertical

output = conv2d(image, kernel, padding=1)
print(f"Input shape: {image.shape}")
print(f"Output shape: {output.shape}")

# Visualiser
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(output, cmap='gray')
axes[1].set_title('Convolved (Sobel)')
plt.savefig('convolution_example.png')
```

#### Exercice 3.2: CNN sur MNIST (4h)
```python
"""
Implémenter et entraîner un CNN simple sur MNIST
Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> Softmax
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Définir les couches
        # Conv1: 1 -> 32 channels, kernel 3x3
        # Pool: 2x2
        # Conv2: 32 -> 64 channels, kernel 3x3
        # Pool: 2x2
        # FC: ? -> 10
        pass

    def forward(self, x):
        # TODO: Implémenter forward pass
        pass

# Entraînement
# TODO: DataLoader, boucle d'entraînement, évaluation
```

#### Exercice 3.3: Visualisation des features (2h)
```python
"""
Visualiser les filtres appris et les feature maps
"""
# TODO: Après entraînement, visualiser:
# 1. Les kernels de la première couche conv
# 2. Les activations sur une image exemple
```

### Validation
- [ ] CNN atteint >98% sur MNIST
- [ ] Capable d'expliquer convolution et partage de poids
- [ ] Visualisation des filtres comprise

---

## Semaine 4: Architectures Modernes

### Objectifs
- [ ] Comprendre ResNet et skip connections
- [ ] Introduction aux Transformers et attention
- [ ] Savoir où chaque activation est utilisée en pratique

### Ressources

#### Papers
1. **ResNet**: "Deep Residual Learning for Image Recognition" (2015)
   - https://arxiv.org/abs/1512.03385

2. **Transformer**: "Attention Is All You Need" (2017)
   - https://arxiv.org/abs/1706.03762

#### Blogs
- **The Illustrated Transformer**: http://jalammar.github.io/illustrated-transformer/
- **The Annotated Transformer**: https://nlp.seas.harvard.edu/2018/04/03/attention.html

### Exercices pratiques

#### Exercice 4.1: ResNet block (3h)
```python
"""
Implémenter un bloc résiduel et comprendre pourquoi ça aide
"""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO: Implémenter bloc résiduel
        # Conv -> BN -> ReLU -> Conv -> BN
        # + skip connection
        pass

    def forward(self, x):
        # TODO: Implémenter avec skip connection
        # out = F(x) + x
        pass

# TODO: Créer un mini-ResNet et comparer avec un réseau sans skip connections
```

#### Exercice 4.2: Self-Attention (3h)
```python
"""
Implémenter le mécanisme d'attention from scratch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # TODO: Projections Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        """
        # TODO: Implémenter attention
        # Q = x @ W_q, K = x @ W_k, V = x @ W_v
        # Attention = softmax(Q @ K^T / sqrt(d)) @ V
        pass

# Test
batch, seq_len, embed_dim = 2, 10, 64
x = torch.randn(batch, seq_len, embed_dim)
attn = SelfAttention(embed_dim)
out = attn(x)
print(f"Input: {x.shape}, Output: {out.shape}")
```

#### Exercice 4.3: Comparaison activations sur CIFAR-10 (4h)
```python
"""
Entraîner un petit réseau sur CIFAR-10 avec:
- ReLU
- GELU
- SiLU

Comparer: convergence, accuracy finale, comportement des gradients
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class SmallNet(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()

        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'silu':
            act = nn.SiLU()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            act,
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            act,
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# TODO: Entraîner les 3 variantes et comparer
# Tracer les courbes de loss et accuracy
```

### Validation
- [ ] Comprendre pourquoi skip connections aident
- [ ] Implémenter attention basique
- [ ] Résultats comparatifs des activations

---

# PHASE 2: CRYPTOGRAPHIE & SÉCURITÉ ML

## Semaine 5: Fondamentaux Cryptographie

### Objectifs
- [ ] Revoir les chiffrements par blocs (structure)
- [ ] Comprendre la cryptanalyse différentielle
- [ ] Maîtriser les modèles d'attaque

### Ressources

#### Cours
- **Cryptography I** - Stanford/Coursera (Dan Boneh)
  - https://www.coursera.org/learn/crypto
  - Semaines 1-3 particulièrement

#### Papers
- Biham & Shamir (1990): "Differential Cryptanalysis of DES-like Cryptosystems"

### Exercices pratiques

#### Exercice 5.1: Structure d'un chiffrement par blocs (2h)
```python
"""
Implémenter un chiffrement par blocs jouet pour comprendre la structure
SPN: Substitution-Permutation Network
"""
import numpy as np

# S-box simple (4 bits -> 4 bits)
SBOX = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
        0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]

def substitute(state, sbox):
    """Applique la S-box à chaque nibble"""
    result = 0
    for i in range(4):  # 16 bits = 4 nibbles
        nibble = (state >> (4*i)) & 0xF
        result |= sbox[nibble] << (4*i)
    return result

def permute(state):
    """Permutation bit à bit simple"""
    # TODO: Implémenter une permutation
    pass

def key_schedule(master_key, rounds):
    """Génère les sous-clés"""
    # TODO: Implémenter
    pass

def spn_encrypt(plaintext, key, rounds=4):
    """Chiffrement SPN"""
    state = plaintext
    subkeys = key_schedule(key, rounds)

    for r in range(rounds):
        state ^= subkeys[r]      # AddKey
        state = substitute(state, SBOX)  # SubBytes
        if r < rounds - 1:
            state = permute(state)  # Permutation

    state ^= subkeys[rounds]  # Final key addition
    return state
```

#### Exercice 5.2: Cryptanalyse différentielle basique (4h)
```python
"""
Comprendre la cryptanalyse différentielle sur un exemple simple
"""
import numpy as np
from collections import defaultdict

# Table des différences pour notre S-box
def build_difference_table(sbox):
    """
    DDT[Δin][Δout] = nombre de paires (x, x') telles que
    sbox[x] ^ sbox[x ^ Δin] = Δout
    """
    n = len(sbox)
    ddt = np.zeros((n, n), dtype=int)

    for x in range(n):
        for delta_in in range(n):
            x_prime = x ^ delta_in
            delta_out = sbox[x] ^ sbox[x_prime]
            ddt[delta_in][delta_out] += 1

    return ddt

ddt = build_difference_table(SBOX)
print("Difference Distribution Table:")
print(ddt)

# Trouver les meilleures caractéristiques différentielles
# (delta_in, delta_out) avec haute probabilité
```

#### Exercice 5.3: Modèles d'attaque (1h)
```
Sur papier, décrire et comparer:
1. Attaque à clair connu (Known Plaintext Attack)
2. Attaque à clair choisi (Chosen Plaintext Attack)
3. Attaque à chiffré choisi (Chosen Ciphertext Attack)

Pour chaque type:
- Que connaît l'attaquant?
- Que peut-il faire?
- Donner un exemple
```

### Validation
- [ ] Comprendre structure SPN
- [ ] Savoir lire une DDT
- [ ] Distinguer les modèles d'attaque

---

## Semaine 6: Sécurité ML - Modèles d'attaque

### Objectifs
- [ ] Comprendre white-box vs black-box
- [ ] Distinguer raw-output vs hard-label
- [ ] Comprendre le concept d'oracle

### Ressources

#### Papers
1. **Tramèr et al. (2016)**: "Stealing Machine Learning Models through Prediction APIs"
   - https://arxiv.org/abs/1609.02943

2. **Survey**: Biggio & Roli - "Wild Patterns: Ten Years After the Rise of Adversarial ML"

### Exercices pratiques

#### Exercice 6.1: Simuler différents niveaux d'accès (2h)
```python
"""
Créer des "oracles" avec différents niveaux d'accès
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelOracle:
    def __init__(self, model, access_level='hard_label'):
        """
        access_level: 'white_box', 'raw_output', 'hard_label'
        """
        self.model = model
        self.model.eval()
        self.access_level = access_level
        self.query_count = 0

    def query(self, x):
        self.query_count += 1

        with torch.no_grad():
            logits = self.model(x)

        if self.access_level == 'white_box':
            # Accès complet: logits + paramètres
            return {
                'logits': logits,
                'params': dict(self.model.named_parameters())
            }
        elif self.access_level == 'raw_output':
            # Logits ou probabilités
            return F.softmax(logits, dim=-1)
        elif self.access_level == 'hard_label':
            # Uniquement la classe prédite
            return logits.argmax(dim=-1)

    def get_query_count(self):
        return self.query_count

# Test
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 5)
)

oracle_hard = ModelOracle(model, 'hard_label')
oracle_raw = ModelOracle(model, 'raw_output')

x = torch.randn(1, 10)
print("Hard label:", oracle_hard.query(x))
print("Raw output:", oracle_raw.query(x))
```

#### Exercice 6.2: Model stealing basique (4h)
```python
"""
Implémenter une attaque de model stealing simple
Stratégie: entraîner un modèle "voleur" sur les réponses de l'oracle
"""
import torch
import torch.nn as nn
import torch.optim as optim

class ModelStealing:
    def __init__(self, oracle, input_dim, output_dim):
        self.oracle = oracle
        self.stolen_model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.optimizer = optim.Adam(self.stolen_model.parameters())

    def steal(self, num_queries=1000, batch_size=32):
        """
        Voler le modèle en utilisant des requêtes aléatoires
        """
        for _ in range(num_queries // batch_size):
            # Générer des entrées aléatoires
            x = torch.randn(batch_size, 10)  # Ajuster dimension

            # Obtenir les labels de l'oracle
            with torch.no_grad():
                if self.oracle.access_level == 'hard_label':
                    labels = self.oracle.query(x)
                else:
                    labels = self.oracle.query(x).argmax(dim=-1)

            # Entraîner le modèle volé
            self.optimizer.zero_grad()
            pred = self.stolen_model(x)
            loss = nn.CrossEntropyLoss()(pred, labels)
            loss.backward()
            self.optimizer.step()

        return self.stolen_model

    def evaluate_fidelity(self, test_x):
        """Mesurer à quel point le modèle volé imite l'original"""
        with torch.no_grad():
            original_pred = self.oracle.query(test_x)
            stolen_pred = self.stolen_model(test_x).argmax(dim=-1)

            if self.oracle.access_level != 'hard_label':
                original_pred = original_pred.argmax(dim=-1)

            fidelity = (original_pred == stolen_pred).float().mean()
        return fidelity.item()

# TODO: Tester avec différents niveaux d'accès
# Comparer le nombre de requêtes nécessaires
```

### Validation
- [ ] Comprendre les différences entre modes d'accès
- [ ] Model stealing basique fonctionnel
- [ ] Mesure de fidélité comprise

---

## Semaine 7: Attaques d'extraction - Théorie

### Objectifs
- [ ] Lire et comprendre CRYPTO 2020 en profondeur
- [ ] Comprendre la notion de point critique
- [ ] Maîtriser l'algèbre linéaire nécessaire

### Ressources

#### Paper principal
- Carlini et al. CRYPTO 2020 (lecture complète ~10h)

#### Prérequis mathématiques
- Algèbre linéaire: SVD, résolution de systèmes
- Analyse: différentiation numérique

### Exercices pratiques

#### Exercice 7.1: Points critiques ReLU (3h)
```python
"""
Comprendre et trouver les points critiques d'un réseau ReLU simple
"""
import numpy as np
import torch
import torch.nn as nn

def find_critical_points_1d(model, x_range=(-5, 5), resolution=10000):
    """
    Pour un réseau 1D, trouver les points où le comportement change
    """
    x = torch.linspace(x_range[0], x_range[1], resolution).reshape(-1, 1)

    with torch.no_grad():
        y = model(x).numpy()

    # Calculer la dérivée numérique
    dy = np.diff(y.flatten())
    dx = (x_range[1] - x_range[0]) / resolution
    derivative = dy / dx

    # Trouver où la dérivée change brusquement (points critiques)
    derivative_change = np.abs(np.diff(derivative))
    threshold = np.mean(derivative_change) + 3 * np.std(derivative_change)
    critical_indices = np.where(derivative_change > threshold)[0]

    critical_points = x[critical_indices + 1].numpy()

    return critical_points

# Réseau simple 1D: 1 -> 3 -> 1
model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

# Initialisation connue pour vérification
with torch.no_grad():
    model[0].weight.copy_(torch.tensor([[1.0], [-0.5], [2.0]]))
    model[0].bias.copy_(torch.tensor([0.5, 1.0, -1.0]))
    model[2].weight.copy_(torch.tensor([[1.0, 1.0, 1.0]]))
    model[2].bias.copy_(torch.tensor([0.0]))

# Les points critiques théoriques sont:
# W[0]*x + b[0] = 0 => x = -b[0]/W[0] = -0.5/1 = -0.5
# W[1]*x + b[1] = 0 => x = -1.0/-0.5 = 2.0
# W[2]*x + b[2] = 0 => x = 1.0/2.0 = 0.5

critical = find_critical_points_1d(model)
print("Points critiques trouvés:", critical.flatten())
print("Points critiques théoriques: [-0.5, 0.5, 2.0]")
```

#### Exercice 7.2: Extraction d'une couche (4h)
```python
"""
Extraire les poids d'un réseau 1 couche cachée ReLU
En utilisant les points critiques et les dérivées
"""
import numpy as np
import torch
import torch.nn as nn

class SimpleExtractor:
    def __init__(self, oracle, input_dim, hidden_dim):
        self.oracle = oracle  # Fonction qui évalue le réseau
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def numerical_gradient(self, x, epsilon=1e-5):
        """Calculer le gradient par différences finies"""
        grad = np.zeros(self.input_dim)
        for i in range(self.input_dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (self.oracle(x_plus) - self.oracle(x_minus)) / (2 * epsilon)
        return grad

    def find_critical_direction(self, x, direction, t_range=(-2, 2), resolution=1000):
        """Trouver un point critique le long d'une direction"""
        t_values = np.linspace(t_range[0], t_range[1], resolution)
        outputs = [self.oracle(x + t * direction) for t in t_values]

        # Détecter les changements de pente
        derivatives = np.diff(outputs) / (t_values[1] - t_values[0])
        changes = np.abs(np.diff(derivatives))

        if changes.max() > 0.1:  # Seuil arbitraire
            idx = changes.argmax()
            return x + t_values[idx + 1] * direction
        return None

    def extract(self, num_samples=100):
        """
        Algorithme d'extraction simplifié:
        1. Trouver des points critiques
        2. Calculer les gradients de part et d'autre
        3. La différence donne information sur les poids
        """
        # TODO: Implémenter extraction complète
        # C'est complexe - voir paper pour détails
        pass

# Test avec un réseau connu
hidden_dim = 3
model = nn.Sequential(
    nn.Linear(2, hidden_dim, bias=True),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1, bias=True)
)

def oracle(x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()

# TODO: Essayer d'extraire les poids
```

#### Exercice 7.3: Lire CRYPTO 2020 (6h)
```
Lecture guidée du paper:
1. Abstract et Introduction (30 min)
   - Quelle est la contribution principale?
   - Pourquoi l'analogie avec la cryptanalyse?

2. Section 2 - Preliminaries (1h)
   - Comment formalisent-ils le problème?
   - Quelles hypothèses font-ils?

3. Section 3 - Attack Overview (2h)
   - Quel est le plan d'attaque général?
   - Comment trouvent-ils les points critiques?

4. Section 4 - Full Attack (2h)
   - Détails techniques
   - Complexité en requêtes

5. Section 5 - Experiments (30 min)
   - Sur quels modèles testent-ils?
   - Quelle précision atteignent-ils?

NOTES À PRENDRE:
- Équations clés
- Hypothèses critiques
- Limitations mentionnées
```

### Validation
- [ ] Capable d'expliquer l'attaque CRYPTO 2020
- [ ] Trouver des points critiques sur réseau simple
- [ ] Notes de lecture complètes

---

## Semaine 8: Attaques d'extraction - Pratique

### Objectifs
- [ ] Reproduire l'attaque sur un petit réseau
- [ ] Comprendre les limitations pratiques
- [ ] Lire EUROCRYPT 2024

### Exercices pratiques

#### Exercice 8.1: Reproduction simplifiée (6h)
```python
"""
Reproduire l'attaque CRYPTO 2020 sur un réseau jouet
Réseau: 2 entrées -> 3 neurones ReLU -> 1 sortie
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

class CryptoAttack2020:
    def __init__(self, oracle_fn, input_dim=2, hidden_dim=3, output_dim=1):
        self.oracle = oracle_fn
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.query_count = 0

    def query(self, x):
        self.query_count += 1
        return self.oracle(x)

    def numerical_jacobian(self, x, eps=1e-7):
        """Jacobien par différences finies"""
        x = np.array(x, dtype=np.float64)
        f0 = self.query(x)
        jac = np.zeros((len(np.atleast_1d(f0)), len(x)))

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = self.query(x_plus)
            jac[:, i] = (np.atleast_1d(f_plus) - np.atleast_1d(f0)) / eps

        return jac

    def find_critical_point(self, x_start, direction, tol=1e-6):
        """
        Trouver un point critique le long d'une direction
        par recherche binaire sur le changement de gradient
        """
        t_low, t_high = -5.0, 5.0

        while t_high - t_low > tol:
            t_mid = (t_low + t_high) / 2

            # Comparer les gradients de part et d'autre
            x_left = x_start + (t_mid - 0.01) * direction
            x_right = x_start + (t_mid + 0.01) * direction

            grad_left = self.numerical_jacobian(x_left)
            grad_right = self.numerical_jacobian(x_right)

            # Si les gradients sont différents, il y a un point critique
            if not np.allclose(grad_left, grad_right, atol=1e-4):
                # Affiner la recherche
                # (simplifié ici - le vrai algorithme est plus sophistiqué)
                return x_start + t_mid * direction

            # Sinon, chercher dans une direction
            t_low = t_mid

        return None

    def extract_weights(self):
        """
        Algorithme principal d'extraction
        """
        # 1. Trouver plusieurs points critiques
        critical_points = []
        gradients_before = []
        gradients_after = []

        for _ in range(self.hidden_dim * 10):
            x_start = np.random.randn(self.input_dim) * 2
            direction = np.random.randn(self.input_dim)
            direction /= np.linalg.norm(direction)

            cp = self.find_critical_point(x_start, direction)
            if cp is not None:
                critical_points.append(cp)

                # Gradients de part et d'autre
                grad_before = self.numerical_jacobian(cp - 0.01 * direction)
                grad_after = self.numerical_jacobian(cp + 0.01 * direction)
                gradients_before.append(grad_before)
                gradients_after.append(grad_after)

        # 2. Extraire les poids à partir des différences de gradients
        # Δgradient = W2 * (0 ou W1) donc Δgradient nous donne W2 @ diag @ W1
        # C'est une simplification - le vrai algorithme résout un système

        print(f"Points critiques trouvés: {len(critical_points)}")
        print(f"Requêtes utilisées: {self.query_count}")

        # TODO: Résoudre le système pour extraire W1, b1, W2, b2
        return critical_points, gradients_before, gradients_after

# Créer un réseau cible
target_model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

def oracle(x):
    with torch.no_grad():
        return target_model(torch.tensor(x, dtype=torch.float32)).numpy()

# Lancer l'attaque
attacker = CryptoAttack2020(oracle)
results = attacker.extract_weights()
```

#### Exercice 8.2: Lire EUROCRYPT 2024 (4h)
```
Lecture guidée:
1. Quelle est l'amélioration par rapport à CRYPTO 2020?
2. Comment passent-ils de temps exponentiel à polynomial?
3. Quelles nouvelles techniques introduisent-ils?
4. Quelles sont les limitations qui restent?
```

### Validation
- [ ] Extraction fonctionnelle sur réseau jouet
- [ ] Comprendre les différences entre les papers
- [ ] Notes comparatives complètes

---

# PHASE 3: SPÉCIALISATION THÈSE

## Semaine 9: Au-delà de ReLU - Analyse

### Objectifs
- [ ] Analyser pourquoi les attaques ReLU échouent sur GELU/SiLU
- [ ] Explorer des adaptations possibles
- [ ] Identifier les pistes de recherche

### Exercices pratiques

#### Exercice 9.1: Tester l'attaque ReLU sur GELU (4h)
```python
"""
Appliquer l'attaque CRYPTO 2020 sur un réseau GELU
Documenter pourquoi ça échoue
"""
import torch
import torch.nn as nn
import numpy as np

# Réseau GELU
gelu_model = nn.Sequential(
    nn.Linear(2, 3),
    nn.GELU(),
    nn.Linear(3, 1)
)

def gelu_oracle(x):
    with torch.no_grad():
        return gelu_model(torch.tensor(x, dtype=torch.float32)).numpy()

# Appliquer l'attaque (de l'exercice 8.1)
# Documenter:
# 1. Les "points critiques" trouvés (ou leur absence)
# 2. La différence de comportement des gradients
# 3. Pourquoi la linéarité par morceaux était cruciale
```

#### Exercice 9.2: Analyse mathématique (3h)
```
Sur papier:
1. Pour GELU: calculer la dérivée seconde ∂²GELU/∂x²
   - Montrer qu'elle n'est jamais nulle (contrairement à ReLU)
   - Qu'est-ce que ça implique pour la méthode des points critiques?

2. Pour SiLU: même analyse
   - SiLU(x) = x·σ(x)
   - Calculer SiLU''(x)

3. Proposer des pistes:
   - Approximation locale par fonction piecewise linear?
   - Utilisation de points d'inflexion au lieu de points critiques?
   - Méthodes d'optimisation différentes?
```

#### Exercice 9.3: Approximation piecewise linear (3h)
```python
"""
Explorer l'idée d'approximer GELU par une fonction piecewise linear
et voir si ça permet d'adapter les attaques
"""
import numpy as np
import matplotlib.pyplot as plt

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def piecewise_linear_approx(x, breakpoints, slopes, intercepts):
    """Approximation PWL avec breakpoints donnés"""
    result = np.zeros_like(x)
    for i in range(len(breakpoints) - 1):
        mask = (x >= breakpoints[i]) & (x < breakpoints[i+1])
        result[mask] = slopes[i] * x[mask] + intercepts[i]
    return result

def fit_pwl_to_gelu(num_segments=5):
    """Trouver la meilleure approximation PWL de GELU"""
    x_range = np.linspace(-3, 3, 1000)
    y_gelu = gelu(x_range)

    # Breakpoints uniformément espacés (simplification)
    breakpoints = np.linspace(-3, 3, num_segments + 1)
    slopes = []
    intercepts = []

    for i in range(num_segments):
        mask = (x_range >= breakpoints[i]) & (x_range < breakpoints[i+1])
        x_seg = x_range[mask]
        y_seg = y_gelu[mask]

        # Régression linéaire sur le segment
        A = np.vstack([x_seg, np.ones_like(x_seg)]).T
        slope, intercept = np.linalg.lstsq(A, y_seg, rcond=None)[0]
        slopes.append(slope)
        intercepts.append(intercept)

    return breakpoints, slopes, intercepts

# Visualiser l'approximation
breakpoints, slopes, intercepts = fit_pwl_to_gelu(num_segments=10)
x = np.linspace(-3, 3, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x, gelu(x), 'b-', label='GELU', linewidth=2)
plt.plot(x, piecewise_linear_approx(x, breakpoints, slopes, intercepts),
         'r--', label='PWL approximation', linewidth=2)
plt.legend()
plt.grid(True)
plt.title('Approximation PWL de GELU')
plt.savefig('gelu_pwl_approx.png')

# Question: Cette approximation permet-elle d'adapter les attaques?
# Quelle erreur d'approximation est acceptable?
```

### Validation
- [ ] Documentation de l'échec sur GELU
- [ ] Analyse mathématique complète
- [ ] Au moins une piste explorée

---

## Semaine 10: Architectures CNN

### Objectifs
- [ ] Comprendre les spécificités des CNN pour l'extraction
- [ ] Analyser le paper side-channel + CNN
- [ ] Identifier les défis spécifiques

### Exercices pratiques

#### Exercice 10.1: Analyse du partage de poids (2h)
```python
"""
Comparer le nombre de paramètres et la structure entre MLP et CNN
"""
import torch
import torch.nn as nn

# MLP pour 28x28 images -> 10 classes
mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# CNN équivalent
cnn = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),  # 28x28 -> 28x28
    nn.ReLU(),
    nn.MaxPool2d(2),                  # 28x28 -> 14x14
    nn.Conv2d(32, 64, 3, padding=1), # 14x14 -> 14x14
    nn.ReLU(),
    nn.MaxPool2d(2),                  # 14x14 -> 7x7
    nn.Flatten(),
    nn.Linear(64*7*7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"MLP: {count_params(mlp):,} paramètres")
print(f"CNN: {count_params(cnn):,} paramètres")

# Question: Comment le partage de poids affecte-t-il l'extraction?
# Le fait qu'un kernel soit réutilisé sur toute l'image donne-t-il
# plus ou moins d'information à l'attaquant?
```

#### Exercice 10.2: Lire le paper side-channel (3h)
```
Lecture guidée de arXiv:2411.10174:
1. Comment combinent-ils side-channel et extraction cryptanalytique?
2. Quelles informations le side-channel révèle-t-il?
3. Comment adaptent-ils la méthode aux CNN?
4. Quelles limitations identifient-ils?
```

#### Exercice 10.3: Extraction CNN simplifiée (5h)
```python
"""
Tenter d'extraire un très petit CNN
"""
import torch
import torch.nn as nn
import numpy as np

# CNN minimal: 1 couche conv + 1 couche dense
tiny_cnn = nn.Sequential(
    nn.Conv2d(1, 2, 3, padding=1),  # 2 filtres 3x3
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2*8*8, 2)  # Pour images 8x8
)

# Initialisation connue pour test
with torch.no_grad():
    tiny_cnn[0].weight.copy_(torch.randn(2, 1, 3, 3))
    tiny_cnn[0].bias.copy_(torch.randn(2))

def cnn_oracle(x):
    """x: (8, 8) numpy array"""
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tiny_cnn(x_tensor).numpy()

# TODO: Adapter l'attaque d'extraction pour ce CNN
# Défis:
# 1. Le même kernel est appliqué à différentes positions
# 2. Comment exploiter cette redondance?
# 3. Le pooling (même absent ici) complique l'analyse
```

### Validation
- [ ] Comprendre les différences MLP vs CNN pour l'extraction
- [ ] Notes sur le paper side-channel
- [ ] Tentative documentée sur CNN

---

## Semaine 11: Défenses

### Objectifs
- [ ] Comprendre MEA-Defender en profondeur
- [ ] Implémenter une défense simple
- [ ] Analyser le trade-off sécurité/performance

### Exercices pratiques

#### Exercice 11.1: Implémenter noise injection (3h)
```python
"""
Défense simple: ajouter du bruit aux réponses
Mesurer l'impact sur les attaques et sur la précision
"""
import torch
import torch.nn as nn
import numpy as np

class NoisyOracle:
    def __init__(self, model, noise_std=0.1):
        self.model = model
        self.noise_std = noise_std

    def query(self, x, mode='hard_label'):
        with torch.no_grad():
            logits = self.model(x)

            # Ajouter du bruit aux logits
            noisy_logits = logits + torch.randn_like(logits) * self.noise_std

            if mode == 'raw_output':
                return torch.softmax(noisy_logits, dim=-1)
            else:
                return noisy_logits.argmax(dim=-1)

    def clean_query(self, x):
        """Pour évaluer la vraie accuracy"""
        with torch.no_grad():
            return self.model(x).argmax(dim=-1)

# Test: impact sur la précision
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Entraîner le modèle sur MNIST (code omis)

# Mesurer accuracy avec différents niveaux de bruit
noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
for noise in noise_levels:
    oracle = NoisyOracle(model, noise_std=noise)
    # TODO: Mesurer accuracy et succès de l'attaque d'extraction
```

#### Exercice 11.2: Implémenter rate limiting (2h)
```python
"""
Défense: limiter le nombre de requêtes
"""
import time
from collections import deque

class RateLimitedOracle:
    def __init__(self, model, max_queries_per_minute=100):
        self.model = model
        self.max_qpm = max_queries_per_minute
        self.query_times = deque()

    def query(self, x):
        current_time = time.time()

        # Nettoyer les vieilles requêtes
        while self.query_times and current_time - self.query_times[0] > 60:
            self.query_times.popleft()

        # Vérifier la limite
        if len(self.query_times) >= self.max_qpm:
            raise Exception("Rate limit exceeded")

        self.query_times.append(current_time)

        with torch.no_grad():
            return self.model(x).argmax(dim=-1)

# Question: Combien de requêtes les attaques d'extraction nécessitent-elles?
# Quel taux de limitation serait efficace tout en permettant l'utilisation légitime?
```

#### Exercice 11.3: Analyser MEA-Defender (5h)
```python
"""
Implémenter une version simplifiée de MEA-Defender
"""
# Lire le paper en détail et implémenter les concepts clés

# Idée principale:
# 1. Créer des watermarks qui ressemblent aux données normales
# 2. Le watermark est un mélange de deux classes
# 3. Loss function qui maintient le watermark dans l'espace des sorties normales

# TODO: Implémenter et tester
```

### Validation
- [ ] Défense par bruit fonctionnelle
- [ ] Compréhension du trade-off
- [ ] Notes sur MEA-Defender

---

## Semaine 12: Synthèse et rédaction

### Objectifs
- [ ] Rédiger l'état de l'art structuré
- [ ] Identifier clairement les contributions potentielles
- [ ] Préparer le plan de thèse détaillé

### Livrables

#### Livrable 12.1: Document État de l'art (20-30 pages)
```
Structure suggérée:
1. Introduction et motivation
2. Contexte: MLaaS et menaces
3. Attaques d'extraction
   3.1 Approches pré-cryptanalytiques
   3.2 CRYPTO 2020 et suivants
   3.3 Extensions et limitations
4. Défenses existantes
   4.1 Watermarking
   4.2 Autres approches
5. Au-delà de ReLU: état actuel et verrous
6. Positionnement et contributions envisagées
```

#### Livrable 12.2: Plan de thèse avec contributions
```
1. Contribution 1: [Titre précis]
   - Question de recherche
   - Méthodologie envisagée
   - Résultats attendus
   - Publication visée

2. Contribution 2: [...]

3. Contribution 3: [...]
```

#### Livrable 12.3: Présentation de mi-parcours (15 slides)
```
Pour présenter à ton directeur:
- Contexte et motivation
- État de l'art
- Questions de recherche
- Méthodologie
- Résultats préliminaires (si disponibles)
- Planning
```

---

## Annexe: Checklist de compétences

### Deep Learning
- [ ] Implémenter MLP from scratch
- [ ] Comprendre et coder backpropagation
- [ ] Connaître les activations (ReLU, GELU, SiLU)
- [ ] Implémenter CNN
- [ ] Comprendre ResNet et Attention

### Cryptographie
- [ ] Comprendre la cryptanalyse différentielle
- [ ] Connaître les modèles d'attaque
- [ ] Faire le lien DNN ↔ chiffrement

### Sécurité ML
- [ ] Comprendre les niveaux d'accès (white/black box)
- [ ] Implémenter model stealing basique
- [ ] Comprendre les attaques d'extraction cryptanalytiques
- [ ] Connaître les défenses (watermarking, etc.)

### Spécifique thèse
- [ ] Lire et comprendre les 5 papers clés
- [ ] Identifier pourquoi GELU/SiLU sont différents
- [ ] Proposer des pistes de recherche originales

---

*Plan d'étude généré le 31 Janvier 2026*
