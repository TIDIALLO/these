# Sprint 2 Semaines — Avant Réunion Directeur
## Tidiane DIALLO | 1er Avril → 14 Avril 2026

> **Objectif**: Comprendre les fondamentaux + présenter au Pr. Ciss une base solide
> **Rythme**: 2-4h/jour | **Niveau**: Débutant absolu → Bases solides

---

## Vue d'ensemble des 2 semaines

```
SEMAINE 1 (1-7 Avril)          SEMAINE 2 (8-14 Avril)
━━━━━━━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━━━━━━━━━━
Maths essentielles pour DL     Deep Learning Pratique
+ Concept ML/DL de base        + Lien avec la thèse
```

---

# SEMAINE 1 — Maths + Concepts ML

## Jour 1 (1 Avril) — Vecteurs & Matrices [2h]

### Pourquoi c'est important pour ta thèse
> Un réseau de neurones = une série de multiplications matricielles. Si tu comprends les matrices, tu comprends comment fonctionne un DNN.

### Ce qu'il faut savoir

#### 1. Vecteur (= liste de nombres)
```
x = [1, 2, 3]   ← vecteur de taille 3
```
Un vecteur représente **une entrée** dans ton réseau (ex: une image 3px)

#### 2. Matrice (= tableau de nombres)
```
W = [[1, 0, 2],
     [3, 1, 0]]   ← matrice 2×3 (2 lignes, 3 colonnes)
```
Une matrice représente **les poids** d'une couche de neurones

#### 3. Multiplication matricielle (FONDAMENTAL)
```
Si W est (2×3) et x est (3×1), alors W·x = (2×1)

Exemple:
W = [[1, 0, 2],    x = [[1],    W·x = [[1×1 + 0×2 + 2×3],   = [[7],
     [3, 1, 0]]         [2],           [3×1 + 1×2 + 0×3]]      [5]]
                        [3]]
```

#### 4. Transposée
```
W  = [[1, 2],     W^T = [[1, 3],
      [3, 4]]            [2, 4]]
```

### Exercice pratique (30min)
```python
import numpy as np

# Exercice 1: Créer un vecteur et une matrice
x = np.array([1, 2, 3])
W = np.array([[1, 0, 2], [3, 1, 0]])

# Calculer W · x (produit matriciel)
result = np.dot(W, x)
print("Résultat:", result)  # Doit afficher [7, 5]

# Exercice 2: Transposée
print("Transposée de W:", W.T)

# Exercice 3: Vrai exemple DNN
# x = image de 784 pixels (28x28 aplatie)
# W = 100 neurones cachés × 784 entrées
x_image = np.random.randn(784)      # 784 pixels
W_couche = np.random.randn(100, 784) # 100 neurones
sortie = np.dot(W_couche, x_image)   # 100 valeurs
print("Forme de la sortie:", sortie.shape)  # (100,)
```

### Ressource vidéo (1h)
- **3Blue1Brown - Essence of Linear Algebra**: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
  - Regarde épisodes 1, 2, 3 (vecteurs, matrices, multiplication)

---

## Jour 2 (2 Avril) —  [2-3h]

### Pourquoi c'est crucial
> L'entraînement d'un DNN = trouver les poids qui minimisent l'erreur. On utilise les dérivées pour savoir **dans quelle direction corriger** les poids.

### Concept 1: Dérivée = pente d'une courbe
```
f(x) = x²
f'(x) = 2x    ← la dérivée

Si x = 3, f'(3) = 6 → la courbe monte avec une pente de 6 à ce point
```

### Concept 2: Gradient = dérivée pour plusieurs variables
```
f(x, y) = x² + y²
∂f/∂x = 2x    ← gradient par rapport à x
∂f/∂y = 2y    ← gradient par rapport à y

gradient = [2x, 2y]  ← vecteur qui pointe vers la montée maximale
```

### Concept 3: Descente de gradient (comment le DNN apprend)
```
nouveau_poids = ancien_poids - α × gradient

α (alpha) = learning rate = "pas d'apprentissage" (ex: 0.01)
```

**Intuition**: Imagine que tu es dans un brouillard sur une montagne et tu veux descendre dans la vallée. Le gradient te dit "tu montes de ce côté". Donc tu vas dans le sens OPPOSÉ.

### Dérivées importantes à connaître
```
f(x) = xⁿ        → f'(x) = n·xⁿ⁻¹
f(x) = eˣ        → f'(x) = eˣ
f(x) = ln(x)     → f'(x) = 1/x
f(x) = sigmoid   → f'(x) = sigmoid(x)·(1 - sigmoid(x))
```

### Exercice pratique (45min)
```python
import numpy as np
import matplotlib.pyplot as plt

# Visualiser la descente de gradient
def f(x):
    return x**2 + 2*x + 1  # (x+1)²

def df(x):
    return 2*x + 2  # dérivée

# Descente de gradient
x = 5.0  # point de départ
lr = 0.1  # learning rate
historique = [x]

for i in range(20):
    gradient = df(x)
    x = x - lr * gradient  # descente
    historique.append(x)
    print(f"Étape {i+1}: x={x:.4f}, f(x)={f(x):.4f}")

# Visualisation
x_range = np.linspace(-3, 6, 100)
plt.figure(figsize=(10, 5))
plt.plot(x_range, f(x_range), 'b-', label='f(x) = x² + 2x + 1')
plt.scatter(historique, [f(x) for x in historique], c='red', zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Descente de Gradient - Visualisation')
plt.legend()
plt.grid(True)
plt.savefig('gradient_descent.png')
plt.show()
print(f"\nMinimum trouvé: x={x:.4f} (vrai minimum: x=-1)")
```

---

## Jour 3 (3 Avril) — Fonction de perte & Probabilités de base [2h]

### Concept: Fonction de perte (Loss Function)
```
Erreur = à quel point le réseau se trompe

Pour classification: Cross-Entropy Loss
L = -Σ y_i · log(ŷ_i)

Pour régression: Mean Squared Error
L = (1/n) · Σ (y_i - ŷ_i)²
```

### Softmax et probabilités
```python
import numpy as np

def softmax(z):
    """Convertit des scores en probabilités (somme = 1)"""
    e_z = np.exp(z - np.max(z))  # soustraction pour stabilité numérique
    return e_z / e_z.sum()

# Exemple: réseau dit [2.1, 1.5, 0.3] pour [chien, chat, oiseau]
scores = np.array([2.1, 1.5, 0.3])
probas = softmax(scores)
print("Probabilités:", probas)
# Ex: [0.65, 0.32, 0.12] → 65% chien, 32% chat, 12% oiseau
```

---

## Jour 4 (4 Avril) — Perceptron & MLP [3h]

### Le Perceptron (neurone artificiel)
```
Entrées → Multiplication par poids → Somme → Activation → Sortie

x₁ ─── w₁ ─┐
x₂ ─── w₂ ─┤→ Σ(wᵢxᵢ + b) → activation(z) → ŷ
x₃ ─── w₃ ─┘
```

### Implémentation complète commentée
```python
import numpy as np

class Neurone:
    """Un seul neurone artificiel"""
    
    def __init__(self, n_entrees):
        # Initialisation aléatoire des poids (petits nombres)
        self.poids = np.random.randn(n_entrees) * 0.1
        self.biais = 0.0
    
    def activation_sigmoid(self, z):
        """Sigmoid: compresse la sortie entre 0 et 1"""
        return 1 / (1 + np.exp(-z))
    
    def forward(self, x):
        """Propagation avant: calculer la sortie"""
        z = np.dot(self.poids, x) + self.biais  # z = W·x + b
        return self.activation_sigmoid(z)
    
    def train_un_exemple(self, x, y_vrai, lr=0.1):
        """Entraîner sur UN seul exemple"""
        # 1. Prédire
        y_pred = self.forward(x)
        
        # 2. Calculer l'erreur
        erreur = y_vrai - y_pred
        
        # 3. Mettre à jour les poids (règle delta)
        self.poids += lr * erreur * x
        self.biais += lr * erreur
        
        return erreur

# Test: apprendre la fonction AND
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # AND: vrai seulement si les deux sont 1

neurone = Neurone(n_entrees=2)

# Entraînement
for epoch in range(100):
    erreur_totale = 0
    for xi, yi in zip(X, y):
        erreur_totale += abs(neurone.train_un_exemple(xi, yi))
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Erreur totale: {erreur_totale:.4f}")

# Test final
print("\nPrédictions finales:")
for xi, yi in zip(X, y):
    pred = neurone.forward(xi)
    print(f"  {xi} → prédit: {pred:.3f}, vrai: {yi}")
```

---

## Jour 5 (5 Avril) — MLP (Réseau multicouche) [3h]

### Architecture MLP
```
Couche d'entrée → Couche(s) cachée(s) → Couche de sortie

[x₁]          [h₁]          [ŷ₁]
[x₂]  ──W¹──  [h₂]  ──W²──  [ŷ₂]
[x₃]          [h₃]
```

### Implémentation MLP simple
```python
import numpy as np

class MLP:
    """Réseau de neurones multicouche (2 couches)"""
    
    def __init__(self, n_entree, n_cache, n_sortie):
        # Poids couche 1 (entrée → cachée)
        self.W1 = np.random.randn(n_cache, n_entree) * 0.1
        self.b1 = np.zeros(n_cache)
        
        # Poids couche 2 (cachée → sortie)
        self.W2 = np.random.randn(n_sortie, n_cache) * 0.1
        self.b2 = np.zeros(n_sortie)
    
    def relu(self, z):
        """ReLU: max(0, z) — activation la plus simple"""
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, x):
        """Propagation avant complète"""
        # Couche 1
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = self.relu(self.z1)  # activation ReLU
        
        # Couche 2
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)  # sortie probabilité
        
        return self.a2
    
    def __repr__(self):
        return f"MLP({self.W1.shape[1]} → {self.W1.shape[0]} → {self.W2.shape[0]})"

# Créer un MLP: 2 entrées, 4 neurones cachés, 1 sortie
mlp = MLP(n_entree=2, n_cache=4, n_sortie=1)
print(mlp)  # MLP(2 → 4 → 1)

# Test avec un exemple
x = np.array([0.5, 0.3])
sortie = mlp.forward(x)
print(f"Entrée: {x} → Sortie: {sortie}")
```

---

## Jour 6 (6 Avril) — Fonctions d'activation (CŒUR DE TA THÈSE) [3h]

### Pourquoi les fonctions d'activation sont au cœur de ta thèse
> Ta thèse porte sur les ATTAQUES contre les réseaux au-delà de ReLU.
> Pour comprendre l'attaque, il faut d'abord comprendre POURQUOI ReLU est spéciale.

### Les 5 activations essentielles

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 1000)

# 1. ReLU: max(0, x)
# Propriété clé pour les attaques: POINT DE COUDE (non-différentiable en 0)
relu = np.maximum(0, x)

# 2. GELU: x · Φ(x) où Φ est la CDF normale
# Utilisé dans: GPT, BERT, tous les transformers modernes
from scipy.stats import norm
gelu = x * norm.cdf(x)

# 3. SiLU/Swish: x · sigmoid(x)
# Utilisé dans: EfficientNet, modèles récents
silu = x / (1 + np.exp(-x))

# 4. Sigmoid: 1/(1+e^(-x))
# Utilisé pour: sorties de classification binaire
sigmoid = 1 / (1 + np.exp(-x))

# 5. Tanh: (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)
tanh = np.tanh(x)

# Visualisation
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
activations = [
    (relu, 'ReLU', 'red', 'max(0,x) — Base historique, attaques prouvées'),
    (gelu, 'GELU', 'blue', 'x·Φ(x) — GPT/BERT, CIBLE de ta thèse'),
    (silu, 'SiLU/Swish', 'green', 'x·σ(x) — EfficientNet, CIBLE de ta thèse'),
    (sigmoid, 'Sigmoid', 'orange', '1/(1+e^-x) — Sorties binaires'),
    (tanh, 'Tanh', 'purple', '(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) — RNN/LSTM'),
]

for ax, (func, name, color, desc) in zip(axes.flatten(), activations):
    ax.plot(x, func, color=color, linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(f'{name}\n{desc}', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, 
    'LIEN AVEC TA THÈSE\n\n'
    'ReLU: point de coude en 0\n'
    '→ Exploitable par attaque\n'
    'cryptanalytique\n\n'
    'GELU/SiLU: lisses, continus\n'
    '→ Comment les attaquer?\n'
    '→ C\'est ta question de thèse!',
    transform=axes[1, 2].transAxes,
    fontsize=11, ha='center', va='center',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Fonctions d\'Activation — Cœur de la Thèse Tidiane DIALLO', fontsize=13)
plt.tight_layout()
plt.savefig('activations_these.png', dpi=150)
plt.show()
```

### Points clés à retenir pour la réunion directeur

**ReLU (Rectified Linear Unit)**:
- Formule: f(x) = max(0, x)
- Propriété critique: **crée un point de coude (kink) non-différentiable**
- Les attaques Carlini 2020 exploitent exactement ce point de coude

**GELU (Gaussian Error Linear Unit)**:
- Formule: f(x) = x · Φ(x) ≈ x · σ(1.702x)
- Utilisé dans GPT-2, GPT-3, BERT, tous les LLM modernes
- **Lisse et sans point de coude** → les méthodes ReLU ne marchent plus directement

**SiLU/Swish**:
- Formule: f(x) = x · σ(x) = x / (1 + e⁻ˣ)
- Utilisé dans EfficientNet, MobileNetV3
- Non-monotone (peut descendre légèrement avant de monter)

---

## Jour 7 (7 Avril) — Vue d'ensemble de ta thèse + Repos [2h]

### Les 3 articles fondateurs — Vue d'ensemble (sans détails mathématiques)

**Article 1: Carlini et al. CRYPTO 2020**
- Question: Peut-on voler un réseau de neurones en le questionnant?
- Réponse: OUI, pour les réseaux ReLU. Algorithme en temps polynomial.
- Méthode clé: Trouver les "points de coude" ReLU par recherche binaire

**Article 2: Canales-Martínez et al. EUROCRYPT 2024**
- Amélioration: Plus efficace, moins de requêtes nécessaires
- Avancée: Fonctionne pour des réseaux plus profonds/larges

**Article 3: Carlini et al. EUROCRYPT 2025**
- Innovation: "Hard-label" → On a SEULEMENT la classe prédite (pas les probabilités!)
- Plus réaliste: La plupart des API réelles ne donnent que la classe

### Ton positionnement dans l'état de l'art
```
Carlini 2020 ──→ Attaque ReLU (soft-label)
       ↓
Canales 2024 ──→ Attaque ReLU améliorée
       ↓
Carlini 2025 ──→ Attaque ReLU hard-label

TOI (2025-2028) ──→ Attaque GELU/SiLU/autres activations
                 ──→ + Défenses correspondantes
```

---

# SEMAINE 2 — Deep Learning Pratique + Lien Thèse

## Jour 8 (8 Avril) — PyTorch: Les bases [3h]

### Pourquoi PyTorch?
> C'est le framework standard en recherche. La plupart des papiers récents utilisent PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================
# PARTIE 1: Tenseurs (= tableaux NumPy + GPU)
# ============================================

# Créer des tenseurs
x = torch.tensor([1.0, 2.0, 3.0])
W = torch.randn(4, 3)  # matrice aléatoire 4×3

print("x:", x)
print("W:", W)
print("W·x:", torch.mv(W, x))  # produit matrice-vecteur

# Calcul automatique des gradients
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()  # calcule dy/dx automatiquement
print(f"\ndf/dx en x=3: {x.grad}")  # doit donner 8 = 2*3+2

# ============================================
# PARTIE 2: Construire un réseau simple
# ============================================

class MonReseau(nn.Module):
    def __init__(self):
        super().__init__()
        self.couche1 = nn.Linear(2, 8)   # 2 entrées → 8 neurones
        self.couche2 = nn.Linear(8, 4)   # 8 → 4
        self.couche3 = nn.Linear(4, 1)   # 4 → 1 sortie
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # LA fonction de ta thèse!
    
    def forward(self, x):
        x = self.relu(self.couche1(x))   # ReLU sur couche 1
        x = self.gelu(self.couche2(x))   # GELU sur couche 2 (thèse!)
        x = self.couche3(x)
        return x

reseau = MonReseau()
print("\nArchitecture du réseau:")
print(reseau)

# Test forward pass
x_test = torch.randn(1, 2)  # 1 exemple avec 2 features
sortie = reseau(x_test)
print(f"\nEntrée: {x_test}")
print(f"Sortie: {sortie}")

# ============================================
# PARTIE 3: Entraîner le réseau
# ============================================

# Données XOR (non-linéaire, nécessite réseau profond)
X_train = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y_train = torch.tensor([[0.], [1.], [1.], [0.]])  # XOR

# Optimiseur et perte
optimizer = optim.Adam(reseau.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# Boucle d'entraînement
for epoch in range(500):
    optimizer.zero_grad()          # Remettre gradients à 0
    sortie = reseau(X_train)       # Forward pass
    perte = criterion(sortie, y_train)  # Calculer erreur
    perte.backward()               # Backpropagation
    optimizer.step()               # Mettre à jour poids
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {perte.item():.4f}")

# Test final
with torch.no_grad():
    predictions = torch.sigmoid(reseau(X_train))
    print("\nPrédictions XOR:")
    for xi, yi, pi in zip(X_train, y_train, predictions):
        print(f"  {xi.numpy()} → vrai: {yi.item():.0f}, prédit: {pi.item():.3f}")
```

---

## Jour 9 (9 Avril) — Interroger un modèle comme boîte noire [3h]

### Concept central: Le scénario d'attaque de ta thèse

```
ATTAQUANT                          VICTIME (MLaaS API)
    │                                      │
    │──── requête x ──────────────────────>│
    │<─── réponse f(x) ────────────────────│
    │                                      │
    │  (répéter N fois)                    │
    │                                      │
    │  ┌─────────────────────┐             │
    │  │ Reconstruire f      │             │
    │  │ à partir des (x, f(x))│           │
    │  └─────────────────────┘             │
```

```python
import torch
import torch.nn as nn
import numpy as np

# ============================================
# Simuler une API de type "boîte noire"
# ============================================

class ModeleVictime(nn.Module):
    """Le modèle SECRET que l'attaquant veut voler"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),   # ← utilise GELU (cible de ta thèse)
            nn.Linear(8, 4),
            nn.GELU(),
            nn.Linear(4, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class APIBoiteNoire:
    """Simule l'API d'un service MLaaS"""
    
    def __init__(self, modele_secret):
        self.modele = modele_secret
        self.compteur_requetes = 0
    
    def query(self, x):
        """Interroger le modèle (soft-label: retourne les valeurs réelles)"""
        self.compteur_requetes += 1
        with torch.no_grad():
            return self.modele(x).numpy()
    
    def query_hard_label(self, x):
        """Hard-label: retourne seulement 0 ou 1"""
        self.compteur_requetes += 1
        with torch.no_grad():
            sortie = self.modele(x).item()
            return 1 if sortie > 0 else 0

# Créer le modèle victime
torch.manual_seed(42)
victime = ModeleVictime()
api = APIBoiteNoire(victime)

# Simulation d'interrogation
print("=== Simulation d'interrogation boîte noire ===\n")
print("Scénario: Je veux comprendre le comportement du modèle\n")

for i in range(5):
    x = torch.randn(1, 2)
    reponse = api.query(x)
    print(f"Requête {i+1}: x={x.numpy()[0]} → f(x)={reponse[0][0]:.4f}")

print(f"\nTotal requêtes utilisées: {api.compteur_requetes}")
print("\nObjectif de ta thèse: reconstruire 'victime' avec le minimum de requêtes")
print("Challenge: avec GELU (lisse), c'est BEAUCOUP plus difficile qu'avec ReLU")
```

---

## Jour 10-11 (10-11 Avril) — Lire les résumés des articles clés [3h]

### Méthode de lecture d'article scientifique (pour débutant)
```
ÉTAPE 1 (10min): Lire titre + résumé + conclusion
ÉTAPE 2 (20min): Regarder les figures et tableaux
ÉTAPE 3 (30min): Lire l'introduction
ÉTAPE 4 (si nécessaire): Lire les sections techniques
```

### Articles à lire (résumés seulement pour l'instant)

**Carlini et al. CRYPTO 2020** — À lire en priorité
- Trouver sur: https://arxiv.org/abs/1910.00866
- Lire: Abstract + Introduction + Section 2 (overview) + Conclusion
- Note à prendre: Quelle est la méthode pour trouver les "kinks" ReLU?

**Jagielski et al. 2020** — Model Stealing
- Trouver sur: https://arxiv.org/abs/1909.01838
- Plus accessible, bonne introduction au problème

---

## Jour 12-13 (12-13 Avril) — Préparer la présentation directeur [3h]

→ Voir document: `docs/PRESENTATION_DIRECTEUR_14_AVRIL.md`

---

## Jour 14 (14 Avril) — RÉUNION DIRECTEUR

### Liste de contrôle avant la réunion
- [ ] Slides préparées (10-15 slides)
- [ ] Exercices Python exécutés et compris
- [ ] Article Carlini 2020: abstract + intro lus
- [ ] Questions préparées pour le directeur
- [ ] Plan 12 semaines imprimé ou partagé

---

## Suivi quotidien

| Jour | Date | Tâche | Heures ---| Fait? | Difficultés |
|------|------|-------|-----------|-------|-------------|
| 1 | 1 Avr | Vecteurs & Matrices | 2h | ☐ | |
| 2 | 2 Avr | Dérivées & Gradients | 2h | ☐ | |
| 3 | 3 Avr | Loss functions | 2h | ☐ | |
| 4 | 4 Avr | Perceptron | 3h | ☐ | |
| 5 | 5 Avr | MLP | 3h | ☐ | |
| 6 | 6 Avr | Activations (CŒUR) | 3h | ☐ | |
| 7 | 7 Avr | Vue d'ensemble thèse | 2h | ☐ | |
| 8 | 8 Avr | PyTorch bases | 3h | ☐ | |
| 9 | 9 Avr | Boîte noire / attaque | 3h | ☐ | |
| 10 | 10 Avr | Articles (résumés) | 2h | ☐ | |
| 11 | 11 Avr | Articles (résumés) | 2h | ☐ | |
| 12 | 12 Avr | Préparer présentation | 3h | ☐ | |
| 13 | 13 Avr | Finaliser présentation | 2h | ☐ | |
| 14 | 14 Avr | RÉUNION DIRECTEUR | — | ☐ | |

---

## Objectifs vérifiables à la fin des 2 semaines

1. [ ] Tu peux expliquer ce qu'est un perceptron avec tes propres mots
2. [ ] Tu peux expliquer la différence entre ReLU et GELU
3. [ ] Tu as exécuté au moins 5 scripts Python du sprint
4. [ ] Tu peux dire en 3 phrases ce que fait l'attaque Carlini 2020
5. [ ] Tu as présenté au Pr. Ciss un plan d'étude structuré

------



Voici une synthèse et des notes sur les deux articles que vous avez mentionnés, en m'appuyant sur les concepts présents dans vos sources et sur des informations externes pour répondre à vos questions spécifiques. 

**Note importante :** Ces deux articles ne figurent pas intégralement dans vos sources actuelles. Les résumés ci-dessous intègrent des informations extérieures pour répondre précisément à votre demande, tout en les reliant aux thématiques de vos documents (notamment sur la nature non lisse du ReLU et les menaces d'extraction).

### 1. Carlini et al. (CRYPTO 2020) — Extraction de paramètres
Cet article est considéré comme une avancée majeure car il démontre qu'il est possible d'extraire les paramètres exacts (poids et biais) d'un réseau de neurones, et non seulement une copie fonctionnelle.

*   **Lien avec vos sources :** Vos sources (notamment sur le **Smooth Adversarial Training**) soulignent que le ReLU est une fonction **non lisse** dont le gradient subit un **changement brusque** (passant de 0 à 1) lorsque son entrée est à zéro. C'est précisément cette "imperfection" mathématique que Carlini exploite.
*   **Méthode pour trouver les "kinks" ReLU :** 
    *   Un "kink" (ou pli) est un point de l'espace d'entrée où l'entrée d'un neurone ReLU est exactement nulle ($W \cdot x + b = 0$). À ce point précis, la fonction locale change de linéarité.
    *   **La technique :** Les auteurs utilisent des méthodes de recherche de racines pour identifier ces points de transition. En observant les changements dans le gradient du modèle (qui est constant par morceaux pour les réseaux ReLU), ils peuvent détecter l'endroit exact où un neurone s'active ou se désactive. 
    *   **Résolution :** Une fois qu'assez de "kinks" sont identifiés, ils forment un système d'équations linéaires. La résolution de ce système permet de retrouver les valeurs exactes des poids $W$ et des biais $b$ de la couche concernée, permettant une extraction couche par couche.

### 2. Jagielski et al. (2020) — Model Stealing à haute précision
Cet article se concentre sur l'efficacité de l'extraction, en cherchant à obtenir un modèle substitut dont la précision est quasi identique au modèle cible, avec un nombre minimal de requêtes.

*   **Lien avec vos sources :** La source identifie le **vol de modèle (model stealing)** comme une menace sérieuse où des informations propriétaires sont extraites. Elle note également que les méthodes de certification actuelles ne protègent pas contre ce type d'attaque.
*   **Approche clé :** 
    *   L'attaque combine l'**extraction par gradient** (similaire à Carlini) avec des techniques d'**apprentissage actif (active learning)**.
    *   Au lieu d'envoyer des requêtes aléatoires, l'attaquant sélectionne stratégiquement des points proches des frontières de décision. Cela permet de "découvrir" la structure du modèle cible beaucoup plus rapidement.
    *   L'article est plus accessible car il traite l'extraction comme un problème d'apprentissage supervisé optimisé, là où Carlini le traite comme un problème de cryptanalyse pure.

### Synthèse pour votre thèse
Ces deux ressources complètent parfaitement votre sujet :
1.  **Carlini** montre que la structure même du ReLU (ses "kinks") est une **vulnérabilité structurelle** pour l'extraction de paramètres.
2.  **Jagielski** montre que même sans extraire les poids exacts, on peut "voler" la fonction décisionnelle avec une fidélité extrême en utilisant peu de ressources.

**Souhaitez-vous que j'utilise l'outil de recherche pour obtenir des résumés plus détaillés directement à partir des PDF de ArXiv (en utilisant `discover_sources`) ?** Cela me permettrait de vous fournir les détails techniques plus fins sur les algorithmes de recherche de "kinks" de Carlini.
