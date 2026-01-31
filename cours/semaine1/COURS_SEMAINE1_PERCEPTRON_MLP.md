# Cours Semaine 1 : Perceptron et Réseaux de Neurones Multicouches (MLP)

**Thèse : Attaques d'extraction et défenses pour les DNN au-delà de ReLU**
**Doctorant : Tidiane DIALLO**
**Date : Janvier 2026**

---

## Table des matières

1. [Introduction aux Réseaux de Neurones](#1-introduction-aux-réseaux-de-neurones)
2. [Le Perceptron Simple](#2-le-perceptron-simple)
3. [Perceptron Multicouches (MLP)](#3-perceptron-multicouches-mlp)
4. [Propagation Avant (Forward Pass)](#4-propagation-avant-forward-pass)
5. [Rétropropagation (Backpropagation)](#5-rétropropagation-backpropagation)
6. [Exercices Pratiques](#6-exercices-pratiques)
7. [Ressources Complémentaires](#7-ressources-complémentaires)

---

## 1. Introduction aux Réseaux de Neurones

### 1.1 Contexte Historique

Les réseaux de neurones artificiels s'inspirent du fonctionnement du cerveau humain. L'histoire commence en **1943** avec le modèle de McCulloch et Pitts, suivi du **perceptron de Rosenblatt en 1958**.

La révolution majeure arrive en **1986** quand **Rumelhart, Hinton et Williams** introduisent l'algorithme de rétropropagation (backpropagation), permettant l'entraînement efficace des réseaux multicouches.

### 1.2 Pourquoi c'est important pour ta thèse

Les attaques d'extraction de modèles exploitent la structure mathématique des réseaux de neurones. Comprendre :
- Comment les poids transforment les données
- Comment les gradients révèlent de l'information
- Pourquoi certaines activations (ReLU) créent des "points critiques"

...est **fondamental** pour comprendre les attaques cryptanalytiques sur les DNN.

### 1.3 Le Neurone Biologique vs Artificiel

```
Neurone Biologique          Neurone Artificiel
─────────────────          ─────────────────
Dendrites (entrées)   →    x₁, x₂, ..., xₙ (inputs)
Corps cellulaire      →    Somme pondérée + biais
Axone (sortie)        →    Fonction d'activation → y
Synapses (connexions) →    Poids w₁, w₂, ..., wₙ
```

---

## 2. Le Perceptron Simple

### 2.1 Définition Mathématique

Le perceptron est le modèle le plus simple d'un neurone artificiel :

```
         ┌─────────────────────────────────────┐
         │                                     │
    x₁ ──┤──→ w₁ ──┐                          │
         │         │                          │
    x₂ ──┤──→ w₂ ──┼──→ Σ + b ──→ f(z) ──→ y │
         │         │                          │
    xₙ ──┤──→ wₙ ──┘                          │
         │                                     │
         └─────────────────────────────────────┘
```

**Équation fondamentale :**

$$z = \sum_{i=1}^{n} w_i \cdot x_i + b = \mathbf{w}^T \mathbf{x} + b$$

$$y = f(z)$$

où :
- **x** = vecteur d'entrée (features)
- **w** = vecteur de poids (weights)
- **b** = biais (bias)
- **f** = fonction d'activation
- **y** = sortie (output)

### 2.2 Fonction d'Activation du Perceptron Original

Le perceptron original utilise une **fonction de seuil (step function)** :

$$f(z) = \begin{cases} 1 & \text{si } z \geq 0 \\ 0 & \text{si } z < 0 \end{cases}$$

### 2.3 Interprétation Géométrique

Le perceptron définit un **hyperplan** dans l'espace des entrées :

$$\mathbf{w}^T \mathbf{x} + b = 0$$

Cet hyperplan **sépare** l'espace en deux régions :
- Points où $\mathbf{w}^T \mathbf{x} + b > 0$ → classe 1
- Points où $\mathbf{w}^T \mathbf{x} + b < 0$ → classe 0

**Important pour ta thèse :** Cette séparation linéaire est la base des "régions linéaires" créées par ReLU !

### 2.4 Algorithme d'Apprentissage du Perceptron

```python
# Pseudo-code de l'algorithme du perceptron
pour chaque époque:
    pour chaque exemple (x, y_vrai):
        y_pred = 1 si (w·x + b) >= 0 sinon 0
        erreur = y_vrai - y_pred

        # Mise à jour des poids
        w = w + η * erreur * x
        b = b + η * erreur
```

où **η** (eta) est le **taux d'apprentissage** (learning rate).

### 2.5 Limitation : Le Problème XOR

Le perceptron simple ne peut résoudre que les problèmes **linéairement séparables**.

```
AND (séparable)          XOR (non séparable)

  1 ──●───●── 1            1 ──○───●── 1
    │   │                    │   │
    │   │                    │   │
  0 ──○───○── 0            0 ──●───○── 0
    0   1                    0   1

● = classe 1, ○ = classe 0
```

Le XOR nécessite **plusieurs couches** → d'où le MLP !

---

## 3. Perceptron Multicouches (MLP)

### 3.1 Architecture

Un MLP est un réseau **feedforward** composé de :
- **Couche d'entrée** : reçoit les features
- **Couches cachées** : transformations non-linéaires
- **Couche de sortie** : produit la prédiction

```
Couche d'entrée    Couche cachée      Couche de sortie
     (2)              (3)                  (1)

    ○ x₁ ─────────→ ○ h₁ ─────────┐
      │╲            │╲             │
      │ ╲           │ ╲            │
      │  ╲──────────│──╲───────────┼──→ ○ y
      │  ╱──────────│──╱───────────┤
      │ ╱           │ ╱            │
    ○ x₂ ─────────→ ○ h₂ ─────────┤
                    │              │
                  → ○ h₃ ─────────┘
```

### 3.2 Notation Mathématique

Pour un MLP avec L couches :

**Couche l :**
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$$

où :
- $\mathbf{W}^{[l]}$ = matrice de poids de la couche l (dimensions : $n_l \times n_{l-1}$)
- $\mathbf{b}^{[l]}$ = vecteur de biais de la couche l
- $\mathbf{z}^{[l]}$ = pré-activation (avant la fonction d'activation)
- $\mathbf{a}^{[l]}$ = activation (après la fonction d'activation)
- $f^{[l]}$ = fonction d'activation de la couche l

**Convention :** $\mathbf{a}^{[0]} = \mathbf{x}$ (l'entrée)

### 3.3 Exemple : MLP 2-2-1 pour XOR

```
Architecture : 2 entrées → 2 neurones cachés → 1 sortie

Paramètres :
- W¹ : matrice 2×2 (couche 1)
- b¹ : vecteur 2×1
- W² : matrice 1×2 (couche 2)
- b² : scalaire

Total : 2×2 + 2 + 1×2 + 1 = 9 paramètres
```

### 3.4 Pourquoi les Couches Cachées sont Nécessaires

**Théorème d'approximation universelle** (Cybenko, 1989) :

> Un MLP avec une seule couche cachée et suffisamment de neurones peut approximer n'importe quelle fonction continue sur un compact.

En pratique, des réseaux **profonds** (plusieurs couches) sont plus efficaces que des réseaux larges (beaucoup de neurones dans une seule couche).

---

## 4. Propagation Avant (Forward Pass)

### 4.1 Algorithme

Le forward pass calcule la sortie du réseau étape par étape :

```python
def forward_pass(x, weights, biases, activations):
    """
    x: entrée (vecteur)
    weights: liste des matrices W pour chaque couche
    biases: liste des vecteurs b pour chaque couche
    activations: liste des fonctions d'activation
    """
    a = x  # activation initiale = entrée

    for l in range(len(weights)):
        z = weights[l] @ a + biases[l]  # pré-activation
        a = activations[l](z)            # activation

    return a  # sortie finale
```

### 4.2 Exemple Détaillé : MLP 2-3-1

Soit un réseau avec :
- Entrée : $\mathbf{x} = [x_1, x_2]^T$
- Couche cachée : 3 neurones avec ReLU
- Sortie : 1 neurone (régression)

**Étape 1 : Couche cachée**
$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$$

$$\mathbf{z}^{[1]} = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}$$

$$\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \max(0, \mathbf{z}^{[1]})$$

**Étape 2 : Couche de sortie**
$$z^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]}$$

$$\hat{y} = z^{[2]}$$ (pas d'activation pour la régression)

### 4.3 Importance pour l'Extraction de Modèles

Lors d'une attaque d'extraction :
1. L'attaquant envoie des **requêtes** (inputs x)
2. Il observe les **réponses** (outputs y)
3. Il essaie de **reconstruire** les poids W et biais b

**Point clé :** Avec ReLU, les "points critiques" (où ReLU change de comportement) révèlent de l'information sur les poids !

---

## 5. Rétropropagation (Backpropagation)

### 5.1 Principe

La rétropropagation calcule les **gradients** de la fonction de perte par rapport à chaque paramètre, en utilisant la **règle de la chaîne** (chain rule).

**Objectif :** Minimiser la fonction de perte $L(\hat{y}, y)$

**Méthode :** Descente de gradient
$$\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}$$

### 5.2 La Règle de la Chaîne

Si $y = f(g(x))$, alors :
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

Pour un réseau de neurones :
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### 5.3 Algorithme de Backpropagation

```
1. FORWARD PASS : Calculer toutes les activations a[l]
2. CALCUL DE LA PERTE : L = Loss(y_pred, y_vrai)
3. BACKWARD PASS :
   - Calculer δ[L] = ∂L/∂z[L]  (couche de sortie)
   - Pour l = L-1 jusqu'à 1 :
       δ[l] = (W[l+1])ᵀ δ[l+1] ⊙ f'(z[l])
4. GRADIENTS :
   - ∂L/∂W[l] = δ[l] (a[l-1])ᵀ
   - ∂L/∂b[l] = δ[l]
5. MISE À JOUR :
   - W[l] = W[l] - η * ∂L/∂W[l]
   - b[l] = b[l] - η * ∂L/∂b[l]
```

### 5.4 Dérivation Complète pour un MLP 2-2-1

Soit :
- Entrée : $\mathbf{x} = [x_1, x_2]^T$
- Couche cachée : 2 neurones, activation $\sigma$ (sigmoid)
- Sortie : 1 neurone, pas d'activation
- Perte : MSE $L = \frac{1}{2}(\hat{y} - y)^2$

**Forward Pass :**
$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$$
$$\mathbf{a}^{[1]} = \sigma(\mathbf{z}^{[1]})$$
$$z^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]}$$
$$\hat{y} = z^{[2]}$$

**Backward Pass :**

**Couche de sortie :**
$$\delta^{[2]} = \frac{\partial L}{\partial z^{[2]}} = \frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$

$$\frac{\partial L}{\partial \mathbf{W}^{[2]}} = \delta^{[2]} (\mathbf{a}^{[1]})^T$$

$$\frac{\partial L}{\partial b^{[2]}} = \delta^{[2]}$$

**Couche cachée :**
$$\boldsymbol{\delta}^{[1]} = (\mathbf{W}^{[2]})^T \delta^{[2]} \odot \sigma'(\mathbf{z}^{[1]})$$

où $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ pour la sigmoid.

$$\frac{\partial L}{\partial \mathbf{W}^{[1]}} = \boldsymbol{\delta}^{[1]} \mathbf{x}^T$$

$$\frac{\partial L}{\partial \mathbf{b}^{[1]}} = \boldsymbol{\delta}^{[1]}$$

### 5.5 Lien avec la Cryptanalyse

**Observation clé de Carlini et al. (CRYPTO 2020) :**

En utilisant des **différences finies** (numerical differentiation), un attaquant peut estimer les gradients sans accès direct au modèle :

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$

Ces gradients révèlent de l'information sur la structure du réseau !

---

## 6. Exercices Pratiques

### Exercice 1 : Perceptron pour AND (Code complet)

```python
"""
Exercice 1.1 : Implémenter un perceptron pour la fonction AND
"""
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        """
        Initialise le perceptron avec des poids aléatoires

        Args:
            n_inputs: nombre d'entrées
            learning_rate: taux d'apprentissage η
        """
        # Initialisation aléatoire des poids (petites valeurs)
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.lr = learning_rate

    def activation(self, z):
        """Fonction de seuil (step function)"""
        return 1 if z >= 0 else 0

    def forward(self, x):
        """
        Calcule la sortie du perceptron

        Args:
            x: vecteur d'entrée
        Returns:
            sortie (0 ou 1)
        """
        z = np.dot(self.weights, x) + self.bias
        return self.activation(z)

    def train(self, X, y, epochs=100):
        """
        Entraîne le perceptron

        Args:
            X: matrice d'entrées (n_samples, n_features)
            y: vecteur de labels
            epochs: nombre d'époques
        """
        history = []

        for epoch in range(epochs):
            total_error = 0

            for xi, yi in zip(X, y):
                # Prédiction
                y_pred = self.forward(xi)

                # Calcul de l'erreur
                error = yi - y_pred
                total_error += abs(error)

                # Mise à jour des poids (règle du perceptron)
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

            history.append(total_error)

            # Arrêt si convergence
            if total_error == 0:
                print(f"Convergence à l'époque {epoch + 1}")
                break

        return history

# ========== TEST ==========

# Données pour AND
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND

# Créer et entraîner le perceptron
perceptron = Perceptron(n_inputs=2, learning_rate=0.1)
history = perceptron.train(X, y, epochs=100)

# Vérification
print("\nRésultats après entraînement:")
print(f"Poids: {perceptron.weights}")
print(f"Biais: {perceptron.bias}")
print("\nPrédictions:")
for xi, yi in zip(X, y):
    pred = perceptron.forward(xi)
    print(f"  {xi} -> {pred} (attendu: {yi}) {'✓' if pred == yi else '✗'}")
```

### Exercice 2 : MLP pour XOR (Code complet)

```python
"""
Exercice 1.2 : MLP 2-2-1 pour résoudre XOR
Avec backpropagation manuelle
"""
import numpy as np

def sigmoid(x):
    """Fonction sigmoid avec protection contre overflow"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Dérivée de sigmoid: σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

class MLP:
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        """
        MLP avec une couche cachée
        Architecture: input_size -> hidden_size -> output_size
        """
        # Couche 1 (entrée -> caché)
        # Xavier initialization pour une meilleure convergence
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))

        # Couche 2 (caché -> sortie)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

        # Pour stocker les valeurs intermédiaires (nécessaire pour backprop)
        self.cache = {}

    def forward(self, X):
        """
        Propagation avant

        Args:
            X: entrées (n_features, n_samples)
        Returns:
            sortie du réseau
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
        Rétropropagation et mise à jour des poids

        Args:
            X: entrées (n_features, n_samples)
            Y: sorties attendues (1, n_samples)
            learning_rate: taux d'apprentissage
        """
        m = X.shape[1]  # nombre d'exemples

        # Récupérer les valeurs du cache
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        Z1 = self.cache['Z1']

        # ===== BACKWARD PASS =====

        # Couche de sortie (L=2)
        # δ² = ∂L/∂Z² = (A² - Y) * σ'(Z²)
        # Pour MSE avec sigmoid: dL/dA2 = (A2 - Y), dA2/dZ2 = sigmoid_derivative
        dZ2 = (A2 - Y) * sigmoid_derivative(self.cache['Z2'])

        # Gradients de W2 et b2
        dW2 = (1/m) * dZ2 @ A1.T
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Couche cachée (L=1)
        # δ¹ = (W²)ᵀ δ² ⊙ σ'(Z¹)
        dZ1 = (self.W2.T @ dZ2) * sigmoid_derivative(Z1)

        # Gradients de W1 et b1
        dW1 = (1/m) * dZ1 @ X.T
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # ===== MISE À JOUR DES POIDS =====
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, Y, epochs=10000, learning_rate=0.5, print_every=1000):
        """
        Entraînement du MLP
        """
        losses = []

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Calcul de la perte (MSE)
            loss = np.mean((output - Y) ** 2)
            losses.append(loss)

            # Backward pass
            self.backward(X, Y, learning_rate)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return losses

    def predict(self, X):
        """Prédiction avec seuil à 0.5"""
        return (self.forward(X) > 0.5).astype(int)

# ========== TEST SUR XOR ==========

# Données XOR (format: n_features x n_samples)
X = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])
Y = np.array([[0, 1, 1, 0]])  # XOR

# Créer et entraîner le MLP
np.random.seed(42)  # Pour reproductibilité
mlp = MLP(input_size=2, hidden_size=4, output_size=1)  # 4 neurones cachés

print("Entraînement du MLP pour XOR...")
losses = mlp.train(X, Y, epochs=10000, learning_rate=1.0, print_every=2000)

# Résultats
print("\n" + "="*50)
print("Résultats finaux:")
print("="*50)
output = mlp.forward(X)
predictions = mlp.predict(X)

print("\nEntrées -> Sortie (attendu)")
for i in range(4):
    print(f"  [{X[0,i]}, {X[1,i]}] -> {output[0,i]:.4f} ≈ {predictions[0,i]} (attendu: {Y[0,i]})")

print(f"\nPoids finaux:")
print(f"W1:\n{mlp.W1}")
print(f"b1: {mlp.b1.flatten()}")
print(f"W2: {mlp.W2}")
print(f"b2: {mlp.b2.flatten()}")
```

### Exercice 3 : Dérivation Manuelle (Sur papier)

**Objectif :** Calculer les gradients à la main pour un MLP 2-3-1

**Réseau :**
- Entrée : $x = [x_1, x_2]^T$
- Couche cachée : 3 neurones, ReLU
- Sortie : 1 neurone, pas d'activation
- Perte : MSE

**Questions :**

1. Écrire les équations du forward pass
2. Calculer $\frac{\partial L}{\partial W^{[2]}}$
3. Calculer $\frac{\partial L}{\partial W^{[1]}}$ en utilisant la chain rule
4. Que se passe-t-il pour $\frac{\partial \text{ReLU}}{\partial z}$ quand $z < 0$ ?

**Solution :**

1. **Forward pass :**
$$z^{[1]} = W^{[1]} x + b^{[1]}$$
$$a^{[1]} = \text{ReLU}(z^{[1]}) = \max(0, z^{[1]})$$
$$\hat{y} = W^{[2]} a^{[1]} + b^{[2]}$$
$$L = \frac{1}{2}(\hat{y} - y)^2$$

2. **Gradient de la couche de sortie :**
$$\frac{\partial L}{\partial W^{[2]}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W^{[2]}}$$
$$= (\hat{y} - y) \cdot (a^{[1]})^T$$

3. **Gradient de la couche cachée :**
$$\frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial a^{[1]}} \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial W^{[1]}}$$
$$= (\hat{y} - y) \cdot W^{[2]T} \cdot \mathbb{1}_{z^{[1]} > 0} \cdot x^T$$

4. **Dérivée de ReLU :**
$$\frac{\partial \text{ReLU}(z)}{\partial z} = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{si } z < 0 \end{cases}$$

**Note importante :** Quand $z < 0$, le gradient est 0. Cela signifie que le neurone ne contribue pas au gradient → c'est le phénomène des "**dying ReLU**".

---

## 7. Ressources Complémentaires

### 7.1 Vidéos Recommandées

1. **3Blue1Brown - Neural Networks** (Playlist)
   - Excellente visualisation intuitive
   - https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

2. **Andrej Karpathy - Micrograd** (2h30)
   - Implémentation from scratch
   - https://www.youtube.com/watch?v=VMj-3S1tku0
   - Code : https://github.com/karpathy/micrograd

3. **fast.ai - Practical Deep Learning, Lesson 13**
   - Backpropagation et MLP
   - https://course.fast.ai/Lessons/lesson13.html

### 7.2 Lectures

1. **"Deep Learning" - Goodfellow, Bengio, Courville**
   - Chapitre 6: Deep Feedforward Networks
   - Gratuit : https://www.deeplearningbook.org/

2. **Tutoriels en ligne :**
   - [DataCamp - Multilayer Perceptrons Guide](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning)
   - [The Multilayer Perceptron - Pablo Insente](https://pabloinsente.github.io/the-multilayer-perceptron)

### 7.3 Code de Référence

1. **Micrograd de Karpathy**
   - https://github.com/karpathy/micrograd
   - Autograd minimaliste en ~150 lignes

2. **Neural Networks: Zero to Hero**
   - https://karpathy.ai/zero-to-hero.html
   - Cours complet avec notebooks

### 7.4 Pour aller plus loin

- **MDPI Journal Article** : "Perceptron: Learning, Generalization, Model Selection, Fault Tolerance, and Role in the Deep Learning Era"
  - https://www.mdpi.com/2227-7390/10/24/4730

---

## Résumé

| Concept | Équation clé | Importance pour la thèse |
|---------|--------------|--------------------------|
| Perceptron | $y = f(w^Tx + b)$ | Base de tout neurone |
| Forward pass | $a^{[l]} = f(W^{[l]}a^{[l-1]} + b^{[l]})$ | Ce que l'attaquant observe |
| Backpropagation | $\delta^{[l]} = (W^{[l+1]})^T\delta^{[l+1]} \odot f'(z^{[l]})$ | Révèle la structure |
| Gradients numériques | $\frac{\partial f}{\partial x} \approx \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$ | Outil d'attaque |

**Point clé pour ta thèse :** La structure mathématique du réseau (poids, biais, activations) est ce que les attaques d'extraction cherchent à reconstruire. Comprendre le forward/backward pass est essentiel pour comprendre quelles informations "fuient" à travers l'API du modèle.

---

*Cours créé le 31 Janvier 2026 - Thèse Tidiane DIALLO*
