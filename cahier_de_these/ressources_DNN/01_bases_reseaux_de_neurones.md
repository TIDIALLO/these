# 01 — Les bases des réseaux de neurones (pour débuter de zéro)

> Tu es développeur Python mais débutant en deep learning. Ce chapitre te donne le **minimum vital** pour comprendre les articles d'extraction. On va du neurone unique au réseau profond, en restant concret.

## 1.1 Le neurone : une fonction affine + une non-linéarité

Un neurone prend un vecteur d'entrée `x = (x₁,…,xₙ)`, calcule une **combinaison linéaire** avec des **poids** `w` et un **biais** `b`, puis applique une **fonction d'activation** `σ` :

```
pré-activation :  z = w·x + b   =  w₁x₁ + w₂x₂ + … + wₙxₙ + b
sortie         :  a = σ(z)
```

- `w·x + b` est exactement l'équation d'un **hyperplan** dans l'espace d'entrée. Retiens ça : **chaque neurone définit un hyperplan**. C'est LA clé géométrique de toutes les attaques d'extraction.
- `σ` introduit la **non-linéarité** : sans elle, empiler des couches ne servirait à rien (composer des fonctions linéaires donne une fonction linéaire).

En Python pur :

```python
import numpy as np
def neuron(x, w, b, sigma):
    return sigma(np.dot(w, x) + b)
```

## 1.2 La couche : plusieurs neurones en parallèle

Une **couche** applique `m` neurones à la même entrée. On empile les poids dans une **matrice** `W` (m×n) et les biais dans un vecteur `b` (m) :

```
z = W x + b          (z, b ∈ ℝ^m)
a = σ(z)             (σ appliquée composante par composante)
```

## 1.3 Le réseau profond (DNN) : empiler les couches

Un **réseau de neurones profond** (DNN) enchaîne les couches. La sortie d'une couche est l'entrée de la suivante :

```
a⁰ = x                              (entrée)
a¹ = σ(W¹ a⁰ + b¹)                  (couche cachée 1)
a² = σ(W² a¹ + b²)                  (couche cachée 2)
…
aᴸ = W_L a^{L-1} + b_L              (couche de SORTIE : souvent SANS activation)
```

Points cruciaux pour ta thèse :

- Les **paramètres** = tous les `(Wⁱ, bⁱ)`. **Les extraire, c'est voler le modèle.**
- La **couche de sortie n'a généralement pas de ReLU** → c'est pourquoi elle résiste aux attaques (f.07) et pourquoi ton article fondateur (f.08) la traite à part.
- En **classification**, on prend `argmax(aᴸ)` = la **classe prédite** (le « label »). En **hard-label**, l'attaquant ne voit QUE ce label, pas le vecteur `aᴸ`.

## 1.4 « Logits », « softmax », « hard-label » : le vocabulaire qui compte

- **Logits / raw-output** = le vecteur `aᴸ` brut (scores réels). Les attaques f.03–05, f.10–11 supposent y accéder.
- **Softmax** = transforme les logits en probabilités. Certaines API renvoient ça.
- **Hard-label** = l'API ne renvoie **que** `argmax`, c.-à-d. « chat » ou « voiture ». C'est le cadre **réaliste** et **difficile** (f.06–09) — le cœur de ta thèse.

## 1.5 Ce qu'est l'extraction (intuition)

L'attaquant ne connaît pas `(Wⁱ, bⁱ)`. Il **interroge** le réseau sur des entrées choisies et observe les sorties. En exploitant le fait qu'**un réseau ReLU est affine par morceaux** (voir chapitre 02), il reconstruit les hyperplans des neurones un par un, couche par couche. C'est un puzzle géométrique — d'où l'analogie avec la **cryptanalyse différentielle**.

## 1.6 À retenir absolument

1. Neurone = **hyperplan** `w·x+b` + activation `σ`.
2. DNN = couches empilées ; paramètres = `(Wⁱ, bⁱ)`.
3. Couche de sortie souvent **sans activation**.
4. **Hard-label** = on ne voit que `argmax` (réaliste, dur).
5. Extraire = reconstruire les hyperplans par requêtes → **géométrie**.

➡️ Suite : `02_activations_et_geometrie.md` — le chapitre le plus important pour ta thèse.
