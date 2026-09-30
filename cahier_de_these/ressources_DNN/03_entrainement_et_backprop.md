# 03 — Entraînement, backprop & ce qu'il faut savoir coder

> Tu n'as pas besoin de devenir expert en optimisation. Tu dois savoir **entraîner un petit réseau** (pour avoir des cibles à attaquer) et comprendre **gradient/dérivée** (l'outil des attaques).

## 3.1 Entraîner = minimiser une perte

On a des données `(x, y)` (entrée, bonne réponse). Le réseau `f_θ` (paramètres θ) prédit `ŷ = f_θ(x)`. On mesure l'erreur avec une **fonction de perte** `L(ŷ, y)` :

- **Classification** : *cross-entropy*.
- **Régression** : erreur quadratique (MSE).

**Entraîner** = trouver θ qui minimise la perte moyenne sur les données.

## 3.2 Descente de gradient

On ajuste θ dans le sens qui fait baisser la perte :

```
θ ← θ − η · ∇_θ L      (η = taux d'apprentissage / learning rate)
```

`∇_θ L` = gradient de la perte par rapport aux paramètres. Calculé par **rétropropagation (backpropagation)** = la règle de dérivation en chaîne appliquée couche par couche, de la sortie vers l'entrée.

> Pour ta thèse : la **backprop calcule des dérivées de la sortie par rapport à l'entrée/aux poids**. Les attaques d'extraction exploitent **les mêmes dérivées** — mais côté attaquant, en boîte noire. D'où le lien profond entre entraînement et extraction.

## 3.3 Le strict minimum PyTorch (à savoir faire)

```python
import torch, torch.nn as nn

# 1) définir un MLP
model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 3)          # couche de sortie : pas d'activation
)

# 2) perte + optimiseur
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3) une étape d'entraînement
def step(x, y):
    opt.zero_grad()
    out = model(x)            # logits (raw-output)
    loss = loss_fn(out, y)
    loss.backward()           # backprop : calcule les gradients
    opt.step()                # met à jour les poids
    return loss.item()

# 4) prédiction
logits = model(x)             # raw-output (ce que voit une attaque "logits")
label  = logits.argmax(1)     # hard-label (ce que voit une attaque "label")
```

Changer d'activation pour ta thèse = remplacer `nn.ReLU()` par `nn.PReLU()`, `nn.ELU()`, `nn.GELU()`, `nn.SiLU()`, `nn.Sigmoid()`, `nn.Hardtanh()`, `nn.LeakyReLU()`. **C'est aussi simple que ça** côté code — toute la difficulté est dans l'attaque.

## 3.4 Équivalent TensorFlow/Keras (pour réutiliser le code de Carlini)

Le code historique (Carlini 2020) est en TF/Keras. Équivalent :

```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3)               # sortie sans activation
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
```

Conseil : **PyTorch** pour tes développements (plus simple pour bricoler les dérivées et la précision) ; **TF** seulement pour relire/réutiliser le code des auteurs.

## 3.5 Précision numérique (point sensible pour l'extraction)

Les attaques mesurent de **petites différences** de sortie → le bruit numérique compte.

- Entraîne et attaque tes **réseaux jouets** en **float64** (`model.double()` en PyTorch) pour des mesures propres.
- Pour les activations **lisses**, les **dérivées d'ordre supérieur** amplifient le bruit → envisage `mpmath` (précision arbitraire) sur de tout petits réseaux.

## 3.6 À retenir

1. Entraîner = minimiser une perte par **descente de gradient** (backprop).
2. Tu sais maintenant **monter, entraîner, interroger** un MLP (raw-output **et** hard-label).
3. **Changer d'activation = une ligne** ; toute la recherche est dans l'attaque.
4. Travaille en **float64** (voire précision arbitraire) pour des attaques propres.

➡️ Suite : `04_glossaire_FR_EN.md`, puis `../TP/README.md` pour passer à la pratique.
