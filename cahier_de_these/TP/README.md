# TP — Travaux pratiques d'extraction de DNN

> Série de TP **exécutables** pour reproduire l'état de l'art puis explorer tes pistes « au-delà de ReLU ». Tu es dev Python : ici on **code** les attaques, pas seulement on les lit.

## Installation

```bash
# 1) Python 3.10+ recommandé
python3 --version

# 2) Dépendances minimales (TP2 à TP7 : NumPy pur)
pip install numpy

# 3) Pour TP1 (entraînement de la cible) :
pip install torch            # PyTorch (principal)
# (optionnel, pour relire le code historique de Carlini)
pip install tensorflow
```

Les TP2 à TP7 **n'ont besoin que de NumPy** : ils utilisent un MLP « maison »
(`common.py`) en float64, idéal pour des mesures d'attaque propres et déterministes.
TP1 utilise PyTorch pour te familiariser avec un vrai entraînement.

## Progression conseillée (suis l'ordre)

| TP | Fichier | Ce que tu apprends | Article(s) |
|----|---------|--------------------|-----------|
| **TP1** | `code/tp1_mlp_pytorch.py` | Construire/entraîner un MLP, interroger en raw-output et hard-label | — (mise en jambe) |
| **TP2** | `code/tp2_points_critiques.py` | Voir le réseau ReLU **affine par morceaux** ; détecter les **points critiques** | f.03 |
| **TP3** | `code/tp3_signature.py` | **Récupérer la signature** (direction des poids) de la 1ʳᵉ couche | f.03 |
| **TP4** | `code/tp4_signe.py` | Comprendre le **problème du signe** (ambiguïté ±) | f.03, f.04 |
| **TP5** | `code/tp5_hardlabel.py` | Attaque **hard-label** : trouver des **points de transition** | f.06, f.07, f.08 |
| **TP6** | `code/tp6_beyond_relu.py` | **Au-delà de ReLU** : identifier l'activation, fuite de signe, mur du lisse | f.10, f.11, f.12 |
| **TP7** | `code/tp7_defense.py` | **Défense** : casser l'unicité des neurones (similarité intra-couche) | f.13 |

Lance par exemple :

```bash
cd code
python3 tp2_points_critiques.py
python3 tp3_signature.py
python3 tp6_beyond_relu.py
```

## La brique partagée : `common.py`

- `MLP(layer_sizes, activation=...)` : la **cible** (boîte noire). Change `activation`
  parmi `relu, leaky_relu, prelu, elu, hardtanh, sigmoid, tanh, gelu, silu`.
- `oracle_raw(x)` → logits ; `oracle_label(x)` → classe seule (**hard-label**).
- `net.n_queries` : **compteur de requêtes** (métrique clé à toujours rapporter).
- `fidelity_error(a, b)` et `cosine(u, v)` : pour **évaluer** tes attaques.

> Les méthodes `forward`/`W`/`preactivations` servent à **vérifier** (boîte blanche).
> Une vraie attaque n'utilise que `oracle_raw` / `oracle_label`.

## Reproduire l'existant → ce que prouve chaque TP

- **TP3** récupère les 5 signatures de couche 1 avec `|cos| = 1.000000` (direction exacte).
- **TP5** trouve des points de la frontière de décision avec un écart de logits ~1e-13.
- **TP6** classe correctement les 9 activations en « par morceaux » vs « lisse »,
  montre que Leaky/PReLU/ELU **fuient des deux côtés** (signe plus facile), et que
  le **cas lisse n'a pas de coude** exploitable.
- **TP7** montre que rapprocher les poids (défense f.13) fait chuter le nombre de
  neurones séparables par l'attaque (5/5 → 3/5).

## Aller plus loin → tes contributions (voir `../IDEE.md`)

Chaque TP se termine par un bloc **PISTE THÈSE** qui pointe l'idée associée :

- **TP4/TP6 → idée B1** : extraction hard-label des activations **par morceaux** non-ReLU
  (PReLU/ELU : coude ≠ 0, estimer α, exploiter la fuite de signe).
- **TP5/TP6 → idée B2** : extraction hard-label des activations **lisses** — inventer un
  substitut aux dérivées d'ordre supérieur via la **courbure de la frontière**.
- **TP6 → idée B4** : **identifier l'activation** en hard-label (classifieur de frontière).
- **TP7 → idées B6/B7** : tester la défense hors ReLU, en concevoir une pour le lisse.

## Exercices guidés (à faire dans l'ordre)

1. **TP3 en plus profond** : passe la cible à `[4, 5, 5, 2]`. La signature de la
   couche 2 est-elle aussi directe ? (indice : il faut d'abord « peler » la couche 1.)
2. **TP3 → PReLU** : mets `activation="prelu"`. La signature tient-elle ? Ajoute
   l'**estimation de α** (fiche 12).
3. **TP5 → lisse** : mets `activation="sigmoid"`. Les points de transition existent
   toujours, mais que devient la **forme** de la frontière ? Mesure sa courbure.
4. **TP7-bis (PyTorch)** : implémente la **vraie** régularisation de similarité
   intra-couche dans la perte de TP1, réentraîne, vérifie la perte de précision (<1%),
   puis relance TP3 sur le modèle protégé.

## Bonnes pratiques (rappel)

- Fixe les **seeds**, travaille en **float64** (déjà le cas dans `common.py`).
- Rapporte toujours **(1) nb de requêtes, (2) temps, (3) erreur de fidélité**.
- Vérifie l'**équivalence fonctionnelle** sur des entrées indépendantes, pas juste
  la proximité des poids.
