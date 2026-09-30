# Sprint Semaines 3 & 4 — L'Attaque en Profondeur + Vers GELU
## Tidiane DIALLO | 28 Avril → 11 Mai 2026

> **Tu as terminé**: Fondamentaux DL, PyTorch, aperçu thèse
> **Objectif ces 2 semaines**: Comprendre l'attaque Carlini 2020 de A à Z + commencer à voir pourquoi GELU résiste
> **Rythme**: 2-4h/jour | **Niveau**: Bases acquises → Intermédiaire recherche

---

## Vue d'ensemble

```
SEMAINE 3 (28 Avr - 4 Mai)         SEMAINE 4 (5 Mai - 11 Mai)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━━━━━━━━━━━━
L'Attaque Carlini 2020              Vers GELU/SiLU
Décryptée pas à pas                 Pourquoi c'est plus dur
+ Code complet de l'attaque         + Premières pistes de recherche
``` 

---

# SEMAINE 3 — L'Attaque Carlini 2020 Décryptée

## 🎯 Objectif de la semaine
> À la fin de cette semaine, tu pourras expliquer **précisément** comment on "vole" un réseau ReLU:
> comment on trouve les points de coude, comment on en déduit les poids, et pourquoi ça marche.

---

## Jour 15 (28 Avril) — Les Régions Linéaires de ReLU [2-3h]

### Pourquoi c'est le fondement de tout
> **Idée centrale**: Un réseau ReLU n'est pas une fonction lisse. C'est une fonction **linéaire par morceaux**.
> L'attaque exploite exactement cette structure.

### Concept 1: ReLU crée des "morceaux linéaires"

Rappel: `ReLU(x) = max(0, x)`

Quand on enchaîne plusieurs ReLU, le réseau entier devient **linéaire par morceaux**:

```
                    ●─────────────
                   /
──────────────────●
                  0

Sur chaque "morceau", la fonction est LINÉAIRE (droite)
Le "point de coude" (kink) = là où ça change de comportement
```

**Intuition visuelle pour 1 neurone:**
```
Neurone h(x) = ReLU(w·x + b)

Si w·x + b < 0:  h(x) = 0  (neurone "éteint")
Si w·x + b > 0:  h(x) = w·x + b  (neurone "allumé")

Le point de transition: w·x + b = 0  → x = -b/w
                        ↑
                    C'EST LE KINK!
```

### Concept 2: Pour un réseau entier, les régions linéaires

Pour un MLP avec ReLU, l'espace d'entrée est **partitionné** en régions:

```
Espace d'entrée 2D (x₁, x₂)
┌─────────────────────────────┐
│    ╲       /                │
│  A  ╲  B  /   C             │  Chaque région = une fonction linéaire différente
│      ╲   /                  │  
│       ╲ /    D              │  Dans la région A: f(x) = W_A · x + b_A
│        X                    │  Dans la région B: f(x) = W_B · x + b_B
│       / ╲                   │  Aux frontières: kinks (discontinuités du gradient)
└─────────────────────────────┘
```

### Concept 3: Pattern d'activation

Chaque région linéaire correspond à un **pattern d'activation** unique:

```python
import torch
import torch.nn as nn
import numpy as np

def get_activation_pattern(model, x):
    """
    Pour un réseau ReLU, retourne quels neurones sont allumés/éteints
    pour une entrée x donnée.
    
    C'est l'EMPREINTE DIGITALE de la région linéaire.
    """
    patterns = []
    
    def hook(module, input, output):
        # Pour chaque neurone: 1 si actif (>0), 0 si inactif
        pattern = (output > 0).float()
        patterns.append(pattern.detach().numpy())
    
    # Enregistrer des hooks sur chaque couche ReLU
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook))
    
    # Forward pass
    with torch.no_grad():
        model(x)
    
    # Supprimer les hooks
    for h in hooks:
        h.remove()
    
    return patterns

# Exemple concret
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)

# Deux points proches mais dans des régions différentes
x1 = torch.tensor([[0.5, 0.3]])
x2 = torch.tensor([[0.5, -0.3]])  # même x1, x2 négatif

p1 = get_activation_pattern(model, x1)
p2 = get_activation_pattern(model, x2)

print("Point x1 =", x1.numpy())
print("Pattern d'activation:", [p.flatten().astype(int).tolist() for p in p1])
print()
print("Point x2 =", x2.numpy())
print("Pattern d'activation:", [p.flatten().astype(int).tolist() for p in p2])
print()
print("Même région?", all(
    np.array_equal(a, b) for a, b in zip(p1, p2)
))
```

### Exercice: Compter les régions linéaires
```python
import torch
import torch.nn as nn
import numpy as np

def compter_regions(model, n_echantillons=10000, dimension=2):
    """
    Estime le nombre de régions linéaires en échantillonnant
    aléatoirement l'espace d'entrée.
    """
    regions = set()
    
    for _ in range(n_echantillons):
        x = torch.randn(1, dimension)
        pattern = get_activation_pattern(model, x)
        # Convertir le pattern en tuple (hashable)
        cle = tuple(tuple(p.flatten().tolist()) for p in pattern)
        regions.add(cle)
    
    return len(regions)

# Petit réseau
petit = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))

# Grand réseau
grand = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1)
)

print(f"Petit réseau (2→4→1): ~{compter_regions(petit)} régions linéaires")
print(f"Grand réseau (2→8→8→1): ~{compter_regions(grand)} régions linéaires")
print("\nPlus le réseau est profond → PLUS il y a de régions → PLUS l'attaque est complexe")
```

### Ressources vidéo (1h)
- **Visualisation des régions linéaires de ReLU**: https://playground.tensorflow.org/
  - Joue avec les couches et observe comment les frontières se forment
- **Kolter & Madry - Lecture on ReLU Networks** (MIT 6.S978):
  https://www.youtube.com/watch?v=0TXkDGHGNCc

---

## Jour 16 (29 Avril) — La Recherche de Kinks par Différences Finies [3h]

### Pourquoi les kinks sont exploitables

> **Idée de l'attaque**: Si la fonction est linéaire PAR MORCEAUX,
> alors **aux points de transition (kinks)**, les dérivées changent brutalement.
> On peut détecter ces transitions en mesurant comment la sortie change.

### Concept: Différences finies

Une **différence finie** approxime la dérivée en mesurant le changement:

```
Dérivée en x ≈ [f(x + ε) - f(x)] / ε   (différence avant)

Dérivée du 2ème ordre ≈ [f(x+ε) - 2f(x) + f(x-ε)] / ε²
```

**Pourquoi la dérivée du 2ème ordre?**

```
Dans une région linéaire:  f(x) = a·x + b
                           f'(x) = a  (constante)
                           f''(x) = 0  (ZÉRO !)

Au niveau d'un kink:       f''(x) ≠ 0  (GRAND !)

Donc:  f''(x) ≈ 0  →  on est dans une région linéaire
       |f''(x)| grand  →  on est PRÈS d'un kink !
```

### Algorithme de détection de kinks

```
ENTRÉE: modèle f, direction de recherche d, point de départ x₀
SORTIE: position t* du kink sur la droite x₀ + t·d

1. Balayer t de -1 à 1 par petits pas δ
2. Pour chaque t, calculer la dérivée numérique de f
3. Chercher les sauts de dérivée → ce sont les kinks
4. Affiner avec recherche binaire
```

### Implémentation complète

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Créer un réseau ReLU à 1 couche cachée (simple)
# ============================================
torch.manual_seed(123)

# Réseau SECRET (ce qu'on veut attaquer)
reseau_victime = nn.Sequential(
    nn.Linear(1, 6),   # 1 entrée → 6 neurones cachés
    nn.ReLU(),
    nn.Linear(6, 1)    # 6 → 1 sortie
)

def f(x_val):
    """Interroger le réseau victime (boîte noire)"""
    with torch.no_grad():
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32)
        return reseau_victime(x_tensor).item()

# ============================================
# Détecter les kinks par différences finies
# ============================================

def derivee_seconde(f, x, epsilon=1e-3):
    """
    Approxime la dérivée du 2ème ordre en x.
    C'est GRANDE près d'un kink, PROCHE DE ZÉRO ailleurs.
    """
    return (f(x + epsilon) - 2*f(x) + f(x - epsilon)) / (epsilon**2)

def detecter_kinks_brut(f, x_min=-3, x_max=3, n_points=1000):
    """
    Balaye l'espace pour trouver où |f''(x)| est grand.
    """
    xs = np.linspace(x_min, x_max, n_points)
    derivees2 = np.array([abs(derivee_seconde(f, x)) for x in xs])
    
    # Normaliser pour visualisation
    seuil = np.percentile(derivees2, 90)  # 90ème percentile
    kinks_bruts = xs[derivees2 > seuil]
    
    return xs, derivees2, kinks_bruts

# Détecter
xs, d2, kinks = detecter_kinks_brut(f)
print(f"Kinks détectés (brut): {len(kinks)} positions candidates")

# ============================================
# Affiner avec recherche binaire
# ============================================

def recherche_binaire_kink(f, a, b, tolerance=1e-6):
    """
    Affine la position d'un kink entre a et b.
    
    Idée: si le pattern d'activation change entre a et b,
    alors il y a un kink quelque part entre les deux.
    On cherche par dichotomie.
    
    Pour simplifier: on utilise la magnitude de la dérivée 2ème ordre.
    """
    for _ in range(50):  # 50 itérations de bisection
        m = (a + b) / 2
        d2_a = abs(derivee_seconde(f, a))
        d2_m = abs(derivee_seconde(f, m))
        
        if d2_m > d2_a:
            b = m
        else:
            a = m
        
        if abs(b - a) < tolerance:
            break
    
    return (a + b) / 2

# Visualisation
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 1. La fonction f(x)
f_vals = [f(x) for x in xs]
axes[0].plot(xs, f_vals, 'b-', linewidth=2)
axes[0].set_title("f(x) — Réseau ReLU (fonction linéaire par morceaux)")
axes[0].set_ylabel("f(x)")
axes[0].grid(True, alpha=0.3)

# 2. La dérivée 2ème ordre |f''(x)|
axes[1].plot(xs, d2, 'r-', linewidth=1)
axes[1].set_title("|f''(x)| — Les pics = les kinks !")
axes[1].set_ylabel("|f''(x)|")
axes[1].grid(True, alpha=0.3)

# 3. Positions des kinks détectés
axes[2].plot(xs, f_vals, 'b-', linewidth=2, alpha=0.5)
for kink in kinks[:20]:  # Afficher les 20 premiers
    axes[2].axvline(x=kink, color='red', alpha=0.7, linewidth=1)
axes[2].set_title("Kinks détectés (lignes rouges)")
axes[2].set_xlabel("x")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kink_detection.png', dpi=150)
plt.show()

print("\n✅ Les kinks sont détectables sans connaître les poids du réseau !")
print("→ C'est la clé de l'attaque Carlini 2020")
```

### Ce que les kinks nous révèlent

```
Kink k = position où le pattern d'activation change

→ Ce changement est causé par UN neurone qui s'allume ou s'éteint

→ Le neurone qui change obéit à: w·x + b = 0

→ Donc le kink nous donne une équation LINÉAIRE sur les poids !

Avec assez de kinks (autant d'équations que d'inconnues),
on peut résoudre le système et TROUVER LES POIDS EXACTS.
```

### Ressources
- **Article original Carlini 2020**: https://arxiv.org/abs/1910.00866 (Section 3: High-level overview)
- **Blog d'explication accessible**: https://nicholas.carlini.com/writing/2020/extracting-nn-models.html

---

## Jour 17 (30 Avril) — Extraire les Poids à partir des Kinks [3h]

### L'équation clé

Rappel: Un kink en position `t*` signifie qu'il existe un neurone `j` tel que:

```
w_j · x(t*) + b_j = 0

Avec x(t*) = x₀ + t* · d  (point sur la droite de recherche)

→ w_j · x₀ + t* · (w_j · d) + b_j = 0
```

Si on fait plusieurs recherches le long de **différentes directions** `d₁, d₂, ...`:
```
t₁* → w_j · x₀ + t₁* · (w_j · d₁) + b_j = 0
t₂* → w_j · x₀ + t₂* · (w_j · d₂) + b_j = 0
...

C'est un système linéaire en (w_j, b_j) !
```

### Algorithme d'extraction (couche 1 seulement)

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq

# ============================================
# RÉSEAU CIBLE (1 couche cachée, entrée 2D)
# ============================================
torch.manual_seed(42)

VRAIES_DIMENSIONS = {'entree': 2, 'cache': 4, 'sortie': 1}

victime = nn.Sequential(
    nn.Linear(2, 4),   # poids à récupérer: 4×2 + 4 biais
    nn.ReLU(),
    nn.Linear(4, 1)
)

# Afficher les VRAIS poids (normalement inconnus de l'attaquant)
print("=== VRAIS POIDS (inconnus de l'attaquant) ===")
W1_vrai = victime[0].weight.data.numpy()
b1_vrai = victime[0].bias.data.numpy()
print(f"W1 =\n{W1_vrai}")
print(f"b1 = {b1_vrai}")

def oracle(x_np):
    """L'oracle = notre seul accès au réseau"""
    with torch.no_grad():
        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        return victime(x_t).item()

# ============================================
# ÉTAPE 1: Trouver les kinks en 2D
# ============================================

def chercher_kink_1d(oracle, x0, direction, t_range=(-2, 2), n_points=500, eps=1e-3):
    """
    Cherche un kink le long de la droite x0 + t*direction.
    Retourne les valeurs de t où la dérivée change.
    """
    ts = np.linspace(t_range[0], t_range[1], n_points)
    
    def f_1d(t):
        x = x0 + t * direction
        return oracle(x)
    
    # Différence seconde ordre
    d2 = []
    for t in ts:
        val = (f_1d(t + eps) - 2*f_1d(t) + f_1d(t - eps)) / eps**2
        d2.append(abs(val))
    
    d2 = np.array(d2)
    seuil = np.mean(d2) + 2 * np.std(d2)
    kinks_t = ts[d2 > seuil]
    
    return kinks_t

# Point de base
x0 = np.zeros(2)

# Chercher des kinks dans plusieurs directions
directions = [
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 1.0]) / np.sqrt(2),
    np.array([1.0, -1.0]) / np.sqrt(2),
]

print("\n=== RECHERCHE DE KINKS ===")
tous_kinks = []
for i, d in enumerate(directions):
    kinks = chercher_kink_1d(oracle, x0, d)
    print(f"Direction {i+1} {d}: {len(kinks)} kink(s) trouvé(s) à t ≈ {kinks[:5]}")
    for t in kinks[:3]:  # Garder les 3 premiers
        point_kink = x0 + t * d
        tous_kinks.append((point_kink, d, t))

print(f"\nTotal: {len(tous_kinks)} kinks collectés")

# ============================================
# ÉTAPE 2: Construire le système linéaire
# ============================================
# Pour chaque kink trouvé: w_j · x_kink + b_j = 0
# On construit une matrice A où chaque ligne est [x_kink | 1]
# Et on cherche le noyau (nullspace) de A

if len(tous_kinks) >= 3:
    # Construire la matrice des contraintes
    A = np.array([[*kink[0], 1.0] for kink in tous_kinks[:10]])
    print(f"\nMatrice de contraintes A (forme: {A.shape}):")
    print(A)
    
    # Le noyau de A contient les vecteurs [w_j, b_j]
    # On peut le calculer avec SVD
    U, S, Vt = np.linalg.svd(A)
    print(f"\nValeurs singulières: {S}")
    print("→ Les petites valeurs singulières indiquent des solutions au système")
```

### Ce que l'algorithme complet fait (vue simplifiée)

```
PHASE 1: Extraction couche 1
    Pour chaque neurone j de la couche 1:
        1. Trouver des kinks causés par j (en variant x dans plusieurs directions)
        2. Résoudre le système linéaire → obtenir (w_j, b_j)
    
    → On connaît maintenant TOUS les poids de la couche 1 !

PHASE 2: "Déplier" la couche 1
    Maintenant qu'on connaît la couche 1, on peut construire
    un oracle "virtuel" qui expose la couche 2:
    
    f(x) = W2 · ReLU(W1·x + b1) + b2
    
    En fixant les neurones de la couche 1 dans un état connu
    (les garder tous actifs), on peut isoler W2·a₁ + b2,
    une simple régression linéaire.
    
    → On connaît maintenant W2 et b2 !

RÉSULTAT: On a extrait tous les poids du réseau !
```

### Points d'ambiguïté à connaître (important!)

```
L'extraction n'est pas parfaite à cause de 2 ambiguïtés:

1. SIGNE: Si on multiplie w_j par -1 et b_j par -1,
   ReLU(w_j·x + b_j) peut rester identique.
   → On ne peut pas savoir le "signe" de chaque neurone directement.

2. SCALING: Si on multiplie w_j par c et qu'on divise
   la matrice de sortie par c, le réseau est identique.
   → On ne peut pas connaître l'échelle absolue.

Carlini 2020 résout ces ambiguïtés avec des requêtes supplémentaires.
```

### Ressources
- **Carlini 2020, Section 4**: Algorithme complet pour couche 1
  https://arxiv.org/abs/1910.00866
- **Code officiel** (référence): https://github.com/google/cryptanalytic-model-extraction

---

## Jour 18 (1er Mai) — Récupération du Signe & Implémentation Jouet [3h]

### Le problème du signe

Quand on extrait les poids d'un neurone, on trouve `(±w_j, ±b_j)` — on ne sait pas le signe.

```
POURQUOI?

On détecte les kinks de neurone j là où: w_j · x + b_j = 0
        Mais:  (-w_j) · x + (-b_j) = 0  aussi !

Les deux solutions donnent le MÊME hyperplan séparateur.

La valeur de ReLU change cependant:
    ReLU(w_j·x + b_j)  peut valoir 5
    ReLU(-w_j·x - b_j) peut valoir 0 (si le réseau utilise -w_j)
```

### Récupération du signe par "witness"

```python
def recuperer_signe(oracle, x0, w_j_candidat, b_j_candidat, epsilon=1e-4):
    """
    Détermine si w_j ou -w_j est le vrai poids.
    
    Méthode: On perturbe x dans la direction de w_j et on observe
    si la sortie augmente ou diminue.
    
    Intuition: Si on est dans la région où le neurone j est actif,
    la sortie f(x + ε·w_j) devrait augmenter d'une façon prévisible.
    """
    # Trouver un point où le neurone j est actif (w_j·x + b_j > 0)
    # Pour ça, on prend un point légèrement au-delà du kink dans la bonne direction
    
    # Point dans la région active (neurone "allumé")
    x_actif = x0 + (np.abs(b_j_candidat) / np.dot(w_j_candidat, w_j_candidat)) * w_j_candidat + 0.1 * w_j_candidat
    
    # Perturber dans la direction de w_j
    f_plus = oracle(x_actif + epsilon * w_j_candidat)
    f_moins = oracle(x_actif - epsilon * w_j_candidat)
    
    # Si la dérivée directionnelle est positive, w_j est dans la bonne direction
    derivee = (f_plus - f_moins) / (2 * epsilon)
    
    return np.sign(derivee)

# Test conceptuel (simplifié)
print("=== TEST CONCEPTUEL: RÉCUPÉRATION DU SIGNE ===\n")

torch.manual_seed(0)
reseau_test = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

W_vrai = reseau_test[0].weight.data.numpy()
print("Vrais poids de la couche 1:")
print(W_vrai)
print()

# Simuler qu'on a trouvé les poids mais avec signe inconnu
W_estime_sans_signe = np.abs(W_vrai)  # On simule qu'on a perdu le signe
print("Poids estimés (signe perdu):")
print(W_estime_sans_signe)
print()
print("→ Il faut récupérer les signes avec des requêtes supplémentaires")
```

### Bilan de l'attaque sur ReLU (vue complète)

```
ATTAQUE CARLINI 2020 — RÉSUMÉ COMPLET
══════════════════════════════════════

ENTRÉE:
  • Accès oracle à f(x) [soft-label]
  • Connaissance de l'architecture (nombre de couches, neurones)

SORTIE:
  • Tous les poids W₁, b₁, W₂, b₂, ..., Wₙ, bₙ

ALGORITHME (couche par couche):
  
  Pour la couche 1:
  ┌─────────────────────────────────────────────────────┐
  │ Pour chaque neurone j ∈ {1,...,n₁}:                 │
  │   1. Chercher kinks le long de ~n₁ directions       │
  │   2. Identifier quels kinks viennent du neurone j   │
  │   3. Résoudre système linéaire → w_j, b_j (±signe)  │
  │   4. Requête "witness" → récupérer le signe         │
  └─────────────────────────────────────────────────────┘
  
  Pour les couches suivantes:
  ┌─────────────────────────────────────────────────────┐
  │   Fixer les couches précédentes dans un pattern     │
  │   d'activation constant → couche suivante devient   │
  │   une régression linéaire simple                    │
  └─────────────────────────────────────────────────────┘

COMPLEXITÉ:
  • Nombre de requêtes: O(n² · W²)
  • n = nombre de couches, W = taille couche max
  • Polynomial en la taille du réseau ✅
  
RÉSULTAT:
  • Extraction EXACTE (à ε-près numérique)
```

### Ressources
- **Talk vidéo de Carlini** (30min, très accessible):
  https://www.youtube.com/watch?v=ysnkKCfNyZY
- **Article complet**: https://arxiv.org/abs/1910.00866

---

## Jour 19 (2 Mai) — Lire Carlini 2020 avec Guide [3h]

### Guide de lecture de l'article

> L'article est dense. Voici exactement **quoi lire** et **dans quel ordre**.

#### Étape 1 (30min): Contexte général
```
✅ Lire: Abstract (résumé) — Page 1
✅ Lire: Introduction complète — Pages 1-4
✅ Regarder: Figure 1 — Vue d'ensemble de l'attaque

Questions à répondre après:
  □ Quel est le modèle de menace? (ce que l'attaquant sait/peut faire)
  □ Qu'est-ce qui est "extraction exacte" vs "approximative"?
  □ Quelle est la contribution principale par rapport à avant?
```

#### Étape 2 (45min): Vue d'ensemble technique
```
✅ Lire: Section 2 "Overview" — Pages 4-7
✅ Regarder: Figure 2 — Comment les kinks sont trouvés
✅ Regarder: Figure 3 — La structure mathématique

Questions:
  □ Qu'est-ce qu'une "critical point" dans l'article?
  □ Comment l'article définit-il une "witness"?
  □ Quelle est la différence entre Phase 1 et Phase 2?
```

#### Étape 3 (45min): L'algorithme principal
```
✅ Lire: Section 3 "Layer 1 extraction" — Pages 7-11
✅ Lire: Section 4 "Sign recovery" — Pages 11-13

Questions:
  □ Comment trouve-t-on les kinks d'un neurone spécifique?
  □ Pourquoi a-t-on besoin de la "witness" pour le signe?
  □ Combien de requêtes utilise chaque étape?
```

#### Ce qu'on peut IGNORER pour l'instant
```
❌ Section 5 (deep networks) — complexe, lire plus tard
❌ Preuves formelles — peuvent attendre
❌ Annexes mathématiques — pour quand tu travailles sur ta propre contribution
```

### Tes notes de lecture (à remplir)

```
=== MES NOTES — CARLINI 2020 ===

Date de lecture: ___________

1. Contribution principale:
   ________________________________________

2. Modèle de menace:
   - L'attaquant sait: _______________________
   - L'attaquant peut faire: _________________
   - L'attaquant ne sait pas: ________________

3. Méthode en 3 mots:
   ________________________________________

4. Limites / ce qui ne marche pas:
   ________________________________________

5. Questions que j'ai encore:
   ________________________________________
```

---

## Jour 20 (3 Mai) — Canales-Martínez 2024: Améliorations [2h]

### Qu'apporte Canales-Martínez 2024?

> **Article**: "Polynomial-Time Cryptanalytic Extraction of Neural Network Models"
> **Eurocrypt 2024** | https://eprint.iacr.org/2024/297

```
Carlini 2020                     Canales-Martínez 2024
─────────────────────────────    ────────────────────────────────
✅ Extraction exacte             ✅ Extraction exacte (maintenu)
⚠️  Requêtes: O(n²·W²)          ✅ Requêtes réduites: O(n·W²)
⚠️  Problèmes pour réseaux      ✅ Fonctionne sur réseaux 
    profonds (>3 couches)            plus profonds
⚠️  Difficultés numériques      ✅ Plus stable numériquement
```

### Améliorations clés à comprendre

#### Innovation 1: Meilleure détection des kinks

```python
# Carlini 2020: recherche exhaustive dans toutes directions
def detecter_kinks_v1(oracle, x0, n_directions=100):
    """Coûteux: teste 100 directions aléatoires"""
    kinks = []
    for _ in range(n_directions):
        d = np.random.randn(dim)
        d = d / np.linalg.norm(d)
        kinks.extend(chercher_kink_1d(oracle, x0, d))
    return kinks  # Beaucoup de doublons !

# Canales 2024: recherche adaptative guidée
def detecter_kinks_v2(oracle, x0, neurones_connus=[]):
    """
    Guidé: utilise les neurones déjà trouvés pour orienter la recherche.
    Moins de directions testées, moins de doublons.
    """
    # Utiliser les poids déjà extraits pour pointer vers les kinks attendus
    # (détails dans l'article, Section 4)
    pass
```

#### Innovation 2: Extraction par systèmes surdéterminés

```
Carlini 2020: résoudre système carré (n équations pour n inconnues)
              → Solution unique mais sensible aux erreurs numériques

Canales 2024: système SURDÉTERMINÉ (+ d'équations que d'inconnues)
              → Résolution par moindres carrés → plus robuste
              
Outil mathématique: Pseudo-inverse de Moore-Penrose
              W = A† · b   (si A est n×m avec n >> m)
```

### Guide de lecture Canales 2024

```
✅ Lire: Abstract + Introduction (pages 1-5)
✅ Lire: Section 2 "Preliminaries" (bases mathématiques)
✅ Regarder: Figure 1 (comparaison avec Carlini)
❌ Ignorer pour l'instant: preuves formelles, Section 5+
```

### Ressources
- **Article Canales 2024**: https://eprint.iacr.org/2024/297
- **Présentation Eurocrypt 2024**: chercher sur YouTube "Canales neural network extraction eurocrypt 2024"

---

## Jour 21 (4 Mai) — Carlini 2025 Hard-Label + Révision Semaine 3 [2h]

### Hard-label: Le scénario plus réaliste

> **Carlini 2025 (Eurocrypt)**: "Extracting Neural Networks with Hard-Label Queries"

```
SOFT-LABEL (Carlini 2020, 2024):
  Requête: x = [0.3, 0.7]
  Réponse: f(x) = [0.82, 0.15, 0.03]  ← probabilités complètes
  
HARD-LABEL (Carlini 2025):
  Requête: x = [0.3, 0.7]  
  Réponse: "CLASSE 0"  ← seulement la classe gagnante !
```

#### Pourquoi c'est plus dur?

```
Avec soft-label:
  On voit les VALEURS NUMÉRIQUES de f(x)
  → On peut calculer des dérivées numériques précises
  → Détection de kinks possible directement
  
Avec hard-label:
  On voit seulement 0 ou 1 (discret!)
  → Pas de dérivées numériques directes
  → Il faut inférer les valeurs continues à partir de comparaisons

Exemple:
  f(x₁) = [0.82, 0.15, 0.03] → "CLASSE 0"   ← même réponse!
  f(x₂) = [0.51, 0.48, 0.01] → "CLASSE 0"   ← pourtant très différents
```

#### La clé: Utiliser les frontières de décision

```python
# En hard-label, on peut trouver EXACTEMENT où la classification change
# en faisant une recherche binaire entre deux points de classes différentes

def trouver_frontiere_decision(oracle_hl, x_classe_0, x_classe_1, tol=1e-6):
    """
    Trouve le point exact sur la frontière de décision
    entre x_classe_0 et x_classe_1.
    
    Utilise la recherche binaire: O(log(1/tol)) requêtes seulement!
    """
    a, b = 0.0, 1.0
    
    while b - a > tol:
        m = (a + b) / 2
        x_m = (1-m) * x_classe_0 + m * x_classe_1  # interpolation
        
        if oracle_hl(x_m) == 0:   # même classe que x_classe_0
            a = m
        else:
            b = m
    
    t = (a + b) / 2
    return (1-t) * x_classe_0 + t * x_classe_1  # point frontière

# Ces points frontière remplacent les kinks de Carlini 2020!
print("Avec hard-label: on cherche des FRONTIÈRES DE DÉCISION")
print("plutôt que des kinks de dérivée")
print("→ Même idée, outil différent")
```

### Bilan de la semaine 3

```
TU SAIS MAINTENANT:
✅ Ce que sont les régions linéaires d'un réseau ReLU
✅ Comment détecter les kinks par différences finies
✅ Comment extraire les poids à partir des kinks
✅ L'idée de la récupération du signe
✅ Les 3 papiers de la lignée Carlini (2020, 2024, 2025)
✅ La différence soft-label vs hard-label

PROCHAINE ÉTAPE (Semaine 4):
→ Pourquoi ces méthodes NE MARCHENT PAS directement pour GELU/SiLU
→ Quelles nouvelles idées sont nécessaires
→ C'est là que commence TA contribution de thèse
```

---

# SEMAINE 4 — Vers GELU : Pourquoi C'est Plus Dur

## 🎯 Objectif de la semaine
> Comprendre précisément POURQUOI les attaques ReLU échouent sur GELU,
> identifier les propriétés mathématiques qui font la différence,
> et explorer les premières pistes pour ta thèse.

---

## Jour 22 (5 Mai) — Propriétés Mathématiques de GELU & SiLU [3h]

### Rappel: Pourquoi GELU/SiLU dans les réseaux modernes?

```
GPT-2, GPT-3, GPT-4, BERT, LLaMA... → tous utilisent GELU
EfficientNet, MobileNetV3...          → utilisent SiLU (Swish)

Raison pratique: GELU/SiLU donnent de MEILLEURES performances que ReLU
                 (meilleure convergence, meilleures prédictions)
                 
Raison théorique: La lisseur permet un meilleur flux de gradient
```

### GELU en détail

```
GELU(x) = x · Φ(x)

où Φ(x) = CDF de la loi normale standard
         = (1/2) · [1 + erf(x/√2)]

En pratique, on utilise l'approximation:
GELU(x) ≈ x · σ(1.702 · x)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)

# Définition exacte
gelu = x * norm.cdf(x)

# Approximation courante (tanh)
gelu_approx = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Approximation sigmoid
gelu_sigmoid = x * (1 / (1 + np.exp(-1.702 * x)))

# Comparaison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, gelu, 'b-', linewidth=2, label='GELU exact')
plt.plot(x, gelu_approx, 'r--', linewidth=2, label='Approx tanh')
plt.plot(x, gelu_sigmoid, 'g:', linewidth=2, label='Approx sigmoid')
plt.title("GELU et ses approximations")
plt.legend()
plt.grid(True, alpha=0.3)

# Dérivée de GELU
dgelu = norm.cdf(x) + x * norm.pdf(x)  # dérivée exacte

plt.subplot(1, 2, 2)
plt.plot(x, dgelu, 'b-', linewidth=2, label="GELU'(x)")
plt.plot(x, np.ones_like(x) * 0, 'k--', alpha=0.5)  # ligne zéro
plt.title("Dérivée de GELU")
plt.ylabel("GELU'(x)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('gelu_properties.png', dpi=150)
plt.show()

# Vérifications numériques importantes
print("=== PROPRIÉTÉS MATHÉMATIQUES DE GELU ===\n")
print(f"GELU(0) = {norm.cdf(0) * 0:.4f}  (vaut exactement 0)")
print(f"GELU(-∞) ≈ {gelu[0]:.4f}  (s'approche de 0 par en-dessous!)")
print(f"GELU'(0) = {norm.cdf(0):.4f}  (≈ 0.5, non-nul en 0!)")
print(f"\nGELU est C∞ (infiniment dérivable) → PAS DE KINKS !")
```

### La différence fondamentale: lisseur vs kinks

```
ReLU                                  GELU
────────────────────────────          ────────────────────────────
f(x) = max(0, x)                      f(x) = x · Φ(x)

DISCONTINUITÉ DU GRADIENT en x=0     INFINIMENT DIFFÉRENTIABLE partout

f'(x):  0 pour x<0                   f'(x) = Φ(x) + x·φ(x)
        1 pour x>0                   (toujours continu et lisse)
        UNDEFINED en x=0 !!!

Conséquence pour l'attaque:           Conséquence pour l'attaque:

✅ Les kinks sont DETECTABLES         ❌ Pas de kinks à détecter!
   par dérivées numériques                La dérivée seconde → 0 partout
                                          
✅ Position exacte du kink            ❌ Pas de "point de coude"
   → équation sur les poids               → pas d'équation directe

BREF: L'attaque Carlini 2020 suppose
      l'existence de kinks. GELU en a PAS.
```

### SiLU / Swish

```python
# SiLU (Sigmoid Linear Unit) = Swish
def silu(x):
    return x / (1 + np.exp(-x))

def dsilu(x):
    sig = 1 / (1 + np.exp(-x))
    return sig + x * sig * (1 - sig)

def d2silu(x):
    """Dérivée seconde de SiLU"""
    sig = 1 / (1 + np.exp(-x))
    dsig = sig * (1 - sig)
    return 2 * dsig + x * (dsig - 2 * sig * dsig + dsig * (1 - sig))  # simplifiable

x = np.linspace(-5, 5, 1000)

print("\n=== PROPRIÉTÉS DE SiLU/SWISH ===")
print(f"SiLU est non-monotone! Minimum à x ≈ {x[np.argmin(silu(x))]:.2f}")
print(f"SiLU(minimum) ≈ {np.min(silu(x)):.4f}")
print("→ SiLU peut DÉCROÎTRE avant de croître (contrairement à ReLU)")
print("→ Cette non-monotonicité rend l'analyse encore plus complexe")
```

### Ressources
- **Paper original GELU** (Hendrycks & Gimpel, 2016): https://arxiv.org/abs/1606.08415
- **Paper SiLU/Swish** (Ramachandran et al., 2017): https://arxiv.org/abs/1710.05941
- **Vidéo "Beyond ReLU"** (Stanford): https://www.youtube.com/watch?v=UQdlkFGf4kk

---

## Jour 23 (6 Mai) — Pourquoi les Méthodes ReLU Échouent sur GELU [3h]

### Expérience: Tester l'attaque ReLU sur un réseau GELU

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Comparer la détection de kinks sur ReLU vs GELU
# ============================================

torch.manual_seed(42)

# Réseau avec ReLU
reseau_relu = nn.Sequential(nn.Linear(1, 6), nn.ReLU(), nn.Linear(6, 1))

# Réseau avec GELU (mêmes poids!)
reseau_gelu = nn.Sequential(nn.Linear(1, 6), nn.GELU(), nn.Linear(6, 1))

# Copier les mêmes poids
reseau_gelu[0].weight.data = reseau_relu[0].weight.data.clone()
reseau_gelu[0].bias.data = reseau_relu[0].bias.data.clone()
reseau_gelu[2].weight.data = reseau_relu[2].weight.data.clone()
reseau_gelu[2].bias.data = reseau_relu[2].bias.data.clone()

def derivee_seconde_reseau(reseau, x_val, eps=1e-3):
    def f(x):
        with torch.no_grad():
            return reseau(torch.tensor([[x]], dtype=torch.float32)).item()
    return abs((f(x_val + eps) - 2*f(x_val) + f(x_val - eps)) / eps**2)

xs = np.linspace(-3, 3, 1000)
d2_relu = [derivee_seconde_reseau(reseau_relu, x) for x in xs]
d2_gelu = [derivee_seconde_reseau(reseau_gelu, x) for x in xs]

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Valeurs de f(x)
f_relu = [reseau_relu(torch.tensor([[x]], dtype=torch.float32)).item() for x in xs]
f_gelu = [reseau_gelu(torch.tensor([[x]], dtype=torch.float32)).item() for x in xs]

axes[0, 0].plot(xs, f_relu, 'r-', linewidth=2)
axes[0, 0].set_title("Réseau ReLU: f(x)\n(linéaire par morceaux, kinks visibles)")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(xs, f_gelu, 'b-', linewidth=2)
axes[0, 1].set_title("Réseau GELU: f(x)\n(lisse, aucun kink!)")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(xs, d2_relu, 'r-', linewidth=1)
axes[1, 0].set_title("|f''(x)| pour ReLU\n→ PICS NETS = kinks détectables ✅")
axes[1, 0].set_ylabel("|f''(x)|")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(xs, d2_gelu, 'b-', linewidth=1)
axes[1, 1].set_title("|f''(x)| pour GELU\n→ LISSE, pas de pics nets ❌")
axes[1, 1].set_ylabel("|f''(x)|")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("POURQUOI L'ATTAQUE RELU ÉCHOUE SUR GELU", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('relu_vs_gelu_attack.png', dpi=150)
plt.show()

# Quantification
print("=== QUANTIFICATION DE L'ÉCHEC ===\n")
seuil_relu = np.mean(d2_relu) + 2*np.std(d2_relu)
seuil_gelu = np.mean(d2_gelu) + 2*np.std(d2_gelu)

kinks_relu = sum(1 for d in d2_relu if d > seuil_relu)
kinks_gelu = sum(1 for d in d2_gelu if d > seuil_gelu)

print(f"Réseau ReLU: {kinks_relu} 'kinks' détectés (pour 6 neurones cachés) ✅")
print(f"Réseau GELU: {kinks_gelu} 'kinks' détectés (pour 6 neurones cachés)")
print()
print("Pour GELU:")
print("  → Pas de kinks → pas d'équations sur les poids")
print("  → Attaque Carlini 2020 ne peut pas s'appliquer directement")
```

### Les 3 obstacles précis à surmonter

```
OBSTACLE 1: Pas de kinks détectables
─────────────────────────────────────
ReLU: kink = changement brusque de la dérivée
GELU: tout est lisse, aucun changement brusque

Comment le surmonter?
→ Piste A: Chercher des "pseudo-kinks" (inflexions de la dérivée 3ème ordre?)
→ Piste B: Utiliser des propriétés différentes de GELU pour isoler les neurones

OBSTACLE 2: Pas de séparation claire des neurones
──────────────────────────────────────────────────
ReLU: neurone = allumé OU éteint (binaire)
      → facile d'identifier quel neurone cause quel kink

GELU: neurone toujours "actif" à différents degrés
      → les effets de plusieurs neurones se mélangent
      → comment les séparer?

OBSTACLE 3: Pas de pattern d'activation
─────────────────────────────────────────
ReLU: pour chaque entrée x, on peut définir un "état" {0,1}^n
      → cet état change aux kinks

GELU: les "états" sont continus
      → concept de région linéaire n'existe plus
```

---

## Jour 24 (7 Mai) — Piste 1: Approximation Locale par ReLU [3h]

### Idée: GELU peut être approximé localement par ReLU

```
OBSERVATION:
  GELU(x) ≈ 0      pour x << 0   (≈ région "éteinte" de ReLU)
  GELU(x) ≈ x      pour x >> 0   (≈ région "allumée" de ReLU)
  
  La transition est lisse (pas brusque comme ReLU)
  MAIS: On peut approximer GELU par une ReLU "élargie"

Question de thèse: cette approximation est-elle assez précise
                   pour permettre une attaque?
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# ============================================
# Approximation de GELU par ReLU par morceaux
# ============================================

x = np.linspace(-4, 4, 10000)
gelu = x * norm.cdf(x)

def relu_generalise(x, alpha, beta):
    """
    ReLU généralisée: max(alpha*x, beta*x) + décalage
    Essaie d'approximer GELU
    """
    return np.maximum(alpha * x, beta * x)

# Approximation par morceaux: 3 régions
def approx_gelu_relu(x, seuil_bas=-1.0, seuil_haut=1.0):
    """
    Approxime GELU par une fonction linéaire par morceaux à 3 segments.
    """
    result = np.zeros_like(x)
    
    # Région gauche (x < seuil_bas): GELU ≈ pente faible
    mask_gauche = x < seuil_bas
    pente_gauche = (norm.cdf(seuil_bas) + seuil_bas * norm.pdf(seuil_bas))
    result[mask_gauche] = gelu[mask_gauche][0] + pente_gauche * (x[mask_gauche] - seuil_bas)
    
    # Région centrale (seuil_bas ≤ x ≤ seuil_haut): interpolation
    mask_centre = (x >= seuil_bas) & (x <= seuil_haut)
    result[mask_centre] = x[mask_centre] * norm.cdf(x[mask_centre])
    
    # Région droite (x > seuil_haut): GELU ≈ identité
    mask_droit = x > seuil_haut
    result[mask_droit] = x[mask_droit] * norm.cdf(x[mask_droit])
    
    return result

approx = approx_gelu_relu(x)
erreur = np.abs(gelu - approx)

print("=== APPROXIMATION RELU DE GELU ===")
print(f"Erreur max: {np.max(erreur):.4f}")
print(f"Erreur moyenne: {np.mean(erreur):.4f}")
print(f"Erreur RMS: {np.sqrt(np.mean(erreur**2)):.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x, gelu, 'b-', linewidth=2, label='GELU exact')
plt.plot(x, approx, 'r--', linewidth=2, label='Approx ReLU')
plt.legend()
plt.title("Approximation de GELU par ReLU")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, erreur, 'g-', linewidth=2)
plt.title("Erreur d'approximation |GELU - approx|")
plt.ylabel("Erreur")
plt.xlabel("x")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gelu_relu_approx.png', dpi=150)
plt.show()

print("\n→ Piste de thèse: Si on peut approximer GELU par ReLU,")
print("  peut-on appliquer l'attaque Carlini sur l'approximation?")
print("  Quelle est la précision d'extraction?")
```

### Analyse de la piste 1

```
AVANTAGES de cette approche:
  ✅ Réutilise les outils Carlini 2020 (attaque connue, code disponible)
  ✅ Complexité comparable (même ordre de magnitude)
  ✅ Applicable aux réseaux GELU réels (GPT, BERT)

LIMITATIONS / QUESTIONS OUVERTES:
  ❓ L'erreur d'approximation se propage-t-elle dans l'extraction?
  ❓ Combien de segments ReLU faut-il pour une bonne approximation?
  ❓ L'erreur est-elle bornée? (requis pour une preuve formelle)
  
FAISABILITÉ POUR UNE THÈSE:
  → Potentiellement publiable si l'analyse de l'erreur est rigoureuse
  → Lien direct avec la littérature existante
  → Résultats expérimentaux comparables à mesurer
```

---

## Jour 25 (8 Mai) — Piste 2: Méthodes Basées sur les Dérivées d'Ordre Supérieur [3h]

### Idée: GELU a des propriétés analytiques exploitables

```
OBSERVATION CLÉ:
  ReLU est linéaire par morceaux → pas de dérivée 2ème ordre dans les régions
  GELU est analytique → des dérivées à TOUS les ordres existent

  Mais... ces dérivées de GELU dépendent des poids d'une façon exploitable !
```

### La dérivée de f(x) pour un réseau GELU

```
Pour un réseau 1 couche cachée: f(x) = W₂ · GELU(W₁·x + b₁) + b₂

Dérivée par rapport à x:
df/dx = W₂ · diag(GELU'(W₁·x + b₁)) · W₁

         ↑ matrice       ↑ gradient elementwise    ↑ poids couche 1
         poids sortie      de GELU appliqué à       (ce qu'on veut)
                          chaque neurone caché
```

```python
import torch
import torch.nn as nn
import numpy as np

# ============================================
# Calculer les dérivées d'un réseau GELU par autograd
# ============================================

torch.manual_seed(42)
reseau = nn.Sequential(
    nn.Linear(2, 4),
    nn.GELU(),
    nn.Linear(4, 1)
)

def jacobien(reseau, x):
    """Calcule le Jacobien de f en x: matrice des dérivées partielles"""
    x = x.clone().requires_grad_(True)
    y = reseau(x)
    
    # Gradient de la sortie par rapport à chaque entrée
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return grad

def hessien(reseau, x):
    """Calcule la matrice Hessienne (dérivées du 2ème ordre)"""
    x = x.clone().requires_grad_(True)
    grads = jacobien(reseau, x)
    
    H = []
    for g in grads.flatten():
        grad2 = torch.autograd.grad(g, x, retain_graph=True)[0]
        H.append(grad2.detach().numpy().flatten())
    
    return np.array(H)

# Calculer aux différents points
points = [
    torch.tensor([[0.0, 0.0]]),
    torch.tensor([[1.0, 0.5]]),
    torch.tensor([[-1.0, 2.0]]),
]

print("=== DÉRIVÉES DU RÉSEAU GELU ===\n")
for i, x in enumerate(points):
    J = jacobien(reseau, x)
    H = hessien(reseau, x)
    print(f"Point {i+1}: x = {x.numpy()[0]}")
    print(f"  Jacobien (df/dx): {J.detach().numpy()[0]}")
    print(f"  Hessien (|H|_F): {np.linalg.norm(H):.4f}")
    print()

print("→ Idée: si la Hessienne contient les poids de façon exploitable,")
print("  peut-on les extraire à partir des Hessiennes mesurées en différents points?")

# ============================================
# Piste: Le Hessien d'un réseau GELU
# révèle des infos sur les poids
# ============================================

# Hessien analytique pour 1 couche: f(x) = W₂ · GELU(W₁x + b₁) + b₂
# d²f/dx² = W₁ᵀ · diag(W₂ · diag(GELU''(z₁))) · W₁
#
# Cela donne: H(x) = W₁ᵀ · D(x) · W₁
# où D(x) est diagonale et dépend des valeurs intermédiaires
#
# Si on mesure H(x) en de nombreux points:
# H(x₁), H(x₂), ..., H(xₙ)
# peut-on récupérer W₁ ?

print("\nHessien analytique:")
print("H(x) = W₁ᵀ · D(x) · W₁")
print("Avec D(x) diagonal dépendant de GELU''(W₁x + b₁)")
print()
print("Si D était connu → problème de factorisation matricielle (difficile)")
print("Si on mesure H en BEAUCOUP de points → système surdéterminé")
print("→ Piste de recherche active !")
```

### Analyse de la piste 2

```
AVANTAGES:
  ✅ Exploite la structure analytique de GELU (information riche)
  ✅ Potentiellement plus précis que l'approximation ReLU
  ✅ Approche originale (peu de travail dans ce sens)

DÉFIS:
  ⚠️  Mathématiques plus avancées (décomposition tensorielle)
  ⚠️  Nombre de requêtes potentiellement élevé
  ⚠️  Stabilité numérique des dérivées d'ordre élevé
  
LIEN AVEC LA LITTÉRATURE:
  → Lien avec l'extraction par gradient (Jagielski et al. 2020)
  → Lien avec les méthodes de factorisation tensorielle
```

---

## Jour 26 (9 Mai) — Piste 3: Exploitation des Symétries [2h]

### Une idée élégante: les symétries de GELU

```
OBSERVATION: GELU et SiLU ont des symétries exploitables

SiLU(-x) = -x · σ(-x) = -x · (1 - σ(x)) ≠ -SiLU(x) en général
MAIS: SiLU a un minimum local en x ≈ -1.28

Ce minimum est UNIQUE et sa POSITION NE DÉPEND PAS de x.
Pour un neurone: SiLU(w·x + b) a son minimum là où w·x + b = -1.28

→ En cherchant les minima, on obtient des équations sur w et b !
```

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def silu(x):
    return x / (1 + np.exp(-x))

def dsilu(x):
    s = 1 / (1 + np.exp(-x))
    return s + x * s * (1 - s)

# Trouver le minimum de SiLU
from scipy.optimize import minimize_scalar
result = minimize_scalar(silu, bounds=(-3, 0), method='bounded')
x_min = result.x
print(f"Minimum de SiLU: x = {x_min:.6f}")
print(f"SiLU({x_min:.3f}) = {silu(x_min):.6f}")
print(f"SiLU'({x_min:.3f}) = {dsilu(x_min):.6f} (doit être 0)")

# Visualisation
x = np.linspace(-4, 2, 1000)
plt.figure(figsize=(8, 5))
plt.plot(x, silu(x), 'b-', linewidth=2, label='SiLU(x)')
plt.axvline(x=x_min, color='r', linestyle='--', label=f'Minimum à x={x_min:.3f}')
plt.axhline(y=silu(x_min), color='g', linestyle=':', label=f'f(min)={silu(x_min):.3f}')
plt.scatter([x_min], [silu(x_min)], c='red', s=100, zorder=5)
plt.title("SiLU — Le minimum local comme point d'ancrage")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('silu_minimum.png', dpi=150)
plt.show()

print(f"\n→ IDÉE: Pour un neurone SiLU avec poids w et biais b:")
print(f"  SiLU(w·x + b) est minimal quand w·x + b ≈ {x_min:.3f}")
print(f"  Donc le minimum de la fonction vue de l'extérieur se produit quand:")
print(f"  x = ({x_min:.3f} - b) / w")
print(f"  En trouvant ce minimum par optimisation → équation sur w et b !")
```

---

## Jour 27 (10 Mai) — Synthèse: Définir Ta Contribution de Thèse [3h]

### L'état de l'art en une carte

```
                          ATTAQUES D'EXTRACTION DE RÉSEAUX
                          ══════════════════════════════════

    SOFT-LABEL                          HARD-LABEL
    (l'API retourne f(x) complet)       (l'API retourne juste la classe)
         │                                     │
         ▼                                     ▼
  ┌──────────────────┐                ┌──────────────────┐
  │ Carlini 2020     │                │ Carlini 2025     │
  │ ReLU, polynomial │                │ ReLU, hard-label │
  │ [CRYPTO 2020]    │                │ [EUROCRYPT 2025] │
  └──────────────────┘                └──────────────────┘
         │                                     │
         ▼                                     ▼
  ┌──────────────────┐                ┌──────────────────┐
  │ Canales 2024     │                │ ???              │
  │ ReLU, meilleur   │                │ GELU hard-label  │
  │ [EUROCRYPT 2024] │                │ OUVERT           │
  └──────────────────┘                └──────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────┐
  │ TON TRAVAIL (2025-2028)                  │
  │                                          │
  │ GELU/SiLU soft-label → Piste prioritaire │
  │                                          │
  │ Piste A: Approximation ReLU              │
  │ Piste B: Dérivées d'ordre élevé          │
  │ Piste C: Symétries de l'activation       │
  └──────────────────────────────────────────┘
```

### Questions de recherche à présenter à ton directeur

```
QUESTION PRINCIPALE:
"Peut-on effectuer une extraction cryptanalytique exacte
 d'un réseau de neurones utilisant des fonctions d'activation
 lisses (GELU, SiLU) en temps polynomial?"

SOUS-QUESTIONS:
  Q1: Quelles propriétés de ReLU sont ESSENTIELLES pour l'attaque
      Carlini 2020, et lesquelles peuvent être remplacées?
  
  Q2: Peut-on approximer efficacement GELU par une combinaison
      de ReLU pour permettre l'utilisation des méthodes existantes?
      
  Q3: Les dérivées d'ordre élevé d'un réseau GELU contiennent-elles
      assez d'information pour extraire les poids?
  
  Q4: Peut-on définir un analogue des "kinks" pour les réseaux GELU
      (ex: points d'inflexion, minima locaux)?

CONTRIBUTIONS POSSIBLES:
  1. Preuve négative: montrer formellement pourquoi les méthodes
     ReLU ne peuvent PAS s'appliquer directement à GELU
  2. Algorithme d'extraction pour réseaux GELU (soft-label)
  3. Borne inférieure sur le nombre de requêtes nécessaires
  4. Application à des modèles réels (GPT-2 avec GELU)
```

### Préparer tes questions pour la prochaine réunion directeur

```
□ Question 1: "Quelle piste (A, B ou C) vous semble la plus prometteuse
               pour une contribution publiable en 18 mois?"

□ Question 2: "Dois-je d'abord prouver l'impossibilité des méthodes
               existantes, ou aller directement vers une nouvelle attaque?"

□ Question 3: "Y a-t-il des papiers récents (2024-2025) que vous
               recommandez sur l'extraction pour GELU spécifiquement?"

□ Question 4: "Pour les expériences: dois-je travailler sur des réseaux
               jouets ou sur des modèles réels (GPT-2) dès le début?"
```

---

## Jour 28 (11 Mai) — Révision + Mini-Projet d'Expérimentation [3h]

### Mini-projet: Tester les 3 pistes expérimentalement

```python
"""
MINI-PROJET: Comparaison des 3 pistes d'attaque sur un réseau GELU 1D
             
Réseau: f: ℝ → ℝ (entrée 1D pour pouvoir visualiser)
Réseau: 1 couche cachée, 4 neurones GELU
Objectif: Mesurer la qualité de récupération des poids avec chaque piste
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

torch.manual_seed(2026)

# ============================================
# Réseau GELU cible
# ============================================
reseau_gelu = nn.Sequential(
    nn.Linear(1, 4),
    nn.GELU(),
    nn.Linear(4, 1)
)

W1_vrai = reseau_gelu[0].weight.data.numpy().flatten()  # 4 poids
b1_vrai = reseau_gelu[0].bias.data.numpy()              # 4 biais
W2_vrai = reseau_gelu[2].weight.data.numpy().flatten()  # 4 poids sortie
b2_vrai = reseau_gelu[2].bias.data.numpy()[0]           # 1 biais sortie

print("=== VRAIS PARAMÈTRES ===")
print(f"W1: {W1_vrai}")
print(f"b1: {b1_vrai}")
print(f"W2: {W2_vrai}")
print(f"b2: {b2_vrai:.4f}")

def oracle(x_val):
    with torch.no_grad():
        return reseau_gelu(torch.tensor([[x_val]], dtype=torch.float32)).item()

# ============================================
# PISTE A: Approximation par régression
# (Baseline: ajustement de courbe sur les sorties)
# ============================================
from scipy.optimize import curve_fit

def modele_approxime(x, *params):
    """Modèle à ajuster: même architecture que le vrai réseau"""
    W1, b1, W2, b2 = params[:4], params[4:8], params[8:12], params[12]
    z = np.outer(x, W1) + b1  # (n, 4)
    a = z * norm.cdf(z)         # GELU
    return a @ W2 + b2          # (n,)

# Données d'entraînement
n_data = 200
x_data = np.linspace(-3, 3, n_data)
y_data = np.array([oracle(x) for x in x_data])

# Compter les requêtes
requetes_piste_a = n_data
print(f"\n=== PISTE A: Régression directe ===")
print(f"Requêtes utilisées: {requetes_piste_a}")

try:
    p0 = np.random.randn(13) * 0.1
    params_opt, _ = curve_fit(modele_approxime, x_data, y_data, p0=p0, maxfev=10000)
    
    W1_est = params_opt[:4]
    y_pred = modele_approxime(x_data, *params_opt)
    erreur_a = np.sqrt(np.mean((y_data - y_pred)**2))
    print(f"Erreur RMS (reconstruction): {erreur_a:.6f}")
    print(f"W1 estimé: {W1_est}")
    print(f"W1 vrai:   {W1_vrai}")
except Exception as e:
    print(f"Difficultés d'optimisation: {e}")
    print("→ Ce problème est non-convexe, difficile à optimiser!")

# ============================================
# Visualisation finale
# ============================================
x_plot = np.linspace(-4, 4, 1000)
f_vrai = [oracle(x) for x in x_plot]

plt.figure(figsize=(10, 5))
plt.plot(x_plot, f_vrai, 'b-', linewidth=3, label='Réseau GELU (vrai)')
plt.scatter(x_data[::10], y_data[::10], c='red', s=20, alpha=0.5, label='Données d\'observation')
plt.title("Mini-projet: Observer un réseau GELU en boîte noire")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('miniprojet_gelu.png', dpi=150)
plt.show()

print("\n=== BILAN DU MINI-PROJET ===")
print("Ce qu'on apprend:")
print("  1. Un réseau GELU est beaucoup plus 'lisse' à observer")
print("  2. La régression directe est difficile (non-convexe)")
print("  3. On a besoin d'une méthode plus intelligente → thèse!")
```

---

## Bilan des 2 Semaines

```
SEMAINE 3 — CE QUE TU SAIS MAINTENANT
═══════════════════════════════════════
✅ Régions linéaires de ReLU (concept fondamental)
✅ Détection de kinks par différences finies
✅ Extraction des poids à partir des kinks
✅ Problème du signe et comment le résoudre
✅ Les 3 papiers: Carlini 2020, Canales 2024, Carlini 2025
✅ Différence soft-label vs hard-label

SEMAINE 4 — CE QUE TU SAIS MAINTENANT
═══════════════════════════════════════
✅ Propriétés mathématiques de GELU et SiLU
✅ Pourquoi les attaques ReLU échouent sur GELU (3 obstacles précis)
✅ Piste A: Approximation locale par ReLU
✅ Piste B: Dérivées d'ordre élevé (Hessien)
✅ Piste C: Exploitation des symétries (minima locaux)
✅ Ta position dans l'état de l'art
✅ Les bonnes questions à poser à ton directeur

PROCHAINES ÉTAPES (Semaines 5-6):
═══════════════════════════════════
→ Choisir une piste avec ton directeur
→ Étudier les outils mathématiques spécifiques à cette piste
→ Reproduire l'attaque Carlini 2020 sur des réseaux jouets
→ Premier rapport de recherche
```

---

## Suivi quotidien

| Jour | Date | Tâche | Durée | Fait? | Notes |
|------|------|-------|-------|-------|-------|
| 15 | 28 Avr | Régions linéaires ReLU | 2-3h | ☐ | |
| 16 | 29 Avr | Recherche de kinks | 3h | ☐ | |
| 17 | 30 Avr | Extraction des poids | 3h | ☐ | |
| 18 | 1 Mai | Signe + implémentation | 3h | ☐ | |
| 19 | 2 Mai | Lire Carlini 2020 guidé | 3h | ☐ | |
| 20 | 3 Mai | Canales 2024 | 2h | ☐ | |
| 21 | 4 Mai | Carlini 2025 + révision | 2h | ☐ | |
| 22 | 5 Mai | Propriétés GELU/SiLU | 3h | ☐ | |
| 23 | 6 Mai | Pourquoi ReLU échoue | 3h | ☐ | |
| 24 | 7 Mai | Piste A: approx ReLU | 3h | ☐ | |
| 25 | 8 Mai | Piste B: Hessien | 3h | ☐ | |
| 26 | 9 Mai | Piste C: Symétries | 2h | ☐ | |
| 27 | 10 Mai | Synthèse + questions | 3h | ☐ | |
| 28 | 11 Mai | Mini-projet expérimental | 3h | ☐ | |

---

## Ressources Complètes

### Articles à lire (par priorité)
| Priorité | Article | Où | Quand lire |
|----------|---------|-----|------------|
| ⭐⭐⭐ | Carlini et al. CRYPTO 2020 | arxiv.org/abs/1910.00866 | Jour 19 |
| ⭐⭐⭐ | Canales-Martínez et al. 2024 | eprint.iacr.org/2024/297 | Jour 20 |
| ⭐⭐⭐ | Carlini et al. EUROCRYPT 2025 | À venir | Jour 21 |
| ⭐⭐ | Hendrycks & Gimpel (GELU) | arxiv.org/abs/1606.08415 | Jour 22 |
| ⭐⭐ | Jagielski et al. 2020 (stealing) | arxiv.org/abs/1909.01838 | Semaine 5 |
| ⭐ | Ramachandran et al. (SiLU) | arxiv.org/abs/1710.05941 | Semaine 5 |

### Vidéos essentielles
| Vidéo | Durée | Pour quel jour |
|-------|-------|----------------|
| Carlini - Talk sur l'extraction | ~30min | Jour 19 |
| 3Blue1Brown - Essence of Calculus (ep. 1-4) | ~1h | Révision |
| Andrej Karpathy - Micrograd from scratch | ~2h30 | Semaine 5 |
| MIT OpenCourseWare - 6.S978 (Adversarial ML) | ~50min/cours | Semaine 5+ |

### Outils Python à maîtriser
```python
# Cette semaine tu vas utiliser:
import torch              # Réseaux et autograd
import numpy as np        # Calcul numérique
from scipy.stats import norm    # CDF normale pour GELU
from scipy.optimize import minimize, minimize_scalar, curve_fit
import matplotlib.pyplot as plt  # Visualisation
```

### Code de référence à explorer
```
GitHub: google/cryptanalytic-model-extraction
        ↑ Code officiel Carlini 2020

GitHub: bethgelab/foolbox
        ↑ Boîte à outils d'attaques adversariales

GitHub: Trusted-AI/adversarial-robustness-toolbox
        ↑ IBM ART: framework complet sécurité ML
```

---

## Objectifs Vérifiables à la Fin des 2 Semaines

1. ☐ Tu peux expliquer en 2 minutes ce qu'est une "région linéaire" d'un réseau ReLU
2. ☐ Tu peux coder un détecteur de kinks par différences finies
3. ☐ Tu as lu Abstract + Introduction + Section 2 de Carlini 2020
4. ☐ Tu peux expliquer pourquoi GELU n'a pas de kinks (en termes mathématiques)
5. ☐ Tu peux nommer 3 pistes pour attaquer un réseau GELU
6. ☐ Tu as préparé 4 questions concrètes pour ton prochain meeting directeur
7. ☐ Tu as exécuté au moins 5 scripts Python de ce sprint
