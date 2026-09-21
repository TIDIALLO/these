# Cahier de thèse — Tidiane DIALLO

**Extraction cryptanalytique de réseaux de neurones profonds : au-delà de ReLU**
École Polytechnique de Thiès — EDPT — LTISI / CRISIN'2D
Directeur : Pr. Abdoul Aziz Ciss

> **Comment utiliser ce cahier.** Ce n'est pas un document à lire d'un trait. C'est une référence à consulter chaque fois qu'un terme te bloque dans un article. Chaque entrée suit le même schéma : *définition formelle → intuition → exemple chiffré → pourquoi ça compte pour ta thèse*. Les sections 6 et 7 sont un plan d'apprentissage : coche les cases.

---

## Table des matières

1. [Notations de référence](#1-notations)
2. [Glossaire — Deep Learning](#2-glossaire-deep-learning)
3. [Glossaire — Géométrie de l'extraction](#3-glossaire-geometrie)
4. [Glossaire — Cryptanalyse et sécurité](#4-glossaire-cryptanalyse)
5. [Les mécanismes d'attaque, expliqués pas à pas](#5-mecanismes)
6. [Compétences à maîtriser — checklist](#6-competences)
7. [Ressources d'apprentissage](#7-ressources)
8. [Boîte à outils pratique](#8-outils)
9. [Erreurs classiques et pièges](#9-pieges)

---

<a name="1-notations"></a>
## 1. Notations de référence

Ces notations sont celles de Carlini et al. et de ton article de départ. **Utilise-les systématiquement**, y compris dans tes propres notes : la cohérence de notation est ce qui distingue un manuscrit lisible d'un manuscrit pénible.

| Symbole | Signification |
|---|---|
| `f` | le réseau complet, `f : X → Y` |
| `r` | profondeur (nombre de couches cachées) — on parle de « réseau r-profond » |
| `f_i` | i-ème couche affine : `f_i(x) = A^(i) x + b^(i)` |
| `σ` | fonction d'activation |
| `d_i` | nombre de neurones de la couche i ; `d_0` = dimension d'entrée ; `d_{r+1}` = nombre de sorties |
| `A^(i)` | matrice de poids de la couche i, taille `d_i × d_{i−1}` |
| `b^(i)` | vecteur de biais de la couche i, taille `d_i` |
| `A^(i)_j` | j-ème **ligne** de `A^(i)` = poids du j-ème neurone |
| `A_{*,k}` | k-ème **colonne** de A |
| `N = Σ_k d_k` | nombre total de neurones |
| `θ` | l'ensemble des paramètres (poids + biais) |
| `θ̂`, `Â`, `b̂` | paramètres **récupérés** par l'attaque (le chapeau = « estimé ») |
| `η` | un neurone |
| `V(η; x)` | valeur de η en x **avant** activation |
| `S(η; x)` | état de η en x : *actif* si `V > 0`, *inactif* sinon |
| `z(·)` | fonction hard-label (argmax après softmax) |
| `F_{i−1}` | fonction d'entrée : les couches 1..i−1 déjà récupérées |
| `G_{i+1}` | fonction de sortie : les couches i+1..r+1 non encore récupérées |
| `Γ, β` | matrice et vecteur de la transformation affine effective dans un voisinage linéaire |
| `I^(ℓ)` | matrice diagonale des états (1 = actif, 0 = inactif) de la couche ℓ |
| `m`, `m_on`, `m_off` | vecteurs normaux à la frontière de décision |
| `Δ_on`, `Δ_off` | distances mesurées de part et d'autre d'un hyperplan critique |
| `D`, `D'` | sous-espaces de points duaux |
| `α` | constante multiplicative inconnue dans la signature |

**Composition du réseau :**

```
f = f_{r+1} ∘ σ ∘ f_r ∘ … ∘ σ ∘ f_2 ∘ σ ∘ f_1
```

Note bien : **il n'y a pas de σ après `f_{r+1}`**. La couche de sortie est purement affine. C'est *exactement* la raison pour laquelle la technique de signature d'EUROCRYPT 2025 ne s'y applique pas, et donc la raison d'être de ton article de départ.

**Décomposition en trois blocs** (le point de vue de l'attaquant, à mémoriser) :

```
f  =  G_{i+1}  ∘  σ ∘ f_i  ∘  F_{i−1}
      ↑            ↑           ↑
   inconnu     la cible     déjà connu
```

---

<a name="2-glossaire-deep-learning"></a>
## 2. Glossaire — Deep Learning

### 2.1 Neurone

**Définition.** Le j-ème neurone de la couche i est la fonction

```
η(x) = σ( A^(i)_j · x + b^(i)_j )
```

**Intuition.** Un produit scalaire (« à quel point l'entrée ressemble-t-elle à mon vecteur de poids ? »), plus un décalage, passé dans une non-linéarité.

**Exemple chiffré.** `A_j = (2, −1, 0.5)`, `b_j = −1`, entrée `x = (1, 1, 2)`.
Pré-activation : `2(1) + (−1)(1) + 0.5(2) − 1 = 2 − 1 + 1 − 1 = 1`. Avec ReLU : `σ(1) = 1`. Le neurone est **actif**.
Avec `x = (0, 3, 0)` : `0 − 3 + 0 − 1 = −4`, `ReLU(−4) = 0`. Le neurone est **inactif**.

**Pourquoi ça compte.** Toute l'attaque consiste à retrouver les `A_j` et `b_j`. Le passage de « actif » à « inactif » est **le seul événement observable** qui trahit ces valeurs.

---

### 2.2 Fonctions d'activation — la classification qui structure ta thèse

C'est le concept central de ton sujet. Range-les en **trois familles**, car c'est cette structure qui détermine si une attaque marche.

#### Famille A — Piecewise-linéaires (linéaires par morceaux)

| Nom | Formule | Points de non-différentiabilité |
|---|---|---|
| **ReLU** | `max(x, 0)` | 1 (en x = 0) |
| **Leaky ReLU** | `x si x>0, ax sinon` (a ≈ 0.01) | 1 |
| **PReLU** | idem mais `a` est **appris** | 1 |
| **HardTanh** | `clip(x, −1, 1)` | 2 (en −1 et +1) |
| **Step / Heaviside** | `1 si x>0, 0 sinon` | 1 (discontinue) |

**Propriété clé :** le réseau entier est une **fonction affine par morceaux**. L'espace d'entrée est découpé en régions polyédriques ; dans chacune, `f` est exactement affine. Les frontières sont des **hyperplans**.

C'est cette propriété qui rend l'extraction *facile*, et c'est pourquoi toute la littérature 2020–2025 vit là.

#### Famille B — Lisses quasi-linéaires ⭐ *(ton terrain)*

| Nom | Formule | Comportement |
|---|---|---|
| **GELU** | `x · Φ(x)` où Φ = CDF de la loi normale | → 0 en −∞, → x en +∞ |
| **SiLU / Swish** | `x · sigmoid(x)` | idem |
| **ELU** | `x si x>0, a(eˣ−1) sinon` | → −a en −∞, → x en +∞ |
| **SELU** | ELU mise à l'échelle (auto-normalisante) | idem |
| **Softplus** | `ln(1 + eˣ)` | → 0 en −∞, → x en +∞ |
| **Mish** | `x · tanh(softplus(x))` | idem |

**Propriété clé et pourquoi c'est important :** ces fonctions sont **quasi-linéaires en dehors d'une bande étroite** autour de zéro. GELU(5) ≈ 5.0000 (à 3·10⁻⁷ près), GELU(−5) ≈ −1.5·10⁻⁶ ≈ 0. Donc à distance de l'origine, GELU **est** ReLU.

C'est précisément l'observation qu'exploite ePrint 2026/253 : « les activations convergent vers un comportement linéaire en dehors de zones non-linéaires étroites ». La géométrie de l'extraction survit — le pli net devient une **zone de courbure concentrée** de largeur ~2–3 unités de pré-activation.

**Exemple chiffré (à faire toi-même, c'est très instructif).**

| x | ReLU | GELU | SiLU | tanh |
|---|---|---|---|---|
| −5 | 0 | −0.0000015 | −0.0336 | −0.9999 |
| −2 | 0 | −0.0455 | −0.2384 | −0.9640 |
| −1 | 0 | −0.1587 | −0.2689 | −0.7616 |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0.8413 | 0.7311 | 0.7616 |
| 2 | 2 | 1.9545 | 1.7616 | 0.9640 |
| 5 | 5 | 5.0000 | 4.9665 | 0.9999 |

Lis ce tableau en colonnes : GELU rejoint ReLU dès |x| ≈ 4. tanh, lui, **sature** — il ne rejoint jamais l'identité. Voilà pourquoi tanh sera bien plus dur (voire impossible) à attaquer que GELU. **Ce tableau contient l'intuition centrale de ta thèse.**

#### Famille C — Analytiques saturantes ⭐⭐ *(le vrai défi)*

| Nom | Formule | Image |
|---|---|---|
| **Sigmoïde** | `1/(1+e^{−x})` | (0, 1) |
| **Tanh** | `(eˣ−e^{−x})/(eˣ+e^{−x})` | (−1, 1) |

**Propriété clé :** bornées, **jamais linéaires nulle part**, analytiques (infiniment dérivables). Il n'y a plus aucune région linéaire, donc plus d'hyperplan critique, plus de point critique au sens de la Définition 8.

Résultat classique à connaître : **Fefferman (1994)** a montré que sous certaines conditions un réseau tanh est déterminé de façon **unique** (à symétries près) par sa fonction de sortie. Donc l'information *est* là. La question est de savoir si on peut l'extraire **efficacement**, et **en hard-label**.

**Mon pronostic honnête, à discuter avec ton directeur :** pour tanh/sigmoïde en hard-label, tu obtiendras probablement un résultat d'**impossibilité pratique** plutôt qu'une attaque. C'est une bonne nouvelle — c'est un résultat, et il fait le pont direct vers ton volet défense (« utiliser une activation saturante *est* une défense »).

---

### 2.3 Couche entièrement connectée (fully connected / dense)

`f_i(x) = A^(i) x + b^(i)`, avec `A^(i) ∈ ℝ^{d_i × d_{i−1}}`.
Chaque neurone voit **toutes** les entrées de la couche précédente. Aucun paramètre partagé.

### 2.4 Couche convolutive

Chaque neurone ne voit qu'une **fenêtre locale** et les poids sont **partagés** entre positions.

**Ce que ça change pour l'attaque** — et c'est subtil :
- **Partage de poids** = beaucoup moins de paramètres à récupérer (bonne nouvelle pour l'attaquant), mais les hyperplans critiques d'un même noyau sont **parallèles** entre eux, ce qui crée des ambiguïtés (mauvaise nouvelle).
- Techniquement, une couche convolutive **est** une couche dense avec une matrice creuse très structurée. ePrint 2026/139 la formalise comme une matrice **BTTB** (Block-Toeplitz-with-Toeplitz-Blocks).
- 2026/1164 exploite le partage de poids par un **clustering centré noyau** (kernel-centric) plutôt que centré neurone.

### 2.5 Pooling

- **Average pooling** : linéaire → se fond dans la couche affine, ne pose aucun problème nouveau.
- **Max pooling** : **non-linéaire**. C'est une deuxième source de non-linéarité en plus de ReLU, et elle **masque** les points critiques de ReLU. Résultat de 2026/464 : un point critique de max-pooling capture la **différence entre deux neurones convolutifs** (le point où deux d'entre eux s'égalisent). 2026/241 prouve formellement que les CNN à max-pooling restent piecewise linéaires.

### 2.6 Softmax

```
s(u)_i = e^{u_i} / Σ_j e^{u_j}
```

**Deux propriétés qui gouvernent toute la Section 3 de ton article de départ :**

1. **Invariance par translation.** `s(u + c̄) = s(u)` pour tout vecteur `c̄` à coordonnées toutes égales à `c`. Preuve immédiate : `e^{u_i+c} / Σ e^{u_j+c} = (e^{u_i}e^c)/(e^c Σ e^{u_j}) = s(u)_i`.
2. **Non-invariance par mise à l'échelle, mais préservation de l'ordre.** `s(cu) ≠ s(u)`, mais pour `c > 0` la plus grande coordonnée reste la plus grande. Donc **l'argmax — le hard-label — est invariant**.

**Conséquence directe :** on peut translater n'importe quelle colonne de `A^(r+1)`, translater `b^(r+1)`, et multiplier le tout par `c > 0`, sans changer aucune prédiction. Cela fait **`d_r + 2` degrés de liberté**. D'où : rang maximal `d_{r+1}(d_r+1) − (d_r+2)`, et nécessité de fixer arbitrairement `d_r + 2` variables.

**Exemple chiffré.** `d_r = 64`, `d_{r+1} = 10` (le réseau CIFAR-10 de l'article) :
- inconnues : `10 × 65 = 650`
- rang maximal : `650 − 66 = 584` ✓ (c'est le chiffre de l'article)
- les auteurs fixent `Â^(5)_1 = (0,…,0)`, `b̂^(5)_1 = 0`, `Â^(5)_{2,1} = 1` → 64 + 1 + 1 = 66 variables fixées ✓

---

### 2.7 Réseaux équivalents et symétries

**Définition.** Deux réseaux sont *fonctionnellement équivalents* si `f_θ̂(x) = f_θ(x)` pour tout `x` (ou, version étendue de Chen et al. ASIACRYPT 2024 : `f_θ̂(x) = c · f_θ(x)` pour un `c > 0` fixe).

**Les symétries d'un réseau ReLU :**

| Transformation | Préserve la fonction ? |
|---|---|
| Permuter les neurones d'une couche | ✅ oui |
| Multiplier `(A_j, b_j)` par `c > 0` et diviser la colonne correspondante de la couche suivante par `c` | ✅ oui — car `ReLU(cx) = c·ReLU(x)` pour `c > 0` |
| Multiplier par `c < 0` | ❌ **non** — `ReLU(−x) ≠ −ReLU(x)` |

**C'est toute l'histoire du problème du signe.** L'attaque récupère naturellement `Â_j = α A_j` avec `α = 1/A_{j,1}`, une constante de signe **inconnu**. Si `α > 0` c'est un réseau équivalent ; si `α < 0` le réseau est faux. D'où deux sous-problèmes toujours traités séparément :

- **Récupération de signature** : trouver `A_j` à une constante multiplicative près.
- **Récupération de signe** : trouver le signe de cette constante.

**Attention conceptuelle importante pour ton créneau :** cette homogénéité positive `σ(cx) = cσ(x)` est **spécifique à ReLU**. GELU, SiLU, tanh **ne sont pas homogènes**. Donc chez ces activations, la mise à l'échelle n'est plus une symétrie — **le groupe de symétries se réduit aux seules permutations**. Conséquence pratique : *le problème du signe disparaît* pour les activations lisses (on récupère les poids exacts, pas un multiple), mais *l'extraction de la signature devient plus dure* car on ne peut plus normaliser librement. C'est un point à énoncer clairement dans ton premier article — beaucoup de lecteurs ne l'ont pas en tête.

---

<a name="3-glossaire-geometrie"></a>
## 3. Glossaire — Géométrie de l'extraction

C'est le cœur technique. Prends le temps de **dessiner** chacune de ces notions.

### 3.1 Voisinage linéaire

**Définition.** `{u ∈ X | S(η; x) = S(η; u) pour tout neurone η}` — l'ensemble des points où **tous** les neurones ont le même état qu'en `x`.

**Intuition.** Un « motif d'activation » est un vecteur binaire de longueur N (actif/inactif pour chaque neurone). Le voisinage linéaire est la région de l'espace d'entrée qui produit ce même motif. Dans cette région, chaque ReLU est soit « identité » soit « zéro » — donc constant — donc le réseau entier **se réduit à une transformation affine** `Γx + β`.

**Conséquence opérationnelle (Équation 3 de l'article) :**
```
F_{i,j}(x + Δ) − F_{i,j}(x) = Γ Δ
```
La constante `β` s'annule. **C'est le fondement de toutes les attaques différentielles** : on prend des différences pour éliminer les biais et isoler la partie linéaire, exactement comme en cryptanalyse différentielle on prend des différences de textes clairs pour éliminer les clés de tour.

**Exemple.** Réseau 2 entrées → 3 neurones → 1 sortie. Motif (actif, inactif, actif) → une région polygonale de ℝ². Dans cette région, `f(x) = γ₁x₁ + γ₂x₂ + β`. Franchis un bord → le motif change → `γ` change brusquement.

---

### 3.2 Point critique

**Définition 8.** `x` est critique pour `η` si `V(η; x) = 0` (pré-activation exactement nulle).

**Intuition.** Le neurone est *pile* sur son seuil de basculement. Une variation infinitésimale le fait changer d'état.

**Ce qu'un point critique révèle** — et c'est énorme : `A_j · x + b_j = 0`. C'est une **équation linéaire exacte** sur les paramètres inconnus. Collecte `d_{i−1}` points critiques indépendants pour le même neurone → système linéaire → signature.

**Pour ton sujet :** cette définition **n'a plus de sens** chez tanh ou GELU. `V(η; x) = 0` ne produit plus aucun événement observable, puisque la fonction est lisse en 0. Il faut lui substituer autre chose. C'est le premier obstacle formel de ta thèse, et il faut savoir l'énoncer en une phrase.

---

### 3.3 Hyperplan critique

**Définition.** L'ensemble `{x : A_j x + b_j = 0}` — l'ensemble de tous les points critiques d'un neurone donné. C'est un hyperplan de dimension `d₀ − 1`.

**Intuition géométrique (Figure 1a de l'article).** Chaque neurone dessine un hyperplan dans l'espace d'entrée. L'ensemble de ces hyperplans découpe l'espace en régions polyédriques = les voisinages linéaires.

**Subtilité cruciale.** Pour la **première couche**, les hyperplans sont *plats* (ce sont de vrais hyperplans). Pour les couches **profondes**, ils sont **pliés** par les hyperplans des couches précédentes. Regarde la Figure 1a de ton article : `η₄` et `η₅` (couche 2) sont visiblement cassés là où ils croisent `η₁, η₂, η₃` (couche 1). C'est pour ça que l'extraction procède couche par couche en « pelant » (peeling) : une fois la couche 1 connue, on peut *déplier* mentalement l'espace.

---

### 3.4 Point de transition ⭐ *(la notion centrale en hard-label)*

**Définition 10.** `x` où la décision change de classe : `z(f(x+ε)) ≠ z(f(x−ε))`.

**Différence capitale avec un point critique :**

| | Point critique | Point de transition |
|---|---|---|
| Condition | un neurone vaut 0 | deux logits sont égaux |
| Observable en raw-output ? | ✅ (cassure de pente) | ✅ |
| Observable en hard-label ? | ❌ **invisible** | ✅ **le seul événement visible** |
| Révèle | les poids du neurone | une relation entre logits |

**Le drame du hard-label est là :** l'événement qu'on peut voir (transition) n'est **pas** celui qui contient l'information sur les neurones internes (critique). Un point de transition ne correspond à aucun changement d'état interne. Toute l'ingéniosité de Carlini et al. EUROCRYPT 2025 consiste à **fabriquer** des points critiques à partir de points de transition.

---

### 3.5 Frontière de décision

**Définition.** L'ensemble des points de transition entre deux classes. Dimension `d₀ − 1`.

**La propriété exploitée (Figure 1b) :** puisque le réseau est affine par morceaux, la frontière de décision est **elle-même** affine par morceaux — et elle **se plie** exactement là où elle traverse un hyperplan critique.

**Voilà l'idée maîtresse à retenir.** La frontière de décision est *observable* en hard-label. Les hyperplans critiques ne le sont pas. Mais les hyperplans critiques **laissent leur empreinte** sur la frontière de décision, sous forme de plis. Donc : *observer les plis de la frontière de décision revient à observer les hyperplans critiques*. C'est la phrase à savoir réciter.

**Et pour les activations lisses ?** La frontière ne se plie plus — elle **s'incurve**. L'empreinte est toujours là, mais elle est étalée sur une largeur ~ε au lieu d'être ponctuelle. **C'est là que se joue ta thèse.**

---

### 3.6 Point dual

**Définition 11.** Un point à la fois **de transition** et **critique** pour un neurone.

**Pourquoi c'est le graal :** c'est un point observable en hard-label (transition) qui contient de l'information sur un neurone interne (critique). Le pont entre les deux mondes.

**Comment on en fabrique un** (procédure de la Figure 2, à savoir dessiner) :
1. Trouver un point de transition `x₁` (recherche binaire entre deux entrées de classes différentes).
2. Marcher le long de la frontière de décision dans une direction `δ`.
3. À un moment on traverse un hyperplan critique — la frontière se plie — on sort de la frontière.
4. Projeter sur la frontière « de l'autre côté » → point de transition `x₂`.
5. Estimer les normales `m_off` (avant) et `m_on` (après) par échantillonnage local.
6. **L'intersection des deux frontières** (avant/après pli) est un sous-espace `D` de dimension `d₀ − 2`, **entièrement contenu dans l'hyperplan critique**.
7. Recommencer ailleurs → un autre sous-espace `D'` pour le même neurone.
8. `D` et `D'` engendrent l'hyperplan critique complet (dimension `d₀ − 1`) → **signature**.

Note l'astuce de dimension : un seul `D` donne `d₀−2` dimensions, il en manque une ; deux sous-espaces suffisent (avec très haute probabilité).

---

### 3.7 Signature

**Définition 13.** La signature du j-ème neurone de la couche i est `α A^(i)_j` pour un `α ∈ ℝ` quelconque.

**Intuition.** La *direction* du vecteur de poids, sans son échelle ni son signe. Géométriquement : la **normale** à l'hyperplan critique. Un hyperplan détermine sa normale à un scalaire près — d'où l'indétermination.

**Normalisation usuelle :** `A^(i)_j / A^(i)_{j,1}` (diviser par la première coordonnée), donc `α = 1/A^(i)_{j,1}`.

**Et le biais ?** Une fois la signature connue et les couches précédentes récupérées, `b̂^(i)_j = b^(i)_j / A^(i)_{j,1}` s'obtient directement — même `α`, cohérent.

---

### 3.8 Récupération de signe

Le second sous-problème. Trois méthodes historiques :

**(a) Freeze** (CRYPTO 2020, §4.2)
Nécessite une couche **contractive** (`d_{i−1} > d_i`). On construit `x₊` et `x₋` qui ne modifient que la j-ème coordonnée de la pré-activation, tous les autres neurones étant « gelés ». Si `f(x₋) = f(x*)` → signe correct (car `σ(h − εe_j) = σ(h)` quand le neurone est inactif). Si `f(x₊) = f(x*)` → inverser.
*Condition technique* : il faut pouvoir calculer des pré-images, donc `Â^(i) F^(i−1)_x` doit être de rang `d_i`. En pratique on exige `d_{i−1} > d_i` car `F^(i−1)_x` n'est presque jamais de rang plein (à cause des ReLU inactives).

**(b) SOE — System Of Equations** (EUROCRYPT 2024)
Récupère les signes de **toute la couche simultanément** en résolvant `c · y_k = z_k`. Le j-ème neurone est inactif ssi `c_j = 0`. Nécessite des couches « suffisamment contractives » consécutives.

**(c) Neuron wiggle** (EUROCRYPT 2024)
Méthode générale, applicable en principe à n'importe quelle couche. On maximise la variation du neurone cible et on observe si le ReLU l'a bloquée.

**Les versions hard-label** (Sections 4.2 et 4.3 de ton article de départ) sont les contributions (b) et (c) de Canales-Martínez & Santos. Voir §5.3 ci-dessous.

**Le point de performance à retenir :** SOE avec le vecteur normal `m` déjà connu → **0,22 à 0,50 s par couche**, sans aucune requête supplémentaire. C'est le meilleur résultat de l'article et il vient d'une observation presque triviale (`⟨m, Δ_k⟩ = 0`). Leçon de méthode : les meilleures contributions viennent souvent de la réutilisation d'une quantité déjà calculée.

---

<a name="4-glossaire-cryptanalyse"></a>
## 4. Glossaire — Cryptanalyse et sécurité

### 4.1 L'analogie DNN ↔ chiffrement par blocs

C'est le cadre conceptuel de tout le domaine, formulé par Carlini et al. en 2020. Sache le réciter :

| Chiffrement par blocs | Réseau de neurones |
|---|---|
| Tours successifs | Couches successives |
| Opération linéaire sur clé secrète | `A^(i)x + b^(i)` avec poids secrets |
| S-box (non-linéarité publique) | ReLU (non-linéarité publique) |
| Clé de tour | Paramètres du neurone |
| Attaque différentielle à clairs choisis | Requêtes `x`, `x+Δ` et analyse des différences |
| Récupérer la clé | Récupérer θ |

**Ce que ça implique méthodologiquement :** les outils de la cryptanalyse s'appliquent — attaques différentielles, techniques « meet-in-the-middle », épluchage couche par couche (analogue au *peeling* d'un chiffrement itératif). C'est ce qui rend ton sujet légitime dans une conférence de crypto (EUROCRYPT, CRYPTO, ASIACRYPT) plutôt qu'en ML pur.

### 4.2 Modèle d'oracle — les cinq scénarios (Jagielski et al., USENIX 2020)

Du plus facile au plus dur pour l'attaquant :

1. **Raw output / logits** — le vecteur de réels brut. Le plus facile.
2. **Probabilités complètes** — après softmax.
3. **Top-k probabilités** — les k meilleures avec leurs scores.
4. **Top-1 + score** — la classe et sa confiance.
5. **Hard-label** — **uniquement l'étiquette** de la classe la plus probable. Le plus dur, et le plus réaliste.

**Toujours annoncer ton modèle d'oracle en premier** dans un article. C'est la première question que pose un reviewer.

### 4.3 Fidélité vs. Précision (fidelity vs. accuracy)

- **Extraction en précision (accuracy)** : le modèle volé doit être *bon* (bonne accuracy sur la tâche). On se fiche qu'il ressemble à l'original.
- **Extraction en fidélité (fidelity)** : le modèle volé doit **reproduire l'original**, y compris ses erreurs.
- **Extraction fonctionnellement équivalente** : `f_θ̂(x) = f_θ(x)` pour tout `x`. Le plus fort.

**Ton domaine est le troisième.** Ne confonds jamais avec les attaques de « model stealing » du monde ML (Knockoff Nets, etc.) qui visent le premier. Cette distinction est le premier paragraphe de ton état de l'art.

### 4.4 MLaaS

*Machine Learning as a Service* : le modèle est déployé derrière une API. C'est le modèle de menace réaliste : accès illimité en requête, aucun accès aux poids. C'est le contexte qui justifie tout le domaine.

### 4.5 Complexité en requêtes vs. complexité en temps

Deux mesures **indépendantes** — c'est une source de confusion permanente chez les débutants :

| Article | Requêtes | Temps |
|---|---|---|
| CRYPTO 2020 | polynomial | **exponentiel** (signes) |
| EUROCRYPT 2024 | polynomial | polynomial |
| ASIACRYPT 2024 (hard-label) | polynomial | **exponentiel** |
| EUROCRYPT 2025 (hard-label) | polynomial | polynomial |
| **Ito et al. 2025** | — | **conteste le « polynomial » quand la profondeur croît** |

En pratique une requête à une API coûte de l'argent et du temps réseau ; le calcul local coûte du CPU. Les deux comptent, et les défenses ciblent surtout les requêtes (rate limiting, proof-of-work).

### 4.6 Neurone difficile (difficult neuron)

Neurone dont le signe est **systématiquement** mal récupéré, quelle que soit la répétition. À distinguer d'une **erreur de précision** flottante, qui produit des erreurs **différentes** à chaque exécution. Ton article de départ fait explicitement cette distinction : les signes erronés de SOE changeaient d'une exécution à l'autre, donc précision et non neurone difficile — d'où la stratégie de **10 exécutions + vote majoritaire**.

### 4.7 Neurone quasi-toujours actif (Ito et al. 2025)

Un neurone dont l'état ne bascule presque jamais sur la distribution des entrées. **Il bloque l'attaque hard-label** : impossible d'observer son basculement, donc impossible de récupérer ses paramètres, et l'ignorer introduit une erreur non négligeable. La probabilité d'observer un basculement décroît **exponentiellement avec la profondeur**.
→ C'est à la fois la faille théorique de l'état de l'art et une **piste de défense** (créer volontairement de tels neurones).

---

<a name="5-mecanismes"></a>
## 5. Les mécanismes d'attaque, expliqués pas à pas

### 5.1 L'attaque raw-output (CRYPTO 2020 + EUROCRYPT 2024)

```
Pour i = 1 à r+1 :
    1. Trouver des points critiques pour les neurones de la couche i
       (chercher les discontinuités de ∂²f/∂x² le long de segments aléatoires)
    2. Récupérer la signature de chaque neurone
       (dérivées secondes partielles → système linéaire)
    3. Récupérer les biais
    4. Récupérer les signes (freeze / SOE / neuron wiggle)
    5. "Peler" la couche i : elle rejoint F_i
```

**Pourquoi ça marche :** `f` est affine par morceaux, donc `∂²f/∂x²` est nulle **presque partout sauf aux points critiques**. Ces points sont détectables par simple observation de la sortie brute. L'argument du **collectionneur de coupons** garantit qu'en échantillonnant assez de paires `(x₁, x₂)`, on trouve plusieurs points critiques pour chaque neurone.

### 5.2 L'attaque hard-label (EUROCRYPT 2025)

```
Pour i = 1 à r :
    1. Trouver un point de transition x₁ (recherche binaire)
    2. Marcher le long de la frontière de décision dans une direction δ
    3. Détecter le pli → projeter → obtenir x₂
    4. Estimer m_off et m_on
    5. Intersection → sous-espace de points duaux D (dimension d₀−2)
    6. Répéter ailleurs → D'
    7. D + D' → hyperplan critique complet → SIGNATURE
    8. Signe : mesurer Δ_on et Δ_off, en général Δ_on < Δ_off
    9. Peler
# Il manque la couche r+1 → c'est ton article de départ
```

**L'intuition de la récupération de signe (étape 8), à bien comprendre :** du côté *actif* du ReLU, le neurone transmet sa variation aux couches suivantes ; du côté *inactif*, il est bloqué à zéro et ne transmet rien. Donc du côté actif, les valeurs des neurones suivants changent **plus vite**, donc on rencontre plus tôt le prochain hyperplan critique, donc **`Δ_on < Δ_off`**. C'est un argument statistique, pas déterministe : d'où la nécessité de tester un grand nombre de points duaux avant de trancher.

### 5.3 Ce que fait ton article de départ (résumé opérationnel)

**Couche de sortie, cas multi-sorties :** en un point de transition entre classes i et j,
```
A_i y + b_i − A_j y − b_j = 0
```
Étendu à tous les paramètres, chaque ligne de la matrice `C` du système s'écrit
```
(y, 1) ⊗ Ĩ_{i,j}
```
où `Ĩ_{i,j}` est la matrice bloc de taille `(d_r+1) × d_{r+1}(d_r+1)` dont seuls les blocs i et j sont non nuls (`+Ĩ` et `−Ĩ`).
On collecte `d_{r+1}(d_r+1) − (d_r+2)` équations indépendantes, on fixe `d_r+2` variables, on résout.

**Freeze hard-label :** pour déterminer l'état d'un neurone sans voir la sortie brute — prendre `x'`, puis tester `x'+Δ` et `x'−Δ`.
- Labels **différents** → `x'` est sur la frontière → la sortie de couche 1 n'a pas changé → le neurone est **inactif**.
- Labels **identiques** → `x'` a quitté la frontière → le neurone est **actif**.
*Condition sur `Δ`* : assez petit pour ne pas traverser la frontière. Valeurs de l'article : `ε = 10⁻⁵`, `‖Δ‖ = 10⁻⁹`.
Complexités : temps `d_i(t_T + t₁)`, requêtes `d_i(q_T + 2)`.

**SOE hard-label :** choisir `Δ_k` tel que `x` et `x+Δ_k` soient **tous deux** des points de transition entre les **mêmes** classes α et β. Les termes `f_β(x) − f_α(x)` s'annulent des deux côtés → système **homogène** `c · y_k = 0`. Puis, en réutilisant la normale `m` déjà calculée : `c = m (A^(i)F^(i−1)_x)⁻` (pseudo-inverse) est une solution **sans aucune requête**.
Complexités : temps `(d_i+1)t_T + t₁`, requêtes `(d_i+1)q_T` (version sans `m`) ; **0 requête** avec `m`.

---

<a name="6-competences"></a>
## 6. Compétences à maîtriser — checklist

Coche au fur et à mesure. **Ordre indicatif : les blocs 1 à 4 sont des prérequis pour lire les articles, le bloc 5 pour les écrire.**

### Bloc 1 — Algèbre linéaire (prérequis absolu)
- [ ] Espaces vectoriels, sous-espaces, base, dimension
- [ ] Rang, noyau, image, théorème du rang
- [ ] Systèmes linéaires : existence, unicité, sous-détermination
- [ ] **Pseudo-inverse de Moore-Penrose** — utilisée telle quelle dans SOE
- [ ] **SVD** (décomposition en valeurs singulières) — utilisée pour les tests de rang dans toutes les implémentations
- [ ] **Conditionnement** d'une matrice — c'est ce qui gouverne la propagation d'erreurs, et donc le succès de l'attaque. **C'est le point le plus sous-estimé par les débutants.**
- [ ] Hyperplans, vecteurs normaux, projections orthogonales
- [ ] Matrices de Toeplitz / BTTB (pour la partie CNN)

### Bloc 2 — Géométrie et analyse
- [ ] Convexité, polyèdres, arrangements d'hyperplans
- [ ] Fonctions affines par morceaux
- [ ] Dérivées directionnelles, gradient, hessienne
- [ ] Différentiabilité, points de non-différentiabilité
- [ ] **Courbure d'une hypersurface, seconde forme fondamentale** ⭐ *(nécessaire pour ton idée n°2 — c'est ce qui te distinguera)*
- [ ] Notions de base de géométrie différentielle : variété, espace tangent

### Bloc 3 — Analyse numérique ⭐ *(bloc critique, ne le saute pas)*
- [ ] Norme IEEE 754, précision machine (`2⁻⁵³ ≈ 1.1×10⁻¹⁶` en double)
- [ ] Propagation d'erreurs, annulation catastrophique
- [ ] **Pourquoi une distance résiduelle de 10⁻¹³ est proche de la limite du double**, et ce que ça implique
- [ ] Arithmétique multi-précision (`mpmath`, `gmpy2`)
- [ ] Moindres carrés, régularisation, résolution robuste
- [ ] Recherche binaire en précision finie

### Bloc 4 — Deep Learning
- [ ] Perceptron multicouche, forward pass
- [ ] Rétropropagation (comprendre, pas nécessairement réimplémenter)
- [ ] SGD, momentum, learning rate, batch size
- [ ] Fonctions de perte (cross-entropy, sparse categorical cross-entropy)
- [ ] Toutes les activations de §2.2, **avec leurs dérivées première et seconde**
- [ ] CNN : convolution, stride, padding, pooling
- [ ] Datasets : MNIST, CIFAR-10 (formats, normalisation, dimensions)
- [ ] Notions sur les transformers (pour la perspective finale)

### Bloc 5 — Cryptanalyse et sécurité
- [ ] Cryptanalyse différentielle des chiffrements par blocs
- [ ] Structure SPN, notion de S-box, épluchage par tour
- [ ] Modèles d'adversaire, hypothèses de sécurité
- [ ] Attaques par canaux auxiliaires (culture générale — c'est un domaine voisin actif)
- [ ] Confidentialité différentielle (bases, pour le volet défense)
- [ ] Watermarking de modèles (bases, pour l'état de l'art des défenses)

### Bloc 6 — Programmation
- [ ] Python avancé : NumPy vectorisé, SciPy (`linalg`, `optimize`)
- [ ] PyTorch **ou** TensorFlow (choisis-en un et maîtrise-le)
- [ ] Profilage (`cProfile`, `line_profiler`) — indispensable quand une expérience prend 13 h
- [ ] Parallélisation (`multiprocessing`, `joblib`) — l'article note que ses résultats sont **non parallélisés**, il y a du gain immédiat à prendre
- [ ] Git, versionnement d'expériences
- [ ] LaTeX (classe `llncs` pour Springer) + BibTeX
- [ ] Reproductibilité : seeds, journalisation, environnements figés

### Bloc 7 — Métier de chercheur
- [ ] Lire un article de crypto efficacement (méthode §3 de la note stratégique)
- [ ] Tenir un **journal de recherche daté** — non négociable
- [ ] Rédiger en LaTeX au format conférence
- [ ] Présenter en 15–20 min
- [ ] Connaître le calendrier des conférences (voir §8.4)

---

<a name="7-ressources"></a>
## 7. Ressources d'apprentissage

### 7.1 Algèbre linéaire et analyse numérique
| Ressource | Lien | Commentaire |
|---|---|---|
| Gilbert Strang, *Linear Algebra* (MIT 18.06) | https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/ | La référence. Chapitres sur SVD et pseudo-inverse en priorité. |
| 3Blue1Brown, *Essence of Linear Algebra* | https://www.3blue1brown.com/topics/linear-algebra | Pour l'intuition géométrique. À voir **avant** Strang. |
| Trefethen & Bau, *Numerical Linear Algebra* | (livre) | Le chapitre sur le conditionnement est directement applicable à ton problème. |
| Goldberg, *What Every Computer Scientist Should Know About Floating-Point* | https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html | Lecture obligatoire vu ton ε = 10⁻¹³. |

### 7.2 Deep Learning
| Ressource | Lien |
|---|---|
| Goodfellow, Bengio, Courville — *Deep Learning* | https://www.deeplearningbook.org/ |
| 3Blue1Brown — série réseaux de neurones | https://www.3blue1brown.com/topics/neural-networks |
| CS231n (Stanford, CNN) | https://cs231n.github.io/ |
| Tutoriels PyTorch | https://pytorch.org/tutorials/ |
| Hanin & Rolnick, *Deep ReLU Networks Have Surprisingly Few Activation Patterns* (NeurIPS 2019) | https://arxiv.org/abs/1906.00904 |

> Le dernier est **très important pour toi** : il quantifie le nombre de régions linéaires réellement atteintes en pratique, ce qui est directement lié à la difficulté de trouver des points critiques — et donc au phénomène des neurones quasi-toujours actifs.

### 7.3 Cryptanalyse
| Ressource | Lien |
|---|---|
| Katz & Lindell, *Introduction to Modern Cryptography* | (livre) |
| Heys, *A Tutorial on Linear and Differential Cryptanalysis* | Cherche « Heys tutorial differential cryptanalysis » |
| Boneh & Shoup, *A Graduate Course in Applied Cryptography* | https://toc.cryptobook.us/ |
| IACR ePrint (veille quotidienne) | https://eprint.iacr.org/ |

### 7.4 Le domaine lui-même
| Ressource | Lien |
|---|---|
| Code EUROCRYPT 2025 (hard-label) | https://github.com/jchavezsaab/hard-label-dnn-extraction |
| Code CRYPTO 2020 (Carlini) | https://github.com/google-research/cryptanalytic-model-extraction |
| Beyond Slow Signs (implémentation optimisée) | https://arxiv.org/abs/2406.10011 |
| Blog Carlini (LLM stealing) | https://not-just-memorization.github.io/partial-model-stealing.html |
| Survey systématique attaques/défenses | https://arxiv.org/abs/2508.15031 |

### 7.5 Veille — à mettre en place cette semaine
- Alerte **arXiv** : catégories `cs.CR` + `cs.LG`, mots-clés « model extraction », « cryptanalytic extraction », « hard-label ».
- **IACR ePrint** : flux RSS, à parcourir chaque jour (10 minutes).
- **Google Scholar** : alertes sur *citations* des articles [7] et [8]. C'est le signal le plus fiable pour voir qui travaille sur ton sujet.
- Suivre : Nicholas Carlini, Adi Shamir, Isaac Canales-Martínez, Xiaoyang Dong, Yi Chen, Thomas Peyrin, Christina Boura, Yosuke Todo, Aydin Aysu.

---

<a name="8-outils"></a>
## 8. Boîte à outils pratique

### 8.1 Environnement recommandé
```bash
python -m venv venv-these && source venv-these/bin/activate
pip install numpy scipy matplotlib torch torchvision jupyter
pip install mpmath gmpy2          # arithmétique multi-précision
pip install tqdm joblib           # progression, parallélisation
pip install pandas seaborn        # analyse de résultats
```

### 8.2 Structure de dépôt suggérée
```
these-extraction/
├── README.md
├── notebooks/          # exploration interactive
├── src/
│   ├── networks/       # définitions et entraînement des réseaux cibles
│   ├── oracles/        # raw-output, top-k, hard-label
│   ├── attacks/
│   │   ├── critical_points.py
│   │   ├── transition_points.py
│   │   ├── signature.py
│   │   ├── sign_recovery.py     # freeze, SOE, wiggle
│   │   └── output_layer.py
│   ├── defenses/
│   └── utils/          # précision, métriques, journalisation
├── experiments/        # un dossier par expérience, daté, avec config figée
├── results/
├── papers/             # PDF annotés
└── journal/            # journal de recherche, un fichier par semaine
```

### 8.3 Métriques à journaliser systématiquement
Pour chaque expérience d'attaque :
- nombre de requêtes à l'oracle (**compteur dans l'oracle lui-même**, pas une estimation)
- temps mural, temps CPU, séparés par phase
- erreur max sur les paramètres `max |θ − θ̂|`
- accord de prédiction sur N entrées aléatoires (le protocole de l'article : 5 000 tirages uniformes dans (−1,1))
- nombre de signes corrects / total
- rang du système obtenu vs. rang théorique maximal
- seed, version du code (hash git), configuration matérielle

### 8.4 Conférences visées et calendrier indicatif

| Conférence | Rang | Période de soumission (indicatif) |
|---|---|---|
| **EUROCRYPT** | A* | octobre |
| **CRYPTO** | A* | février |
| **ASIACRYPT** | A* | mai |
| **CHES / TCHES** | A | soumissions par tours (4/an) |
| **LATINCRYPT** | B | (biennale) — **ton article de départ y est paru** |
| **AfricaCrypt** | B | **stratégique** : proximité géographique, réseau africain |
| **IACR ToSC** | journal | tours réguliers |
| **ACNS, ESORICS, USENIX Security, NDSS** | A/A* | variable |
| **ePrint IACR** | prépublication | **à tout moment — à utiliser pour dater tes résultats** |

**Conseil tactique fort :** dans un domaine où quatre équipes publient en parallèle, **poste sur ePrint dès qu'un résultat tient**, avant même la soumission. C'est gratuit, instantané, et cela établit l'antériorité. C'est la norme du domaine, pas une entorse.

---

<a name="9-pieges"></a>
## 9. Erreurs classiques et pièges

**1. Confondre extraction de fonctionnalité et extraction de paramètres.**
Les articles ML « model stealing » entraînent un modèle substitut. Toi, tu récupères les poids **exacts**. Ne cite pas les deux au même niveau sans le dire.

**2. Sous-estimer la précision numérique.**
Ton article de départ attribue explicitement ses erreurs de signe à la précision flottante 64 bits, avec une distance résiduelle estimée à 10⁻¹³. Prévois la multi-précision **dès le premier prototype**, pas en rattrapage.

**3. Croire que « polynomial » veut dire « pratique ».**
6 h 30 pour construire un système de rang 584, 13 h pour un système de rang 278. C'est polynomial. Ce n'est pas rapide. Et Ito et al. montrent que le polynôme lui-même est peut-être mal caractérisé en profondeur.

**4. Oublier de compter les requêtes.**
Instrumente ton oracle dès le début. Un reviewer demandera toujours le nombre exact de requêtes.

**5. Négliger l'architecture supposée connue.**
Presque toutes ces attaques supposent l'architecture (nombre de couches, tailles) **connue**. Dis-le explicitement. Sinon un reviewer te le reprochera. (À noter : TPUXtract et les attaques par canaux auxiliaires récupèrent précisément les hyperparamètres — la combinaison est un travail futur naturel.)

**6. Ne pas distinguer neurone difficile et erreur de précision.**
Test discriminant : rejoue l'expérience avec une autre seed. Si le **même** neurone échoue → neurone difficile. Si un **autre** échoue → précision. Ton article de départ fait exactement ce raisonnement.

**7. Comparer des temps d'exécution non comparables.**
33,41 s (freeze, non parallélisé) vs. 880 s (EUROCRYPT 2025, parallélisé sur 4 lots) — l'article annonce ×6,5, mais le calcul est délicat. Sois toujours explicite sur : parallélisation, matériel, ce qui est inclus.

**8. Attendre la perfection avant de publier.**
Domaine à quatre équipes concurrentes. Un résultat partiel bien écrit et daté sur ePrint vaut mieux qu'un résultat complet publié trois mois trop tard.

**9. Ne pas tenir de journal.**
Dans six mois tu ne te souviendras plus pourquoi tu avais choisi `ε = 10⁻⁵`. Une entrée datée par jour de travail, même de trois lignes.

---

## 10. Auto-évaluation : les questions auxquelles tu dois savoir répondre

À la fin de l'année 1, tu dois pouvoir répondre à chacune **sans notes**, au tableau. Utilise cette liste comme entraînement à la soutenance.

1. Pourquoi la technique de signature d'EUROCRYPT 2025 ne s'applique-t-elle pas à la couche de sortie ?
2. Pourquoi ne peut-on pas obtenir un système de rang plein pour la couche de sortie ? Combien de degrés de liberté, et d'où viennent-ils exactement ?
3. Quelle est la différence entre un point critique, un point de transition et un point dual ?
4. Pourquoi la frontière de décision se plie-t-elle en traversant un hyperplan critique ?
5. Pourquoi `Δ_on < Δ_off` en général, et pourquoi seulement « en général » ?
6. Pourquoi SOE avec le vecteur normal `m` ne coûte-t-il aucune requête ?
7. Que devient la notion de point critique pour GELU ? Pour tanh ?
8. Pourquoi le problème du signe disparaît-il pour les activations non homogènes ?
9. Quelle hypothèse d'EUROCRYPT 2025 Ito et al. remettent-ils en cause, et par quel mécanisme ?
10. Pourquoi une défense par bruit aléatoire indépendant à chaque requête échoue-t-elle contre un attaquant adaptatif ?
11. Quel est le goulot d'étranglement réel de l'attaque : l'algèbre ou la recherche de points de transition ? Chiffres à l'appui.
12. Comment distinguer expérimentalement un neurone difficile d'une erreur de précision flottante ?

---

*Cahier vivant — à compléter au fil de la thèse. Dernière mise à jour : août 2026.*
