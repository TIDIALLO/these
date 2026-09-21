# Documentation — `Reproduction_attaques.ipynb`

**Tidiane DIALLO** — EPT Thiès / EDPT — 6 août 2026

---

## 1. À quoi sert ce notebook, et à quoi il ne sert pas

**Il sert à** comprendre le *mécanisme* de chaque attaque en le voyant tourner, sur des réseaux
assez petits pour que chaque étape soit inspectable. C'est un instrument pédagogique, et c'est
aussi le support d'une démonstration de dix minutes devant ton directeur.

**Il ne sert pas à** produire des chiffres publiables. Les réseaux ont quelques dizaines de
neurones, les dimensions d'entrée sont de 2 à 40, il n'y a pas de réplication statistique sérieuse.
Les implémentations de référence sont ailleurs :

| Article | Code officiel |
|---|---|
| CRYPTO 2020 | https://github.com/google-research/cryptanalytic-model-extraction |
| ASIACRYPT 2024 | https://github.com/AI-Lab-Y/NN_cryptanalytic_extraction |
| EUROCRYPT 2025 | https://github.com/jchavezsaab/hard-label-dnn-extraction |

**Ce notebook ne remplace pas la reproduction du code officiel d'EUROCRYPT 2025.** Celle-là reste
au programme du semestre 2, et il faut lui budgéter deux mois pleins.

---

## 2. Lancer le notebook

### Sur Google Colab

1. https://colab.research.google.com → *Importer un notebook* → déposer `Reproduction_attaques.ipynb`
2. *Exécution → Tout exécuter*
3. Aucun GPU. Le runtime CPU gratuit suffit.

### En local

```bash
pip install numpy scipy matplotlib jupyter
jupyter notebook Reproduction_attaques.ipynb
```

### Temps d'exécution indicatif — mesuré sur un cœur

| Cellule | Durée |
|---|---|
| § 0 socle | instantané |
| § 1 CRYPTO 2020 | ~1 min |
| § 2 signe | instantané |
| § 3 marche sur la frontière | ~4 min |
| § 4 couche de sortie, 4 activations | ~3 min |
| § 5 mesure de ρ, 6 activations | ~10 min |
| § 6 test E1, 3 valeurs de h | ~15 min |

**Total : 30 à 35 minutes.** Pour une démonstration en réunion, exécuter les § 0 à 4 seulement,
ou réduire `N_TRACES` et `n_traces` à 3.

### Aucune dépendance exotique — et c'est délibéré

NumPy, SciPy, Matplotlib. Rien d'autre. Pas de PyTorch, pas de JAX. Raison : la reproductibilité
à cinq ans. Un notebook qui dépend d'une version précise d'un framework est illisible dans deux ans ;
celui-ci tournera encore.

---

## 3. Ce que fait chaque section

### § 0 — Le socle

- Six activations avec leurs dérivées analytiques : `relu`, `lrelu`, `gelu`, `silu`, `softplus`, `tanh`.
- Une classe `MLP` : propagation avant, entraînement SGD avec momentum, rétropropagation manuelle.
  Pas de framework, donc chaque ligne est vérifiable.
- Deux oracles, `RawOracle` et `HardLabelOracle`, qui **comptent les requêtes**.
- Une méthode `preact` marquée « vérité terrain » : elle ne sert **qu'à l'évaluation des métriques**,
  jamais à l'attaque.

> **Le comptage des requêtes n'est pas du confort.** C'est la métrique qui rend une attaque
> comparable à la littérature. Une attaque qui ne compte pas ses requêtes n'est pas évaluable.
> C'est aussi la première chose à ajouter au code de recherche existant du projet, qui ne la fait pas.

### § 1 — CRYPTO 2020, la signature de la couche 1

Trouve les points critiques le long de droites aléatoires par détection de cassure de pente, puis
récupère la signature par différence de gradient de part et d'autre.

**Sortie attendue :** 5 neurones sur 5 récupérés, erreur ~10⁻¹¹.

**Trois choses à observer, et elles sont plus instructives que le résultat lui-même :**

1. **Une seule droite ne suffit pas.** Elle ne croise pas tous les hyperplans. D'où la boucle sur
   14 droites — et d'où l'explosion du coût en requêtes.
2. **Certains points critiques sont rejetés.** Le contrôle de stabilité compare deux estimations
   obtenues avec deux `ε` différents. Quand deux hyperplans sont proches, la fenêtre les enjambe
   tous les deux et les estimations divergent. C'est le **neurone difficile** de la littérature,
   observé en direct.
3. **Ce contrôle n'utilise aucune vérité terrain.** C'est un test que le vrai attaquant peut faire.
   La comparaison avec `net1.A[0]` sert uniquement à *nous* dire si le contrôle a bien fonctionné.

**Paramètre sensible :** `eps` dans `signature_at`. Trop grand → enjambement d'hyperplan.
Trop petit → le bruit d'arrondi domine. C'est le compromis central de toute la littérature.

### § 2 — Le problème du signe

Vérifie numériquement que `ReLU(λx) = λ·ReLU(x)` à l'erreur machine près pour `λ > 0`,
et que **ce n'est pas le cas de GELU** — l'écart est de l'ordre de 10⁻¹ et croît avec `λ`.

C'est la démonstration en trois lignes de l'argument à retenir : **le problème du signe est une
conséquence de l'homogénéité positive de ReLU, et il disparaît pour les activations qui ne
l'ont pas.**

### § 3 — EUROCRYPT 2025, la marche sur la frontière

Entraîne un réseau `2−8−8−2` sur *two moons*, puis, **uniquement avec des labels** :

1. `start_on_boundary` — dichotomie entre deux points de classes différentes ;
2. `initial_tangent` — balayage angulaire pour trouver la direction de la frontière ;
3. `trace_boundary` — pas tangentiel `h`, puis reprojection par dichotomie transversale ;
4. `curvature` — dérivée de l'angle tangent par rapport à l'abscisse curviligne.

**Sortie attendue :** `κ_max` de l'ordre de 100 à 200 pour ReLU et LeakyReLU, de 1 à 25 pour
les activations lisses. **Deux ordres de grandeur.**

**Paramètres sensibles :**

| Paramètre | Effet |
|---|---|
| `h = 0.004` | pas tangentiel. **Fixe la résolution** : un pic plus étroit que `h` est sous-estimé |
| `halfwidth = 0.08` | demi-largeur de la fenêtre de reprojection. Trop petit → la trace décroche |
| `iters = 60` | itérations de dichotomie. 60 donne ~2⁻⁶⁰, largement au-delà du besoin |
| `n_steps` | longueur de la trace |

**Les traces courtes sont écartées** (`len(pts) < 150`). C'est nécessaire : quand la reprojection
décroche, la trace s'arrête au bout de deux ou trois points et la courbure calculée n'a aucun sens.
Sans ce filtre, les résultats sont ininterprétables.

### § 4 — LATINCRYPT 2025, la couche de sortie

C'est la reproduction de l'article de départ, et du WP2.

Architecture `40 − 24 − 16 − 10 − 4`. Inconnues : `4 × 11 = 44`. Degrés de liberté : `d_r + 2 = 12`.
**Rang maximal : 32.**

Points de transition obtenus **uniquement par requêtes hard-label**. Système construit ligne par
ligne, `d_r + 2` variables fixées, résolution par moindres carrés.

**Sortie attendue :** rang 32/32, résidu ~10⁻¹⁴, accord 5000/5000 pour ReLU, GELU, SiLU et tanh.

**Le résultat secondaire, et c'est celui qui vaut d'être montré.** En fixant `Â₂,₁ = 1`, on choisit
implicitement `c = 1/A₂,₁`. Si le vrai `A₂,₁` est négatif, `c < 0` et le réseau reconstruit classe
à l'envers. **Symptôme : accord 0/5000 au lieu des ~25 % du hasard sur 4 classes.**

> **Règle de diagnostic à retenir pour toute la thèse :** un taux de réussite très *inférieur* au
> hasard n'est jamais un échec de la méthode. C'est une convention de signe ou d'orientation
> inversée. Le chercher là en premier.

Le correctif coûte **une requête**. L'article traite ce cas pour le réseau à une sortie
— « if the outputs are different, we flip the signs of Â and b̂ » — mais ne le mentionne pas en
multi-sorties, où il est tout aussi nécessaire.

**Paramètre sensible :** `need = 6 * RANK_MAX`. Avec seulement `3 × RANK_MAX` points, le système
peut être de rang déficient sur ReLU. Raison probable : la **parcimonie** de ReLU. Beaucoup de
coordonnées de `y` sont nulles, donc les lignes du système sont moins indépendantes.
**C'est une observation à creuser** — si elle se confirme, elle dit que la parcimonie induite par
ReLU est en soi un léger facteur de protection, ce qui est directement pertinent pour le volet défense.

### § 5 — La mesure de ρ

Calcule le désalignement entre la courbure de la frontière et les hyperplans critiques.

```
      distance moyenne au croisement le plus proche, PONDÉRÉE PAR LA COURBURE
ρ  =  ───────────────────────────────────────────────────────────────────────
      distance moyenne au croisement le plus proche, pondérée par la longueur
```

`ρ → 0` : la courbure est **sur** les hyperplans → attaque possible.
`ρ = 1` : elle les ignore → aucune information.

**Les croisements sont la vérité terrain, et ils servent uniquement à évaluer la métrique.**
L'attaque simulée, elle, ne voit que des labels. Cette phrase doit être dite dans toute présentation :
sans elle, un auditeur attentif soupçonnera à juste titre que l'évaluation utilise de l'information
interdite à l'attaquant.

### § 6 — E1, le test d'invalidation

Recalcule ρ pour `h ∈ {0,016 ; 0,004 ; 0,001}`, à longueur d'arc constante.

C'est un **test conçu pour pouvoir détruire la conclusion** du § 5. C'est exactement pour cela
qu'il passe en premier au semestre 2.

---

## 4. Résultats obtenus en exécutant ce notebook — et ce qu'ils changent

Exécution du 6 août 2026, réseaux `2−8−8−2`, 8 traces par activation.

| Activation | ρ médian | κ_max médian |
|---|---|---|
| ReLU | **0,012** | 100,2 |
| LeakyReLU | **0,008** | 103,3 |
| SiLU | 0,695 | 4,7 |
| GELU | 0,856 | 1,2 |
| Softplus | 0,925 | 1,3 |
| tanh | **1,224** | 7,9 |

Et le test E1 :

| Activation | h = 0,016 | h = 0,004 | h = 0,001 |
|---|---|---|---|
| ReLU | 0,043 | 0,023 | 0,007 |
| SiLU | 0,464 | 0,622 | 0,622 |
| GELU | 0,773 | 0,913 | 0,904 |
| tanh | 1,219 | 1,039 | 1,033 |

### Deux conclusions, et la seconde est importante

**1. L'écart linéaire-par-morceaux / lisse est robuste.** ReLU et LeakyReLU restent à
`ρ ≤ 0,05` à toutes les résolutions ; toutes les activations lisses restent au-dessus de 0,46.
Ce résultat-là tient. C'est celui que tu peux présenter.

**2. L'ordre *entre* activations lisses n'est pas robuste — et il contredit ta campagne du 3 août.**

| | Campagne du 3 août | Ce notebook, 6 août |
|---|---|---|
| Meilleur des lisses pour l'attaquant | **tanh** ρ = 0,379 | **SiLU** ρ = 0,695 |
| Pire des lisses pour l'attaquant | **GELU** ρ = 1,249 | **tanh** ρ = 1,224 |

Les deux expériences utilisent la même architecture et la même métrique, avec des graines
différentes. **tanh passe du meilleur au pire.**

C'est une **confirmation directe de la limite n° 2** que tu avais toi-même énoncée : *« les
différences entre activations lisses sont indicatives seulement »*. Elle est maintenant démontrée,
plus seulement supposée.

**Conséquence pratique, et elle est nette :** ne présente **pas** l'inversion GELU / raw-output
comme un résultat. Le classement des activations lisses dépend de l'instance du réseau, pas
seulement de l'activation. Avant toute affirmation à ce sujet, il faut agréger sur **plusieurs
graines de réseau**, et non sur plusieurs traces d'un même réseau — c'est un ajout au protocole
d'E1 qui n'était pas prévu.

**Ce que tu peux dire au directeur :** « l'écart entre activations non différentiables et toutes
les autres est robuste à la résolution et à la graine — deux ordres de grandeur. L'ordre à
l'intérieur des activations lisses, lui, change d'une graine à l'autre : je ne peux rien en
conclure pour l'instant, et j'ajoute la variation de graine au protocole d'E1. »

C'est un meilleur discours que le précédent. Un résultat qui résiste à une tentative de
destruction vaut plus qu'un résultat qui n'a jamais été attaqué.

---

## 5. Ce qu'il faut ajouter au code avant de l'utiliser sérieusement

- [ ] **Compteur de requêtes par phase**, pas seulement global — c'est ce qui permet de vérifier
      le déséquilibre annoncé par l'article de départ entre recherche de transitions et algèbre
- [ ] **Graines dans un fichier de configuration**, pas en dur
- [ ] **Hash git journalisé** avec chaque résultat
- [ ] **Agrégation sur plusieurs graines de réseau**, et pas seulement plusieurs traces — voir § 4
- [ ] **Arithmétique multi-précision** (`mpmath`, `gmpy2`) dès qu'on vise 10⁻¹³
- [ ] **Parallélisation de la recherche de points de transition** — c'est trivialement parallèle,
      et c'est là que part tout le temps

---

## 6. Les six exercices du notebook, et ce que chacun prépare

| Ex. | Sujet | Prépare |
|---|---|---|
| 1 | float32 vs float64 | Comprendre pourquoi la précision est le vrai verrou |
| 2 | Extension à la couche 2 | La notion d'hyperplan *plié* et le peeling |
| 3 | Compteur par phase | La reproduction des chiffres de l'article de départ |
| 4 | ρ en fonction de β pour softplus_β | **La figure centrale du premier article visé** |
| 5 | Variation de l'échelle des poids | L'expérience E3 et le remplacement de la taxonomie |
| 6 | Bruit sur la frontière | **Le premier résultat du volet défense** |

Les exercices 4 et 6 ne sont pas des exercices : ce sont les deux premières expériences de tes
futurs articles, déjà cadrées.

---

*Documentation établie le 6 août 2026. Les résultats du § 4 ont été obtenus en exécutant
le notebook, pas estimés.*
