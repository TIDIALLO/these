# Note de lecture : six travaux 2026 sur l'extraction de réseaux de neurones

Tidiane Diallo · état au 5 octobre 2026

---

## 0. Comment lire cette note

Six travaux de 2026 touchent le même sujet que la thèse. Cette note dit, pour chacun : ce qu'il fait, dans quel cadre, avec quels chiffres, et ce qu'il change pour mon travail.

**Source des informations.** Les textes ont été lus par extraction automatique du contenu (arXiv et ePrint IACR). Les descriptions de méthode et les chiffres sont donc fiables au niveau de la structure, mais **chaque citation et chaque chiffre doit être recoupé dans le PDF avant d'être utilisé dans un document formel**. Les points marqués **[à vérifier]** sont ceux où l'extraction ne suffit pas à trancher.

**Vocabulaire minimal.**
- *Oracle raw-output* : l'attaquant reçoit le vecteur des sorties réelles (logits).
- *Oracle hard-label* : l'attaquant reçoit seulement la classe prédite.
- *Activation lisse* : GELU, SiLU, SELU, sigmoïde, tanh. Pas de coude net.
- *Activation par morceaux* : ReLU, Leaky ReLU, PReLU, HardTanh, ELU, Step.

---

## 1. Vue d'ensemble

| # | Référence | Oracle | Activations | Architecture | Ce qu'il apporte | Lien avec la thèse |
|---|---|---|---|---|---|---|
| 1 | Hasan & Vassilev, arXiv 2608.28843 (août 2026, v2 septembre) | **raw-output** | GELU, SiLU (ReLU exclu) | FFN de transformers, 2 couches | Fuite de courbure (Hessienne projetée) qui révèle les directions cachées | Très proche de l'idée de courbure. Oracle différent. |
| 2 | Chen, Tang, Gao, Su, Qin, Dong, arXiv 2608.05736 (août, v2 septembre) | **hard-label** | ReLU seul | FCN et CNN à max-pooling (LeNet-5) | Clustering algébrique des points duaux (NRC, ASV), beaucoup plus rapide | Outil réutilisable pour mon étape de regroupement. Pas de lissé. |
| 3 | Asselineau, Derbez, Fouque, Minaud, ePrint 2026/253 (février) | **non précisé** [à vérifier] | GELU, SiLU, SELU, sigmoïde, ReLU | non précisée | Attaque boîte noire pour activations non linéaires, poids et biais | Concurrence directe sur le lissé. Oracle décisif à vérifier. |
| 4 | Qi, Lei, Wei, Sun, Wang, ePrint 2026/178 (ToSC 2026) | **raw-output et hard-label** | ReLU, PReLU, Leaky, HardTanh, ELU, Step | non précisée | Cadre par famille d'activation, dont PReLU en hard-label | **Concurrence directe sur mon contribution 1.** |
| 5 | Ito, Miura, Todo, ePrint 2026/2097 | **hard-label** | ReLU | MLP, largeur 16, 4 à 6 couches | Récupération du signe de bout en bout, sans requêtes dédiées | Même groupe que la critique « polynomial ? ». Verrou du signe. |
| 6 | Tang, Chen, Su, Gao, Qin, Dong, ePrint 2026/2061 (« Normal Alignment ») | **hard-label** | ReLU | MLP et CIFAR-10 (192-64×8-10), MNIST (64-96×3-32-10) | Signe en temps polynomial par alignement des normales | Verrou du signe. Même groupe que 2. |

---

## 2. Les deux travaux les plus proches de l'idée « courbure »

### 2.1 Hasan & Vassilev, arXiv 2608.28843 : « Curvature Cryptanalysis of Smooth Transformer Feed-Forward Networks »

**Ce que le papier cherche à faire.** Un bloc FFN de transformer est un petit réseau à deux couches avec une activation lisse. Les auteurs montrent que les dérivées secondes de la sortie, mesurées depuis l'entrée, laissent fuir la structure des poids. L'attaquant n'accède à aucun paramètre.

**Modèle de menace (Définition 1, section 3.1).** L'attaquant choisit les entrées et reçoit le vecteur de sortie réel, bruité ou arrondi selon un paramètre. Un vecteur retourné compte pour une requête. Ce n'est donc **pas** un oracle hard-label. C'est le point qui différencie ce travail de ma thèse.

**Pourquoi ReLU est exclu (Corollaire 2, section 4.6).** Hors des points de coude, la dérivée seconde d'un ReLU est nulle. La fuite de courbure disparaît presque partout. Le papier le dit explicitement : ReLU relève d'un autre mécanisme et sort du périmètre.

**La construction, étape par étape.**

1. *Hessienne projetée.* Pour une projection de sortie c, on pose s_c(x) = cᵀ g(x). Sa Hessienne en entrée vaut :

   ∇²ₓ s_c(x) = Σⱼ (cᵀ vⱼ) φ''(wⱼᵀ x + b₁,ⱼ) wⱼ wⱼᵀ

   où wⱼ sont les lignes de la première couche, vⱼ celles de la seconde, et φ l'activation. Les directions cachées wⱼ apparaissent comme facteurs de rang un, pondérés par φ''.

2. *Mesure par différences finies.* Pour chaque entrée de probe, on estime la Hessienne avec des différences centrées sur les coordonnées (lemme et équations 13 et 14). Le nombre de requêtes est **Q = 2d² + 1** par Hessienne. Pour d = 64, cela fait 8 193 requêtes.

3. *Réutilisation du stencil.* Tous les T Hessiennes d'un même point de probe partagent les mêmes 2d² + 1 appels. On forme ensuite les projections c par combinaison linéaire. C'est une optimisation de coût, pas de principe.

4. *Décomposition.* L'ensemble des Hessiennes forme un tenseur partiellement symétrique :

   H̃ₜ ≈ Σⱼ aₜ,ⱼ uⱼ uⱼᵀ

   Les uⱼ sont unitaires. La recherche des uⱼ est une optimisation conjointe sur les facteurs.

5. *Identifiabilité locale (Théorème 1, équation 10).* La solution est localement unique si la différentielle restreinte est injective. Un comptage de dimensions donne T ≥ m(d − 1) / [d(d + 1)/2 − m]. Pour d = 64 et m = 128, cela fait T ≥ 5.

**Résultats (Tableau 2, section 6.1).**

| Activation | Direction retrouvée (cosinus moyen) | Part des directions avec cosinus > 0,90 |
|---|---|---|
| GELU | 0,919 ± 0,014 | 86,7 % |
| SiLU | 0,940 ± 0,012 | 91,2 % |

Budget : 8 193 requêtes structurelles et 12 000 de complétion, soit 20 193 au total. Expériences sur des ViT entraînés sur CIFAR-10. La substitution fonctionnelle atteint plus de 92 % d'accord en top-1 (résumé). Le bruit de sortie et l'arrondi font baisser le rappel. Un ajustement adaptatif du pas de différence le restaure en partie (0,919 pour GELU, 0,908 pour SiLU).

**Limites reconnues.**
- Pas d'expérience sur un MLP autonome : tout est sur des FFN de transformers.
- Deux couches seulement dans le modèle d'étude.
- L'efficacité dépend de la lissité de l'activation.
- Le bruit de sortie est une défense partielle.
- Généralisation à d'autres configurations non explorée.

**Ce que ça change pour moi.**
- L'idée « la courbure révèle les poids » n'est pas nouvelle en raw-output. Ma contribution devrait donc porter sur **la même information accessible en hard-label**, ou sur un autre objet.
- Ce papier suggère une question que je n'ai pas encore posée : le rapport d'échelle β que je mesure est-il déjà une quantité qu'ils contrôlent via φ'' ? [à vérifier]
- Il fournit une référence de coût (8 193 requêtes par Hessienne) à comparer avec mes mesures de requêtes.

---

### 2.2 Chen, Tang, Gao, Su, Qin, Dong, arXiv 2608.05736 : « Algebraic Cryptanalytic Extraction on Hard-Label Neural Networks »

**Ce que le papier cherche à faire.** Accélérer l'attaque hard-label géométrique de Carlini et al. (EUROCRYPT 2025). Le verrou identifié est le regroupement des points duaux.

**Rappel du cadre.** Un point dual est l'intersection entre un hyperplan critique (un neurone change d'état) et la frontière de décision. Autour de ce point, deux facettes de frontière ont des normales m_ℓ et m_r. Ces normales portent l'information sur le neurone. Le problème : savoir quels points duaux appartiennent au même neurone. La méthode d'origine utilise une décomposition en valeurs singulières, de coût O(n² · d³).

**Modèle de menace.** Oracle **hard-label** explicite : « le modèle retourne l'étiquette de la classe la plus probable, correspondant au maximum de la sortie brute » (section 2.3). Hypothèses : architecture connue, paramètres secrets, requêtes sur tout le domaine, précision flottante suffisante.

**Activations.** **ReLU uniquement.** Toutes les σ^(k) sont des ReLU (section 2.3). Pas de lissé.

**Deux nouvelles méthodes de regroupement.**

1. *Normal Rank Check (NRC).* Au lieu de tester le rang de 2d vecteurs, on forme la matrice N de 4 normales unitaires issues de deux points duaux, N ∈ ℝ^(4 × d). Si rang(N) < 4, les deux points sont sur le même neurone. Si rang(N) = 4, non. Coût moyen O(n · d³).

2. *Approximate Signature Vector (ASV).* Pour un point dual, on pose v = n_ℓ − (n_ℓ · n_r) n_r. C'est une approximation de la direction du poids du neurone. Deux points appartiennent au même neurone si leurs ASV sont presque parallèles :

   DGap(v₁, v₂) = 1 − |v₁ · v₂| / (‖v₁‖ ‖v₂‖) < τ

   La proposition 1 borne l'écart moyen par 1/d pour des réseaux équilibrés à poids aléatoires unitaires. Le coût devient un produit scalaire O(d) au lieu d'une SVD.

**Résultats (Tableau 1, extrait).**

| Modèle | Couche | Méthode | Temps (h) | Requêtes (log₂) |
|---|---|---|---|---|
| 64-64×4-10 | F₁ | ASV | 0,05 | 24,3 |
| 64-64×4-10 | F₁ | Carlini et al. | 0,92 | 24,3 |
| 64-64×4-10 | F₂ | NRC | 4,22 | 30,8 |
| 64-64×4-10 | F₂ | Carlini (estimé) | ≈ 1 478 | 30,8 |
| 3072-256×3-64-10 | F₁ | ASV | 6,36 | 22,8 |
| 3072-256×3-64-10 | F₁ | Carlini (estimé) | ≈ 825 846 | 22,8 |
| LeNet-5 (max-pool) | C₁, C₂ | ASV par noyau | 12,65 | 31,0 |

Ce sont des **estimations** de Carlini pour les lignes marquées, pas des mesures. Le gain annoncé sur F₂ est d'environ 2¹⁵, soit environ 850 fois. Il s'agit de précision numérique de l'ordre 2⁻³⁹ à 2⁻⁴⁴ pour les petits réseaux.

**Limites reconnues (section 4.5).**
- L'ASV perd en discrimination quand d diminue.
- L'ASV devient inopérant quand le nombre de neurones actifs partagés est très faible (moins de 10).
- Deux points duaux qui partagent la même facette ont la même normale : cela produit des faux négatifs dans NRC.

**Nouveauté revendiquée.** Premier travail d'extraction hard-label sur un CNN à max-pooling.

**Ce que ça change pour moi.**
- **Mon TP8 a exactement ce problème à résoudre.** Quand je regroupe les points de pli autour d'un germe, j'utilise un ajustement robuste (rejet d'outliers) qui est un bricolage. NRC et ASV sont des critères de regroupement précis. Je dois les tester dans mon code.
- Le coût de regroupement que je n'ai pas mesuré est justement celui qu'ils optimisent.
- Ce travail est **entièrement ReLU**. Il ne menace pas directement le volet lissé, mais il menace le volet « par morceaux » sur le plan de l'efficacité.

---

## 3. Les quatre autres travaux

### 3.1 Asselineau, Derbez, Fouque, Minaud, ePrint 2026/253 : « Cryptanalytic Extraction of Deep Neural Networks with Non-Linear Activations »

- **Activations.** GELU, SiLU, SELU, sigmoïde et « d'autres ». ReLU est couvert au sens où le travail étend les techniques ReLU à une classe plus large.
- **Technique (résumé).** Dérivées d'ordre supérieur et analyse des zones linéaires adjacentes, sans supposer de non-dérivabilité. Récupération simultanée des poids et des biais.
- **Oracle.** **Non précisé dans l'extraction.** Le vocabulaire (dérivées d'ordre supérieur) suggère raw-output, comme ma fiche 11. **[à vérifier]** : c'est la question la plus importante de cette note pour la thèse. Si l'oracle est hard-label, ce travail est en concurrence frontale avec le volet lissé.
- **Chiffres.** Non extraits.
- **Verdict.** À lire en premier dans le PDF, uniquement pour trancher la question de l'oracle.

### 3.2 Qi, Lei, Wei, Sun, Wang, ePrint 2026/178 (ToSC 2026) : « Cryptanalytic Extraction of Neural Networks with Various Activation Functions »

- **Activations.** ReLU, PReLU, Leaky ReLU, HardTanh, ELU, Step. Cadre systématique par famille.
- **Oracle.** **Raw-output et hard-label.** L'extraction (citation du résumé) : « la première étude d'extraction pour ces fonctions d'activation et les réseaux à PReLU en setting hard-label ». **[à vérifier dans le PDF].**
- **Chiffres et architectures.** Non extraits.
- **Correction importante.** Ma fiche 10 disait que ce travail était raw-output seulement. C'était faux, ou du moins incomplet. Je l'avais déduit de ma lecture du résumé, sans le texte intégral.
- **Conséquence.** Ma contribution 1 (hard-label par morceaux non-ReLU) est **au moins en partie prise**. Il faut lire le texte pour savoir si leur hard-label couvre la détection des points de transition avec coude non nul, et l'estimation de la pente α, ou seulement une partie.

### 3.3 Ito, Miura, Todo, ePrint 2026/2097 : « End-to-End Hard-Label Cryptanalytic Model Extraction Using Efficient Sign Recovery »

- **Groupe.** NTT Social Informatics Laboratories, le même que celui de la critique « Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial? » (ma fiche 09).
- **Oracle.** Hard-label. Citation : « un adversaire ne peut observer que l'étiquette finale, par exemple 'chien' ou 'chat' ».
- **Activations.** ReLU. Architecture : MLP à plusieurs couches cachées.
- **Idée.** Récupération du signe par un principe « complètement différent » de Carlini et al., qui ne demande **aucune requête dédiée**. Combinée à la récupération des poids pour une extraction de bout en bout.
- **Chiffres.** Plus de 98 % d'accord sur les labels. Modèles MNIST et Fashion-MNIST, largeur 16, 4 à 6 couches cachées.
- **Limites.** Non détaillées dans le résumé.
- **Verdict.** Très important pour le verrou du signe, qui n'est pas mon centre actuel. Le fait qu'ils disent « sans requêtes dédiées » est un argument à comprendre précisément.

### 3.4 Tang, Chen, Su, Gao, Qin, Dong, ePrint 2026/2061 : « Normal Alignment: Improved Cryptanalytic Sign Recovery on Hard-Label Networks »

- **Groupe.** Le même que le travail 2608.05736 (Tsinghua et Shandong).
- **Oracle.** Hard-label (S1). Activations : ReLU.
- **Idée.** Le signe d'un neurone se lit dans l'alignement des normales projetées des facettes adjacentes. Les facettes voisines ont des normales de longueurs différentes, selon le signe. Cela produit un vote.
- **Résultat (résumé).** Le vote est plus précis que celui de Carlini et al., qui produisait de « fausses prédictions de signe à haute confiance » dans les couches profondes. Combiné à eSOE, le papier revendique une récupération complète du signe en temps polynomial. Il évite des énumérations de 2⁵² (CIFAR-10) et 2⁸² (MNIST).
- **Chiffres.** Architectures CIFAR-10 192-64×8-10 et MNIST 64-96×3-32-10. Pas de nombre de requêtes extrait.
- **Verdict.** Verrou du signe, en hard-label et ReLU. Pour ma thèse, c'est une étape que je dois savoir citer et comprendre, pas une voie où je dois entrer.

---

## 4. Ce que disent les six ensemble

**Ce qui est pris, en ReLU hard-label.** Les quatre travaux du groupe Tsinghua/Shandong et NTT couvrent le regroupement (NRC, ASV), le signe (alignement, sans énumération), et l'extraction de bout en bout. Ce terrain est très encombré. Je ne dois pas y placer ma contribution principale.

**Ce qui est en partie pris, en par morceaux non-ReLU hard-label.** Qi et al. revendiquent PReLU en hard-label. Ma contribution 1 doit être redéfinie par rapport à ce travail, après lecture complète.

**Ce qui n'est pas clairement pris, en lissé hard-label.** Dans les six, aucun ne traite explicitement l'extraction de poids d'un réseau lissé en hard-label. Les deux travaux lissés (2608.28843 et 2026/253) sont en raw-output, ou leur oracle n'est pas établi. **C'est la seule lacune nette**, et elle reste à confirmer dans le PDF de 2026/253.

**Corrections apportées à mes notes précédentes.**
1. Ma fiche 10 (Qi 2026) disait « raw-output seulement » : à corriger, le travail revendique aussi le hard-label.
2. Ma cartographie du 5 octobre présentait 2608.28843 comme « très proche de B2, risque majeur » sans préciser l'oracle. C'est vrai pour le sujet, faux pour le cadre : il est raw-output.
3. Ma cartographie plaçait 2608.05736 dans le volet lissé. C'est faux : il est entièrement ReLU.

---

## 5. Plan de lecture recommandé (dans l'ordre)

1. **ePrint 2026/253 (Asselineau et al.)**, lire le PDF en entier, uniquement pour fixer l'oracle. Cela tranche la concurrence sur le lissé.
2. **ePrint 2026/178 (Qi et al.)**, lire la section des expériences hard-label et la section PReLU. Cela définit ce qui reste à faire pour ma contribution 1.
3. **arXiv 2608.28843 (Hasan & Vassilev)**, lire les sections 4 et 5 (Hessienne projetée et mesure). C'est la base du raisonnement courbure.
4. **arXiv 2608.05736 (Chen et al.)**, lire les sections 3 et 4 (NRC et ASV) et les tests de coût. À reproduire sur mon TP8 pour le regroupement.
5. **ePrint 2026/2097 et 2026/2061**, lire la partie signe. À citer, pas à reproduire pour l'instant.

**Objectif de la lecture.** Pour chaque travail, répondre par écrit à trois questions : quel oracle, quelles activations, et quelle question laisse-t-il ouverte pour moi ?

---

## 6. Questions à poser au directeur

1. Faut-il redéfinir la contribution 1 après la lecture de Qi et al. (2026/178), ou la déplacer vers la mesure d'échelle β ?
2. Acceptons-nous que le volet signe reste hors périmètre, faute de temps, en ne faisant que le citer ?
3. Peut-on réutiliser NRC ou ASV dans notre cadre, en les citant et en les comparant à notre ajustement robuste ?
4. Le dépôt d'une prépublication sur la couche de sortie (résultat vérifié) doit-il être fait avant de lire la suite ?

---

*Note de travail. Les descriptions proviennent d'une lecture automatisée des textes et doivent être recoupées dans les PDF avant toute citation formelle.*
