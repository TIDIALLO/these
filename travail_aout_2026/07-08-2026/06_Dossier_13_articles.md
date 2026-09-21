# Dossier d'articles — extraction cryptanalytique de réseaux de neurones

**Tidiane DIALLO** — EPT Thiès / EDPT — dossier établi le 6 août 2026
Ordre de lecture : CRYPTO 2020 → EUROCRYPT 2024 → ASIACRYPT 2024 → EUROCRYPT 2025 → LATINCRYPT 2025 → vague 2026

---

## Avertissement à lire avant tout le reste

Tu m'as demandé de t'aider à **expliquer ces articles à ton directeur sans les avoir ouverts**.

Je ne vais pas t'aider à faire ça, et voici pourquoi — ce n'est pas de la morale, c'est du calcul de risque.

Un directeur de thèse ne teste pas ta capacité à résumer un abstract. Il pose une question de
second niveau : *« et comment est-ce qu'ils gèrent les neurones qui ne s'activent presque jamais ? »*.
À ce moment-là, soit tu as lu, soit la conversation s'arrête — et ce qui se casse n'est pas
la réunion, c'est la confiance, pour trois ans. Tu as toi-même écrit dans tes instructions de projet :
*« ne jamais affirmer un détail technique d'article sans l'avoir vu dans la source »*. C'est la bonne règle.

**Ce que je peux faire, et qui est plus utile :** te donner une compréhension réelle du
mécanisme de chaque article — assez pour en parler avec justesse — **plus** les formules exactes
pour dire où s'arrête ta connaissance sans que cela te décrédibilise. Un doctorant qui dit
« j'ai lu l'abstract et la section 3, je n'ai pas encore vérifié la preuve du lemme 4 » passe
pour rigoureux. Un doctorant pris en flagrant délit de bluff passe pour autre chose.

La section finale, **« Parler juste de ce qu'on n'a pas lu »**, te donne les phrases à employer.

### Convention de niveau de source

Chaque fiche indique ce que **j'ai** réellement vu :

- **[PDF]** — texte intégral lu. Les rubriques *algorithme* et *preuve* sont fiables.
- **[ABS+]** — résumé, page ePrint et extraits substantiels vus. Le mécanisme est fiable ; les
  détails de preuve sont **reconstruits par analogie** et signalés comme tels.
- **[ABS]** — résumé seul. *Algorithme* et *preuve* sont des **reconstructions plausibles**, non vérifiées.
- **[CIT]** — connu par ce qu'en disent d'autres articles.

**Une rubrique marquée « reconstruction » n'est pas une source. Ne la cite jamais.**

---

# Étape 1 — Le fondateur

## 1. Carlini, Jagielski, Mironov — *Cryptanalytic Extraction of Neural Network Models*

**CRYPTO 2020** · LNCS 12172, p. 189–218
https://arxiv.org/abs/2003.04884 · code : https://github.com/google-research/cryptanalytic-model-extraction
**Niveau de source : [ABS]** — abstract et page arXiv vérifiés le 6 août 2026

**Résumé.** Les auteurs soutiennent que l'extraction de modèle est un problème de cryptanalyse
déguisé et doit être étudié comme tel. Avec un accès oracle au réseau, ils construisent une
attaque différentielle qui récupère les paramètres à la précision du flottant près. L'attaque
repose sur le fait que les réseaux ReLU sont des fonctions affines par morceaux : les requêtes
effectuées **aux points critiques** révèlent de l'information sur les paramètres. Résultats
annoncés : modèles 2²⁰ fois plus précis et 100 fois moins de requêtes que l'état de l'art
antérieur ; un réseau MNIST de 100 000 paramètres extrait en 2²¹·⁵ requêtes en moins d'une heure,
avec une erreur pire cas de 2⁻²⁵.

**Problème.** Étant donné un oracle qui renvoie les sorties brutes d'un réseau ReLU, peut-on
récupérer *exactement* ses poids et biais, avec un nombre raisonnable de requêtes ?

**Hypothèses.** Oracle **raw-output** (les logits, en précision flottante). Activation ReLU.
Architecture entièrement connectée et **connue** de l'attaquant. Requêtes en entrée arbitraire
et non bruitées.

**Idée principale.** L'analogie cryptographique, à retenir mot pour mot :

| Réseau de neurones | Chiffrement par blocs |
|---|---|
| couche linéaire `A x + b` | addition de clé de tour |
| ReLU | S-box |
| profondeur `r` | nombre de tours |
| poids et biais | clé secrète |

Et donc : cryptanalyse différentielle. On prend des différences pour éliminer tout ce qui est
commun aux deux branches et isoler la cible.

**Algorithme.** *(mécanisme reproduit et vérifié numériquement dans le notebook, § 1 — la
correspondance ligne à ligne avec l'article n'a pas été vérifiée)*

1. Tirer une droite aléatoire `x₀ + t·u` et évaluer `f` le long de cette droite.
   `f` y est affine par morceaux : les **cassures de pente** sont les points critiques.
2. Raffiner chaque cassure par recherche dichotomique sur la pente.
3. En un point critique `x*` du neurone `j`, comparer le gradient de part et d'autre :
   `Δᵢ = ∂f/∂xᵢ(x*+εu) − ∂f/∂xᵢ(x*−εu)`. Tous les autres neurones sont dans le même état
   des deux côtés : **leurs contributions s'annulent**. Il reste `Δ ∝ A_j`.
4. Normaliser : `signature = Δ/Δ₁ = A_j/A_j,1`.
5. **Récupération de signe** : décider entre `+A_j` et `−A_j`. Par recherche exhaustive.
6. « Peeling » : une fois la couche 1 connue, on la retire et on recommence sur la couche 2.

**Preuve.** L'argument de correction est géométrique, pas algébrique. Dans un voisinage linéaire,
`f` est exactement affine, donc son gradient est constant. Franchir un hyperplan critique fait
basculer **un seul** neurone, donc modifie le gradient d'une quantité proportionnelle au vecteur
de poids de ce neurone, pondérée par les coefficients de la chaîne aval. La différence de gradients
élimine cette chaîne comme facteur commun et laisse la direction de `A_j`.
*Structure vérifiée numériquement dans le notebook ; l'énoncé formel de l'article n'a pas été lu.*

**Limites.**
1. **L'étape de signe est exponentielle** en le nombre de neurones. C'est *le* point faible,
   et c'est exactement ce que corrige EUROCRYPT 2024. Le résumé de cet article le dit explicitement :
   CRYPTO 2020 demande « a polynomial number of queries but an exponential amount of time ».
2. Oracle raw-output : irréaliste. Une API réelle renvoie une classe.
3. Suppose l'architecture connue.
4. Se dégrade avec la profondeur : les hyperplans des couches profondes sont *pliés*.
5. Sensible à la précision : deux hyperplans proches font échouer la différenciation.

> **Attention à un raccourci répandu.** Le résumé de LATINCRYPT 2025 présente CRYPTO 2020 et
> EUROCRYPT 2024 comme tous deux polynomiaux en temps *et* en requêtes. C'est un raccourci de
> rédaction. La formulation d'EUROCRYPT 2024 est la bonne. **Savoir relever cette nuance est
> une excellente réponse à une question de jury** — elle prouve que tu as lu les deux.

---

# Étape 2 — La polynomialité

## 2. Canales-Martínez, Chávez-Saab, Hambitzer, Rodríguez-Henríquez, Satpute, Shamir — *Polynomial Time Cryptanalytic Extraction of Neural Network Models*

**EUROCRYPT 2024** · LNCS 14653, p. 3–33
https://arxiv.org/abs/2310.08708 · https://eprint.iacr.org/2023/1526
**Niveau de source : [ABS]** — abstract vérifié le 6 août 2026

**Résumé.** Des milliards de dollars et d'innombrables heures GPU sont consacrés à l'entraînement
de DNN ; il est donc essentiel de déterminer la difficulté d'en extraire tous les paramètres à
partir d'une implémentation boîte noire. La meilleure attaque à ce jour sur les DNN à ReLU est
celle de CRYPTO 2020 ; elle ressemble à une attaque différentielle à clairs choisis sur un
cryptosystème dont la clé secrète est enfouie dans l'implémentation, et **requiert un nombre
polynomial de requêtes mais un temps exponentiel** en le nombre de neurones. Les auteurs
développent plusieurs techniques nouvelles qui permettent d'extraire, avec une précision
arbitrairement élevée, tous les paramètres réels d'un DNN à ReLU en **temps polynomial** et
avec un nombre polynomial de requêtes.

**Problème.** Rendre polynomiale l'étape de récupération de signe.

**Hypothèses.** Identiques à CRYPTO 2020 : oracle raw-output, ReLU, architecture connue.

**Idée principale — le *neuron wiggle*.** Le signe d'un neurone est décidé non plus par essai
exhaustif mais par une mesure locale : on « secoue » l'entrée dans une direction choisie pour
maximiser l'influence du neurone cible, et on observe la réponse. L'asymétrie de la réponse
révèle l'orientation.

**La racine du problème, à savoir expliquer.** ReLU est **positivement homogène** :
`ReLU(λx) = λ·ReLU(x)` pour `λ > 0`. On peut donc multiplier les poids d'un neurone par `λ > 0`
et diviser d'autant la colonne correspondante de la couche suivante : le réseau calcule
exactement la même fonction. La signature n'est donc définie qu'à un facteur **positif** près.
Reste `+A_j` ou `−A_j` — et ce choix-là, lui, **change** la fonction, puisqu'il fait basculer
l'état du neurone.

**Algorithme.** *(reconstruction — non vérifiée)* Signature comme en 2020, puis pour chaque neurone :
construire une direction de « wiggle » qui maximise sa contribution relative, mesurer la réponse
du réseau des deux côtés, en déduire le signe, propager. Complexité annoncée polynomiale.

**Preuve.** **Non lue.** Ce que l'on peut dire sans risque : la preuve doit établir que la mesure
de wiggle sépare les deux hypothèses de signe avec une marge qui ne se dégrade pas
exponentiellement avec le nombre de neurones. *Ne pas aller plus loin sans avoir ouvert le PDF.*

**Limites.**
1. Toujours en raw-output.
2. **Point important et documenté par la littérature 2026 :** la méthode échoue sur les
   « neurones de faible confiance », ce qui a contraint des travaux ultérieurs (Foerster et al.)
   à réintroduire une recherche exhaustive — la promesse de polynomialité n'est donc pas
   entièrement réalisée en pratique. **[CIT]** — vu dans l'introduction d'ePrint 2026/1025,
   à vérifier avant citation.
3. **Pour ta thèse :** les activations non positivement homogènes (GELU, SiLU, tanh, ELU)
   n'ont pas cette symétrie. **Le problème du signe y disparaît.** C'est l'un des rares points
   où quitter ReLU *simplifie* la tâche de l'attaquant. Vérifié numériquement au § 2 du notebook.

---

# Étape 3 — Le passage au hard-label

## 3. Chen, Dong, Guo, Shen, Wang, Wang — *Hard-Label Cryptanalytic Extraction of Neural Network Models*

**ASIACRYPT 2024** · LNCS 15491, p. 207–236
https://arxiv.org/abs/2409.11646 · code : https://github.com/AI-Lab-Y/NN_cryptanalytic_extraction
**Niveau de source : [ABS+]** — abstract et extraits d'introduction vus

**Résumé.** Le problème de l'extraction des paramètres est posé depuis près de trente ans, et
l'**extraction fonctionnellement équivalente** en est l'objectif central. Quand l'adversaire
accède aux sorties brutes, plusieurs attaques (CRYPTO 2020, EUROCRYPT 2024) l'atteignent. Ce
n'est pas le cas dans le cadre **hard-label**, où la sortie brute est inaccessible. Les auteurs
proposent la première attaque qui atteint théoriquement cet objectif en hard-label, pour les
réseaux ReLU, validée sur MNIST et CIFAR-10. Pour un réseau de 10⁵ paramètres, l'attaque demande
quelques heures sur un seul cœur.

**Problème.** Le hard-label était considéré comme **une défense** contre l'extraction
fonctionnellement équivalente. Est-ce vrai ?

**Hypothèses.** Oracle hard-label : seule la classe prédite est renvoyée. ReLU. Architecture connue.

**Idée principale.** Deux notions nouvelles : le **motif d'activation du modèle** et la
**signature du modèle**. Un réseau ReLU se décompose en un nombre fini de transformations affines,
chacune associée à un motif d'activation, et chacune fuit une partie de l'information sur les
paramètres. Résultat marquant : pour un réseau à `n` neurones, **`n + 1` motifs d'activation
bien choisis suffisent** à révéler toute l'information.

Les auteurs élargissent aussi la définition de l'équivalence fonctionnelle : le modèle extrait
`f_θ̂` satisfait `f_θ̂(x) = c · f_θ(x)` pour une constante `c > 0` fixée — un facteur d'échelle
global, sans effet sur l'argmax.

**Algorithme.** Cadre en trois étapes : **localisation de points duaux** → **extraction de
signature** → **extraction de signe**. *(structure rapportée par la littérature 2026 ; les détails
n'ont pas été vérifiés dans le PDF)*

**Preuve.** **Non lue.** Le cœur porte sur le décompte `n+1` : il faut montrer que ces motifs
d'activation particuliers engendrent un système de rang suffisant. *Ne rien affirmer de plus.*

**Limites.**
1. **Temps exponentiel** en pratique — c'est le point que corrige EUROCRYPT 2025.
2. ReLU uniquement, entièrement connecté.
3. Architecture supposée connue.

> **Pour ton article sur les défenses, c'est le meilleur premier paragraphe disponible :**
> « le hard-label a longtemps été considéré comme une défense ; Chen et al. ont montré que non.
> L'intuition *moins d'information en sortie = plus de sécurité* est fausse, il faut raisonner
> géométriquement. »

---

## 4. Carlini, Chávez-Saab, Hambitzer, Rodríguez-Henríquez, Shamir — *Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting*

**EUROCRYPT 2025** · LNCS 15601, p. 364–396
https://arxiv.org/abs/2410.05750 · code : https://github.com/jchavezsaab/hard-label-dnn-extraction
**Niveau de source : [ABS+]** — abstract vérifié ; mécanisme décrit dans l'article de départ, lu

**Résumé.** Les travaux de CRYPTO 2020 et EUROCRYPT 2024 ont rapproché l'extraction de paramètres
de l'extraction de clé d'un chiffrement par blocs par attaque à clairs choisis, mais reposaient sur
la valeur numérique exacte des logits, qui permet d'en calculer les dérivées. Chen et al.
(ASIACRYPT 2024) ont abordé le cadre hard-label, plus réaliste. Cet article présente une attaque
d'extraction en hard-label **polynomiale en temps et en requêtes**, applicable à des DNN à grand
nombre de paramètres.

**Problème.** L'obstacle est structurel :

| | Point **critique** | Point de **transition** |
|---|---|---|
| Condition | un neurone vaut exactement 0 | deux logits sont égaux |
| Visible en raw-output | oui | oui |
| Visible en **hard-label** | **non** | **oui — le seul événement visible** |
| Révèle | les poids du neurone | une relation entre logits |

L'événement observable n'est pas l'événement informatif.

**Hypothèses.** Oracle hard-label. ReLU. Architecture connue. Requêtes en précision arbitraire —
hypothèse forte, on y revient dans les limites.

**Idée principale.** La géométrie de la frontière de décision. La frontière **se plie** en
traversant un hyperplan critique : la cassure est observable *en marchant le long de la frontière*,
uniquement avec des labels. Les **points duaux** — à la fois critiques et de transition — sont
la clé : ils lient l'événement visible à l'information cachée.

**Algorithme.** *(mécanisme reproduit au § 3 du notebook ; description tirée de la section 2 de
l'article de départ, [PDF])*

1. Trouver un point de la frontière par **dichotomie** entre deux points de classes différentes.
2. **Marcher** le long de la frontière : pas tangentiel, puis reprojection par dichotomie.
3. Détecter les plis. Chaque famille de points duaux vit dans un sous-espace `D` qui donne
   de l'information sur le neurone. En répétant, on obtient un second sous-espace `D′` pour le
   même neurone, et leur intersection contraint la signature.
4. Regroupement des points duaux par neurone — via des décompositions en valeurs singulières.
5. Récupération de signe, puis peeling couche par couche.

**Preuve.** **Non lue intégralement.** Ce qui est sûr, parce que rapporté dans l'article de départ
que j'ai lu : la garantie est polynomiale en temps **et** en requêtes, et l'attaque récupère
toutes les couches **sauf la couche de sortie**, « due to its lack of ReLUs ».

**Limites — les auteurs les listent eux-mêmes comme travaux futurs :**
1. pas d'implémentation de bout en bout ;
2. le passage des points de transition à la signature reste inefficace ;
3. **les activations non linéaires par morceaux ne sont pas traitées** — *c'est littéralement ton sujet* ;
4. les autres architectures (transformeurs) ne sont pas traitées.

Deux limites supplémentaires venues de la suite :
5. la **couche de sortie** n'est pas récupérée → LATINCRYPT 2025 ;
6. les hypothèses de polynomialité se dégradent en profondeur → Ito, Miura, Todo (fiche 12).

---

# Étape 4 — L'article de départ

## 5. Canales-Martínez & Santos — *Extracting Some Layers of Deep Neural Networks in the Hard-Label Setting*

**LATINCRYPT 2025** · LNCS 16129 · doi 10.1007/978-3-032-06754-8_15
https://eprint.iacr.org/2025/1118
**Niveau de source : [PDF]** — texte intégral lu (23 pages, présent dans le projet)

**Résumé.** L'extraction de paramètres de DNN a connu ses avancées majeures récemment. CRYPTO 2020
et EUROCRYPT 2024 l'ont traitée en raw-output ; EUROCRYPT 2025 en hard-label, mais en récupérant
toutes les couches **sauf la couche de sortie** — les techniques employées n'y sont pas applicables
faute de ReLU. Cet article comble ce trou. Il présente en outre des méthodes plus efficaces quand
le réseau possède des **couches contractives**, c'est-à-dire quand le nombre de neurones y décroît.
Les méthodes sont appliquées à des réseaux entraînés sur CIFAR-10. Asymptotiquement, elles sont
polynomiales en temps et en requêtes : une attaque complète combinant Carlini et al. et ces
techniques reste donc polynomiale.

**Problème.** La couche de sortie n'a pas d'activation avant le softmax. Pas de point critique
à y chercher, donc pas de signature à extraire. Comment la récupérer ?

**Hypothèses.**
- (H1) en un point de transition entre les classes `i` et `j` : `A_i y + b_i = A_j y + b_j` ;
- (H2) `y = f₁..ᵣ(x)` est calculable, c'est-à-dire que les couches précédentes sont déjà extraites
  — **hypothèse d'oracle partiel, assumée** ;
- (H3) le softmax est invariant par translation et préserve l'ordre sous facteur positif.

**Idée principale.** Renoncer à la signature et passer à un **système linéaire global**.
Un point de transition ne dit rien sur un neurone, mais il dit que deux logits sont égaux —
et c'est une équation linéaire exacte sur les coefficients de la couche de sortie.

**Algorithme.**

*Cas à une seule sortie (section 3.2).* En un point de transition, `Ay + b = 0`. Avec `d_r` points
de transition dont les `y_k` sont linéairement indépendants, le système `A y_k = −b` a une solution
unique. Comme `b` est inconnu, on résout le système équivalent `b⁻¹A y_k = −1`, ce qui donne
`Â = b⁻¹A` et `b̂ = 1`. Si `b > 0` le réseau récupéré est équivalent ; si `b < 0` il classe à
l'envers, et **il suffit de comparer une sortie et d'inverser les signes de `Â` et `b̂`**.

*Cas multi-sorties (section 3.3).* Chaque point de transition entre `i` et `j` donne
`(A_i − A_j) y + (b_i − b_j) = 0`, soit une ligne du système où **toutes** les inconnues de la
couche sont des variables. On empile, on fixe `d_r + 2` variables, on résout par moindres carrés.

**Preuve — celle-ci, je l'ai lue, et c'est celle que tu dois pouvoir refaire au tableau.**

Le point démontré est le décompte des **`d_r + 2` degrés de liberté**. Il repose sur deux
propriétés du softmax `s`, établies explicitement dans la section 3.1 :

1. **Invariance par translation.** Pour `c̄` de coordonnées toutes égales à `c` :
   `s(u + c̄)ᵢ = e^(uᵢ+c) / Σⱼ e^(uⱼ+c) = e^uᵢ e^c / (e^c Σⱼ e^uⱼ) = s(u)ᵢ`.
2. **Préservation de l'ordre.** Le softmax n'est pas invariant par changement d'échelle, mais
   pour `c > 0` les coordonnées les plus grandes le restent : `z(c(Ay+b)) = z(Ay+b)`,
   où `z` désigne l'argmax.

De là, trois familles de transformations laissent le réseau équivalent :
- ajouter un vecteur constant `c̄ⱼ` à n'importe laquelle des `d_r` colonnes de `A` — car
  `(Ãy)ᵢ = Σₖ Aᵢ,ₖyₖ + yⱼc`, et `yⱼc̄` translate la sortie ;
- ajouter un vecteur constant `c̄_b` au biais `b` ;
- multiplier `A` et `b` par un `c > 0`.

Et l'article montre que **toute combinaison** de ces transformations laisse aussi le réseau
équivalent. Total : `d_r` (une par colonne) `+ 1` (le biais) `+ 1` (l'échelle) = **`d_r + 2`**.

D'où le rang maximal du système : `d_{r+1}(d_r + 1) − (d_r + 2)`.

Ce n'est pas une preuve difficile. C'est un décompte de symétries — et c'est exactement pour
cela qu'il faut savoir le refaire : c'est le genre de chose qu'un jury demande au tableau.

**Les chiffres à connaître par cœur.**

| Élément | Valeur |
|---|---|
| Réseau CIFAR-10 | 3072 − 256×3 − 64 − 10 |
| Rang maximal | 10(64+1) − 66 = **584** |
| Construction du système | **≈ 6 h 30**, non parallélisé |
| Résolution du système | fraction de seconde |
| Validation | 5 000 entrées, **100 %** de correspondance |
| Freeze hard-label, couche 1 (256 neurones) | **33,41 s** |
| SOE sans `m` | 358,27 s, quelques signes erronés → 10 répétitions et vote majoritaire |
| SOE avec `m` | **< 1 s** |
| Second réseau | 3072 − 1000 − 350 − 100 − 30 − 10, rang max 278, **≈ 13 h** |
| Distance résiduelle à la frontière | **≈ 10⁻¹³** |

**Le fait le plus important de tout l'article.** Presque tout le temps d'exécution sert à
**trouver des points de transition**, pas à faire de l'algèbre. Ce fait a deux conséquences
opposées : côté attaque, c'est le levier d'optimisation évident, trivialement parallélisable ;
côté **défense**, c'est la cible naturelle — rendre la localisation des transitions plus coûteuse
ou plus bruitée attaque le goulot d'étranglement réel.

**Limites.**
1. **Suppose les couches cachées connues.** Ce n'est pas une attaque de bout en bout, c'est un maillon.
2. Les méthodes freeze et SOE en hard-label ne sont applicables qu'à la **première couche**.
3. Les gains dépendent d'une architecture **contractive**, ce que tous les réseaux n'ont pas.
4. Précision de 10⁻¹³ requise — extrêmement fragile.
5. **Trouvé en reproduisant (WP2) :** l'ambiguïté de signe global est traitée pour le cas
   à une sortie, **pas pour le cas multi-sorties**, où elle est tout aussi nécessaire.
   Symptôme : accord 0/5000 au lieu de ~25 % du hasard. Correctif : une requête.

---

# Étape 5 — La vague 2026

## 6. *Cryptanalytic Extraction of Deep Neural Networks with Non-Linear Activations*

**ePrint 2026/253** (NTU Singapour) · https://eprint.iacr.org/2026/253
**Niveau de source : [ABS]** — résumé et notice de séminaire NTU vérifiés le 5 août 2026

**Résumé.** Les attaques existantes visent essentiellement ReLU ; ce travail étend le vol de
modèle à une large classe d'activations non linéaires : GELU, SiLU, SELU, sigmoïde et d'autres.
Les auteurs présentent la **première attaque boîte noire universelle** capable de récupérer poids
**et** biais pour des réseaux dont les activations convergent vers un comportement linéaire en
dehors de zones non linéaires étroites. La méthode généralise les approches géométriques
antérieures en exploitant les **dérivées d'ordre supérieur** et l'**analyse des zones linéaires
adjacentes**, ce qui permet de se passer de la non-différentiabilité. Pour plusieurs activations,
la signature serait **plus facile** à récupérer que dans le cas ReLU ; la fonction d'activation
elle-même peut être identifiée lorsqu'elle n'est pas publique.

**Problème.** La non-différentiabilité était le pivot de toutes les attaques. Que faire sans elle ?

**Hypothèses.** Oracle **raw-output**. Activation quasi-linéaire hors d'une bande étroite.

**Idée principale.** GELU(5) ≈ 5 et GELU(−5) ≈ 0 : loin de l'origine, GELU **est** ReLU.
Le pli net devient une zone de courbure concentrée de largeur finie, et les dérivées d'ordre
supérieur y prennent le relais de la discontinuité de dérivée première.

**Algorithme et preuve.** **Non lus.** Ne rien affirmer au-delà du résumé.

**Limites.**
1. **Raw-output uniquement.** C'est le point décisif pour toi : ton créneau reste ouvert.
2. Ne couvre pas les activations sans zone linéaire adjacente.
3. **Résultat en tension avec le tien :** ce qui est facile en raw-output (GELU, SiLU)
   semble être le pire cas en hard-label. Cette **inversion** est ton observation la plus
   attrayante — et donc la plus dangereuse. Tant que l'expérience E1 n'est pas faite,
   ne la présente jamais comme un fait.

---

## 7. *Cryptanalytic Extraction of Neural Networks with Various Activation Functions*

**ePrint 2026/178** · https://eprint.iacr.org/2026/178
**Niveau de source : [ABS+]** — résumé et un passage technique du PDF vus

**Résumé.** Au-delà des méthodes centrées sur ReLU, les auteurs traitent la diversité pratique des
activations par une **classification unifiée** et un cadre de récupération de paramètres valable
dans plusieurs scénarios. Outre ReLU et PReLU, ils étudient Leaky ReLU, HardTanh, ELU et la
fonction Step — première étude à le faire, et en particulier pour PReLU dans le cadre S1.
Ils examinent aussi les implications de sécurité du choix de l'activation et explorent les
**activations composites et mixtes comme moyen de renforcement**.

**Problème.** Généraliser au-delà de ReLU dans la famille linéaire par morceaux.

**Hypothèses.** Raw-output, scénarios S1 et S5 de la taxonomie des oracles.

**Idée principale — l'astuce vue dans le PDF, et elle est bonne.** Dans le cadre S5, la différence
de dérivée première ne suffit plus : sur la branche négative, cette différence est non nulle et
non constante, donc les contributions des autres neurones ne s'annulent pas. Les auteurs
construisent alors une entrée telle que **seul le neurone cible** soit sur la branche négative,
tous les autres — même couche et couches suivantes — étant sur la branche positive. Sous ce motif
d'activation particulier, la couche est **localement linéaire** et la différence de dérivée isole
à nouveau le neurone cible.

**Algorithme.** Une procédure par activation, chacune reposant sur ce principe de fabrication
d'un motif d'activation favorable. *Détails non vérifiés au-delà du passage cité.*

**Preuve.** **Non lue.**

**Limites.**
1. Raw-output.
2. Reste dans le monde linéaire par morceaux — les activations lisses ne sont pas la cible.
3. **Pour ta thèse :** c'est le **voisin le plus proche de ton créneau défense**. Ils explorent
   déjà les activations composites comme protection. À lire intégralement **avant** de rédiger
   quoi que ce soit sur les défenses, sous peine de réinventer leur section.

---

## 8. Ma, Chen, Zhang, Yu, Wang — *Breaking Slope and Structure Restrictions: Broadening Hard-Label Cryptanalytic Extraction of PReLU Neural Networks*

**ePrint 2026/1066** (Tsinghua) · https://eprint.iacr.org/2026/1066
**Niveau de source : [ABS+]** — résumé et extraits d'introduction vus

**Résumé.** Deux apports. D'abord un nouvel **isomorphisme de réseau**, appelé *flip-and-scaling*,
qui lève la restriction sur la pente et permet de construire un nouveau cadre d'extraction.
Ensuite, l'identification de **contraintes linéaires sur les états internes** des réseaux PReLU
expansifs, avec le décompte exact de ces contraintes.

**Problème.** PReLU a une pente négative **apprise**, donc inconnue de l'attaquant. Les cadres
existants supposaient cette pente connue ou fixée.

**Hypothèses.** Oracle **hard-label**. PReLU (linéaire par morceaux).

**Idée principale.** Les isomorphismes de réseau — les transformations qui laissent la fonction
inchangée — déterminent ce qui est récupérable. En élargissant le groupe d'isomorphismes considéré
(flip-and-scaling), on peut traiter une pente inconnue.

**Algorithme et preuve.** **Non lus.**

**Limites et importance stratégique.**
1. PReLU reste **linéaire par morceaux** : les points critiques restent bien définis, les
   frontières restent des hyperplans pliés. **Ton créneau n'est pas fermé.**
2. **Mais** c'est la publication la plus récente qui repousse la frontière *hard-label × au-delà
   de ReLU*, et elle vient de Tsinghua — l'une des quatre équipes concurrentes. **Ton créneau se
   referme par le côté linéaire-par-morceaux.** Argument de plus pour publier sur ePrint dès
   qu'un résultat partiel tient, sans attendre une conférence.

---

## 9. *Algebraic Cryptanalytic Extraction on Hard-Label Neural Networks*

**ePrint 2026/1164** · https://eprint.iacr.org/2026/1164
**Niveau de source : [ABS+]** — résumé et section « travaux liés » lus

**Résumé.** L'attaque hard-label d'EUROCRYPT 2025 est polynomiale en théorie, mais son
regroupement de points duaux repose sur une SVD de complexité `O(n²·(d^(k))³)`, d'où un temps
d'exécution considérable en pratique. Les auteurs transposent l'attaque géométrique dans un
**cadre algébrique** et proposent la méthode du **vecteur de signature approché (ASV)**, qui
remplace la vérification de rang par SVD par de simples **produits scalaires**, ramenant la
complexité du regroupement à `O(n·(d^(k))³)` en moyenne.

**Problème.** L'écart entre polynomialité théorique et faisabilité pratique.

**Hypothèses.** Hard-label, ReLU. Et deux hypothèses structurelles explicites :
- en grande dimension, des vecteurs aléatoires sont **presque orthogonaux** ;
- les neurones des réseaux réels tendent à apprendre des **caractéristiques démêlées**.

**Idée principale.** Si deux signatures approchées sont presque orthogonales dès qu'elles
correspondent à des neurones différents, alors un simple produit scalaire suffit à les regrouper —
plus besoin de tester le rang.

**Algorithme et preuve.** **Non lus.** Noter que la seconde hypothèse est *empirique* : elle
porte sur les réseaux entraînés réels, pas sur tous les réseaux. C'est une bonne question à poser.

**Limites.**
1. Repose sur des hypothèses empiriques sur les réseaux entraînés.
2. ReLU.
3. **Utilité pour toi :** sa section « travaux liés » est la **meilleure carte du domaine à jour**
   que j'aie trouvée. C'est là que j'ai relevé l'existence d'un travail de **Liu et al. à
   EUROCRYPT 2026** étendant l'extraction aux couches profondes — **[CIT]**, non vérifié séparément.

---

## 10. Li, Gong, Li, Zhuang, Tang, Lv, Yan — *End-to-End Polynomial-Time Cryptanalytic Extraction of Convolutional Neural Networks in the Hard-Label Setting*

**ePrint 2026/902** · https://eprint.iacr.org/2026/902
**Niveau de source : [ABS]**

**Résumé.** Attaque d'extraction hard-label **de bout en bout** pour des classifieurs CNN à ReLU
et **average pooling**, d'architecture connue. Contribution algorithmique principale : une
récupération **au niveau du canal**, guidée par une optimisation discrète sur candidats retenus (SVGR).

**Problème.** Une couche convolutive partage ses poids entre positions : la récupération
neurone par neurone n'a plus de sens.

**Hypothèses.** Hard-label, ReLU, architecture connue, **average pooling**.

**Idée principale.** Passer du neurone au **canal** comme unité de récupération.

**Algorithme et preuve.** **Non lus.**

**Limites.**
1. Average pooling seulement. D'après la section « travaux liés » d'ePrint 2026/1164, les attaques
   sur CNN à **max pooling** exigent le raw-output — **[CIT]**, à vérifier. **Si cela se confirme,
   il reste là une ouverture : CNN à max pooling en hard-label.** À garder en réserve.
2. Architecture connue.

---

## 11. Duan & Tang — *Geometric Critical Point Screening: Clustering-Free Cryptanalytic Extraction of Neural Network Models*

**ePrint 2026/1025** · https://eprint.iacr.org/2026/1025
**Niveau de source : [ABS]**

**Résumé.** Une méthode qui supprime entièrement l'étape de regroupement. Sur un réseau
`784 − 8⁽⁸⁾ − 1`, le temps d'extraction des signatures des neurones des première et deuxième
couches cachées tombe à **1,7 %** et **3,7 %** de celui des méthodes existantes, et le coût en
requêtes à **3,2 %** et **0,1 %** de l'état de l'art. La méthode n'est pas limitée à ReLU et
s'étend aux autres activations linéaires par morceaux.

**Problème.** Le regroupement (clustering) des points critiques est le goulot d'étranglement.

**Hypothèses.** Raw-output ; activations linéaires par morceaux.

**Idée principale.** Un **criblage géométrique** des points critiques qui les attribue directement
au bon neurone, sans passer par une phase de regroupement.

**Algorithme et preuve.** **Non lus.**

**Limites.**
1. Raw-output.
2. Linéaire par morceaux.
3. Chiffres annoncés sur une architecture particulière — la généralité reste à établir.
4. **Pour toi :** redéfinit la référence de performance à citer, et ajoute une pierre du côté
   « piecewise ».

---

## 12. Ito, Miura, Todo — *Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial?*

**arXiv 2510.06692** · https://arxiv.org/abs/2510.06692
**Niveau de source : [ABS]**

**Résumé.** Une remise en cause des hypothèses de polynomialité d'EUROCRYPT 2025. Certaines
hypothèses deviennent impraticables pour les réseaux profonds — en particulier autour des neurones
**quasi toujours actifs** — et les auteurs proposent une stratégie d'extraction **inter-couches**
pour y remédier.

**Problème.** « Polynomial » ne veut pas dire « faisable ».

**Hypothèses.** Hard-label, ReLU.

**Idée principale.** Un neurone qui reste actif sur presque tout l'espace d'entrée n'a
pratiquement jamais de point critique atteignable. La probabilité de tomber sur son hyperplan
décroît avec la profondeur, et le facteur constant explose.

**Algorithme et preuve.** **Non lus.**

**Limites.** C'est un article de critique et de correction ; sa portée constructive est plus
étroite que sa portée critique. **Pour toi :** c'est le fondement du **créneau C**, ton créneau
de repli, et la source de la notion de neurone quasi toujours actif utilisée dans ton cahier.

---

## 13. *Train to Defend: First Defense Against Cryptanalytic Neural Network Parameter Extraction Attacks*

**arXiv 2509.16546** · https://arxiv.org/abs/2509.16546
**Niveau de source : [ABS+]** — résumé et introduction lus

**Résumé.** Première défense contre les attaques d'extraction cryptanalytique de paramètres.
L'idée : **éliminer l'unicité des neurones**, dont ces attaques ont besoin pour réussir, par un
entraînement conscient de l'extraction. Concrètement, un terme de régularisation ajouté à la
fonction de perte minimise la distance entre les poids des neurones d'une même couche. La défense
a donc un **coût nul en surface et en latence à l'inférence**. Variation d'accuracy inférieure à
1 % après réentraînement à architecture identique. Les réseaux non protégés sont extraits en
**14 minutes à 4 heures** ; la défense tient sur des durées d'extraction prolongées. Les auteurs
fournissent aussi un cadre théorique pour quantifier la probabilité de succès de l'attaque.

**Problème.** Toutes les attaques supposent implicitement que les neurones sont **distinguables**.
Et si on les rendait indistinguables ?

**Hypothèses.** Défense côté **entraînement** ; attaques visées en **raw-output**.
Le défenseur contrôle l'entraînement — hypothèse forte : elle exclut la protection d'un modèle
déjà entraîné.

**Idée principale.** Régulariser vers la redondance plutôt que vers la performance pure.

**Algorithme.** `L_total = L_tâche + λ · Σ_couches Σ_{j,k} d(A_j, A_k)`, minimisant la distance
entre poids de neurones d'une même couche. *(forme générale d'après le résumé ; la formulation
exacte du terme n'a pas été vérifiée)*

**Preuve.** Cadre théorique de quantification de la probabilité de succès annoncé. **Non lu.**

**Limites — et c'est là que ta thèse s'insère :**

| Axe | Train to Defend | Ce qui reste ouvert |
|---|---|---|
| Oracle visé | raw-output | **hard-label** |
| Point d'action | entraînement (les poids) | **déploiement** (l'oracle) |
| Mécanisme visé | l'unicité des neurones | **la localisation des transitions à 10⁻¹³** |
| Levier | régularisation | **choix de l'activation** |
| Modèles couverts | à réentraîner | **déjà entraînés** |

Ton résultat sur l'activation-comme-défense est **complémentaire, pas concurrent**.
C'est un argument à faire valoir dès l'introduction de ton futur article.

---

# La case vide

| # | Réf. | Oracle | Activation | Temps poly. |
|---|---|---|---|---|
| 1 | CRYPTO 2020 | raw | ReLU | **non** (signe exponentiel) |
| 2 | EUROCRYPT 2024 | raw | ReLU | oui |
| 3 | ASIACRYPT 2024 | **hard** | ReLU | **non** |
| 4 | EUROCRYPT 2025 | **hard** | ReLU | oui |
| 5 | **LATINCRYPT 2025** | **hard** | quelconque* | oui |
| 6 | 2026/253 | raw | **lisses** | oui |
| 7 | 2026/178 | raw | PL variées | oui |
| 8 | 2026/1066 | **hard** | **PReLU** | oui |
| 9 | 2026/1164 | **hard** | ReLU | oui |
| 10 | 2026/902 | **hard** | ReLU (CNN) | oui |
| 11 | 2026/1025 | raw | PL | oui |
| 12 | 2510.06692 | hard | ReLU | **conteste** |
| 13 | 2509.16546 | raw | — | *défense* |
| **?** | **—** | **hard** | **lisses** | **← la case vide** |

\* couche de sortie uniquement ; l'indépendance vis-à-vis de l'activation est **ton** apport (WP2).

---

# Parler juste de ce qu'on n'a pas lu

## Le principe

Un chercheur expérimenté dit sans arrêt « je ne l'ai pas lu en détail ». Ce n'est pas un aveu de
faiblesse, c'est un **marqueur de métier** : cela signale qu'il distingue ce qu'il sait de ce
qu'il suppose. Un doctorant qui ne le dit jamais paraît, au choix, naïf ou peu fiable.

## Les quatre formules à avoir en bouche

1. **« J'ai lu le résumé et la section des travaux liés ; je n'ai pas encore ouvert la preuve. »**
   Précise et neutre. La plupart du temps, elle clôt la question.

2. **« Ce que je sais de cet article vient de la façon dont [autre article] le cite — je dois le vérifier à la source. »**
   Indispensable pour tout ce qui est marqué [CIT] dans ce dossier.

3. **« Sur ce point je fais une hypothèse par analogie avec [article lu], je n'ai pas vérifié qu'ils procèdent ainsi. »**
   La formule qui te permet de **quand même raisonner à voix haute** sans t'engager.

4. **« Je ne sais pas. Je le regarde et je te réponds cette semaine. »**
   La plus forte des quatre. Elle ne coûte rien si tu réponds vraiment.

## Ce que tu dois savoir sans notes — la vraie liste

Ce sont les questions qu'on te posera. Aucune ne demande d'avoir lu les treize PDF.

| Question | Où est la réponse |
|---|---|
| Pourquoi ReLU rend-il l'extraction possible ? | Fiche 1, « idée principale » |
| Différence point critique / point de transition ? | Fiche 4, le tableau |
| Pourquoi la couche de sortie résiste-t-elle ? | Fiche 5, « problème » |
| D'où viennent les `d_r + 2` degrés de liberté ? | **Fiche 5, la preuve — à savoir refaire au tableau** |
| Pourquoi le signe est-il un problème, et pourquoi disparaît-il hors de ReLU ? | Fiche 2 |
| Quel est le goulot d'étranglement réel de l'attaque ? | Fiche 5, « le fait le plus important » |
| Qu'est-ce qui reste ouvert, et pourquoi ? | La case vide |
| Qu'est-ce qui a bougé en 2026 ? | Fiches 6 à 11 |

**Si tu maîtrises ces huit lignes, tu tiendras une heure de discussion.** C'est très différent
de « connaître treize articles », et bien plus solide.

## Le piège à éviter absolument

Ne jamais dire, d'un article non lu, comment il **démontre** quelque chose. Le *quoi* peut venir
d'un résumé ; le *comment* ne s'invente pas, et c'est précisément ce qu'un directeur teste.
Dans ce dossier, chaque fois que tu lis « **Non lu** » sous *Preuve*, ce n'est pas un manque —
c'est la limite exacte de ce que tu as le droit d'affirmer.

---

*Dossier établi le 6 août 2026. Une fiche n'est close que lorsque son marqueur passe à [PDF].*
