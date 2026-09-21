# Note d'avancement de thèse — version vérifiée

*Extraction de modèles en mode hard-label — activations alternatives et défenses*

Tidiane Diallo — EPT Thiès / EDPT — doctorat 2025-2028
Document préparé pour la réunion avec le directeur de thèse — 20 septembre 2026
**Révision du 21 septembre 2026** : chaque affirmation ci-dessous a été rejouée sur le code réellement présent sur disque avant d'être conservée. Voir la section 0 pour la méthode, et l'annexe A pour la traçabilité complète (fichier, commande, date).

---

## 0. Méthode de vérification (à lire en premier)

Ce document reprend la note du 20 septembre, mais rien n'y est affirmé sans avoir été retrouvé dans le code ou réexécuté. Trois marqueurs sont utilisés :

- **[VÉRIFIÉ]** — rejoué le 21/09/2026 à partir du code existant, résultat reproduit à l'identique ou dans les mêmes ordres de grandeur.
- **[REQUALIFIÉ]** — l'affirmation originale était trop forte ; la version corrigée est donnée avec la preuve qui la corrige.
- **[NON RETROUVÉ]** — présent dans la note mais aucune trace dans le code ou les fichiers du projet ; à vérifier avant de l'affirmer au directeur.

Cette discipline suit celle que tu as toi-même posée dans `07-08-2026/06_Dossier_13_articles.md` : *« ne jamais affirmer un détail technique sans l'avoir vu dans la source »*. Elle s'applique ici aux résultats expérimentaux, pas seulement aux articles.

**Précision de périmètre.** Le travail réel du semestre 2 ne vit pas dans `cahier_de_these` (qui reflète l'état de juin 2026, essentiellement pédagogique). Il a été rangé le 21/09/2026 dans `these/travail_aout_2026/` (ex-`D:\thése\07-08-2026\`, `codes\`, `new\`, dispersés et non versionnés jusque-là) ; le dépôt officiel est en sous-module Git dans `these/external/hard-label-dnn-extraction/`. Voir §5, point 6.

---

## 1. Où on en est

Ça fait maintenant huit mois, un peu plus, que le travail a démarré. Le premier semestre a servi à poser les bases : revue de littérature, prise en main du code officiel, mise en place de l'environnement de travail. Le deuxième semestre produit les premiers résultats propres.

Le domaine a bougé plus vite que prévu : entre janvier et juin 2026, plusieurs équipes (NTU Singapour, Tsinghua, TII — ePrint 2026/253, ToSC 2026/1) ont publié des attaques sur les activations alternatives en **raw-output**, ce qui recoupait un axe initial du projet. La thèse se recentre sur l'intersection **hard-label × activations lisses**, qui reste ouverte, avec un volet défense avancé plus tôt que prévu. Ce recentrage est documenté dans `new/02_Plan_Annee_1_detaille.md` (2 août 2026) — **[VÉRIFIÉ]**, le document existe et détaille la même justification.

## 2. Ce qui est acquis et vérifié

### 2.1 Premier semestre — infrastructure

- **[VÉRIFIÉ]** Le code officiel d'EUROCRYPT 2024 / *Journal of Cryptology* (Chávez-Saab, Canales-Martínez et al., dépôt `Jchavezsaab/hard-label-dnn-extraction`) est cloné, désormais en sous-module Git dans `these/external/hard-label-dnn-extraction/`, avec les réseaux CIFAR-10 (`cifar10_3x256_64_10_float64.keras`, 0,52 de précision selon son propre README) et les points duaux précalculés. Le cache `__pycache__` de `sign_recovery/common.py` est daté du 3 août 2026, signe d'une exécution réelle, pas d'un simple clonage.
- Reformuler : *« le code d'EUROCRYPT 2025 tourne »* devrait plutôt dire **EUROCRYPT 2024 / Journal of Cryptology** — c'est le nom exact du papier associé à ce dépôt (signature recovery + sign recovery). L'article EUROCRYPT 2025 (hard-label, Carlini-Shamir) est un article différent, couvert par ta propre reproduction plus bas (§2.3).
- **[NON RETROUVÉ]** « État de l'art structuré autour de 21 références, tableau à 7 colonnes » : le dossier `cahier_de_these/articles/` contient 16 fiches ; un dossier distinct (`07-08-2026/06_Dossier_13_articles.md`) en couvre 13 avec un système de niveaux de confiance par source ([PDF]/[ABS+]/[ABS]/[CIT]). Aucun tableau maître à 21 lignes × 7 colonnes n'a été retrouvé. À vérifier : soit il existe ailleurs et il faut le retrouver, soit c'est un livrable cible (`new/02_Plan_Annee_1_detaille.md` le fixe comme objectif du semestre 1) pas encore atteint — dans ce cas, ne pas le présenter comme fait.

### 2.2 Campagne du 3 août — ce qui tient

- **[VÉRIFIÉ] Récupération de la couche de sortie en hard-label, indépendamment de l'activation.** Réexécuté le 21/09/2026 avec `travail_aout_2026/07-08-2026/attaque_couche_sortie.py` sur l'architecture 24-20-16-8-4 (36 inconnues, 26 de rang maximal), **pour les 4 activations** (SiLU ajoutée le 21/09, voir §5 point 2) :

  | Activation | Rang atteint | Résidu max | Accord après correction (/5000) |
  |---|---|---|---|
  | ReLU | 26/26 | 5,7 × 10⁻¹⁴ | 5000/5000 |
  | GELU | 26/26 | 4,6 × 10⁻¹⁴ | 5000/5000 |
  | SiLU | 26/26 | 1,6 × 10⁻¹³ | 5000/5000 |
  | tanh | 26/26 | 8,0 × 10⁻¹⁵ | 5000/5000 |

- **[VÉRIFIÉ] Ambiguïté de signe global en sortie multi-classes, corrigée en une requête.** Confirmé en direct : ReLU et GELU avaient un facteur d'échelle négatif (`c < 0`), donnant un accord de 0/5000 *avant* correction — exactement la signature attendue (le hasard sur 4 classes est 25 %, un score de 0 % n'est jamais un échec de la méthode mais une inversion de convention). Une seule requête sonde suffit à la détecter et à corriger. C'est une observation correcte et allant au-delà de l'article de départ, qui ne traite ce cas que pour la sortie binaire (`08_Documentation_notebook.md`, §4).

- **[VÉRIFIÉ, mais requalifié — voir §3] Signal de courbure pour les activations lisses.** Le signal existe bel et bien : la marche sur la frontière de décision (`Reproduction_attaques.ipynb` §3) mesure une courbure maximale médiane de l'ordre de 100 pour ReLU/LeakyReLU contre 1 à 25 pour les activations lisses — un écart de deux ordres de grandeur, confirmé indépendamment ci-dessous avec 5 graines de réseau différentes.

## 3. Ce qui doit être retiré ou requalifié

**Affirmation originale (note du 20/09) :** *« GELU et SiLU, qui sont les cibles les plus faciles en raw-output, sont au contraire les plus vulnérables face à un attaquant hard-label. tanh s'en sort mieux que les deux. »*

**Statut : [REQUALIFIÉ] — à ne pas présenter comme un résultat.**

Ta propre documentation du 6 août (`08_Documentation_notebook.md`, §4) avait déjà réexécuté cette mesure trois jours après la campagne du 3 août et trouvé l'ordre inversé (SiLU meilleure, tanh pire), concluant explicitement : *« ne présente pas l'inversion GELU/raw-output comme un résultat […] avant toute affirmation, il faut agréger sur plusieurs graines de réseau »*. Cette agrégation n'avait pas encore été faite.

**Elle a été faite aujourd'hui (21/09/2026)**, avec le code exact de la fonction `measure()` du notebook (§5), sur 5 graines de réseau indépendantes (architecture `2-8-8-2`, *two moons*, 5 traces de frontière par graine) :

| Graine réseau | ρ ReLU | ρ GELU | ρ SiLU | ρ tanh | Classement (meilleur→pire) |
|---|---|---|---|---|---|
| 7 | 0,023 | 0,913 | 0,622 | 1,039 | relu < silu < gelu < **tanh** |
| 17 | 0,004 | 0,836 | 0,677 | 0,605 | relu < **tanh** < silu < gelu |
| 27 | 0,005 | 0,832 | 0,958 | 0,406 | relu < tanh < gelu < **silu** |
| 37 | 0,010 | 0,443 | 0,848 | 0,490 | relu < gelu < tanh < **silu** |
| 47 | 0,025 | 0,526 | 0,796 | 0,842 | relu < gelu < silu < **tanh** |

**Conclusion sans ambiguïté : sur 5 graines, chacune des trois activations lisses (GELU, SiLU, tanh) se retrouve au moins une fois la *pire* pour l'attaquant, et au moins une fois parmi les deux meilleures.** Il n'existe aucun ordre stable entre activations lisses avec ce protocole. Le classement de la note originale n'est pas une hypothèse fragile : c'est un artefact d'une seule graine de réseau, aujourd'hui infirmé par construction.

**Ce qui reste robuste, et c'est ce qu'il faut présenter :** l'écart entre ReLU/LeakyReLU (ρ médian 0,010, jamais au-dessus de 0,025 sur les 5 graines) et n'importe quelle activation lisse (ρ toujours ≥ 0,41) est net, stable, et reproduit indépendamment aujourd'hui. **Le vrai clivage est différentiable / non-différentiable, pas une taxonomie fine entre activations lisses** — ce point précis de la note originale (fin de §2.2) est correct et se trouve renforcé, pas affaibli, par cette vérification.

## 3 bis. Nouvelle piste vérifiée le 21/09 : l'échelle des poids, pas l'activation

Hypothèse posée au §3 de la note originale (point 3) : *« le vrai facteur pourrait être un rapport entre l'échelle du signal et la largeur de la zone non linéaire, plutôt que l'activation seule »*. **Testée aujourd'hui, et elle tient.**

**Protocole.** Réseau `2-8-8-2` entraîné normalement sur *two moons*, puis seule la première couche (poids + biais) est multipliée par un facteur β avant de mesurer ρ. ReLU sert de contrôle négatif : `ReLU(β·z) = β·ReLU(z)`, donc ρ ne devrait presque pas bouger avec β pour cette activation.

**Défaut de protocole trouvé et corrigé en cours de route :** à β = 0,25 et β = 0,5, l'accuracy s'effondre (ReLU : 0,530 à β=0,25, c'est-à-dire le hasard sur 2 classes) parce que le biais de la 2ᵉ couche, non recalibré, domine quand le signal de la 1ʳᵉ couche rétrécit. Ces deux points sont écartés de l'analyse. **Ne pas les citer.**

**Résultat, sur β ∈ {1, 2, 4, 8, 16} où l'accuracy reste ≥ 0,84 pour toutes les activations (médiane sur 3 graines réseau) :**

| Activation | β=1 | β=2 | β=4 | β=8 | β=16 |
|---|---|---|---|---|---|
| ReLU (contrôle) | 0,011 | 0,018 | 0,016 | 0,028 | 0,020 |
| GELU | 0,902 | 0,784 | 0,571 | 0,469 | 0,265 |
| SiLU | 0,656 | 0,872 | 0,622 | 0,459 | 0,360 |
| tanh | 0,994 | 0,594 | 0,304 | 0,155 | 0,111 |

**Lecture.** ReLU reste plat (contrôle validé). Les trois activations lisses convergent vers le régime ReLU quand β augmente — tanh passe d'un facteur ~90 au-dessus de ReLU à un facteur ~5 seulement. **C'est un résultat plus solide et plus intéressant que le classement abandonné en §3 : ce n'est pas « quelle activation », c'est « à quelle échelle de pré-activation le réseau opère ».**

**À faire avant de le présenter comme acquis :**
- Refaire le test avec un protocole qui compense correctement le biais de couche 2 (ou qui scale les deux couches ensemble), pour pouvoir aussi couvrir β < 1 sans casser l'accuracy.
- Plus de graines (3 actuellement) — la tendance est nette mais SiLU est bruitée (non monotone entre β=1 et β=2).
- Vérifier que la tendance se maintient sur l'architecture de couche de sortie (§2.2), pas seulement sur le petit réseau *two moons*.

## 4. Conséquences pour la suite (WP1, WP3, WP4 — mise à jour)

| Lot de travail | Note du 20/09 | Mise à jour du 21/09 |
|---|---|---|
| WP1 — taxonomie | Reconstruire sur la courbure et sa propagation | **Mis à jour deux fois aujourd'hui.** La variable pertinente n'est ni « différentiable/non-différentiable » seul, ni l'identité de l'activation : c'est le **rapport échelle du signal / largeur de la zone non linéaire** (§3 bis, vérifié). C'est un résultat continu et quantitatif, plus fort qu'une classification par familles — c'est probablement l'axe central de l'article taxonomie (Art. 3 du plan de publication). |
| WP3 — attaque par courbure | Comprendre pourquoi la non-différentiabilité protège autant | Inchangé, et la nouvelle table du §3 en est une preuve supplémentaire, indépendante de la campagne du 3 août. |
| WP4 — défense | Gagne en priorité : le choix de l'activation comme défense gratuite | **Idée de défense élargie par le §3 bis** : ce n'est pas seulement « choisir la bonne activation », mais aussi **contraindre l'échelle des pré-activations** (ex. régularisation de poids, normalisation) — un levier disponible même si l'activation est imposée par ailleurs (ex. réseau pré-entraîné). À creuser comme deuxième piste de défense, en plus de l'écart différentiable/non-différentiable. |

## 5. Vérifications et implémentations à faire avant la présentation

Priorités reprises et complétées à partir de la note du 20/09 et de la check-list du 6 août (`08_Documentation_notebook.md`, §5) :

1. **[FAIT le 21/09, à consolider] Tester l'échelle des poids.** Résultat obtenu (§3 bis) : confirme l'hypothèse. À refaire avec un protocole qui compense le biais de couche 2, sur plus de graines, avant citation ferme.
2. **[FAIT le 21/09] Unifier SiLU dans `attaque_couche_sortie.py`.** Les 4 activations (relu/gelu/silu/tanh) tournent maintenant dans le même script canonique, vérifiées individuellement (rang 26/26, accord 5000/5000 chacune).
3. **[PRIORITÉ ABSOLUE, pas encore fait] Contrôler le pas de discrétisation `h`.** Le notebook a un test partiel (E1, §6) qui documente que l'écart ReLU/reste résiste à `h`, mais pas le classement interne aux lisses — logique, puisque ce classement n'existe pas. À refaire maintenant en croisant `h` avec β (le facteur d'échelle du §3 bis) plutôt qu'avec l'identité de l'activation.
4. **Étendre le test multi-graines à 15-20 graines**, désormais organisé autour de la variable β (échelle des poids) plutôt que de l'identité de l'activation.
5. **Monter en dimension** (20 puis 100 entrées). **Ce n'est pas juste changer un paramètre** : en dimension > 2, la frontière est une hypersurface et la courbure devient un tenseur ; le code actuel (`trace_boundary`, `curvature`) est écrit pour une courbe 2D. Deux options : (a) des coupes 2D aléatoires de l'espace à haute dimension — rapide, approximatif ; (b) une vraie mesure tensorielle — plus long, plus rigoureux. Choix à faire avant d'implémenter.
6. **[FAIT le 21/09] Mettre tout ce travail sous contrôle de version.** `07-08-2026/`, `codes/`, `new/` et les fichiers uniques du clone isolé `D:\thése\hard-label-dnn-extraction\` sont rangés dans `these/travail_aout_2026/` (voir son `README.md`) ; le dépôt officiel est en sous-module Git dans `these/external/hard-label-dnn-extraction/`. **Reste à faire** : supprimer les copies d'origine et les deux `.venv` (~1,8 Go) une fois ce rangement vérifié, et committer.
7. **Compteur de requêtes par phase et graines en fichier de config**, déjà listés le 6 août, toujours pas faits d'après l'état actuel des fichiers.
8. **[PARTIELLEMENT FAIT le 21/09] CIFAR-10 / réseau réel.** Deux bugs trouvés et corrigés dans `travail_aout_2026/exploration_repo_officiel/` : (a) `UnicodeEncodeError` sur emoji (encodage Windows) dans `analysis.py` et `demo_attack.py`, corrigé en forçant l'UTF-8 en sortie ; (b) `demo_attack.py` chargeait silencieusement un **modèle aléatoire** — son architecture supposait `64-256-256-256-64-10`, alors que le vrai `tiny.pth` (vérifié par inspection directe des tenseurs) est `64-64-64-64-64-10`, 17 290 paramètres. Corrigé : le vrai modèle CIFAR-10 charge maintenant. **Ce que ça ne donne toujours pas** : ce script n'exécute pas la vraie attaque (`signature_recovery/find_duals.py` → `recover_weights.py` → `sign_recovery/`), et il interroge le modèle avec du **bruit gaussien aléatoire**, pas de vraies images CIFAR-10 prétraitées — les prédictions observées (quasi toujours classe 7) ne disent rien du comportement réel du modèle. La reproduction complète de l'attaque sur CIFAR-10 reste à faire.

## 6. Ce qu'il faut trancher avec le directeur

- Valider le recentrage de la thèse sur l'intersection hard-label × activations lisses (déjà proposé, cohérent avec les résultats vérifiés).
- Décider si le résultat de couche de sortie (§2.2, indépendant de l'activation, ambiguïté de signe multi-classes) part sur ePrint maintenant — c'est le résultat le plus solide et le mieux vérifié de ce document.
- Acter que le classement fin entre activations lisses n'est **pas** un résultat, mais que le facteur d'échelle des poids (§3 bis) en est probablement l'explication réelle — décider si cette piste devient l'axe de l'article taxonomie (Art. 3).
- Statuer sur la demande de calcul (poste multi-cœurs) pour les contrôles restants (dimension, échelle des poids, graines).
- **[FAIT le 21/09]** Le travail non versionné est rangé dans `these/travail_aout_2026/` ; le dépôt officiel est en sous-module Git (`these/external/hard-label-dnn-extraction/`). Reste à supprimer les copies d'origine (`D:\thése\07-08-2026\`, `codes\`, `new\`, `hard-label-dnn-extraction\` + 2 `.venv`, ~1,8 Go) une fois ce rangement vérifié.
- Discuter d'une introduction auprès des auteurs de l'article de départ.

---

## Annexe A — Traçabilité des vérifications du 21/09/2026

| Affirmation | Source du code | Commande exécutée | Résultat |
|---|---|---|---|
| Couche de sortie, 4 activations, rang et accord | `travail_aout_2026/07-08-2026/attaque_couche_sortie.py` | `python attaque_couche_sortie.py --activation {relu,gelu,silu,tanh} --seed 5 --sans-diagnostic` | rang 26/26, résidu ≤ 1,6e-13, accord 5000/5000 après correction, pour les 4 |
| Ambiguïté de signe multi-classes | idem, étape 8 | idem | ReLU, GELU, SiLU : `c < 0`, corrigé en 1 requête ; tanh : `c > 0`, pas d'inversion |
| Classement instable entre activations lisses | Fonctions extraites de `travail_aout_2026/07-08-2026/Reproduction_attaques.ipynb`, cellules 2/9/15 (`measure()`) | script `robustesse_rho.py`, 5 graines réseau {7,17,27,37,47}, 5 traces/graine | aucun ordre stable ; ρ ReLU ∈ [0,004 ; 0,025], ρ lisses ∈ [0,41 ; 1,04] |
| Code officiel EUROCRYPT 2024 cloné et importé | `these/external/hard-label-dnn-extraction/` (sous-module depuis le 21/09) | `git remote -v`, horodatage `__pycache__` | dépôt `Jchavezsaab/hard-label-dnn-extraction`, `.pyc` daté du 3 août 2026 |
| 21 références / tableau 7 colonnes | — | recherche dans tous les fichiers du projet | non retrouvé ; 16 fiches dans `cahier_de_these/articles/`, 13 dans `travail_aout_2026/07-08-2026/06_Dossier_13_articles.md` |
| SiLU unifiée dans le script canonique | `travail_aout_2026/07-08-2026/attaque_couche_sortie.py` (modifié le 21/09) | `python attaque_couche_sortie.py --activation silu --seed 5 --sans-diagnostic` | rang 26/26, résidu 1,6e-13, `c<0` corrigé en 1 requête, accord 5000/5000 |
| Effet de l'échelle des poids (β) sur ρ | `travail_aout_2026/07-08-2026/test_echelle_poids.py` | 4 activations × 3 graines {7,17,27} × β ∈ {0,25 ... 16}, 4 traces/mesure | ρ décroît avec β pour gelu/silu/tanh (converge vers ReLU) ; ReLU reste plat (contrôle) ; β=0,25 et 0,5 invalidés par une chute d'accuracy (0,53 pour ReLU à β=0,25) |
| Fichiers uniques du clone isolé, dont un plantage CIFAR-10 | `travail_aout_2026/exploration_repo_officiel/` | lecture directe de `RAPPORT_EXECUTION.txt` | `UnicodeEncodeError` sur un emoji (`analysis.py` ligne 13) — le script a planté avant tout résultat, ce n'est pas une reproduction CIFAR-10 réussie |

*Document établi le 21 septembre 2026, à partir de la note du 20 septembre 2026 et d'une vérification directe du code présent sur disque. Complété le même jour après le test sur l'échelle des poids.*
