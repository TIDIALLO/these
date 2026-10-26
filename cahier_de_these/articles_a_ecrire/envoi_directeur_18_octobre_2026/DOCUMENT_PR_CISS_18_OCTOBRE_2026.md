# Document pour Pr. Abdoul Aziz Ciss — préparation de l'Article 1

Tidiane Diallo · suite à la réunion du 18 octobre 2026

---

## 0. Ce que contient ce document

Vous m'avez demandé d'envoyer un document et les tests, pour préparer le premier article. Ce document couvre les deux résultats présentés en réunion, chacun avec : le problème, la méthode, le protocole, les chiffres mesurés aujourd'hui (pas des chiffres recopiés d'une exécution plus ancienne), et les limites honnêtes. Les tests eux-mêmes — le code et la sortie brute de chaque exécution — sont dans `logs/`, décrits en section 4.

Les deux résultats sont à des stades très différents de maturité. Le résultat A est vérifié et stable. Le résultat B est un premier prototype, avec des limites claires. Je les présente dans cet ordre volontairement.

---

## 1. Résultat A — La couche de sortie se récupère indépendamment de l'activation des couches cachées

### Le problème

Carlini et al. (EUROCRYPT 2025) récupèrent toutes les couches d'un réseau ReLU en hard-label, sauf la couche de sortie : elle n'a pas d'activation, donc aucun coude à exploiter. Mon article de départ (Canales-Martínez & Santos 2025) comble ce trou, mais seulement dans un cadre où le reste du réseau est ReLU. La question que je teste : la méthode de récupération de la couche de sortie dépend-elle réellement de ReLU, ou seulement du softmax final ?

### La méthode

La récupération repose sur trois hypothèses, posées à un point de transition entre deux classes `i` et `j` :

- **(H1)** `A_i·y + b_i = A_j·y + b_j`, où `y` est l'entrée de la couche de sortie.
- **(H2)** `y = f_{1..r}(x)` est calculable, c'est-à-dire que les couches cachées sont déjà connues.
- **(H3)** la fonction de décision (softmax) est invariante par translation et préserve l'ordre des scores.

Aucune de ces trois hypothèses ne mentionne l'activation des couches cachées. (H3) ne dépend que du softmax final. Si c'est vrai, la méthode doit fonctionner à l'identique quelle que soit l'activation cachée — ReLU, GELU, SiLU ou tanh — dès que (H2) est satisfaite.

**Protocole.** Réseau `24-20-16-8-4` (profondeur 4, 4 classes en sortie), construit avec la même architecture et la même graine (`seed=5`) pour les quatre activations testées. L'attaquant dispose de l'oracle hard-label et suppose les couches `1..r` déjà extraites (hypothèse H2, explicite). Il collecte des points de transition, résout le système linéaire qui en découle, puis corrige l'ambiguïté de signe global avec une seule requête supplémentaire.

### Résultats mesurés aujourd'hui (18/10/2026, seed=5)

| Activation | Rang atteint | Écart max \|Â − A normalisé\| | Facteur d'échelle `c` | Accord avant correction (/5000) | Accord après correction (/5000, uniforme + gaussien) | Requêtes totales | Temps |
|---|---|---|---|---|---|---|---|
| ReLU | 26 / 26 | 4,297 × 10⁻¹⁴ | −2,035713 (négatif) | 0 / 5000 | 5000 / 5000 et 5000 / 5000 | 16 583 | 8,1 s |
| GELU | 26 / 26 | 1,306 × 10⁻¹³ | −2,831870 (négatif) | 0 / 5000 | 5000 / 5000 et 5000 / 5000 | 16 581 | 16,4 s |
| SiLU | 26 / 26 | 2,132 × 10⁻¹⁴ | −4,920914 (négatif) | 0 / 5000 | 5000 / 5000 et 5000 / 5000 | 16 589 | 9,3 s |
| tanh | 26 / 26 | 6,217 × 10⁻¹⁵ | 0,548012 (positif) | 5000 / 5000 | 5000 / 5000 et 5000 / 5000 | 16 577 | 8,4 s |

**Lecture.** Le rang atteint (26 sur 26, le maximum théorique pour cette architecture) et l'écart résiduel (de l'ordre de 10⁻¹³ à 10⁻¹⁵, proche de la précision machine) sont identiques pour les quatre activations. Pour ReLU, GELU et SiLU, le facteur d'échelle global `c` ressort négatif : la méthode reconstruit alors le réseau « à l'envers » (l'accord tombe à 0 sur 5000, bien en dessous du hasard à 25 %, ce qui est la signature attendue d'une inversion de convention, pas un échec). Une seule requête supplémentaire suffit à détecter et corriger ce cas, pour les trois activations concernées. Pour tanh, `c` ressort positif directement, donc aucune correction n'est nécessaire. Après correction, l'accord est parfait sur deux distributions de test indépendantes (entrées uniformes et gaussiennes), pour les quatre activations.

Le coût en requêtes et en temps est, lui, quasi identique entre les quatre activations (16 500 à 16 600 requêtes, 8 à 16 secondes). La quasi-totalité du coût est dans la recherche des points de transition, qui est strictement la même opération quelle que soit l'activation cachée — cohérent avec le fait que cette étape ne regarde que la couche de sortie.

### Limites honnêtes

- Le résultat suppose les couches cachées `1..r` déjà connues (hypothèse H2). Ce n'est pas encore une attaque de bout en bout sur un réseau dont tout est inconnu.
- Testé sur un seul réseau jouet par activation, une seule graine. Le réseau CIFAR-10 réel de l'article de départ n'a pas encore été rejoué avec cette méthode.
- L'ambiguïté de signe global en sortie multi-classes (étape 8 du script) va au-delà de ce que traite l'article de départ, qui ne discute ce cas que pour la sortie binaire — c'est une observation qui mériterait d'être vérifiée plus formellement avant d'être présentée comme un résultat à part entière.

---

## 2. Résultat B — Extraction hard-label de neurones non-ReLU (premier prototype, TP8)

### Le problème

Les attaques hard-label existantes (Carlini 2025, mon article de départ) sont construites pour ReLU. La question : la même primitive géométrique — balayer des droites voisines autour d'un point de la frontière de décision pour détecter où cette frontière se plie — fonctionne-t-elle sans modification quand l'activation cachée a un coude non nul (Leaky ReLU, PReLU), voire pas de coude du tout (ELU, en partie) ?

### La méthode

Trois étapes, codées dans `tp8_hardlabel_beyond_relu.py` :

1. **Détection de pli.** Le long d'une droite proche d'un point de frontière déjà trouvé, on cherche où la position de la frontière, en fonction du décalage, change brusquement de pente. Ce changement de pente révèle un point situé exactement sur l'hyperplan d'un neurone caché — le même principe que pour ReLU, mais rien dans le calcul ne suppose que l'activation soit ReLU.
2. **Ajustement d'hyperplan robuste.** Plusieurs points de pli trouvés près du même germe sont assemblés par moindres carrés, avec un rejet des points aberrants (une forme légère de RANSAC), pour reconstruire la direction du neurone.
3. **Mesure d'un ratio de pente bilatéral**, une piste empirique pour estimer la pente négative propre à chaque activation — présentée comme une observation, pas comme un estimateur validé (voir la note en bas du fichier de code pour la raison précise).

### Protocole

Réseau jouet `3 entrées – 4 neurones cachés – 2 sorties`, même graine réseau (`seed=7`) pour les quatre activations testées, même graine d'attaque (`seed=3`). Oracle hard-label uniquement — aucun accès aux logits.

### Résultats mesurés aujourd'hui (18/10/2026)

| Activation | Neurones retrouvés / 4 | \|cos\| moyen avec la vraie direction | Requêtes | Temps |
|---|---|---|---|---|
| ReLU | 3 | 0,9785 | 351 647 | 11,7 s |
| Leaky ReLU (α vrai = 0,1) | 1 | 0,9792 | 413 740 | 17,7 s |
| PReLU (α vrai = 0,25) | 3 | 0,6858 | 508 247 | 22,4 s |
| ELU | 0 | — | 466 147 | 23,2 s |

**Lecture.** Quand la méthode trouve un neurone, la direction retrouvée est très proche de la vraie direction pour ReLU et Leaky ReLU (|cos| proche de 0,98), ce qui montre que la primitive fonctionne bien au-delà de ReLU dans son principe. Pour PReLU, la précision est nettement plus faible (0,69). Pour ELU, aucun neurone n'a pu être retrouvé dans ces réglages : la partie négative d'ELU n'est pas linéaire, donc la frontière de décision y est légèrement courbe au lieu d'être plate, ce qui casse l'hypothèse de base de la méthode.

### Limites honnêtes

- Une seule graine réseau testée. Les résultats peuvent varier fortement d'un réseau à l'autre, en particulier pour PReLU où la variance observée précédemment était grande.
- Il existe un compromis net entre le rayon local de recherche, le taux de neurones retrouvés et le nombre de requêtes (resserrer le rayon améliore la précision mais fait chuter le taux de réussite et multiplie le coût par 10 à 100). Ce compromis n'est pas encore caractérisé formellement.
- Le ratio de pente bilatéral (piste pour estimer α) n'est pas un estimateur démontré : le calcul montre qu'il mélange la pente propre au neurone et le comportement du reste du réseau autour de lui, exactement l'ambiguïté d'échelle que Carlini 2020 traite séparément pour le signe.
- Aucune reconstruction complète d'un réseau, aucune expérience sur CIFAR-10.

---

## 3. Ce que ça donne pour l'Article 1

Le résultat A est aujourd'hui le plus solide des deux : chiffres stables, écart proche de la précision machine, comportement identique sur quatre activations. C'est un candidat direct pour une section d'article, voire pour une prépublication ePrint à part, qui daterait la contribution avant que quelqu'un d'autre ne publie la même généralisation.

Le résultat B est la direction la plus proche de ma contribution 1 annoncée (« Hard-Label Extraction Beyond ReLU »), mais il reste à un stade de prototype. Un point important découvert cette semaine, en dehors de ce document : un travail de 2026 (Qi et al., ePrint 2026/178, ToSC) revendique déjà le hard-label sur PReLU. Avant d'investir davantage de temps sur le résultat B, je dois lire ce travail en entier pour savoir ce qu'il reste réellement à faire.

---

## 4. Les tests (code et sortie brute)

Chaque résultat est accompagné du script qui l'a produit et du journal complet de son exécution d'aujourd'hui, pas d'un résumé.

| Résultat | Script | Log |
|---|---|---|
| A — couche de sortie, ReLU | `travail_aout_2026/07-08-2026/attaque_couche_sortie.py --activation relu --seed 5` | `logs/couche_sortie_relu.txt` |
| A — couche de sortie, GELU | même script `--activation gelu` | `logs/couche_sortie_gelu.txt` |
| A — couche de sortie, SiLU | même script `--activation silu` | `logs/couche_sortie_silu.txt` |
| A — couche de sortie, tanh | même script `--activation tanh` | `logs/couche_sortie_tanh.txt` |
| B — hard-label non-ReLU (4 activations) | `cahier_de_these/TP/code/tp8_hardlabel_beyond_relu.py` | `logs/tp8_hardlabel_beyond_relu.txt` |

Pour rejouer : `python attaque_couche_sortie.py --activation <nom> --seed 5 --sans-diagnostic` depuis `travail_aout_2026/07-08-2026/`, ou `python tp8_hardlabel_beyond_relu.py` depuis `cahier_de_these/TP/code/`. Les deux scripts fixent leurs graines : les chiffres ci-dessus se reproduisent à l'identique.

---

## 5. Questions ouvertes pour la prochaine étape

1. Le résultat A part-il en prépublication ePrint maintenant, séparément du reste ?
2. Dois-je lire Qi et al. (2026/178) en entier avant de continuer le résultat B, ou continuer en parallèle ?
3. Le réseau CIFAR-10 réel de l'article de départ doit-il être la prochaine cible pour le résultat A, avant d'élargir le résultat B ?

---

*Document préparé le 18 octobre 2026, à partir d'exécutions réelles des scripts cités, pas de chiffres mémorisés d'une exécution antérieure.*
