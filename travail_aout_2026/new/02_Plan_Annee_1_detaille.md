# Programme détaillé — Année 1

**Tidiane DIALLO — 2025/2028 — EPT Thiès**
Version révisée après cartographie de la littérature (août 2026)

---

## Vue d'ensemble

Le calendrier de l'annexe du projet initial prévoyait :
- **S1** : revue de littérature + reproduction d'une attaque ReLU existante → survey court ou workshop
- **S2** : analyse théorique des activations alternatives + prototypes d'attaque → premier article

Ce plan reste valable dans sa structure. Il est révisé sur deux points :

1. **La cible du premier article change.** « Attaques d'extraction sur activations alternatives » a été publié en raw-output par deux équipes début 2026 (ePrint 2026/253 ; ToSC 2026/1). La cible devient l'**intersection hard-label × activations lisses**, qui reste ouverte.
2. **Le volet défense est avancé.** Il était prévu en année 3. Comme c'est le créneau le moins concurrentiel, il faut y démarrer les expériences exploratoires dès la fin de l'année 1, quitte à ne publier qu'en année 2.

---

## Semestre 1 (déjà écoulé) — bilan et consolidation

### Ce qui devait être produit

| Livrable | Statut à viser |
|---|---|
| État de l'art structuré | Tableau maître 7 colonnes × 21 références |
| Maîtrise pratique des attaques | Code d'EUROCRYPT 2025 installé, exécuté, compris |
| Environnement de travail | Dépôt Git, pipeline d'entraînement, oracles instrumentés |
| Journal de recherche | Une entrée par jour de travail |

### Si des éléments manquent : plan de rattrapage sur 6 semaines

C'est fréquent en première année, ce n'est pas grave, mais il faut le combler avant d'attaquer le S2.

| Semaines | Action |
|---|---|
| 1–2 | Lire **intégralement** Carlini et al. EUROCRYPT 2025 (arXiv 2410.05750). Rédiger un résumé de 4 pages en français avec tes propres figures. |
| 2–3 | Cloner et **faire tourner** `github.com/jchavezsaab/hard-label-dnn-extraction`. Objectif minimal : reproduire l'extraction d'une couche sur un petit réseau. Documenter chaque blocage. |
| 3–4 | Réimplémenter la Section 3 de ton article de départ (couche de sortie) sur le réseau 3072−256×3−64−10. Cible : rang 584, accord 100 % sur 5 000 entrées. |
| 4–5 | Réimplémenter Freeze et SOE hard-label (Sections 4.2 et 4.3). Reproduire les temps : ~33 s, <1 s avec `m`. |
| 5–6 | Rédiger le tableau maître de l'état de l'art. Écrire une note de 10 pages « Où en est le domaine en 2026 ». |

**Point de vigilance :** l'étape « faire tourner le code existant » prend systématiquement 3 à 5 fois plus longtemps que prévu. Ce sont des prototypes de recherche, souvent sans documentation d'installation. Ne culpabilise pas et ne le cache pas à ton directeur : c'est le vécu de tout le monde.

---

## Semestre 2 — le programme détaillé

Quatre lots de travail (WP) menés en partie en parallèle.

---

### WP1 — Taxonomie géométrique des activations *(4 semaines)*

**Objectif :** produire le cadre formel qui structurera tout le reste de la thèse. C'est la contribution « cadre d'analyse » annoncée dans le projet.

**Question centrale :** pour une activation σ donnée, quelle est la structure de la frontière de décision, et quel signal reste observable en hard-label ?

**Tâches :**

1. Pour chaque σ ∈ {ReLU, LeakyReLU, PReLU, HardTanh, ELU, SELU, GELU, SiLU, Softplus, Mish, tanh, sigmoïde} :
   - tracer σ, σ′, σ″
   - mesurer la **largeur de la bande non-linéaire** : définir `w(σ, δ) = |{x : |σ″(x)| > δ}|` — la mesure de l'ensemble où la courbure dépasse un seuil
   - caractériser le comportement asymptotique en ±∞

2. Classer en trois familles (voir §2.2 du cahier) et **justifier formellement** la classification par `w(σ, δ)` :
   - `w = 0` → piecewise-linéaire → point critique bien défini
   - `w` fini et petit → quasi-linéaire → **courbure concentrée**
   - `w` infini → saturante → pas de localisation

3. Énoncer et prouver la **proposition sur les symétries** : pour σ non homogène positivement, le groupe de symétries se réduit aux permutations, donc le problème du signe disparaît mais la normalisation de signature devient contrainte.

**Livrable :** section théorique de 8–12 pages + figures. Réutilisable directement comme section 2 du premier article et comme chapitre 3 du manuscrit.

**Risque :** faible. C'est du travail sûr, qui produit un résultat même si tout le reste échoue.

---

### WP2 — Généralisation de la couche de sortie *(3 semaines, en parallèle de WP1)*

**Objectif :** premier résultat propre et rapide. Voir Idée 1 de la note stratégique.

**Thèse à démontrer :** la technique de récupération de la couche de sortie de Canales-Martínez & Santos est **indépendante de l'activation des couches cachées**.

**Tâches :**

1. Réexaminer la preuve de la Section 3 en isolant précisément les hypothèses utilisées :
   - (H1) en un point de transition, `A_i y + b_i = A_j y + b_j`
   - (H2) `y = f_{1..r}(x)` est calculable
   - (H3) la fonction de probabilité est invariante par translation et préserve l'ordre (softmax)
   Vérifier qu'**aucune** ne mentionne ReLU.
2. Énoncer le théorème généralisé : *pour tout σ, sous H2, la couche de sortie est récupérable en temps polynomial en hard-label, avec `d_r+2` degrés de liberté.*
3. Vérifier que l'analyse des degrés de liberté est inchangée (elle ne dépend que de softmax).
4. **Validation expérimentale.** Entraîner sur CIFAR-10 quatre réseaux d'architecture `3072−256×3−64−10`, identiques sauf l'activation : ReLU, GELU, SiLU, tanh. Supposer les couches 1..r connues (oracle partiel — hypothèse explicite et assumée). Appliquer la méthode. Mesurer : rang atteint, temps de construction, accord sur 5 000 entrées.

**Prédiction à tester :** l'accord devrait être de 100 % pour les quatre. **En revanche, le temps de construction du système peut varier fortement**, parce que trouver des points de transition dépend de la géométrie de la frontière. Mesurer et expliquer cet écart est un résultat en soi.

**Livrable :** théorème + preuve + tableau expérimental à 4 lignes. C'est une section d'article, pas un article.

**Risque :** très faible. C'est ton filet de sécurité : quoi qu'il arrive, tu auras ce résultat.

---

### WP3 — Attaque hard-label par courbure ⭐ *(8–10 semaines — le cœur)*

**Objectif :** la contribution principale. Voir Idée 2 de la note stratégique.

**Hypothèse de travail :** chez les activations quasi-linéaires, l'empreinte d'un neurone sur la frontière de décision n'est plus un pli mais un **pic de courbure** de largeur proportionnelle à `w(σ, δ)`.

**Tâches, dans l'ordre :**

**3.1 — Expérience préliminaire (1 semaine, à faire en tout premier).**
Réseau jouet : 2 entrées → 4 neurones → 2 sorties. **Tracer la frontière de décision dans le plan** pour σ = ReLU, GELU, SiLU, tanh. Superposer les hyperplans critiques (connus, puisque tu contrôles le réseau).

> **Fais cette expérience avant toute autre chose.** En une semaine, tu sauras si l'intuition tient. Si le pic de courbure est visible à l'œil pour GELU, tout WP3 est justifié. S'il ne l'est pas, tu l'apprends au bout d'une semaine et non de trois mois. C'est le meilleur rapport information/effort de toute l'année.

**3.2 — Estimateur de courbure en hard-label (2–3 semaines).**
Contrainte : on ne peut mesurer la normale `m` qu'indirectement, en échantillonnant des points de transition voisins.
- Échantillonner k points de transition dans une boule de rayon ρ.
- Ajuster un hyperplan local aux moindres carrés → estimation de `m(s)`.
- Marcher le long de la frontière, estimer `‖dm/ds‖`.
- **Analyser le compromis fondamental :** ρ petit → estimation de `m` bruitée ; ρ grand → le pic de courbure est lissé et disparaît. Il existe un ρ optimal. **Le caractériser est en soi un résultat.**

**3.3 — Détection et localisation du neurone (2–3 semaines).**
Détecter les maxima locaux de courbure le long de la marche → localiser l'« hyperplan critique flou ». Adapter la construction du sous-espace de points duaux : l'intersection nette devient une intersection approchée → résolution aux moindres carrés au lieu d'une résolution exacte.

**3.4 — Pipeline complet sur petits réseaux (2–3 semaines).**
Cibler d'abord `d₀ = 10`, une couche cachée de 20 neurones, 2 sorties. Puis MNIST (784 entrées). CIFAR-10 seulement si tout tient.

**3.5 — Étude de dégradation (1–2 semaines).**
Faire varier σ le long de l'axe « lissage » : LeakyReLU → ELU → GELU → SiLU → Softplus(β) avec β décroissant → tanh. Tracer :
- nombre de requêtes nécessaires vs. `w(σ, δ)`
- erreur d'extraction vs. `w(σ, δ)`

**C'est cette courbe qui est le vrai résultat de l'article**, même si l'attaque échoue pour tanh. Elle répond quantitativement à la question : *à quel point le lissage protège-t-il ?*

**Critères d'arrêt honnêtes (à fixer avec ton directeur dès le début) :**
- Si à la fin de 3.1 aucun signal n'est visible même pour GELU en 2D → arrêter WP3, réallouer sur WP4.
- Si 3.4 ne converge pas sur le réseau jouet après 3 semaines → publier 3.1+3.2+3.5 comme **étude d'impossibilité pratique**, ce qui reste un article.

**Livrable :** article 1. Cible : ePrint immédiatement, puis AfricaCrypt / LATINCRYPT / ACNS. EUROCRYPT/CRYPTO si le résultat est fort.

---

### WP4 — Défense : étude exploratoire *(4 semaines, fin de S2)*

**Objectif :** poser les fondations du volet défense, sans viser encore la publication. Voir Idée 3 de la note stratégique.

**Tâches :**

1. **Établir la courbe de fragilité.** Prendre l'attaque hard-label ReLU (qui fonctionne, code disponible). Injecter artificiellement un bruit de localisation de frontière d'amplitude τ. Mesurer le taux de succès de l'attaque en fonction de τ.
   → **C'est la mesure la plus importante du WP.** Elle établit combien de bruit est nécessaire. Si l'attaque casse dès τ = 10⁻¹⁰, la défense sera quasi gratuite en utilité.

2. **Mesurer le coût en utilité.** Sur CIFAR-10, calculer la distribution de la marge `g(x) = logit_(1) − logit_(2)` sur le jeu de test. Quelle fraction des entrées a `g(x) < τ` ? Tracer accuracy perdue vs. τ.

3. **Croiser les deux courbes.** S'il existe un τ tel que l'attaque casse et que la perte d'accuracy soit < 1 %, la défense est viable et publiable.

4. **Prototyper la randomisation par PRF.** Implémenter le mécanisme déterministe (HMAC sur les octets de `x`, seuillé) et vérifier qu'il résiste à la répétition de requête.

5. **Lister les attaques adaptatives** à traiter en année 2 : moyennage sur voisinage, régression robuste, vote entre points proches, exploitation de la structure du PRF.

**Livrable :** note interne de 10 pages + jeu de figures. Base de l'article 2 en année 2.

---

## Calendrier synthétique du S2

| Mois | WP1 | WP2 | WP3 | WP4 |
|---|---|---|---|---|
| 1 | Taxonomie, tracés σ/σ′/σ″ | Analyse des hypothèses | **3.1 expérience 2D** | — |
| 2 | Formalisation `w(σ,δ)` | Théorème + preuve | 3.2 estimateur | — |
| 3 | Proposition symétries | Entraînement 4 réseaux | 3.2 / 3.3 | — |
| 4 | Rédaction section | Validation expérimentale | 3.3 localisation | — |
| 5 | — | ✅ terminé | 3.4 pipeline | Courbe de fragilité |
| 6 | Relecture | — | 3.4 / 3.5 dégradation | Coût en utilité, PRF |

**Fin de S2 :** rédaction de l'article 1, soumission ePrint.

---

## Jalons et critères de réussite

| Jalon | Échéance | Critère objectif |
|---|---|---|
| J1 — Reproduction validée | +6 semaines | Rang 584 atteint, accord 100 % sur 5 000 entrées |
| J2 — Signal de courbure observé | +2 mois | Figure 2D montrant un pic de courbure pour GELU |
| J3 — Théorème couche de sortie | +3 mois | Preuve rédigée + 4 réseaux validés |
| J4 — Attaque jouet fonctionnelle | +5 mois | Extraction d'un réseau 10−20−2 GELU en hard-label |
| J5 — Courbe de dégradation | +6 mois | Requêtes vs. `w(σ,δ)` pour ≥ 6 activations |
| J6 — Article 1 sur ePrint | +6 mois | Prépublication en ligne |
| J7 — Courbe de fragilité défense | +6 mois | Taux de succès vs. τ mesuré |

---

## Ce dont tu as besoin (à demander explicitement au directeur)

1. **Calcul.** Les expériences de l'article de départ tournent sur un cœur, en Python/NumPy. Un bon poste de travail (32 Go RAM, CPU multi-cœurs) suffit pour l'année 1. **Mais une expérience de 13 h non parallélisée devient 2 h sur 8 cœurs** — c'est le gain le plus rentable à demander. GPU nécessaire seulement pour l'entraînement des réseaux cibles, ce qui est marginal.
2. **Accès bibliographique.** Springer LNCS (EUROCRYPT, ASIACRYPT, CRYPTO) est payant. Vérifier l'accès EPT. Sinon : versions arXiv/ePrint, ou écrire directement aux auteurs (ils envoient presque toujours leur PDF).
3. **Budget conférence.** Prévoir une mission dès l'année 2. AfricaCrypt est le point d'entrée le plus accessible.
4. **Contacts.** Demander à ton directeur s'il peut faire une introduction auprès d'Isaac Canales-Martínez (TII, Abu Dhabi) ou de l'équipe de Francisco Rodríguez-Henríquez. Une collaboration, même informelle, change tout pour un doctorant isolé.
5. **Comité de suivi.** Si l'EDPT le permet, y inclure quelqu'un du côté ML et non seulement crypto : ton sujet est à cheval, et un regard ML te fera gagner du temps sur les questions d'entraînement et de reproductibilité.

---

## Une remarque de méthode pour finir

Le risque principal de ta thèse n'est pas technique, il est **temporel**. Quatre équipes très bien dotées (Tsinghua, NTU, TII, Versailles/NTU) publient sur ce sujet en continu. Tu ne gagneras pas sur le volume de calcul ni sur la taille de l'équipe.

Tu peux gagner sur deux choses :
- **Le créneau.** L'intersection hard-label × lisse est étroite et difficile — c'est exactement pourquoi elle est encore libre.
- **La rigueur.** Un résultat négatif bien prouvé (« les activations saturantes protègent intrinsèquement ») a une valeur durable qu'aucune course à la performance ne périme.

Et garde en tête que ton article de départ, publié à LATINCRYPT 2025, est signé par deux personnes — dont un stagiaire. La taille de l'équipe n'est pas le facteur déterminant. La clarté de la question l'est.
