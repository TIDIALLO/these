# Année 1 — Reproduire & Maîtriser

> Feuille de route détaillée de la première année. **Tu es à ~1 an d'inscription** : utilise ce document comme un **audit**. Pour chaque phase, coche ce qui est fait, repère les trous, et rattrape l'essentiel. L'objectif de l'année n'est PAS de produire du neuf — c'est d'avoir, à la fin, (1) un **banc d'essai reproductible**, (2) toutes les **baselines ReLU** mesurées, et (3) un **chapitre « état de l'art »** rédigé.

**Principe directeur :** on monte une échelle. Chaque phase retire *une* hypothèse. On ne touche pas au pire cas (lisse × hard-label) cette année — on le prépare.

---

## Vue d'ensemble : 7 phases, 4 jalons

| Phase | Objectif | Cadre | Livrable | Durée |
|-------|----------|-------|----------|-------|
| 0 | Bases DNN + environnement | — | Banc d'essai qui tourne | 3-4 sem |
| 1 | Reproduire signature (raw-output, ReLU) | facile | Carlini 2020 rejoué | 4-5 sem |
| 2 | Reproduire récupération de signe polynomiale | facile | Canales-M. 2024 rejoué | 4-5 sem |
| 3 | Reproduire hard-label (points de transition) | dur | Chen/Carlini 2025 rejoué | 5-6 sem |
| 4 | Reproduire l'article fondateur (couche de sortie) | dur | Canales & Santos 2025 sur CIFAR-10 | 5-6 sem |
| 5 | Reproduire beyond-ReLU raw-output | moyen | Identification + Leaky/ELU | 4 sem |
| 6 | Reproduire la défense | moyen | Train to Defend rejoué | 3-4 sem |
| 7 | Synthèse : état de l'art + banc consolidé | — | Chapitre 3 + rapport d'année | 3-4 sem |

**Jalons :** J1 (M3) banc + signature OK · J2 (M6) signe + hard-label OK · J3 (M9) article fondateur + beyond-ReLU OK · J4 (M12) défense + chapitre état de l'art rédigé.

> **Si tu es en retard** (ce qui est normal et fréquent) : les phases **non négociables** sont 0, 1, 4 (ton article) et 7. Les phases 2, 3, 5, 6 peuvent être faites « en lecture + repro partielle » et complétées en début d'année 2. Mieux vaut 4 reproductions *solides et comprises* que 7 survolées.

---

## Phase 0 — Bases DNN & environnement (3-4 semaines)

**Objectif :** être autonome en PyTorch et avoir un banc d'essai qui tourne.

**À faire :**
- Lire `ressources_DNN/01` à `03` (déjà dans ton cahier). Refaire `TP1` (entraîner un MLP) jusqu'à le coder de mémoire.
- Mettre en place le dépôt : `git init`, structure `src/ tests/ experiments/ data/`, seeds fixés partout, un logger qui écrit les **3 métriques** (requêtes, temps, erreur de fidélité) en CSV.
- Vérifier que `common.py` + TP2-TP7 tournent chez toi (ils sont déjà testés).

**Livrable :** un dépôt git versionné + un MLP entraîné + un script de log des métriques.

**Critère de réussite :** tu sais entraîner un réseau, l'interroger en raw-output ET en hard-label, et logger une expérience reproductible (même seed → même résultat).

**Piège classique :** négliger la précision numérique. Travaille en **float64** dès le départ (`model.double()`), sinon tes mesures d'attaque seront du bruit.

---

## Phase 1 — Signature en raw-output, ReLU (4-5 semaines)

**Objectif :** reproduire le cœur de Carlini 2020 — récupérer la *direction* des poids d'un neurone via les points critiques.

**À faire :**
- Relire `articles/fiche_03` et **réécrire à la main** l'équation du saut de gradient au point critique.
- Refaire `TP2` (détecter les points critiques) et `TP3` (signature, `|cos|=1`). Étends `TP3` à un réseau à **2 couches cachées** : il faut « peler » la couche 1 avant d'attaquer la couche 2.
- Faire tourner le **dépôt officiel** `google-research/cryptanalytic-model-extraction` (JAX) sur un petit MLP MNIST, et comparer ses résultats aux tiens.

**Livrable :** un script qui récupère les signatures d'un MLP à 2 couches, + un tableau de métriques (requêtes vs nb de neurones).

**Critère de réussite :** `|cos|` entre signatures récupérées et vraies lignes de poids > 0,999, et tu sais **expliquer pourquoi** le saut de gradient donne la direction du poids.

**Piège :** confondre signature (direction, OK) et poids complet (il manque l'échelle et le signe → phase 2).

---

## Phase 2 — Récupération de signe polynomiale (4-5 semaines)

**Objectif :** reproduire Canales-Martínez 2024 — passer du signe exponentiel (Carlini) au signe polynomial, ce qui débloque les réseaux profonds.

**À faire :**
- Relire `articles/fiche_04`. Refaire `TP4` (comprendre l'ambiguïté de signe).
- Utiliser le **dépôt unifié** `hannafoe/cryptanalytical-extraction` (il combine Carlini + Canales-Martínez + les optimisations de Foerster) : lance une extraction end-to-end couche par couche sur un réseau à 4-8 couches.
- Documenter **laquelle** des trois techniques de signe tu reproduis, et son coût réel.

**Livrable :** une extraction end-to-end (signature + signe) sur un réseau ≥ 4 couches, métriques à l'appui.

**Critère de réussite :** extraction fonctionnellement équivalente d'un réseau profond ReLU, et tu sais expliquer pourquoi le signe ne se lit plus « directement » quand le neurone est enfoui dans le réseau.

**Piège :** la propagation d'erreur. Plus le réseau est profond, plus l'erreur s'accumule → surveille la fidélité couche par couche.

---

## Phase 3 — Hard-label : points de transition (5-6 semaines)

**Objectif :** reproduire la primitive hard-label (Chen 2024 / Carlini 2025) — n'observer que le label et reconstruire via les frontières de décision.

**À faire :**
- Relire `articles/fiche_06` et `fiche_07`. Refaire `TP5` (points de transition par recherche binaire).
- Reconstruire, à partir de milliers de points de frontière, les hyperplans de la **première couche** d'un petit réseau. (C'est l'étape la plus technique de l'année.)
- Comparer le coût (requêtes) hard-label vs raw-output sur le même réseau.

**Livrable :** récupération des hyperplans de couche 1 en hard-label sur un réseau jouet + comparaison de coût.

**Critère de réussite :** tu retrouves les hyperplans de couche 1 à partir du label seul, et tu sais expliquer la différence point critique (cassure de pente) vs point de transition (changement de classe).

**Piège :** en hard-label tout est plus lent et plus bruité. Reste sur de **petits réseaux** (dim d'entrée ≤ 5, 1-2 couches) cette année.

---

## Phase 4 — Ton article fondateur (5-6 semaines) ★ priorité absolue

**Objectif :** reproduire Canales-Martínez & Santos 2025 — la **récupération de la couche de sortie** (sans ReLU) et l'exploitation des **couches contractives**, sur CIFAR-10.

**À faire :**
- Relire `articles/fiche_08` ligne par ligne. C'est *ton* article : tu dois pouvoir le présenter sans notes.
- Reproduire la technique de couche de sortie sur un petit réseau, puis sur un réseau entraîné sur **CIFAR-10** (architecture contractive).
- Vérifier le gain de temps annoncé pour les architectures contractives.
- Contacter les auteurs si le code n'est pas public (c'est normal et bien vu de demander).

**Livrable :** reproduction de l'extraction de la couche de sortie + mesure du gain contractif, documentée.

**Critère de réussite :** tu reproduis le résultat clé de ton article et tu sais pointer **exactement** où l'absence de ReLU oblige à changer de technique. ← c'est le pont vers tes contributions.

**Piège :** vouloir aller trop vite sur CIFAR-10. Valide d'abord sur un réseau jouet où tu connais la vérité terrain.

---

## Phase 5 — Beyond-ReLU en raw-output (4 semaines)

**Objectif :** reproduire les briques « au-delà de ReLU » dans le cadre facile (raw-output) : identification d'activation + signature pour activations par morceaux non-ReLU.

**À faire :**
- Relire `articles/fiche_10` (Qi 2026) et `fiche_12` (PReLU). Refaire `TP6`.
- Reproduire l'**identification d'activation** (familles par morceaux vs lisse — ton TP6 le fait déjà à 9/9) et la **signature pour Leaky ReLU / ELU**.
- Étendre `TP3` pour **estimer α** (pente négative) sur un réseau PReLU.

**Livrable :** identification d'activation + extraction de signature PReLU/ELU avec estimation de α.

**Critère de réussite :** tu extrais une activation par morceaux non-ReLU en raw-output et tu observes (et expliques) que le **signe y est plus facile** que pour ReLU.

**Note stratégique :** cette phase prépare directement ton **Papier 1** (année 2). Garde des notes propres.

---

## Phase 6 — La défense (3-4 semaines)

**Objectif :** reproduire « Train to Defend » (Kurian & Aysu 2025) — la régularisation de similarité intra-couche.

**À faire :**
- Relire `articles/fiche_13`. Refaire `TP7` (mécanisme).
- Implémenter la **vraie** défense : ajouter le terme de régularisation (distance entre poids intra-couche) dans la perte de `TP1`, réentraîner, vérifier la perte de précision (< 1 %).
- Relancer ton attaque (phase 1-2) sur le modèle protégé et mesurer la dégradation.

**Livrable :** modèle réentraîné avec la défense + courbe « efficacité de l'attaque vs force de la régularisation ».

**Critère de réussite :** tu reproduis l'effet protecteur avec < 1 % de perte de précision, et tu poses déjà la question : *tient-elle hors ReLU ?* (← Papier 2, année 3).

---

## Phase 7 — Synthèse (3-4 semaines)

**Objectif :** transformer l'année en livrables académiques.

**À faire :**
- Rédiger le **chapitre « état de l'art »** à partir de `articles/00_index_bibliographie.md` (le tableau récapitulatif est ton squelette).
- Consolider le **banc d'essai** : un seul point d'entrée, toutes les activations, les deux oracles, export CSV des 3 métriques. (Candidat à une publication « banc d'essai open-source » — idée S3.)
- Écrire le **rapport de 1ʳᵉ année** : ce qui est reproduit, mesuré, compris ; les questions ouvertes ; le plan de l'année 2.

**Livrable :** chapitre 3 (brouillon) + banc consolidé + rapport d'année.

---

## Ce que « maîtriser » veut dire (auto-évaluation)

Tu maîtrises l'année 1 si tu peux, **sans notes** :

1. Expliquer pourquoi un réseau ReLU est affine par morceaux et ce qu'est un point critique.
2. Dériver le saut de gradient et expliquer comment il donne la signature.
3. Expliquer pourquoi le signe est dur, et l'idée d'une technique polynomiale.
4. Expliquer la différence point critique / point de transition (raw vs hard-label).
5. Présenter ton article fondateur et dire **où** l'absence de ReLU change tout.
6. Lancer une extraction end-to-end reproductible et lire ses 3 métriques.
7. Expliquer le principe de la seule défense existante et ses limites.

Si tu coches les 7, tu es prêt pour les contributions de l'année 2.

---

## Discipline permanente (toute l'année)

- **3 métriques** à chaque expérience : nb de requêtes, temps réel, erreur de fidélité (max |Δ|).
- **Reproductibilité** : seed fixé, code versionné, journal de manips daté.
- **Équivalence fonctionnelle** vérifiée sur un jeu de test indépendant, pas seulement la proximité des poids.
- **Lecture active** : pour chaque article, réécrire à la main 1-2 équations clés.
- **Réunion d'encadrement** : apporte à chaque fois une mesure ou une figure, pas seulement « j'ai lu ».

*Dernière mise à jour : 2026-06-18. Coche tes phases au fur et à mesure dans le tableau « Vue d'ensemble ».*
