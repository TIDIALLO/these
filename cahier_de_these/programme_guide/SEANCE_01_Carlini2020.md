# Séance 1 — Carlini 2020 : l'idée fondatrice

> **Objectif de la semaine :** comprendre *pourquoi* extraire un réseau ReLU est possible, et le **prouver toi-même** en code (récupérer la signature d'un neurone). C'est la brique sur laquelle repose TOUT le reste de ta thèse.

**Article :** Carlini, Jagielski, Mironov, *Cryptanalytic Extraction of Neural Network Models*, CRYPTO 2020.
📄 arXiv : https://arxiv.org/abs/2003.04884 · 💻 code : https://github.com/google-research/cryptanalytic-model-extraction
📝 Ta fiche : `articles/fiche_03_carlini2020_crypto.md`

---

## Les 4 idées à emporter (ce que tu dois pouvoir expliquer dimanche soir)

1. **Un réseau ReLU est affine par morceaux.** L'espace d'entrée est découpé en régions ; dans chaque région le réseau est une simple fonction linéaire.
2. **Point critique** = une entrée où un neurone vaut exactement 0. Là, la **pente de la sortie change brutalement**.
3. **Analogie crypto** : couche linéaire (poids secrets) + ReLU (non-linéarité publique) ≈ chiffrement (clé secrète + S-box). Extraire ≈ cryptanalyse différentielle.
4. **Signature** = le vecteur de poids d'un neurone *à un facteur près*. On la lit dans le **saut de gradient** au point critique.

---

## Plan de lecture (Lundi → Vendredi)

**Lundi (1h30–2h30) — Cadrer.**
Lis l'**abstract** + l'**introduction** (sections 1). Note **3 questions** que tu te poses (ex. « c'est quoi un point critique concrètement ? »). Ne cherche pas encore à tout comprendre.

**Mardi (2h) — L'intuition géométrique.**
Lis la partie sur les *réseaux ReLU comme fonctions affines par morceaux* et les *points critiques*. Appuie-toi sur ta fiche `fiche_03` et sur le chapitre `ressources_DNN/02` (section 2.2). Si l'intuition manque : regarde 20 min de 3Blue1Brown sur les réseaux.

**Mercredi (2h) — La récupération de signature.**
Lis comment on passe du point critique aux poids. **Réécris à la main** l'idée : de part et d'autre du point critique, le gradient de la sortie change, et **ce changement pointe dans la direction du vecteur de poids du neurone**. (Détail ci-dessous.)

**Jeudi (2h) — Le problème du signe.**
Lis pourquoi récupérer le **signe** est difficile (recherche exponentielle). Tu n'as pas besoin de tout maîtriser : retiens juste que c'est *le* verrou, et qu'il sera levé en Séance 2.

**Vendredi (1h30) — Consolider.**
Relis tes notes. Réponds par écrit à tes 3 questions de lundi. Fais l'auto-check plus bas.

---

## L'unique équation à maîtriser cette semaine

Note la sortie du réseau `f(x)` (un scalaire, disons le logit 0). Pour un réseau ReLU, `f` est **affine par morceaux**, donc son gradient `∇f(x)` est **constant par morceaux**.

Quand tu traverses le point critique du neurone *i* (lui seul bascule éteint↔allumé), le gradient fait un **saut** :

```
Δg  =  ∇f(juste après)  −  ∇f(juste avant)     ∝     w_i
```

où `w_i` est le vecteur de poids d'entrée du neurone *i*. Donc :

> **la direction du saut de gradient = la direction des poids du neurone = la signature.**

C'est *toute* l'attaque de la première couche en une ligne. Assure-toi de comprendre pourquoi seul le neurone qui bascule contribue au saut (les autres ne changent pas d'état, donc ne changent pas le gradient).

---

## Pratique (Samedi + Dimanche, 4h/jour)

Tout est déjà prêt dans `TP/code/`. Travaille dans ce dossier.

**Samedi — Voir l'affine par morceaux (TP2).**
```bash
cd TP/code
python3 tp2_points_critiques.py
```
- Observe : le réseau ReLU a des **cassures de pente** nettes = les points critiques.
- **Expérience :** dans `tp2_points_critiques.py`, remplace `activation="relu"` par `activation="sigmoid"` et relance. **Que se passe-t-il ?** (Tu ne trouves plus de cassure nette.) Écris 3 lignes sur ce que ça implique pour ta thèse.

**Dimanche — Récupérer la signature (TP3).**
```bash
python3 tp3_signature.py
```
- Observe : `|cos| = 1.000000` → la direction des poids est récupérée exactement.
- **Expérience :** change la cible en `[4, 5, 5, 2]` (2 couches cachées). La couche 1 se récupère-t-elle toujours ? Note le **nombre de requêtes** utilisé.
- Logue tes **3 métriques** : requêtes, temps, `|cos|`.

---

## Auto-check (réponds sans regarder — vendredi ou dimanche)

1. Pourquoi un réseau ReLU est-il « affine par morceaux » ?
2. Qu'est-ce qu'un point critique, physiquement ?
3. Pourquoi le saut de gradient donne-t-il la direction des poids ?
4. Que manque-t-il à la « signature » pour avoir le poids complet ? (2 choses)
5. En une phrase : pourquoi cette attaque échouerait-elle sur une sigmoïde ?

> Si tu réponds aux 5, la Séance 1 est **acquise**. Sinon, relis la section correspondante — c'est normal, ça se construit.

---

## Livrable de la séance (pour toi, et pour ta prochaine réunion d'encadrement)

- Tes **notes de lecture** (1 page) + les réponses aux 3 questions de lundi.
- Le **résultat de TP3** : `|cos|` obtenu + nombre de requêtes, sur le réseau à 2 couches.
- **5 lignes** : « ce que j'ai compris cette semaine + ce qui reste flou ».

## Écueils fréquents

- Vouloir tout comprendre du signe (jeudi) : **non**, c'est pour la Séance 2. Reste sur la signature.
- Sauter la pratique : c'est là que ça rentre vraiment. Le week-end est ton moment clé.
- Oublier le float64 : `common.py` le fait déjà pour toi, mais garde le réflexe.

---

**Quand tu as fini :** coche S1 dans `PROGRAMME_GUIDE.md`, note tes blocages, et demande-moi la **Séance 2** (le signe + le dépôt de Foerster). Bon courage — tu poses cette semaine la première pierre. 🧱
