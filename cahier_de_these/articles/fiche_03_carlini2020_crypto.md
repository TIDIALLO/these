# Fiche 03 — Carlini, Jagielski, Mironov (CRYPTO 2020)

**Réf.** N. Carlini, M. Jagielski, I. Mironov, *Cryptanalytic Extraction of Neural Network Models*, CRYPTO 2020, LNCS 12172, p. 189–218. eprint : à retrouver via la conférence.

> **L'article qui lance le domaine.** Il reformule l'extraction de réseau comme une attaque cryptanalytique différentielle. **À lire en premier et à reproduire (TP3/TP4).**

## Problème

Un attaquant a un **accès oracle** à un DNN (architecture connue, activations ReLU) et peut lire la **sortie brute** (logits, raw-output). Peut-il récupérer **tous les poids et biais** avec un nombre raisonnable de requêtes et en temps raisonnable ?

## Idée clé (l'analogie cryptographique)

Un DNN ReLU est une **fonction affine par morceaux** : l'espace d'entrée est découpé en régions linéaires. Aux frontières entre régions, un neurone passe « exactement par zéro » avant activation : ce sont les **points critiques**. La structure « couche linéaire secrète + non-linéarité publique (ReLU) » est **analogue à un chiffrement par blocs** (clé de tour secrète + S-box publique). L'attaque ressemble alors à de la **cryptanalyse différentielle**.

## Méthode

1. **Trouver des points critiques** d'un neurone (entrées où sa pré-activation = 0), en détectant les cassures de pente de la sortie.
2. **Récupérer la *signature* du neurone** (le vecteur de poids *à un facteur multiplicatif près*) en mesurant les dérivées secondes de part et d'autre du point critique → résolution d'un système linéaire.
3. **« Peeling » couche par couche** : une fois la couche 1 récupérée, on la « pèle » et on recommence sur la couche 2, etc.
4. **Récupération du signe** de chaque neurone (le facteur multiplicatif peut être positif ou négatif) : ici par **recherche exhaustive** → **coût exponentiel**.

## Résultats

- Extraction **fonctionnellement équivalente** de réseaux ReLU avec une **précision quasi-flottante** (jusqu'à ~2⁻³⁰ d'erreur).
- Nombre de requêtes **polynomial** en le nombre de neurones.
- Réseaux jusqu'à ~3 couches en pratique.

## Limites

- La **récupération de signe est exponentielle** → bloque le passage aux réseaux profonds.
- Suppose l'accès aux **logits bruts** (peu réaliste : les API renvoient souvent juste un label).
- ReLU uniquement.

---

## Lien avec ta thèse

- **Socle technique** : « points critiques », « signature », « peeling », « équivalence fonctionnelle » — tu réutiliseras ces notions partout.
- Les deux faiblesses (signe exponentiel + besoin des logits) sont précisément ce que les articles suivants corrigent ; ta thèse hérite de ce fil.
- **À reproduire** : TP3 (signature d'un neurone) et TP4 (problème du signe). C'est la brique de base de toutes tes expériences.

## Mes notes
<!-- AAAA-MM-JJ : … -->
