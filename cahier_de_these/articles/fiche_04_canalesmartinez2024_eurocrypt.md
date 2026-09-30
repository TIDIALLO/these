# Fiche 04 — Canales-Martínez et al. (EUROCRYPT 2024)

**Réf.** I. A. Canales-Martínez, J. Chávez-Saab, A. Hambitzer, F. Rodríguez-Henríquez, N. Satpute, A. Shamir, *Polynomial Time Cryptanalytic Extraction of Neural Network Models*, EUROCRYPT 2024, LNCS 14653, p. 3–33. eprint IACR 2023/1526.

> Résout le **problème du signe** de Carlini 2020 : l'extraction raw-output devient **entièrement polynomiale**. **À lire en 2ᵉ, reproduire en TP4.**

## Problème

Dans Carlini 2020, tout est polynomial sauf la **récupération des signes** des neurones (exponentielle), ce qui empêche d'attaquer les réseaux profonds. Comment rendre cette étape polynomiale ?

## Idée clé

Le signe d'un neurone laisse une **trace mesurable** dans la façon dont la couche suivante réagit. Les auteurs proposent **trois techniques** de récupération de signe en **temps polynomial**, exploitant la géométrie des régions linéaires et le comportement des neurones voisins plutôt qu'une recherche exhaustive.

## Méthode (vue d'ensemble)

- Réutilise la récupération de **signature** (poids à un facteur près) de Carlini 2020.
- Pour le signe : construit des requêtes ciblées qui rendent observable, en sortie, l'effet du signe d'un neurone donné → décision déterministe au lieu d'une énumération 2ⁿ.
- Trois variantes selon la position de la couche et les ressources disponibles (compromis requêtes/temps).

## Résultats

- **Première attaque entièrement polynomiale** (temps **et** requêtes) pour les DNN ReLU en raw-output.
- Démonstration sur des réseaux nettement plus **profonds** (jusqu'à ~8 couches).

## Limites

- Toujours **raw-output** (besoin des logits).
- **ReLU** uniquement.
- Précision et stabilité numérique sensibles à la profondeur.

---

## Lien avec ta thèse

- La **récupération de signe** est le verrou récurrent de tout le domaine. Tes activations « au-delà de ReLU » changent la nature de ce problème : pour certaines (PReLU, ELU), le signe peut être **plus facile** à lire (cf. fiches 10–12) — une piste de contribution.
- Brique réutilisée par Foerster 2024 (end-to-end) et par les attaques hard-label.
- **À reproduire** : TP4 (au moins une des trois techniques de signe, sur petit réseau).

## Mes notes
<!-- AAAA-MM-JJ : … -->
