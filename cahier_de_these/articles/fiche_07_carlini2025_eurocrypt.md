# Fiche 07 — Carlini, Chávez-Saab, Hambitzer, Rodríguez-Henríquez, Shamir (EUROCRYPT 2025)

**Réf.** N. Carlini, J. Chávez-Saab, A. Hambitzer, F. Rodríguez-Henríquez, A. Shamir, *Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting*, EUROCRYPT 2025, LNCS 15601, p. 364–396. **Best Paper.**

> Rend le **hard-label polynomial en temps** (Chen 2024 était exponentiel). Pierre angulaire du sous-domaine. **À lire en 5ᵉ.**

## Problème

Chen 2024 prouve la faisabilité du hard-label mais en **temps exponentiel**. Peut-on extraire en **temps ET requêtes polynomiaux** des DNN profonds, en n'observant que le label ?

## Idée clé

Nouvelles techniques pour exploiter efficacement les **frontières de décision** : reconstruire la géométrie des hyperplans cachés couche par couche sans énumération exponentielle, en propageant l'information des couches déjà extraites.

## Méthode

- Identification massive de points de la frontière de décision.
- Récupération des paramètres **couche par couche** avec contrôle de la précision.
- **Restriction importante** : la méthode récupère **toutes les couches *sauf* la couche de sortie**, car cette dernière n'a pas de ReLU exploitable de la même façon.

## Résultats

- Extraction hard-label en **temps polynomial**, applicable à des DNN à **grand nombre de paramètres**.
- Établit le nouvel **état de l'art** du hard-label.

## Limites

- **La couche de sortie n'est pas récupérée** → c'est exactement le trou que **ton article fondateur (fiche 08) comble**.
- ReLU.
- Hypothèses de « généricité » dont la validité en grande profondeur sera questionnée par Ito 2025 (fiche 09).

---

## Lien avec ta thèse

- **Article-parent direct** de ton sujet : Canales-Martínez & Santos 2025 part de ce résultat et récupère la couche de sortie manquante.
- Comprendre **pourquoi** la couche de sortie résiste (absence de ReLU) ouvre directement sur la question « au-delà de ReLU » : sans ReLU, les outils standards tombent.
- À lire **en parallèle** de la fiche 08.

## Mes notes
<!-- AAAA-MM-JJ : … -->
