# Fiche 05 — Foerster, Mullins, Shumailov, Hayes (NeurIPS 2024)

**Réf.** H. Foerster, R. Mullins, I. Shumailov, J. Hayes, *Beyond Slow Signs in High-fidelity Model Extraction*, NeurIPS 2024, vol. 37, p. 19496–19522.

> Première implémentation **end-to-end** réellement efficace, combinant signature (Carlini 2020) et signe (Canales-Martínez 2024), avec de fortes optimisations. **Référence d'ingénierie pour tes TP.**

## Problème

Les attaques précédentes étaient publiées surtout « sur le papier » ou en preuve de concept partielle. En pratique, la **récupération de signe reste l'étape lente** (« slow signs »). Peut-on construire une chaîne complète et **optimiser** le maillon lent ?

## Idée clé

Réunir les deux briques (signature + signe) en un **pipeline unique**, puis attaquer le goulot d'étranglement : réduire le coût de la récupération de signe par de meilleures heuristiques de sélection de requêtes et un meilleur usage de l'information déjà extraite.

## Méthode

- Pipeline complet : extraction couche par couche, signature puis signe, avec gestion de la **précision numérique** et de la propagation d'erreur en profondeur.
- Optimisations de la phase de signe → gain de vitesse rapporté **≈ 14,8×** par rapport à l'approche directe.

## Résultats

- Extraction **haute-fidélité** end-to-end sur des réseaux plus grands, en temps réduit.
- Met en évidence les vrais coûts pratiques (stabilité numérique, nombre de requêtes effectif).

## Limites

- Raw-output, ReLU.
- La fidélité se dégrade avec la profondeur (accumulation d'erreurs).

---

## Lien avec ta thèse

- **Modèle d'ingénierie** : c'est la base de code/algorithme la plus propre à imiter pour tes propres expériences « beyond ReLU ».
- L'analyse fine de la **propagation d'erreur** est directement transposable : avec des activations lisses (GELU/SiLU), la sensibilité numérique change — à étudier.
- À garder comme **baseline de performance** quand tu mesureras tes propres attaques.

## Mes notes
<!-- AAAA-MM-JJ : … -->
