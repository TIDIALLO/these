# Fiche 14 — Jagielski, Carlini, Berthelot, Kurakin, Papernot (USENIX 2020)

**Réf.** M. Jagielski, N. Carlini, D. Berthelot, A. Kurakin, N. Papernot, *High Accuracy and High Fidelity Extraction of Neural Networks*, USENIX Security 2020, p. 1345–1362.

> Donne le **vocabulaire des objectifs d'attaque** : *fidélité* vs *précision (accuracy)*. Cadre conceptuel à citer dès l'intro.

## Problème

« Voler un modèle » peut vouloir dire deux choses différentes. Lesquelles, et que peut-on atteindre selon le budget de requêtes ?

## Idée clé — deux objectifs distincts

- **Extraction haute *précision* (accuracy)** : obtenir un modèle qui **performe aussi bien** que la cible sur la tâche (peu importe qu'il calcule la même fonction). Approche type « apprentissage par distillation » sur des requêtes.
- **Extraction haute *fidélité* (fidelity)** : obtenir un modèle qui **reproduit exactement la fonction** de la cible (mêmes erreurs comprises). C'est l'objectif des attaques **cryptanalytiques** (fiches 03–12).

## Méthode / résultats

- Méthodes d'apprentissage (requêtes + réentraînement) pour la haute précision.
- Première étape « fonctionnellement fidèle » exploitant la structure ReLU (préfigure Carlini 2020).
- Analyse des **compromis budget de requêtes / qualité**.

## Limites

- L'extraction par apprentissage ne donne pas l'**équivalence exacte** ; la partie cryptanalytique y est embryonnaire.

---

## Lien avec ta thèse

- **Cadre les objectifs** : tes attaques visent la **haute fidélité / équivalence fonctionnelle**. Le dire explicitement situe ta contribution.
- Utile pour la section « modèles de menace » de ton manuscrit (que voit l'attaquant ? que cherche-t-il ?).

## Mes notes
<!-- AAAA-MM-JJ : … -->
