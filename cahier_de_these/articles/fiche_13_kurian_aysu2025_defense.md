# Fiche 13 — ★ DÉFENSE ★ — Kurian & Aysu (NeurIPS 2025)

**Réf.** A. Kurian, A. Aysu (North Carolina State University), *Train to Defend: First Defense Against Cryptanalytic Neural Network Parameter Extraction Attacks*, NeurIPS 2025. arXiv 2509.16546 ; OpenReview `xk9GSBCfcn`.

> **La première défense** contre l'extraction cryptanalytique. Le volet « défenses » de ton sujet repose largement dessus — champ quasi vierge, donc gros potentiel.

## Problème

Toutes les attaques précédentes prospèrent ; **aucune défense dédiée** n'existait. Peut-on **entraîner** un réseau pour qu'il **résiste** à l'extraction de paramètres, **sans surcoût à l'inférence** et sans perdre en précision ?

## Idée clé — « éliminer l'unicité des neurones »

Observation centrale : l'attaque récupère la **magnitude des poids** d'autant plus facilement que les neurones d'une couche sont **distincts** (uniques). Donc : **forcer les neurones d'une même couche à se ressembler** casse le levier de l'attaque.

## Méthode — entraînement « conscient de l'extraction »

- Ajout d'un **terme de régularisation** à la fonction de perte : il **minimise la distance entre les poids des neurones d'une même couche** (contrainte de similarité intra-couche).
- **Zéro surcoût à l'inférence** (tout se passe à l'entraînement ; même architecture).
- Optimisations pour limiter la perte de précision.
- **Cadre théorique** pour quantifier la **probabilité de succès** de l'attaque en fonction de la similarité intra-couche.

## Résultats

- Perte de précision **< 1 %** (réentraînement même architecture).
- Réseaux non protégés extraits en **14 min à 4 h** ; réseaux protégés **résistent** sur des durées prolongées.

## Limites

- Évalué surtout contre des attaques **ReLU** existantes.
- La similarité intra-couche peut être **détectée/contournée** par des attaques adaptatives (course attaque/défense ouverte).
- Compromis précision/sécurité encore peu exploré en profondeur.

---

## Lien avec ta thèse — la moitié « défenses »

- **Brique centrale** du volet défense. Questions directes : cette défense **tient-elle pour les activations non-ReLU** (PReLU, GELU…) ? La similarité intra-couche a-t-elle le même effet quand l'activation change la géométrie ?
- Si la fiche 10/11 montre que **le signe est plus facile** hors ReLU, alors **les défenses doivent être repensées** hors ReLU → contribution.
- Piste : concevoir une **défense spécifique aux activations lisses**, ou une **attaque adaptative** qui bat « Train to Defend ».
- **À reproduire** : TP7 (réentraîner avec la régularisation de similarité, mesurer l'effet sur l'attaque de TP3/TP4).

## Mes notes
<!-- AAAA-MM-JJ : … -->
