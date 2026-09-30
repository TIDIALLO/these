# Fiche 10 — ★ CŒUR DE LA THÈSE ★ — Qi, Lei, Wei, Sun, Wang (ToSC 2026)

**Réf.** X. Qi, H. Lei, L. Wei, X. Sun, M. Wang, *Cryptanalytic Extraction of Neural Networks with Various Activation Functions*, IACR Transactions on Symmetric Cryptology (ToSC), 2026. eprint IACR 2026/178.

> **L'article le plus directement aligné sur ton sujet.** Premier **cadre systématique** d'extraction selon la **catégorie d'activation**, au-delà de ReLU.

## Problème

L'extraction cryptanalytique s'est presque toujours limitée à **ReLU** (et un peu à PReLU). Or les réseaux réels utilisent des activations variées. **Comment adapter l'attaque à chaque famille d'activation ?**

## Idée clé

Classer les activations par **propriétés géométriques** (par morceaux linéaires vs lisses, bornées vs non bornées, différentiables ou non) et concevoir, **pour chaque catégorie**, la bonne stratégie de récupération de signature et de signe. L'attaque n'est plus « une recette ReLU » mais un **cadre paramétré par l'activation**.

## Méthode / activations couvertes

Étend l'extraction à, en plus de ReLU et PReLU :

- **Leaky ReLU** — par morceaux linéaire, pente non nulle des deux côtés.
- **HardTanh** — par morceaux linéaire, **bornée** (deux coudes).
- **ELU** — lisse pour x<0, linéaire pour x>0.
- **Step (fonction échelon)** — discontinue, cas extrême.

Pour chaque famille : où sont les « points informatifs », comment lire la signature, et **comment récupérer le signe** (souvent plus facile que pour ReLU).

## Résultats

- Récupération de paramètres **fonctionnellement équivalente** pour ces activations.
- Constat clé : **pour plusieurs activations, la signature/le signe se récupèrent plus FACILEMENT que pour ReLU** (la dissymétrie ou les bornes donnent plus d'information).
- Possibilité d'**identifier l'activation** quand elle n'est pas connue publiquement.

## Limites

- Cadre **raw-output** (pas encore hard-label) → **trou à combler par toi**.
- Chaque catégorie traitée séparément ; pas de théorie unifiée complète.
- Activations lisses « pures » (sigmoïde/tanh classiques sur tout le réseau) moins centrales que dans la fiche 11.

---

## Lien avec ta thèse — central

- **C'est ton état de l'art principal.** Ta thèse = prolonger ce cadre **vers le hard-label** (combiner fiches 07–08 et fiche 10) et/ou vers les **défenses** (fiche 13).
- Idée de contribution forte : **« extraction hard-label pour activations non-ReLU »** n'existe pas encore → niche claire.
- Le constat « signe plus facile hors ReLU » est une **observation à exploiter** : peut-être que tes attaques non-ReLU battent les attaques ReLU en vitesse.
- **À reproduire** : TP6 (identification d'activation + signature pour Leaky ReLU / ELU).

## Mes notes
<!-- AAAA-MM-JJ : … -->
