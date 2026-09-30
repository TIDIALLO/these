# Fiche 16 — Extraction cryptanalytique de réseaux convolutifs (2026)

**Réf.** *Cryptanalytic Extraction of Convolutional Neural Networks*, Cryptology ePrint Archive 2026/139 (2026).

> Pousse l'extraction au-delà des MLP, vers les **CNN**. Pertinent si ta thèse veut viser des architectures « vision » réalistes.

## Problème

Les attaques d'extraction matures ciblent des **MLP fully-connected**. Les **CNN** (cœur de la vision par ordinateur) ont des **poids partagés** (filtres convolutifs) et une structure très différente. Peut-on les extraire avec les outils cryptanalytiques ?

## Idée clé

Adapter les notions de signature / points critiques à la **structure convolutive** : exploiter le **partage de poids** (un filtre réutilisé sur toute l'image) qui, paradoxalement, **réduit** le nombre de paramètres indépendants à récupérer et fournit des **contraintes redondantes** exploitables.

## Méthode / résultats

- Reformulation de l'extraction tenant compte de la **convolution** et du **pooling**.
- Extraction de filtres de couches convolutives.

## Limites

- Pooling/strides et non-linéarités compliquent l'analyse.
- Souvent encore restreint à des activations classiques (ReLU).

---

## Lien avec ta thèse

- Axe d'extension « architectures » complémentaire de ton axe « activations ».
- Combiner **CNN + activations non-ReLU** (ex. GELU dans des CNN modernes) serait une **double nouveauté** — ambitieux mais valorisable.
- À garder en réserve : utile si un jury demande la portée « vision réelle » de tes méthodes.

## Mes notes
<!-- AAAA-MM-JJ : … -->
