# Articles à écrire — plan de publication

> Ta thèse vise **3 articles principaux** (un par contribution), plus des sorties secondaires possibles. Chaque article ci-dessous a : titre provisoire, question, contributions, ce qu'il faut produire, venue cible, et lien avec le cahier (fiches `f.XX`, idées `BXX`, TP).

**Stratégie d'ensemble.** Tu enchaînes : (1) une **attaque** qui ouvre un trou clair (hard-label beyond ReLU), (2) une **défense** dans le même cadre (champ vierge), (3) une **synthèse/taxonomie** qui devient l'ossature du manuscrit. Les trois se renforcent : l'attaque motive la défense, la taxonomie cadre les deux.

---

## Article 1 — « Hard-Label Extraction Beyond ReLU » ⭐ priorité

**Statut :** contribution principale, la plus neuve et la plus défendable.

**Question de recherche.** Peut-on étendre l'extraction de paramètres au cadre **hard-label** pour des activations **autres que ReLU** — d'abord par morceaux (PReLU, Leaky, ELU), puis lisses (sigmoïde, GELU) ?

**Trou exploité.** Les attaques *beyond ReLU* existantes (Qi 2026 `f.10`, NTU 2026 `f.11`, PReLU `f.12`) sont **toutes en raw-output**. Les attaques hard-label (`f.06–08`) sont **toutes ReLU**. **Personne n'a croisé les deux.**

**Contributions visées.**
1. Une attaque hard-label pour les activations **par morceaux non-ReLU** : adapter la détection de points de transition quand le coude n'est pas en 0, estimer la pente α, exploiter la « fuite de signe » bilatérale (idée `B1`).
2. Une première approche pour les activations **lisses** en hard-label : substituer aux dérivées d'ordre supérieur une mesure de **courbure de la frontière de décision** (idée `B2`).
3. Évaluation : nombre de requêtes, temps, erreur de fidélité, sur réseaux jouets puis CIFAR-10.

**À produire.** Algorithme + preuve(s) de complexité + code (extension de TP5/TP6) + expériences. **Baselines** : reproduire `f.07`/`f.08` (ReLU) pour comparer.

**Venue cible.** EUROCRYPT / ASIACRYPT / IACR ToSC (côté crypto) ; à défaut NeurIPS/ICLR (côté ML sécurité).

**Échéance (cf. PLAN.md).** Rédaction & soumission **mois 22–24**.

**Risque & repli.** Si le cas lisse (contribution 2) est trop dur, l'article tient sur le cas par morceaux (contribution 1) + la taxonomie partielle.

---

## Article 2 — « Defending Non-ReLU Networks against Cryptanalytic Extraction »

**Statut :** volet défense, champ quasi vierge (une seule défense existe : `f.13`).

**Question de recherche.** La défense « Train to Defend » (similarité intra-couche) **résiste-t-elle hors ReLU** ? Sinon, peut-on concevoir une défense **adaptée aux activations lisses** ?

**Contributions visées.**
1. Étude empirique : appliquer la régularisation de similarité (`f.13`) à des réseaux PReLU/ELU/GELU, mesurer l'effet réel sur l'attaque (idée `B6`).
2. Nouvelle défense pour le cas lisse : entraîner pour **aplatir/égaliser la courbure** de la frontière de décision (analogue lisse de la similarité de poids) — idée `B7`.
3. **Attaque adaptative** contre ta propre défense (idée `B9`) pour montrer ses limites honnêtement → renforce la crédibilité.
4. Compromis sécurité/précision quantifié.

**À produire.** Implémentation PyTorch des régularisations (TP7-bis), protocole d'évaluation attaque-vs-défense, courbes précision/résistance.

**Venue cible.** NeurIPS / ICLR / IEEE S&P / USENIX Security (côté ML & sécurité systèmes).

**Échéance.** Rédaction & soumission **mois 28–31**.

**Dépendance.** S'appuie sur les attaques de l'Article 1 (pour évaluer la défense).

---

## Article 3 — « A Taxonomy of Activation Functions for Model Extraction »

**Statut :** synthèse théorie + mesure ; devient l'ossature du chapitre « état de l'art enrichi » du manuscrit.

**Question de recherche.** Quelles propriétés d'une activation rendent l'extraction **plus facile ou plus difficile** ? Les bornes « polynomiales » annoncées **tiennent-elles** en profondeur réelle ?

**Contributions visées.**
1. **Taxonomie** des activations (par morceaux vs lisse, bornée vs non bornée, symétrique vs asymétrique) reliée à un **coût d'extraction mesuré** (idée `B3`).
2. **Audit de complexité** façon Ito 2025 (`f.09`) appliqué aux activations non-ReLU : mesurer le **nombre de requêtes effectif** vs profondeur (idée `B10`).
3. Étude de la **précision numérique** des attaques lisses (dérivées d'ordre supérieur instables) — idée `B11`.

**À produire.** Cadre formel + grande campagne expérimentale (toutes les activations de `common.py`) + tableaux théorie/pratique.

**Venue cible.** IACR ToSC / Journal (IEEE TIFS, journal de cryptologie) — format plus long adapté à une synthèse.

**Échéance.** Rédaction **mois 31–33**, puis intégration au manuscrit.

---

## Sorties secondaires possibles (selon le temps / les résultats)

- **S1 — « Activation Fingerprinting in the Hard-Label Setting » (idée B4).** Identifier l'activation inconnue d'après la **géométrie de la frontière** (anguleuse → ReLU/PReLU ; arrondie → GELU/sigmoïde). Peut être un *workshop paper* ou une section d'Article 1. Démonstration déjà amorcée dans TP6.
- **S2 — Extension CNN (idées B12/B16).** Croiser extraction CNN (`f.16`) et activations non-ReLU. Ambitieux ; plutôt une perspective de fin de thèse.
- **S3 — Note méthodologique reproductibilité.** Publier `common.py` + TP comme **banc d'essai open-source** d'attaques d'extraction multi-activations → visibilité + service à la communauté.

---

## Tableau de bord des publications

| Art. | Titre court | Idées | Venue cible | Soumission (mois) | Statut |
|------|-------------|-------|-------------|-------------------|--------|
| 1 | Hard-Label Beyond ReLU | B1, B2, B3 | EUROCRYPT/ToSC | 22–24 | 🟡 premier prototype (TP8, 2026-09-29) |
| 2 | Defending Non-ReLU | B6, B7, B9 | NeurIPS/S&P | 28–31 | ☐ |
| 3 | Taxonomy for Extraction | B3, B10, B11 | ToSC/journal | 31–33 | ☐ |
| S1 | Activation Fingerprinting | B4 | workshop | flexible | ☐ |
| S2 | CNN beyond ReLU | B12 | — | perspective | ☐ |
| S3 | Banc d'essai open-source | — | dépôt + note | continu | ☐ |

---

## Méthode de rédaction (conseils de directeur)

1. **Écris l'abstract et l'intro AVANT les expériences finales** : ça clarifie ta contribution et évite de partir dans tous les sens.
2. **Une figure « money plot » par article** : pour l'Article 1, ce sera probablement « requêtes vs profondeur, par activation ».
3. **Toujours une baseline ReLU** reproduite : sans comparaison, un reviewer crypto rejettera.
4. **Sois honnête sur les limites** (cf. Ito 2025) : les reviewers crypto détestent les bornes « polynomiales » qui cachent une constante exponentielle.
5. **Construis ta BibTeX** au fil de l'eau depuis `articles/00_index_bibliographie.md`.
6. **Reproductibilité** : publie le code (seeds fixés). C'est un argument fort en review.

*Dernière mise à jour : 2026-06-18.*
