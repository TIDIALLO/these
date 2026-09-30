# Fiche 02 — Approches théoriques de l'apprentissage (Daniely & Granot, Lin et al.)

> Deux références « satellites » : l'une sur l'extraction *exacte* par la théorie de l'apprentissage, l'autre sur ce que les réseaux *fully-connected* peuvent faire. À connaître, pas prioritaires.

## Daniely & Granot (ICLR 2023)

**Réf.** A. Daniely, E. Granot, *An Exact Poly-Time Membership-Queries Algorithm for Extracting a Three-Layer ReLU network*, ICLR 2023.

**Idée clé.** Algorithme à **requêtes d'appartenance** qui extrait **exactement** (pas seulement approximativement) un réseau ReLU à **3 couches** en **temps polynomial**, avec des garanties théoriques propres. Complémentaire de l'approche « cryptanalytique » : même but, outils de **théorie de l'apprentissage**.

**Lien thèse.** Donne un cadre de **garanties exactes** ; utile pour formuler proprement tes propres résultats (équivalence fonctionnelle vs extraction exacte).

## Lin, Memisevic, Konda (2015)

**Réf.** Z. Lin, R. Memisevic, K. R. Konda, *How far can we go without convolution: Improving fully-connected networks*, arXiv 1511.02580.

**Idée clé.** Montre que des réseaux **entièrement connectés** (le type de réseau que ciblent les attaques d'extraction) peuvent atteindre de bonnes performances sans convolution. Justifie que l'étude des **MLP** n'est pas qu'académique.

**Lien thèse.** Rappelle que tes attaques sur **MLP fully-connected** ont une portée réelle ; cadre les hypothèses d'architecture.

## Mes notes
<!-- AAAA-MM-JJ : … -->
