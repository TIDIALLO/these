# IDEE.md — Idées de recherche

> Objectif du fichier : capturer **tout** ce qui peut devenir une contribution. Deux colonnes mentales : **(A) reproduire l'existant** (pour maîtriser et avoir des baselines) et **(B) faire ce qui n'a pas encore été fait** (tes vraies contributions). Marque chaque idée : 🟢 facile / 🟡 moyen / 🔴 ambitieux.

---

## 0. La phrase-thèse (à affiner)

> « Les attaques d'extraction de paramètres et leurs défenses ont été développées presque exclusivement pour les réseaux **ReLU**. Cette thèse étend attaques **et** défenses aux réseaux **au-delà de ReLU** (activations par morceaux non standard et activations lisses), en particulier dans le cadre réaliste **hard-label**, et caractérise quand l'activation **facilite** ou **entrave** l'extraction. »

Les trois mots-clés qui te distinguent : **au-delà de ReLU**, **hard-label**, **défenses**.

---

## 1. Cartographie du domaine (où sont les trous ?)

Matrice **Setting × Activation** — chaque case = un sujet. ✅ = traité, ❓ = partiel, ⬜ = **trou (= opportunité)**.

| | ReLU | PReLU / LeakyReLU | ELU / HardTanh / Step | Lisses (GELU, SiLU, SELU, sigmoïde, tanh) |
|---|---|---|---|---|
| **Raw-output (logits)** | ✅ (f.03–05) | ✅ (f.10, f.12) | ✅ (f.10) | ✅ (f.11) |
| **Hard-label (label seul)** | ✅ (f.06–09) | ⬜ **trou** | ⬜ **trou** | ⬜ **GROS trou** |
| **Défenses** | ❓ (f.13, 1 seul papier) | ⬜ | ⬜ | ⬜ |

**Lecture stratégique :** la ligne « hard-label » et la ligne « défenses » sont presque vides hors ReLU. **C'est là que se trouvent tes contributions les plus sûres.**

---

## 2. Idées — (A) Reproduire l'existant (fondations, ~6 premiers mois)

Indispensable : sans baselines reproductibles, pas de comparaison crédible.

- **A1** 🟢 Reproduire la **signature** d'un neurone ReLU (Carlini 2020, f.03) → TP3.
- **A2** 🟡 Reproduire au moins **une technique de récupération de signe** polynomiale (Canales-Martínez 2024, f.04) → TP4.
- **A3** 🟡 Reproduire l'extraction **hard-label** sur un petit réseau via **points de transition** (Chen 2024 / Carlini 2025, f.06–07) → TP5.
- **A4** 🔴 Reproduire la **récupération de la couche de sortie** (ton article, f.08) et vérifier l'**équivalence fonctionnelle** sur CIFAR-10.
- **A5** 🟢 Reproduire l'**identification d'activation** et la signature pour **Leaky ReLU / ELU** en raw-output (Qi 2026, f.10) → TP6.
- **A6** 🟡 Reproduire la **défense « Train to Defend »** (régularisation de similarité intra-couche, f.13) et mesurer son effet sur A1–A2 → TP7.

> Règle : chaque reproduction produit (1) un script, (2) une mesure (requêtes, temps, erreur), (3) une note « ce que j'ai appris / ce qui casse ».

---

## 3. Idées — (B) Contributions nouvelles (le cœur de la thèse)

### Axe B-I — Hard-label au-delà de ReLU (priorité n°1)

- **B1** 🔴 **Extraction hard-label pour activations par morceaux non-ReLU** (PReLU, Leaky ReLU, ELU). *Pourquoi neuf :* f.10 est raw-output, f.06–09 sont ReLU. Personne n'a croisé les deux. *Plan :* adapter la détection de **points de transition** quand le coude n'est pas en zéro (PReLU/ELU) ; estimer α en plus.
- **B2** 🔴 **Extraction hard-label pour activations lisses** (sigmoïde, GELU, SiLU). *Le défi central :* en hard-label il n'y a **plus de sortie continue**, donc **plus de dérivées d'ordre supérieur** (l'outil clé de f.11). *Idée :* reconstruire une information de courbure à partir de la **forme de la frontière de décision** (les frontières lisses sont courbes, pas polygonales) → estimer la position de la zone non linéaire par la **courbure locale de la frontière**.
- **B3** 🟡 **Caractérisation : quelles activations rendent le hard-label plus dur/plus facile ?** Produire une **taxonomie** (par morceaux vs lisse, bornée vs non bornée) avec, pour chacune, le coût d'extraction mesuré. *Livrable :* un tableau théorie+expérience qui devient un chapitre.

### Axe B-II — Identification d'activation comme primitive

- **B4** 🟡 **Identifier l'activation inconnue en hard-label.** f.10–11 le font en raw-output. En hard-label : la **géométrie de la frontière** (anguleuse → ReLU/PReLU ; lisse/courbe → GELU/sigmoïde) trahit-elle l'activation ? Construire un **classifieur de frontière**.
- **B5** 🟢 **Empreinte d'activation par side-channel** (lien f.15) : la consommation/temps diffèrent selon l'activation → identification physique. Volet « systèmes » optionnel.

### Axe B-III — Défenses au-delà de ReLU (priorité n°2)

- **B6** 🔴 **La défense « Train to Defend » (f.13) tient-elle hors ReLU ?** Tester la régularisation de similarité intra-couche sur des réseaux PReLU/GELU. Hypothèse : l'effet protecteur **dépend de l'activation** → soit ça marche moins bien (et il faut autre chose), soit ça marche mieux (et pourquoi).
- **B7** 🔴 **Concevoir une défense spécifique aux activations lisses.** Ex. : entraîner pour **aplatir la courbure** de la frontière de décision (rendre la zone non linéaire « illisible »), ou **égaliser les courbures** entre neurones (analogue lisse de la similarité de poids).
- **B8** 🟡 **Défense par activation « piège ».** Concevoir/entraîner avec une activation **conçue pour résister** (zones non linéaires multiples, ambiguës) tout en gardant la précision. Compromis sécurité/précision à quantifier.
- **B9** 🟡 **Attaque adaptative contre B6/B7** (course attaque-défense) : montrer les limites de ta propre défense → renforce la crédibilité.

### Axe B-IV — Robustesse & honnêteté des bornes

- **B10** 🟡 **Rejouer la critique d'Ito 2025 (f.09) pour les activations non-ReLU** : les bornes « polynomiales » des attaques beyond-ReLU tiennent-elles en profondeur réelle ? Mesurer le **nombre de requêtes effectif** vs profondeur. Souvent négligé → contribution méthodologique solide.
- **B11** 🟡 **Étude de la précision numérique** des attaques lisses (les dérivées d'ordre supérieur explosent le bruit). Quantifier la fidélité atteignable.

### Axe B-V — Extensions architecture (réserve / ambitieux)

- **B12** 🔴 **CNN à activations non-ReLU** (croiser f.16 et f.10–11). Double nouveauté, gros effort.
- **B13** 🟡 **Couches contractives + activations non-ReLU** : l'astuce de ton article fondateur (f.08) accélère-t-elle aussi hors ReLU ?

---

## 4. Top 3 « contributions cibles » (si tu ne devais en retenir que 3)

1. **B1 + B2** → un papier « **Hard-Label Extraction Beyond ReLU** ». *Le plus aligné, le plus neuf.*
2. **B6 + B7** → un papier « **Defending Non-ReLU Networks against Cryptanalytic Extraction** ». *Le volet défense, champ vierge.*
3. **B3 + B10** → un papier « **A Taxonomy of Activation Functions for Model Extraction (theory + measured complexity)** ». *Synthèse qui structure tout ton manuscrit.*

Ces trois forment naturellement les **trois chapitres de contribution** de la thèse (cf. PLAN.md).

---

## 5. Questions ouvertes à garder sous les yeux

- En hard-label, **sans sortie continue**, quel **substitut** aux dérivées d'ordre supérieur ? (courbure de frontière ? requêtes structurées ?)
- L'observation « **le signe est plus facile hors ReLU** » (f.10–11) : se transfère-t-elle au hard-label ?
- Une **défense unique** peut-elle couvrir toutes les activations, ou faut-il une défense **par famille** ?
- Les **bornes polynomiales** annoncées sont-elles honnêtes en profondeur réelle ? (méthodo f.09)
- Quel **modèle de menace** fixes-tu ? (oracle pur / + side-channel / + architecture connue ou non)

## 6. Hygiène de recherche (à t'appliquer)

- Tout résultat doit être **reproductible** (seed fixé, script versionné).
- Toujours rapporter **3 métriques** : nb de requêtes, temps réel, erreur de fidélité (max |Δ| sur les poids ou sur les sorties).
- Tenir un **journal de manips** daté.
- Vérifier l'**équivalence fonctionnelle** sur un jeu de test indépendant, pas seulement la proximité des poids.

*Dernière mise à jour : 2026-06-18. Ajoute tes idées datées ci-dessous.*

## Mes idées (journal)
<!-- AAAA-MM-JJ : … -->

- **2026-09-29 — première implémentation B1 (TP8, `tp8_hardlabel_beyond_relu.py`).**
  Nouvelle primitive absente de TP5/TP6 : balayage de droites voisines +
  détection de pli (moindres carrés en deux segments) + ajustement
  d'hyperplan robuste (rejet d'outliers façon RANSAC léger). Testée sur
  ReLU/LeakyReLU/PReLU/ELU, dim=3, hidden=4, out=2.
  **Ce qui marche :** direction du neurone récupérée en hard-label pur avec
  |cos| ~0.98 pour ReLU et Leaky ReLU (rayon local serré) — donc la brique
  manquante pour rejouer f.07/f.08 existe et généralise sans changement de
  code à Leaky ReLU/PReLU (accès uniquement au label, aucune hypothèse
  ReLU-spécifique dans le code).
  **Ce qui casse :** (1) compromis net rayon local / taux de récupération /
  nombre de requêtes (x10–x100 selon le rayon) — pas encore caractérisé
  formellement, candidat pour la money plot de l'Article 1 ; (2) PReLU
  (α=0.25) systématiquement moins bon que Leaky ReLU (α=0.1) à réglages
  égaux — hypothèse : le seuil de détection du pli devient plus dur à
  satisfaire quand (1-α) est petit ; (3) ELU jamais récupéré dans ces
  réglages (la partie négative n'est pas linéaire, la frontière y est
  courbe dès qu'on s'écarte du germe) ; (4) le "ratio de pente bilatéral"
  (piste α) reste un indicateur empirique, pas un estimateur — la dérivation
  à la main montre qu'il mélange α du neurone et le jacobien local du reste
  du réseau (même ambiguïté d'échelle que Carlini 2020 pour le signe).
  **Probe B2 (même script) :** mesure de courbure de frontière calculée
  uniquement en hard-label distingue déjà par morceaux (concentration
  ~0.86–1.0) vs lisse (~0.24–0.33) — signal de faisabilité, pas encore une
  récupération de poids.
  **Prochaine étape :** caractériser le compromis rayon/requêtes/précision
  sur plusieurs seeds réseau (un seul seed testé ici) avant de considérer la
  contribution 1 "solide" ; creuser pourquoi ELU échoue systématiquement.
