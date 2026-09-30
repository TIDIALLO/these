# Ressources d'apprentissage — Année 1

> Ressources **vérifiées** pour reproduire & maîtriser. Classées par besoin, avec un **ordre de priorité**. Tu es dev Python : privilégie les ressources « code en main ». Ne lis pas tout — suis le parcours minimal, approfondis selon tes trous.

## Parcours minimal (si tu n'as que peu de temps)

1. Karpathy — *Neural Networks: Zero to Hero* (vidéos, tu codes un réseau de zéro).
2. PyTorch — tutoriels officiels (60-min Blitz).
3. Les 2 dépôts de code des attaques (ci-dessous) : les faire tourner.
4. Carlini 2020 (arXiv 2003.04884) lu en profondeur + son dépôt.

Le reste est de l'approfondissement.

---

## 1. Bases du deep learning (pour combler le niveau débutant)

| Ressource | Format | Pourquoi | Lien |
|-----------|--------|----------|------|
| **3Blue1Brown — Neural Networks** | Vidéos | Intuition visuelle (gradient, backprop). Le meilleur point de départ. | https://www.3blue1brown.com/topics/neural-networks |
| **A. Karpathy — Neural Networks: Zero to Hero** | Vidéos + code | Tu construis un réseau et la backprop **de zéro** en Python. Idéal pour un dev. | https://karpathy.ai/zero-to-hero.html |
| **Dive into Deep Learning (d2l.ai)** | Livre interactif gratuit | Théorie + code PyTorch exécutable. Référence pour approfondir. | https://d2l.ai/ |
| **fast.ai — Practical Deep Learning** | Cours | Approche top-down, très pratique. | https://course.fast.ai/ |
| **Goodfellow, Bengio, Courville — Deep Learning** | Livre gratuit | LA référence théorique (chapitres 6 = réseaux, 8 = optimisation). | https://www.deeplearningbook.org/ |

## 2. PyTorch (ton outil principal)

| Ressource | Pourquoi | Lien |
|-----------|----------|------|
| **Tutoriels officiels PyTorch** | « 60 Minute Blitz » puis « Learn the Basics ». | https://pytorch.org/tutorials/ |
| **Documentation `torch.autograd`** | Comprendre le calcul automatique de dérivées (cœur des attaques). | https://pytorch.org/docs/stable/autograd.html |
| **Documentation des activations `torch.nn`** | `ReLU, PReLU, ELU, GELU, SiLU, Sigmoid, Hardtanh, LeakyReLU`. | https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity |

> TensorFlow/Keras : utile **seulement** pour relire le code historique de Carlini (en JAX/NumPy en réalité). Tutoriels : https://www.tensorflow.org/tutorials

## 3. Maths utiles (au fil de l'eau, pas avant)

| Sujet | Ressource | Lien |
|-------|-----------|------|
| Algèbre linéaire | 3Blue1Brown — *Essence of Linear Algebra* (hyperplans, projections). | https://www.3blue1brown.com/topics/linear-algebra |
| Calcul / dérivées | 3Blue1Brown — *Essence of Calculus*. | https://www.3blue1brown.com/topics/calculus |
| Géométrie des régions linéaires | Montúfar et al., *On the Number of Linear Regions of DNNs* (NeurIPS 2014). | https://arxiv.org/abs/1402.1869 |

## 4. ★ Dépôts de code des attaques (le plus important) ★

| Dépôt | Contenu | Note |
|-------|---------|------|
| **google-research/cryptanalytic-model-extraction** | Code officiel de **Carlini 2020** (JAX/NumPy). Extrait des réseaux fully-connected ReLU. Deps : numpy, scipy, jax, jaxlib, matplotlib, networkx. | À faire tourner en **Phase 1**. https://github.com/google-research/cryptanalytic-model-extraction |
| **hannafoe/cryptanalytical-extraction** | Code de **Foerster 2024** (« Beyond Slow Signs »). **Unifie** Carlini + Canales-Martínez + optimisations de signe (×14,8). Attaque couche par couche. | À faire tourner en **Phase 2**. https://github.com/hannafoe/cryptanalytical-extraction |

> Pour ton article fondateur (Canales-Martínez & Santos 2025) et les papiers beyond-ReLU 2026, le code n'est pas forcément public : **écris aux auteurs** (pratique normale et bien vue). Tes propres TP (`TP/code/`) servent de base reproductible en attendant.

## 5. Les articles à lire (déjà fichés dans `articles/`)

Ordre de lecture en année 1 (détails et résumés dans `articles/00_index_bibliographie.md`) :

1. **Carlini, Jagielski, Mironov 2020** — arXiv 2003.04884 · https://arxiv.org/abs/2003.04884
2. **Canales-Martínez et al. 2024** — eprint 2023/1526 · https://eprint.iacr.org/2023/1526
3. **Foerster et al. 2024** — arXiv 2406.10011 · https://arxiv.org/abs/2406.10011
4. **Chen et al. 2024** (hard-label) — arXiv 2409.11646 · https://arxiv.org/abs/2409.11646
5. **Carlini et al. 2025** (hard-label polynomial) — EUROCRYPT 2025
6. **★ Canales-Martínez & Santos 2025** (ton article) — eprint 2025/1118 · https://eprint.iacr.org/2025/1118
7. **Qi et al. 2026** (various activations) — eprint 2026/178 · https://eprint.iacr.org/2026/178
8. **Kurian & Aysu 2025** (défense) — arXiv 2509.16546 · https://arxiv.org/abs/2509.16546

## 6. Veille (à mettre en place dès maintenant)

| Source | Quoi | Lien |
|--------|------|------|
| **IACR ePrint — Attacks & cryptanalysis** | Les nouveaux papiers crypto (dont extraction). | https://eprint.iacr.org/ |
| **arXiv cs.CR + cs.LG** | Sécurité ML & apprentissage. | https://arxiv.org/list/cs.CR/recent |
| **Page de N. Carlini** | Suit l'un des leaders du domaine. | https://nicholas.carlini.com/papers |
| **OpenReview (NeurIPS/ICLR)** | Côté ML/défense. | https://openreview.net/ |

> Conseil : crée une alerte (Google Scholar ou arXiv) sur « cryptanalytic extraction neural network » et « model extraction activation ».

## 7. Outils & environnement

- **Python 3.10+**, `numpy`, `scipy`, `torch`, `jax`/`jaxlib` (pour le dépôt Carlini), `matplotlib`.
- **Précision** : float64 systématique ; `mpmath` (précision arbitraire) pour les activations lisses.
- **Repro** : `git`, seeds fixés, un fichier `requirements.txt`, journal de manips daté.
- **Calcul** : un GPU n'est pas nécessaire cette année (petits réseaux). CPU suffit.

---

### Comment utiliser ce dossier

Ne le lis pas linéairement. À chaque phase de `ANNEE_1_reproduire_maitriser.md`, reviens ici prendre **la** ressource correspondante (ex. Phase 1 → dépôt Carlini + arXiv 2003.04884). Approfondis les bases (section 1-2) seulement là où tu butes.

*Dernière mise à jour : 2026-06-18.*
