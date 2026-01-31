# Rapport d'Analyse - Thèse de Tidiane DIALLO

## Attaques d'extraction et défenses pour les réseaux de neurones profonds au-delà de ReLU

**Date de génération**: 31 Janvier 2026
**Accompagnement**: Claude (Consultant IA/Cybersécurité)

---

## Table des matières

1. [Informations générales](#1-informations-générales)
2. [Résumé exécutif](#2-résumé-exécutif)
3. [Analyse de la problématique](#3-analyse-de-la-problématique)
4. [Fondements théoriques](#4-fondements-théoriques)
5. [État de l'art détaillé](#5-état-de-lart-détaillé)
6. [Bibliographie annotée](#6-bibliographie-annotée)
7. [Plan de remise à niveau](#7-plan-de-remise-à-niveau)
8. [Opportunités de recherche](#8-opportunités-de-recherche)
9. [Ressources et outils](#9-ressources-et-outils)
10. [Calendrier prévisionnel enrichi](#10-calendrier-prévisionnel-enrichi)

---

## 1. Informations générales

| Élément | Détail |
|---------|--------|
| **Doctorant** | Tidiane DIALLO |
| **Intitulé** | Attaques d'extraction et défenses pour les réseaux de neurones profonds au-delà de ReLU |
| **École doctorale** | École Doctorale Polytechnique Thiès (EDPT) |
| **Directeur** | Pr. Abdoul Aziz Ciss (aaciss@ept.edu.sn) |
| **Laboratoire** | CRISIN'2D / Équipe LTISI |
| **Période** | 2025 – 2028 (3 ans) |
| **Objectif publications** | 3 à 4 articles scientifiques |

### Mots-clés
Deep Learning, réseaux de neurones, fonctions d'activation, ReLU, sécurité des modèles, cryptographie, attaques d'extraction, hard-label, défenses, robustesse.

---

## 2. Résumé exécutif

### 2.1 Contexte

Les réseaux de neurones profonds (DNN) sont massivement déployés via des API (MLaaS - Machine Learning as a Service) dans des domaines critiques: vision par ordinateur, traitement du langage naturel, aide à la décision. Ces modèles représentent un investissement considérable en:
- **Données d'entraînement** (souvent propriétaires)
- **Ressources de calcul** (GPU/TPU pendant des semaines)
- **Expertise humaine** (architecture, hyperparamètres)

### 2.2 Menace

Des travaux récents en cryptographie ont démontré que des **attaques d'extraction de modèles** permettent de reconstruire les paramètres internes d'un DNN en l'interrogeant comme une boîte noire. Ces attaques exploitent la structure mathématique des réseaux, particulièrement ceux utilisant ReLU.

### 2.3 Objectifs de la thèse

**Axe 1 - Vulnérabilité**: Généraliser les attaques d'extraction aux activations modernes (GELU, SiLU/Swish, sigmoïde, tanh) et aux architectures CNN.

**Axe 2 - Protection**: Concevoir des mécanismes de défense efficaces avec un bon compromis sécurité/performance.

---

## 3. Analyse de la problématique

### 3.1 Question centrale

> *Dans quelle mesure est-il possible d'extraire des réseaux de neurones profonds modernes, au-delà des architectures ReLU classiques, en ne disposant que d'un oracle hard-label, et quels mécanismes de défense permettent de réduire efficacement cette vulnérabilité sans dégrader fortement la qualité des prédictions?*

### 3.2 Décomposition en sous-problèmes

#### Volet Vulnérabilité
1. Comment les propriétés mathématiques des activations non-ReLU affectent-elles la faisabilité des attaques?
2. Les techniques basées sur les points critiques peuvent-elles être adaptées aux fonctions non piecewise-linear?
3. Comment extraire des architectures convolutionnelles vs fully-connected?

#### Volet Protection
1. Quel niveau de bruit/randomisation est suffisant pour bloquer les attaques sans dégrader les prédictions?
2. Les techniques de watermarking peuvent-elles détecter les modèles extraits?
3. Comment quantifier le compromis sécurité/performance/coût?

### 3.3 Verrous scientifiques identifiés

```
┌─────────────────────────────────────────────────────────────────────┐
│  VERROU 1: Perte de la linéarité par morceaux                       │
│  ─────────────────────────────────────────────────────────────────  │
│  ReLU divise l'espace en régions linéaires exploitables.            │
│  GELU/SiLU sont des fonctions lisses (C∞) sans discontinuités.      │
│  → Nécessite de nouvelles approches mathématiques                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  VERROU 2: Mode hard-label                                          │
│  ─────────────────────────────────────────────────────────────────  │
│  L'attaquant ne voit que la classe prédite (pas les logits).        │
│  Information très réduite par requête.                              │
│  → Complexité en requêtes significativement plus élevée             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  VERROU 3: Architectures convolutionnelles                          │
│  ─────────────────────────────────────────────────────────────────  │
│  Partage de poids, structure spatiale, pooling.                     │
│  Moins de paramètres mais structure plus complexe.                  │
│  → Adaptation des techniques d'extraction nécessaire                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Fondements théoriques

### 4.1 Les réseaux de neurones comme fonctions mathématiques

Un réseau de neurones profond avec L couches peut s'écrire:

```
f(x) = σL(WL · σL-1(WL-1 · ... σ1(W1 · x + b1) ... + bL-1) + bL)
```

Où:
- `Wi` sont les matrices de poids de la couche i
- `bi` sont les vecteurs de biais
- `σi` sont les fonctions d'activation

### 4.2 Fonctions d'activation - Comparaison

#### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x) = { x  si x > 0
                      { 0  si x ≤ 0
```
**Propriétés**:
- Piecewise linear (linéaire par morceaux)
- Non-différentiable en x=0
- Crée des régions linéaires dans l'espace d'entrée

#### Sigmoid
```
σ(x) = 1 / (1 + e^(-x))
```
**Propriétés**:
- Sortie bornée dans [0, 1]
- Fonction lisse (C∞)
- Gradient vanishing pour |x| grand

#### Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
**Propriétés**:
- Sortie bornée dans [-1, 1]
- Centrée autour de 0
- Fonction lisse (C∞)

#### GELU (Gaussian Error Linear Unit)
```
GELU(x) = x · Φ(x) = x · (1/2)[1 + erf(x/√2)]
```
**Propriétés**:
- Utilisée dans GPT, BERT, ViT
- Approximation: x · σ(1.702x)
- Fonction lisse, non-monotone localement

#### SiLU/Swish
```
SiLU(x) = x · σ(x) = x / (1 + e^(-x))
```
**Propriétés**:
- Utilisée dans EfficientNet, Llama, Llama2
- Auto-gated: la fonction "décide" de son activation
- Fonction lisse (C∞)

### 4.3 L'analogie DNN ↔ Chiffrement par blocs

| Chiffrement par blocs | Réseaux de neurones |
|----------------------|---------------------|
| Texte clair | Entrée x |
| Texte chiffré | Sortie f(x) |
| Clé secrète K | Poids W, biais b |
| S-boxes (non-linéarité) | Fonctions d'activation σ |
| Couches linéaires (MixColumns, etc.) | Couches fully-connected |
| Rounds itératifs | Couches successives |
| Cryptanalyse différentielle | Attaque par points critiques |

### 4.4 Principe des attaques d'extraction sur ReLU

**Observation clé**: Pour un réseau ReLU, l'espace d'entrée est partitionné en régions polytopiques où la fonction est affine.

```
                    Région 2
                   (W2·x + b2)
                  /
                 /
────────────────●────────────────
               /│
              / │
             /  │ Point critique
   Région 1    │ (frontière)
  (W1·x + b1)  │
               │
            Région 3
           (W3·x + b3)
```

**Stratégie d'attaque**:
1. Trouver les **points critiques** (frontières entre régions)
2. Ces points correspondent aux hyperplans où `Wi·x + bi = 0` pour certains neurones
3. Résoudre le système d'équations linéaires pour retrouver les poids

### 4.5 Pourquoi c'est différent pour GELU/SiLU

Les fonctions GELU et SiLU sont **lisses** (pas de discontinuités):
- Pas de régions linéaires distinctes
- Pas de points critiques "nets"
- La dérivée existe partout et varie continûment

**Implications**:
- Les techniques basées sur les discontinuités de ReLU ne s'appliquent pas directement
- Nécessite potentiellement:
  - Approximations linéaires locales
  - Techniques de différentiation numérique
  - Approches basées sur l'optimisation

---

## 5. État de l'art détaillé

### 5.1 Historique des attaques d'extraction

```
1990 ──── Baum: Premiers algorithmes d'apprentissage de réseaux
  │
1992 ──── Blum & Rivest: Entraîner un réseau 3-nœuds est NP-complet
  │
2016 ──── Tramèr et al.: Model stealing via API queries
  │
2020 ──── Carlini et al. (CRYPTO): Cryptanalytic extraction (temps exp.)
  │
2023 ──── Canales-Martínez et al.: Polynomial time (raw-output)
  │
2024 ──── EUROCRYPT: Extension et optimisations
  │       ASIACRYPT: Hard-label pour ReLU
  │
2025 ──── EUROCRYPT: Hard-label polynomial time (Carlini-Shamir)
  │       Oct 2025: Critique "Is it really polynomial?"
```

### 5.2 Attaques principales

#### CRYPTO 2020 - Carlini, Jagielski, Mironov
**"Cryptanalytic Extraction of Neural Network Models"**

- **Innovation**: Premier à reformuler l'extraction comme problème cryptanalytique
- **Méthode**: Différentiation par différences finies + recherche des points critiques
- **Résultat**: Extraction avec précision 2^(-25) sur MNIST
- **Limite**: Temps exponentiel en nombre de neurones par couche

#### EUROCRYPT 2024 - Canales-Martínez, Shamir et al.
**"Polynomial Time Cryptanalytic Extraction"**

- **Innovation**: Algorithme en temps polynomial
- **Méthode**: Exploitation systématique de la structure ReLU
- **Résultat**: 1.2M paramètres extraits en 30 minutes (256 cœurs)
- **Limite**: Nécessite accès aux logits (raw-output)

#### EUROCRYPT 2025 - Carlini, Shamir et al.
**"Hard-Label Setting"**

- **Innovation**: Extraction avec uniquement la classe prédite
- **Méthode**: Recherche binaire sur les frontières de décision
- **Résultat**: ~1000 neurones cachés extraits
- **Limite**: Hypothèse sur la distribution des neurones actifs

### 5.3 Critique récente (Octobre 2025)
**"Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial?"**

**Arguments**:
- L'hypothèse que tous les neurones alternent entre actif/inactif avec probabilité égale est irréaliste
- **Neurones persistants**: Certains neurones sont toujours actifs ou toujours inactifs
- Ces neurones créent du "bruit" non négligeable rendant l'attaque impraticable

**Importance pour ta thèse**: C'est une **opportunité de recherche** - soit réfuter cette critique, soit la confirmer et proposer des améliorations.

### 5.4 Défenses existantes

#### Watermarking
| Technique | Principe | Robustesse extraction |
|-----------|----------|----------------------|
| Backdoor-based | Trigger patterns cachés | Faible (~3%) |
| MEA-Defender | Watermark dans domaine sortie | Forte (83.5%) |
| EWE (Entangled) | Watermark lié aux features | Moyenne |
| ModelShield | Adaptatif | En cours d'évaluation |

#### Autres défenses
- **Rate limiting**: Limiter le nombre de requêtes
- **Noise injection**: Ajouter du bruit aux réponses
- **Randomization**: Varier les réponses pour mêmes entrées
- **Ensemble models**: Combiner plusieurs modèles

---

## 6. Bibliographie annotée

### 6.1 Articles fondamentaux (MUST READ)

#### [1] Carlini, N., Jagielski, M., & Mironov, I. (2020)
**"Cryptanalytic Extraction of Neural Network Models"**
*CRYPTO 2020, LNCS 12172, pp. 189–218*

**Résumé**: Article fondateur établissant le lien entre extraction de modèles et cryptanalyse. Introduit l'attaque différentielle sur réseaux ReLU.

**Concepts clés à retenir**:
- Formulation du problème comme chosen-plaintext attack
- Exploitation de la structure piecewise-linear
- Précision floating-point atteignable

**Lien**: https://link.springer.com/chapter/10.1007/978-3-030-56877-1_7

---

#### [2] Canales-Martínez et al. (2024)
**"Polynomial Time Cryptanalytic Extraction of Neural Network Models"**
*EUROCRYPT 2024, LNCS 14653, pp. 3–33*

**Résumé**: Amélioration majeure réduisant la complexité de exponentielle à polynomiale.

**Concepts clés à retenir**:
- Techniques d'algèbre linéaire efficaces
- Parallélisation de l'attaque
- Validation sur CIFAR-10 (réseau réel)

**Lien**: https://eprint.iacr.org/2023/1526

---

#### [3] Carlini et al. (2025)
**"Polynomial Time Cryptanalytic Extraction in Hard-Label Setting"**
*EUROCRYPT 2025, LNCS 15601, pp. 364–396*

**Résumé**: Extension au cas le plus difficile où seule la classe est visible.

**Concepts clés à retenir**:
- Recherche binaire sur frontières de décision
- Dual points (intersection frontière + hyperplan critique)
- Limitations pratiques identifiées

**Lien**: https://eprint.iacr.org/2024/1580

---

### 6.2 Articles complémentaires

#### [4] Chen et al. (2024) - ASIACRYPT
**"Hard-Label Cryptanalytic Extraction of Neural Network Models"**
*ASIACRYPT 2024*

Approche alternative pour le hard-label. Validation sur MNIST et CIFAR-10.

**Lien**: https://link.springer.com/chapter/10.1007/978-981-96-0944-4_7

---

#### [5] Critique Octobre 2025
**"Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial?"**
*arXiv:2510.06692*

Remet en question les hypothèses de [3]. Important pour positionner ta recherche.

**Lien**: https://arxiv.org/html/2510.06692v1

---

#### [6] Side-Channel Attack (2024)
**"Hard-Label Extraction of Non-Fully Connected DNNs using Side-Channel"**
*arXiv:2411.10174*

Combine side-channel et extraction cryptanalytique pour CNN.

**Lien**: https://arxiv.org/abs/2411.10174

---

### 6.3 Articles sur les défenses

#### [7] MEA-Defender (2024)
**"A Robust Watermark against Model Extraction Attack"**

Meilleure technique de watermarking actuelle (83.5% WSR).

**Lien**: https://arxiv.org/abs/2401.15239

---

#### [8] ModelShield (2025)
**"Adaptive and Robust Watermark Against Model Extraction"**
*IEEE TIFS 2025*

Watermarking adaptatif de dernière génération.

**Lien**: https://dl.acm.org/doi/10.1109/TIFS.2025.3530691

---

### 6.4 Ressources additionnelles

#### Repositories GitHub
- **awesome-deep-model-IP-protection**: https://github.com/ZJZAC/awesome-deep-model-IP-protection
  Liste exhaustive de papers et code sur la protection des modèles

#### Surveys
- Oliynyk et al. (2023) - "I Know What You Trained Last Summer: A Survey on Stealing Machine Learning Models and Defenses"

---

## 7. Plan de remise à niveau

### Phase 1: Fondations Deep Learning (Semaines 1-4)

#### Semaine 1: Perceptron et MLP
**Objectifs**:
- Comprendre le perceptron simple
- Maîtriser la propagation avant (forward pass)
- Comprendre la rétropropagation (backpropagation)

**Ressources**:
- Vidéo: 3Blue1Brown "Neural Networks" (4 épisodes)
- Livre: "Deep Learning" Goodfellow et al., Chapitres 6-8
- Pratique: Implémenter un MLP from scratch en NumPy

**Exercice**:
```python
# Implémenter un MLP 2 couches pour XOR
# Sans utiliser PyTorch/TensorFlow
```

---

#### Semaine 2: Fonctions d'activation
**Objectifs**:
- Comprendre mathématiquement chaque activation
- Calculer les dérivées à la main
- Visualiser les comportements

**Ressources**:
- Paper: "Gaussian Error Linear Units (GELUs)" - Hendrycks & Gimpel
- Paper: "Swish: A Self-Gated Activation Function" - Ramachandran et al.

**Exercice**:
```python
# Implémenter et visualiser:
# - ReLU et sa dérivée
# - GELU (formule exacte et approximation)
# - SiLU/Swish
# Tracer les courbes et leurs dérivées
```

---

#### Semaine 3: Réseaux convolutionnels (CNN)
**Objectifs**:
- Comprendre l'opération de convolution
- Maîtriser pooling, stride, padding
- Comprendre le partage de poids

**Ressources**:
- Cours: Stanford CS231n (disponible sur YouTube)
- Livre: "Deep Learning" Goodfellow, Chapitre 9

**Exercice**:
```python
# Implémenter une convolution 2D from scratch
# Puis utiliser PyTorch pour classifier MNIST avec un CNN simple
```

---

#### Semaine 4: Architectures modernes
**Objectifs**:
- Comprendre ResNet (skip connections)
- Introduction aux Transformers
- Comprendre où sont utilisées les différentes activations

**Ressources**:
- Paper: "Deep Residual Learning" (ResNet)
- Paper: "Attention is All You Need" (Transformer)
- Blog: "The Illustrated Transformer" - Jay Alammar

**Exercice**:
```python
# Entraîner un petit modèle sur CIFAR-10
# Comparer ReLU vs GELU vs SiLU
```

---

### Phase 2: Cryptographie et sécurité (Semaines 5-8)

#### Semaine 5: Fondamentaux crypto
**Objectifs**:
- Revoir les chiffrements par blocs (AES, DES)
- Comprendre la cryptanalyse différentielle
- Modèles d'attaque (CPA, CCA)

**Ressources**:
- Cours: Cryptography I (Coursera - Dan Boneh)
- Paper original: Biham & Shamir "Differential Cryptanalysis of DES"

**Exercice**:
- Implémenter une S-box simple et analyser ses propriétés différentielles

---

#### Semaine 6: Modèles d'attaque ML
**Objectifs**:
- Comprendre les différents niveaux d'accès (white-box, black-box)
- Distinguer raw-output vs hard-label
- Comprendre le concept d'oracle

**Ressources**:
- Survey: "Adversarial Machine Learning" - Biggio & Roli
- Paper: Tramèr et al. "Stealing Machine Learning Models through Prediction APIs"

---

#### Semaine 7: Attaques d'extraction - Théorie
**Objectifs**:
- Lire et comprendre Carlini CRYPTO 2020
- Comprendre la notion de "point critique"
- Maîtriser l'algèbre linéaire sous-jacente

**Ressources**:
- Paper: Carlini et al. CRYPTO 2020 (lecture approfondie)
- Notes: Revoir algèbre linéaire (SVD, résolution systèmes)

**Exercice**:
- Reproduire l'attaque sur un MLP 1 couche cachée (toy example)

---

#### Semaine 8: Attaques d'extraction - Pratique
**Objectifs**:
- Implémenter l'attaque CRYPTO 2020
- Comprendre les limitations
- Lire EUROCRYPT 2024

**Ressources**:
- Code: Chercher implémentations open-source
- Paper: Canales-Martínez et al. EUROCRYPT 2024

**Exercice**:
- Extraire un petit MLP entraîné sur MNIST

---

### Phase 3: Spécialisation thèse (Semaines 9-12)

#### Semaine 9: Au-delà de ReLU - Analyse
**Objectifs**:
- Analyser pourquoi les attaques ReLU ne fonctionnent pas sur GELU/SiLU
- Explorer les approximations possibles
- Identifier les pistes de recherche

**Ressources**:
- Papers sur GELU, SiLU
- Paper: PETS 2024 sur approximations d'activations

**Exercice**:
- Tenter d'appliquer l'attaque ReLU sur un réseau GELU - documenter l'échec

---

#### Semaine 10: Architectures CNN
**Objectifs**:
- Comprendre le paper side-channel + CNN
- Analyser comment le partage de poids affecte l'extraction
- Explorer les vulnérabilités spécifiques aux CNN

**Ressources**:
- Paper: arXiv:2411.10174
- Expérimentation sur petits CNN

---

#### Semaine 11: Défenses
**Objectifs**:
- Comprendre le watermarking (MEA-Defender)
- Analyser les autres défenses
- Comprendre le trade-off sécurité/performance

**Ressources**:
- Papers: MEA-Defender, ModelShield
- Repo: awesome-deep-model-IP-protection

**Exercice**:
- Implémenter une défense simple (noise injection) et mesurer l'impact

---

#### Semaine 12: Synthèse et rédaction
**Objectifs**:
- Rédiger un état de l'art structuré
- Identifier clairement les contributions potentielles
- Préparer le plan de thèse détaillé

**Livrable**:
- Document "État de l'art" (20-30 pages)
- Plan de thèse avec contributions identifiées

---

## 8. Opportunités de recherche

### 8.1 Questions ouvertes identifiées

#### Q1: Extraction de réseaux GELU/SiLU
**État**: Non résolu
**Difficulté**: Élevée
**Approches possibles**:
- Approximation locale par fonctions piecewise-linear
- Techniques d'optimisation (gradient-free)
- Exploitation des propriétés spécifiques de GELU (lien avec distribution gaussienne)

#### Q2: Réponse à la critique Oct 2025
**État**: Débat ouvert
**Opportunité**: Soit confirmer les limitations, soit proposer des améliorations
**Impact**: Fort si résolu

#### Q3: Extraction de CNN
**État**: Partiellement résolu (side-channel)
**Opportunité**: Méthodes purement black-box pour CNN

#### Q4: Défenses adaptatives
**État**: Peu exploré
**Opportunité**: Défenses qui s'adaptent au comportement de l'attaquant

### 8.2 Positionnement suggéré pour ta thèse

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONTRIBUTION 1 (Année 1-2)                                         │
│  ─────────────────────────────────────────────────────────────────  │
│  "Vers l'extraction de réseaux GELU: limites et possibilités"       │
│  → Analyser formellement pourquoi les attaques ReLU échouent        │
│  → Proposer des adaptations ou prouver l'impossibilité              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONTRIBUTION 2 (Année 2)                                           │
│  ─────────────────────────────────────────────────────────────────  │
│  "Extraction de CNN en mode hard-label"                             │
│  → Étendre les travaux side-channel au cas purement black-box       │
│  → Évaluer sur architectures réelles (MobileNet, EfficientNet)      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONTRIBUTION 3 (Année 3)                                           │
│  ─────────────────────────────────────────────────────────────────  │
│  "Défenses robustes avec garanties formelles"                       │
│  → Proposer des défenses avec preuves de sécurité                   │
│  → Quantifier précisément le trade-off sécurité/performance         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Ressources et outils

### 9.1 Environnement de développement

```bash
# Installation recommandée
conda create -n thesis python=3.10
conda activate thesis

# Deep Learning
pip install torch torchvision
pip install jax jaxlib  # Pour différentiation avancée

# Scientifique
pip install numpy scipy matplotlib seaborn
pip install scikit-learn

# Notebooks et documentation
pip install jupyter jupyterlab

# Utilitaires
pip install tqdm wandb  # Tracking expériences
```

### 9.2 Datasets de référence

| Dataset | Taille | Usage |
|---------|--------|-------|
| MNIST | 60k/10k | Prototypage rapide |
| CIFAR-10 | 50k/10k | Validation standard |
| CIFAR-100 | 50k/10k | Complexité accrue |
| ImageNet | 1.2M | Validation finale (si ressources) |

### 9.3 Modèles de référence

- **MLP**: 2-8 couches, 64-256 neurones/couche
- **CNN simple**: LeNet-5, petit VGG
- **CNN moderne**: MobileNetV1 (pour tests scalabilité)

### 9.4 Ressources en ligne

#### Cours
- Stanford CS231n (CNN): http://cs231n.stanford.edu/
- Stanford CS229 (ML): http://cs229.stanford.edu/
- Coursera Cryptography I: https://www.coursera.org/learn/crypto

#### Blogs techniques
- The Illustrated Transformer: http://jalammar.github.io/illustrated-transformer/
- Distill.pub: https://distill.pub/

#### Communautés
- Reddit r/MachineLearning
- Reddit r/crypto
- Twitter/X: Suivre les auteurs des papers clés

---

## 10. Calendrier prévisionnel enrichi

### Année 1 (2025-2026)

| Période | Objectifs | Livrables |
|---------|-----------|-----------|
| **S1** (Jan-Juin) | Remise à niveau + État de l'art | Document état de l'art (30p) |
| | Reproduction attaque ReLU | Code fonctionnel |
| | Premiers tests sur GELU | Rapport intermédiaire |
| **S2** (Juil-Déc) | Analyse formelle GELU/SiLU | Draft article 1 |
| | Premiers résultats théoriques | Soumission workshop |

### Année 2 (2026-2027)

| Période | Objectifs | Livrables |
|---------|-----------|-----------|
| **S1** | Extension aux CNN | Expérimentations MNIST/CIFAR |
| | Comparaison architectures | Draft article 2 |
| **S2** | Affinement résultats | Publication article 1+2 |
| | Début travail sur défenses | Rapport mi-parcours |

### Année 3 (2027-2028)

| Période | Objectifs | Livrables |
|---------|-----------|-----------|
| **S1** | Conception défenses | Prototypes défenses |
| | Évaluation compromis | Draft article 3 |
| **S2** | Rédaction manuscrit | Thèse complète |
| | Préparation soutenance | Soutenance |

---

## Annexe A: Glossaire

| Terme | Définition |
|-------|------------|
| **DNN** | Deep Neural Network - Réseau de neurones profond |
| **MLaaS** | Machine Learning as a Service |
| **Hard-label** | Mode où seule la classe prédite est visible |
| **Raw-output** | Mode où les logits/probabilités sont visibles |
| **Piecewise linear** | Fonction linéaire par morceaux |
| **Critical point** | Point où un neurone change d'état (actif/inactif) |
| **Watermarking** | Technique de marquage pour prouver la propriété |
| **WSR** | Watermark Success Rate |

---

## Annexe B: Contacts et communauté

### Chercheurs clés à suivre
- **Nicholas Carlini** (Google DeepMind) - Attaques ML
- **Adi Shamir** (Weizmann Institute) - Cryptographie + ML
- **Florian Tramèr** (ETH Zurich) - Sécurité ML

### Conférences cibles
- **CRYPTO** / **EUROCRYPT** / **ASIACRYPT** (crypto)
- **NeurIPS** / **ICML** / **ICLR** (ML)
- **IEEE S&P** / **USENIX Security** / **CCS** (sécurité)

---

*Document généré le 31 Janvier 2026*
*Accompagnement: Claude (Anthropic)*
