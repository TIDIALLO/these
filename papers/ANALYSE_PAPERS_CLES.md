# Analyse Détaillée des Papers Clés

## Pour la thèse: "Attaques d'extraction et défenses pour les DNN au-delà de ReLU"

**Date**: 31 Janvier 2026

---

## Table des matières

1. [Paper 1: CRYPTO 2020 - Carlini et al.](#paper-1-crypto-2020)
2. [Paper 2: EUROCRYPT 2024 - Canales-Martínez et al.](#paper-2-eurocrypt-2024)
3. [Paper 3: EUROCRYPT 2025 - Hard-Label Setting](#paper-3-eurocrypt-2025)
4. [Paper 4: Critique Oct 2025 - Is it Really Polynomial?](#paper-4-critique-oct-2025)
5. [Paper 5: MEA-Defender - Défense par Watermarking](#paper-5-mea-defender)
6. [Synthèse et implications pour la thèse](#synthèse)

---

## Paper 1: CRYPTO 2020

### Informations bibliographiques
```
Titre: Cryptanalytic Extraction of Neural Network Models
Auteurs: Nicholas Carlini, Matthew Jagielski, Ilya Mironov
Venue: CRYPTO 2020, LNCS 12172, pp. 189-218
Lien: https://eprint.iacr.org/2020/xxx | https://arxiv.org/abs/2003.04884
```

### Résumé
Premier article établissant le lien formel entre extraction de modèles et cryptanalyse. Les auteurs reformulent le problème d'extraction comme une attaque à clair choisi (chosen-plaintext attack) sur un cryptosystème.

### Idée principale
> "Given oracle access to a neural network, we introduce a differential attack that can efficiently steal the parameters of the remote model up to floating point precision."

### Contribution technique

#### Observation fondamentale
Les réseaux ReLU sont des fonctions **linéaires par morceaux** (piecewise linear):
- L'espace d'entrée est partitionné en régions polytopiques
- Dans chaque région, le réseau calcule une fonction affine
- Les frontières entre régions sont des **hyperplans critiques**

#### Méthode d'attaque
1. **Identifier les points critiques**: Points où un neurone passe de 0 à actif
2. **Différentiation numérique**: Calculer ∂f/∂x aux points critiques
3. **Résoudre le système linéaire**: Retrouver les poids à partir des dérivées

```
Point critique: W·x + b = 0 pour un neurone
                ↓
Changement de comportement du réseau
                ↓
Information sur W et b
```

### Résultats clés

| Métrique | Valeur |
|----------|--------|
| Précision vs prior work | 2^20× meilleure |
| Réduction requêtes | 100× moins |
| MNIST (100k params) | 2^21.5 requêtes, <1h |
| MNIST (4k params) | 2^18.5 requêtes |
| Erreur worst-case | 2^-25 à 2^-40 |

### Limitations
- **Complexité temporelle**: Exponentielle en nombre de neurones par couche
- **Accès raw-output**: Nécessite les logits complets
- **Architectures**: Uniquement MLP fully-connected

### Implications pour ta thèse
- **Fondation théorique**: Ce paper est la base de tout
- **À reproduire**: Exercice essentiel pour comprendre les mécanismes
- **Limitation clé**: Ne fonctionne que pour ReLU → ton opportunité

---

## Paper 2: EUROCRYPT 2024

### Informations bibliographiques
```
Titre: Polynomial Time Cryptanalytic Extraction of Neural Network Models
Auteurs: Isaac A. Canales-Martínez, Jorge Chávez-Saab, Anna Hambitzer,
         Francisco Rodríguez-Henríquez, Nitin Satpute, Adi Shamir
Venue: EUROCRYPT 2024, LNCS 14653, pp. 3-33
Lien: https://eprint.iacr.org/2023/1526
```

### Résumé
Amélioration majeure du CRYPTO 2020: passage de complexité **exponentielle** à **polynomiale**.

### Contribution technique

#### Problème résolu
L'attaque CRYPTO 2020 nécessitait une recherche exhaustive parmi 2^n possibilités pour n neurones.

#### Innovation clé
Nouvelles techniques d'algèbre linéaire permettant de résoudre le système sans recherche exhaustive:

```
CRYPTO 2020:  O(2^n) - Recherche exhaustive
     ↓
EUROCRYPT 2024: O(poly(n)) - Résolution directe
```

#### Méthode
1. **Identification efficace** des hyperplans critiques
2. **Système d'équations structuré** exploitant la géométrie ReLU
3. **Parallélisation** massive de l'attaque

### Résultats clés

| Modèle cible | Paramètres |
|--------------|------------|
| Dataset | CIFAR-10 |
| Entrées | 3,072 dimensions |
| Architecture | 8 couches cachées, 256 neurones/couche |
| Total paramètres | ~1.2 million |
| Temps d'extraction | 30 minutes |
| Hardware | 256 cœurs |

**Comparaison dramatique**:
- CRYPTO 2020 aurait requis: recherche parmi 2^256 possibilités
- EUROCRYPT 2024: 30 minutes

### Limitations
- **Accès raw-output requis**: Les logits doivent être visibles
- **ReLU uniquement**: Pas d'extension aux autres activations

### Implications pour ta thèse
- **État de l'art technique**: Comprendre ces techniques est essentiel
- **Benchmark**: Tes attaques sur GELU/SiLU devront être comparées
- **Code potentiellement disponible**: À rechercher

---

## Paper 3: EUROCRYPT 2025

### Informations bibliographiques
```
Titre: Polynomial Time Cryptanalytic Extraction of Deep Neural Networks
       in the Hard-Label Setting
Auteurs: Nicholas Carlini, Jorge Chávez-Saab, Anna Hambitzer,
         Francisco Rodríguez-Henríquez, Adi Shamir
Venue: EUROCRYPT 2025, LNCS 15601, pp. 364-396
Lien: https://eprint.iacr.org/2024/1580
```

### Résumé
Extension au cas **hard-label**: l'attaquant ne voit que la classe prédite, pas les logits.

### Contribution technique

#### Difficulté du hard-label
```
Raw-output:  f(x) = [0.1, 0.7, 0.2]  → Information riche
Hard-label:  f(x) = "classe 2"       → Information minimale
```

#### Idée clé: Dual Points
Un **dual point** est un point qui appartient simultanément à:
1. La **frontière de décision** entre deux classes
2. Un **hyperplan critique** (où un neurone change d'état)

```
                    Classe A
                      /
        Dual Point → ●────── Hyperplan critique
                    /
                Classe B
```

#### Méthode
1. **Recherche binaire** pour trouver la frontière de décision
2. **Exploration** le long de la frontière pour trouver les dual points
3. **Extraction** des poids à partir des dual points

### Résultats clés

| Métrique | Valeur |
|----------|--------|
| Neurones extraits | ~832 (4 couches cachées) |
| Paramètres totaux | ~1 million |
| Complexité | Polynomiale (queries et temps) |
| Dataset | CIFAR-10 |

**Comparaison avec prior work**:
- Avant: Maximum 4 neurones sur 2 couches
- Maintenant: 832 neurones sur 4 couches

### Limitations (importantes!)
- **Hypothèse forte**: Tous les neurones alternent actif/inactif avec probabilité égale
- **Neurones persistants**: Non traités
- **ReLU uniquement**: Pas d'extension

### Implications pour ta thèse
- **Référence directe**: Ta thèse doit citer et étendre ce travail
- **Limitation = Opportunité**: La question GELU/SiLU est ouverte
- **Hypothèses à challenger**: Voir Paper 4

---

## Paper 4: Critique Oct 2025

### Informations bibliographiques
```
Titre: Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial?
Auteurs: [À vérifier]
Venue: arXiv, Octobre 2025
Lien: https://arxiv.org/abs/2510.06692
```

### Résumé
Critique fondamentale de l'EUROCRYPT 2025 remettant en question la complexité polynomiale.

### Arguments principaux

#### 1. Le problème des neurones persistants

```
Neurone "normal":    Actif 50% du temps, Inactif 50%
                     → Facile à trouver les transitions

Neurone persistant:  Actif 99.9% du temps (ou 0.1%)
                     → Transitions très rares
                     → Requêtes exponentielles pour les trouver
```

#### 2. Analyse mathématique
- La probabilité de transition **décroît exponentiellement** avec la profondeur
- Pour un réseau de 20 couches: certains neurones ont probabilité de switch < 10^-6
- Trouver leurs points d'intersection devient **exponentiel** en pratique

#### 3. Propagation d'erreur
- Un neurone persistant non récupéré → erreur sur toutes les couches suivantes
- L'erreur se propage: O(d^-1/2) où d = largeur de couche
- Même un seul neurone manqué peut ruiner l'extraction

#### 4. Solution proposée: CrossLayer Extraction
- Utiliser les informations des couches profondes
- Exploiter les interactions entre couches
- Récupérer les neurones persistants indirectement

### Implications pour ta thèse

**OPPORTUNITÉ MAJEURE**:
1. Soit **confirmer** cette critique avec des expériences
2. Soit **réfuter** en montrant des conditions où ça marche
3. Soit **améliorer** avec de nouvelles techniques

Cette controverse est une **position idéale** pour ta thèse:
- Sujet d'actualité (débat en cours)
- Impact potentiel élevé
- Combine théorie et expérimentation

---

## Paper 5: MEA-Defender

### Informations bibliographiques
```
Titre: MEA-Defender: A Robust Watermark against Model Extraction Attack
Auteurs: [Équipe de recherche]
Venue: IEEE S&P 2024
Lien: https://arxiv.org/abs/2401.15239
```

### Résumé
Technique de watermarking robuste contre l'extraction de modèles.

### Contribution technique

#### Problème
Les watermarks classiques (backdoor-based) sont **supprimés** lors de l'extraction:
- L'attaquant entraîne un modèle sur les sorties
- Les patterns backdoor ne sont pas transférés
- WSR (Watermark Success Rate) ~3% après extraction

#### Solution: MEA-Defender
Watermarks dont la distribution **ressemble** aux données légitimes:

```
Watermark classique:          MEA-Defender:
┌─────────────────┐           ┌─────────────────┐
│ Données normales│           │ Données normales│
│      ○ ○ ○      │           │      ○ ○ ○      │
│     ○   ○       │           │     ○ ● ○       │ ← Watermark
│    ○     ○      │           │    ○   ● ○      │   intégré
│   ★ (trigger)   │           │   ○     ○       │
└─────────────────┘           └─────────────────┘
Facile à isoler               Indistinguable
```

#### Méthode
1. **Combinaison de classes**: Watermark créé en mélangeant samples de 2 classes
2. **Fonction de perte**: Contraint le watermark dans l'espace des sorties normales
3. **Intégration**: Le watermark "s'enchevêtre" avec les features légitimes

### Résultats clés

| Métrique | Entangled WM | MEA-Defender |
|----------|--------------|--------------|
| WSR après extraction | 3.37% | **83.53%** |
| Impact accuracy | Minimal | Minimal |

### Implications pour ta thèse

**Pour l'axe Défenses**:
- MEA-Defender est le benchmark à battre
- Comprendre pourquoi ça marche → peut-on faire mieux?
- Trade-off sécurité/performance à analyser

---

## Synthèse et implications pour la thèse

### Carte conceptuelle

```
                    CRYPTO 2020
                    (Fondation)
                         │
                         ▼
                  ┌──────────────┐
                  │   ReLU       │
                  │ Extraction   │
                  │ (temps exp)  │
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      EUROCRYPT 2024          EUROCRYPT 2025
      (temps poly)            (hard-label)
              │                     │
              │         ┌───────────┘
              │         ▼
              │    Critique 2025
              │    (neurones persistants)
              │         │
              ▼         ▼
        ┌─────────────────────────┐
        │   OPPORTUNITÉS THÈSE    │
        │                         │
        │  1. Extension GELU/SiLU │
        │  2. Extension CNN       │
        │  3. Résoudre controverse│
        │  4. Défenses robustes   │
        └─────────────────────────┘
```

### Questions de recherche ouvertes

| Question | Difficulté | Impact | Priorité thèse |
|----------|------------|--------|----------------|
| Extraction GELU? | Élevée | Fort | **Priorité 1** |
| Extraction SiLU/Swish? | Élevée | Fort | **Priorité 1** |
| Neurones persistants | Moyenne | Fort | Priorité 2 |
| Extraction CNN | Moyenne | Moyen | Priorité 2 |
| Défenses + preuves | Élevée | Fort | Priorité 3 |

### Positionnement recommandé

**Année 1**:
- Maîtriser CRYPTO 2020 + EUROCRYPT 2024
- Analyser formellement pourquoi ça ne marche pas sur GELU
- Premier article: "Limites de l'extraction cryptanalytique sur activations lisses"

**Année 2**:
- Proposer adaptations ou prouver impossibilité
- Étendre aux CNN si possible
- Deuxième article: comparaison architectures

**Année 3**:
- Concevoir défenses avec le recul des attaques
- Quantifier trade-offs
- Article défenses + manuscrit thèse

---

## Prochaines lectures recommandées

### Ordre de lecture suggéré
1. **CRYPTO 2020** - Fondation (2-3 jours)
2. **EUROCRYPT 2024** - Optimisation (2-3 jours)
3. **EUROCRYPT 2025** - Hard-label (2-3 jours)
4. **Critique 2025** - Nuances (1-2 jours)
5. **MEA-Defender** - Défenses (1-2 jours)

### Papers complémentaires
- Tramèr et al. (2016) - "Stealing ML Models through Prediction APIs"
- Jagielski et al. (2020) - "High Accuracy and High Fidelity Extraction"
- Side-channel paper (2024) - Pour extension CNN

---

*Document généré le 31 Janvier 2026*
