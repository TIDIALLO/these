# 📊 SYNTHÈSE COMPLÈTE - Hard Label DNN Extraction

## 🎯 Vue d'ensemble Exécutive

Ce projet démontre une **attaque cryptanalytique révolutionnaire** capable d'extraire les POIDS COMPLETS d'un réseau de neurones ReLU en utilisant UNIQUEMENT les prédictions (hard-label), sans accès à:
- ❌ Probabilités ou logits
- ❌ Activations internes
- ❌ Gradients
- ✅ JUSTE: Les décisions du réseau (classe prédite)

**Impact**: Démontre que les APIs de prédiction hard-label peuvent être compromise en temps polynomial pratique.

---

## 📚 Informations Générales

| Aspect | Détails |
|--------|---------|
| **Papier** | EUROCRYPT 2024 |
| **Titre** | Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting |
| **Auteur** | Jchavezsaab |
| **Dépôt** | https://github.com/Jchavezsaab/hard-label-dnn-extraction |
| **Modèle Étudié** | CIFAR-10 (petit réseau) |
| **Architecture** | 64 → 256 → 256 → 256 → 64 → 10 (ReLU) |
| **Paramètres** | 165,322 poids totals |

---

## 🏗️ Architecture du Projet

```
hard-label-dnn-extraction/
├── data/
│   ├── cifar10_3x256_64_10_float64.keras (modèle 14 MB)
│   └── dual_points_cifar10_3x256_64_10_float64/ (À TÉLÉCHARGER)
│
├── signature_recovery/  ← PHASE 1: Récupérer les vecteurs de poids
│   ├── find_duals.py (chercher les points duaux)
│   ├── cluster_dual_points.py (grouper les points)
│   ├── recover_weights.py (extraire les poids)
│   └── utils.py (config & utiles)
│
├── sign_recovery/  ← PHASE 2: Récupérer les signes des neurones
│   ├── sign_recovery.py (attaque un neurone)
│   ├── batched_sign_recovery.py (attaque parallélisée)
│   ├── create_tables.py (analyser les résultats)
│   ├── blackbox.py (implémentation de l'attaque)
│   └── whitebox.py (optimisations de démo)
│
├── demo_analysis.py (ANALYSE COMPLÈTE)
└── demo_attack.py (DÉMONSTRATION PRATIQUE)
```

---

## 🔴 PHASE 1: SIGNATURE RECOVERY

### Objectif
Extraire les **vecteurs de poids non-signés** de chaque couche.

### Processus

#### 1️⃣ **find_duals.py** - Trouver les Points Duaux

```python
python3 find_duals.py
```

**Qu'est-ce qu'un point dual?**
- Un point X qui se trouve EXACTEMENT sur la frontière de décision du réseau
- ET qui a au minimum un neurone avec valeur = 0 (zéro exact)
- Propriétés: `score_class1(X) ≈ score_class2(X)` et `W_layer^T * X ≈ 0`

**Processus:**
1. Générer des points aléatoires
2. Les affiner jusqu'à la frontière de décision
3. Vérifier si un neurone a valeur zéro
4. Sauvegarder les points duaux dans `data/dual_points_{model_name}/`

**Résultat:**
- 10,000 points à chaque exécution
- À répéter pour obtenir 10M+ points (pour extraction complète)

#### 2️⃣ **cluster_dual_points.py** - Grouper les Points

```python
python3 cluster_dual_points.py 0  # Pour couche 0
python3 cluster_dual_points.py 1  # Pour couche 1
```

**Processus:**
1. Pour chaque point dual, identifier quel neurone = 0
2. Grouper les points ayant le même neurone zéro
3. Créer une collection par neurone

**Résultat:**
- Fichiers par neurone: `layerX_neuronY.npy`
- Chaque fichier = collection de points duaux pour ce neurone

#### 3️⃣ **recover_weights.py** - Récupérer les Poids

```python
python3 recover_weights.py 0  # Pour couche 0
```

**Processus:**
1. Utiliser les points duaux groupés
2. Résoudre les systèmes d'équations linéaires
3. Extraire les vecteurs de poids (sans signe)

**Limitations:**
- Récupère les MAGNITUDES seulement (pas les signes)
- Nécessite suffisamment de points duaux variés
- Error: "Not enough to fully extract" = besoin plus de points

---

## 🔵 PHASE 2: SIGN RECOVERY

### Objectif
Deviner le **signe de chaque neurone** (activé ON ou désactivé OFF) dans la couche cible.

### Principes Clés

**Concept Central:**
L'attaque exploite l'asymétrie des "patch distances":
- **Quand un neurone est ON**: les perturbations affectent davantage les prédictions
- **Quand un neurone est OFF**: les perturbations affectent moins les prédictions
- Différence ~5% statistiquement mesurable

**Processus Statistique:**
```
Pour chaque neurone cible:
  1. Prendre un point dual proche du neurone
  2. Marcher dans des directions aléatoires (+ et -)
  3. Compter combien de fois la prédiction change
  4. Si ON: taux de changement plus haut
  5. Si OFF: taux de changement plus bas
  6. Test statistique + confidentialité 95%
```

### Exécution

#### Option A: Attaquer UN neurone

```python
python sign_recovery.py \
    --model ../data/cifar10_3x256_64_10_float64.keras \
    --layerID 0 \
    --neuronID 0 \
    --filepath_load_x0 ../data/dual_points_cifar10_3x256_64_10_float64
```

**Arguments:**
- `--nExp 400`: Max nombre d'expériences (par défaut)
- `--nExpMin 25`: Min nombre avant de terminer
- `--nToggles 1`: Changements de neuron = 1
- `--handlePrevLayerToggles True`: Recomputer les directions
- `--choose_dx along_decision_boundary`: Method de marche
- `--analyzeWiggleSensitivity True`: Analyser sensibilité
- `--analyzeSpeed True`: Analyser la vitesse

#### Option B: Attaquer TOUS les neurones (Batch)

```python
python batched_sign_recovery.py
```

**Avantages:**
- Exécution parallélisée sur plusieurs threads
- Automatiquement configure les paramètres optimaux
- Génère un résultat complet pour l'analyse

**Sortie:**
```
results/
  └── model_cifar10_3x256_64_10_float64/
      ├── layerID_0/
      │   ├── neuronID_0/ (fichiers de résultat)
      │   └── neuronID_1/
      └── layerID_1/
          └── ...
```

#### Analyser les Résultats

```python
python create_tables.py
```

**Génère:**
- Tableaux résumés par couche
- Taux de succès pour chaque neurone
- Statistiques globales
- Visualisations (si applicable)

---

## 📊 Résultats Attendus

### Avec le Modèle Réduit (Tiny)
Et points duaux précomputes:

| Couche | Succès | Raison |
|--------|--------|--------|
| 0 | 100% | Dimension faible, beaucoup de points duaux |
| 1 | ~90% | Dimension moyenne |
| 2 | ~25% | Dimension élevée, moins de points |
| 3 | ~0% | Couche finale, besoin énormément de points |

### Modèle Complet (CIFAR-10 plein)
Nécessite:
- Couche 1: 1,000 points duaux
- Couche 2: 100,000 points duaux
- Couche 3: 300,000 points duaux
- Couche 4: 3,000,000 points duaux

Temps: Quelques heures à quelques jours (GPU)

---

## 🚀 Points Clés Techniquement

### 1. Points Duaux - Fondation de l'Attaque

```
Mathématiquement:
  Trouver X tel que:
    • |f_c1(X) - f_c2(X)| < ε  (équilibre entre classes)
    • |W_l^T X| < δ             (un neurone ≈ zéro)
    
Numériquement (USE_GRADIENT=True):
  • Utiliser autograd pour calculer les gradients
  • Newton-Raphson pour converger rapidement
  • Stabilité numérique: ~1e-13 tolérance
```

### 2. Distance de Patch - Mesure de Signature

```
Concept:
  patch_distance(X, direction) = 
    Prob(prediction_change(X + α*direction))
  
Usage:
  • On: patch_distance élevé
  • Off: patch_distance bas
  • Mesurable même en blackbox (requêtes seulement)
```

### 3. Reconstruction des Poids

```
Phase 1 (Signatures):
  W = (X_dual^T X_dual)^-1 X_dual^T y
  
Phase 2 (Signes):
  sign(W) = test_statistique(patch_distances)
  
Résultat:
  W_complet = W * sign(W)
```

---

## 🔐 Implications de Sécurité

### ⚠️ CRITIQUE

**Avant ce travail:**
- On pensait que les APIs hard-label étaient "sûres"
- Les modèles extractibles nécessitaient accès aux logits

**Après ce travail:**
- Les prédictions SEULES suffisent pour extraction
- Temps polynomial connu (pas théorique)
- Applicable à tous les réseaux ReLU

### Recommandations

1. **Ne pas exposer les prédictions brutes:**
   - Utiliser des droits d'accès restrictifs
   - Rate limiting strict (1 requête/utilisateur/jour)
   - Monitoring des requêtes suspectes

2. **Pour les APIs ML:**
   - Ajouter bruit différentiel
   - Limiter les types de requêtes
   - Authentification forte

3. **Pour développeurs:**
   - Ne pas déployer de modèles extractibles
   - Utiliser des défenses robustes
   - Audit régulier

---

## 🛠️ Comment Exécuter

### Préalables
```bash
pip install numpy torch scipy pandas matplotlib
```

### Étapes Complètes

**1. Analyse et Compréhension**
```bash
python analysis.py          # Vue d'ensemble
python demo_attack.py       # Démonstration pratique
```

**2. Télécharger les données**
- Lien: https://drive.google.com/file/d/1mFfKlLgE0ZnGPAYN8tPRtb2iYpP5QgfY
- Extraire dans: `data/dual_points_cifar10_3x256_64_10_float64/`

**3. Signature Recovery (optionnel)**
```bash
cd signature_recovery
python3 find_duals.py
python3 cluster_dual_points.py 0
python3 recover_weights.py 0
```

**4. Sign Recovery (ATTAQUE)**
```bash
cd ../sign_recovery
python batched_sign_recovery.py  # Ou sign_recovery.py pour tester
python create_tables.py
```

**5. Analyser les Résultats**
```bash
# Consulter les fichiers dans results/
# Tableaux générés dans stdout
```

---

## 📈 Flux de Données

```
┌─────────────────┐
│  Modèle Neural  │ (cifar10_3x256_64_10_float64.keras)
│  16,5k params   │
└────────┬────────┘
         │
         └──→ [PHASE 1: SIGNATURE RECOVERY]
             └─→ find_duals.py → Points Duaux (10k chaque)
             └─→ cluster_dual_points.py → Grouper par neurone
             └─→ recover_weights.py → Vecteurs de poids (unsigned)
             └─→ OUTPUT: Wights magnitude
         │
         ├──→ [PHASE 2: SIGN RECOVERY]
         │    └─→ Points duaux précomputes + poids
         │    └─→ sign_recovery.py / batched_sign_recovery.py
         │    └─→ Test statistique (patch distances)
         │    └─→ OUTPUT: Signe de chaque neurone
         │
         └──→ [RÉSULTAT FINAL]
             └─→ Combinaison: W_complet = W_magnitude * sign
             └─→ Extraction complète du modèle ✓
```

---

## 🎓 Apprenez à Partir De

1. **Algorithmes:**
   - Recherche de points duaux (root finding, optimization)
   - Clustering (KMeans, etc.)
   - Tests statistiques (t-test, confidence intervals)

2. **Concepts ML:**
   - Frontières de décision
   - Activation de neurones
   - Inférence partielle

3. **Sécurité ML:**
   - Extraction de modèles
   - Attaques sans accès blanc
   - Implications de la cryptanalyse

---

## 📚 Ressources

- **Dépôt GitHub:** https://github.com/Jchavezsaab/hard-label-dnn-extraction
- **Papier:** EUROCRYPT 2024 (lien dans le dépôt)
- **Dataset:** CIFAR-10 (via script ou manuel)
- **Points Duaux:** https://drive.google.com/file/d/1mFfKlLgE0ZnGPAYN8tPRtb2iYpP5QgfY

---

## ✅ Résumé

Ce projet démontre une attaque polynomiale pour extraire les réseaux de neurones ReLU en utilisant SEULEMENT les prédictions. C'est une contribution importante en cryptanalyse des modèles d'apprentissage automatique et a d'importantes implications de sécurité pour le déploiement des APIs ML.

**Impact:**
- 🔓 Montre la faiblesse des APIs hard-label
- 📈 Complexité polynomial en théorie ET pratique
- 🛡️ Force industrie à reconsidérer défenses
- 📚 Nouveau domaine de recherche: "cryptanalysis vs ML"

---

**Généré pour:** Analyse et exécution du projet
**Date:** 1 Mars 2026
**Auteur de la Démo:** Assistant Copilot
