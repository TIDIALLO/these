#!/usr/bin/env python3
"""
Script de démonstration - Analyse du modèle neural
Extrait et affiche les informations du réseau de neurones CIFAR-10
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import tensorflow as tf
import numpy as np
from pathlib import Path

print("=" * 70)
print("🔍 ANALYSE DU PROJET - Hard Label DNN Extraction")
print("=" * 70)

# -------- 1. Charger le modèle --------
print("\n[1] Chargement du modèle neural...")
model_path = "data/cifar10_3x256_64_10_float64.keras"

try:
    model = tf.keras.models.load_model(model_path)
    print("✓ Modèle chargé avec succès!")
except Exception as e:
    print(f"✗ Erreur lors du chargement: {e}")
    exit(1)

# -------- 2. Architecture du modèle --------
print("\n[2] Architecture du réseau neurone:")
print("-" * 70)
model.summary()

# -------- 3. Informations détaillées sur les couches --------
print("\n[3] Détails des couches:")
print("-" * 70)
for i, layer in enumerate(model.layers):
    print(f"\nCouche {i}: {layer.name}")
    print(f"  Type: {type(layer).__name__}")
    if hasattr(layer, 'units'):
        print(f"  Nombre de neurones: {layer.units}")
    if hasattr(layer, 'activation'):
        print(f"  Activation: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else layer.activation}")
    if hasattr(layer, 'kernel') and layer.kernel is not None:
        weights = layer.kernel.numpy()
        print(f"  Poids - Shape: {weights.shape}, Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")

# -------- 4. Paramètres totaux --------
print("\n[4] Statistiques du modèle:")
print("-" * 70)
total_params = model.count_params()
print(f"Nombre total de paramètres: {total_params:,}")

# -------- 5. Test d'inférence --------
print("\n[5] Test d'inférence:")
print("-" * 70)
test_input = np.random.randn(5, 256, 64, 10).astype(np.float64)
print(f"Entrée de test - Shape: {test_input.shape}")

try:
    output = model.predict(test_input, verbose=0)
    print(f"✓ Sortie du modèle - Shape: {output.shape}")
    print(f"  Prédictions (premiers 5 exemples):\n{output[:5]}")
except Exception as e:
    print(f"✗ Erreur lors de l'inférence: {e}")

# -------- 6. Structure du projet --------
print("\n[6] Structure du projet:")
print("-" * 70)
print("""
📁 hard-label-dnn-extraction/
├── 📄 README.md (description principale)
├── 📁 data/ (données et modèles)
│   └── cifar10_3x256_64_10_float64.keras (modèle CIFAR-10)
├── 📁 signature_recovery/ (Phase 1: Récupération des signatures)
│   ├── find_duals.py (trouver les points duaux)
│   ├── cluster_dual_points.py (grouper les points)
│   ├── recover_weights.py (récupérer les poids)
│   └── utils.py
├── 📁 sign_recovery/ (Phase 2: Récupération des signes)
│   ├── sign_recovery.py (récupération des signes - single)
│   ├── batched_sign_recovery.py (traitement par batch)
│   ├── create_tables.py (analyse des résultats)
│   ├── common.py (fonctions communes)
│   ├── blackbox.py (méthodes boîte noire)
│   └── whitebox.py (méthodes boîte blanche)
""")

# -------- 7. Résumé --------
print("\n[7] 📊 RÉSUMÉ DE L'ATTAQUE:")
print("-" * 70)
print("""
PHASE 1: SIGNATURE RECOVERY (signature_recovery/)
  1️⃣ find_duals.py: Trouver les points duaux (inputs sur la frontière)
  2️⃣ cluster_dual_points.py: Grouper les points par neurone
  3️⃣ recover_weights.py: Récupérer les vecteurs de poids

PHASE 2: SIGN RECOVERY (sign_recovery/)
  1️⃣ Utiliser les points duaux précomptes
  2️⃣ Test statistique pour chaque neurone
  3️⃣ batched_sign_recovery.py: Attaque parallélisée
  4️⃣ create_tables.py: Analyser et résumer les résultats

APPROCHE:
  • Attaque en 2 phases pour extraire complètement les poids
  • Utilise uniquement les prédictions (hard-label setting)
  • Temps polynomial théorique, efficace en pratique
  • Inclut des optimisations whitebox pour la démonstration
""")

print("\n" + "=" * 70)
print("✓ Analyse terminée avec succès!")
print("=" * 70)
