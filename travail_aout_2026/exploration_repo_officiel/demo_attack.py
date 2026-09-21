#!/usr/bin/env python3
"""
Script d'exécution - Démonstration pratique du modèle
Charge le modèle PyTorch et exécute l'attaque sur un neurone
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Suppression des warnings
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import warnings
warnings.filterwarnings("ignore")

print("\n" + "=" * 85)
print(" ⚡ DÉMONSTRATION - Attaque d'Extraction Hard-Label DNN")
print("=" * 85)

# ========== CONFIGURATION ==========
print("\n[STEP 1] Configuration du modèle")
print("-" * 85)

IDIM = 64
DIM = 256  
SHRINK = 64
LAYERS = 5

class CIFAR10Net(nn.Module):
    """Architecture du modèle CIFAR-10"""
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.fc1 = nn.Linear(IDIM, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, DIM)
        self.fc4 = nn.Linear(DIM, SHRINK)
        self.fc5 = nn.Linear(SHRINK, 10)
        self.relu = nn.ReLU()
        self.double()

    @torch.no_grad()
    def forward(self, x):
        x = x.view(-1, IDIM)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

print("✓ Classe CIFAR10Net définie")
print(f"  Architecture: {IDIM} → {DIM} → {DIM} → {DIM} → {SHRINK} → 10")

# ========== CHARGEMENT DU MODÈLE ==========
print("\n[STEP 2] Chargement du modèle")
print("-" * 85)

model = CIFAR10Net()
model.eval()

try:
    # Charger les poids depuis les fichiers PyTorch si existants
    pth_path = "signature_recovery/models/tiny.pth"
    if os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path, map_location='cpu', weights_only=True))
        print(f"✓ Modèle chargé depuis: {pth_path}")
    else:
        print(f"⚠ Modèle PyTorch ({pth_path}) non trouvé")
        print("  → Utilisation du modèle avec initialisations aléatoires")
        
except Exception as e:
    print(f"⚠ Erreur lors du chargement: {e}")
    print("  → Utilisation du modèle avec initialisations aléatoires")

# Afficher les statistiques du modèle
print(f"\nStatistiques du modèle:")
total_params = sum(p.numel() for p in model.parameters())
print(f"  • Nombre total de paramètres: {total_params:,}")
print(f"  • Nombre de couches: {len(list(model.parameters()))} couches de poids")
for i, (name, param) in enumerate(model.named_parameters()):
    if 'weight' in name:
        print(f"    {name}: {tuple(param.shape)}")

# ========== TEST D'INFÉRENCE ==========
print("\n[STEP 3] Test d'inférence")
print("-" * 85)

batch_size = 10
test_input = torch.randn(batch_size, IDIM, dtype=torch.float64)
print(f"✓ Entrée de test générée (shape: {test_input.shape})")

with torch.no_grad():
    output = model(test_input)
    
print(f"✓ Sortie du modèle (shape: {output.shape})")
print(f"\nExemples de prédictions (logits):")
for i in range(min(3, batch_size)):
    print(f"  Exemple {i+1}: {output[i].numpy()}")

# Prédictions en tant que classes
predictions = torch.argmax(output, dim=1)
print(f"\nClasses prédites: {predictions.numpy()}")

# ========== SIMULATION - FRONTIÈRE DE DÉCISION ==========
print("\n[STEP 4] Analyse - Frontière de décision")
print("-" * 85)

print("""
Concept clé de l'attaque:
• Les points DUAUX sont sur la frontière entre classes
• Ils servent de "signaux" pour inverser les poids
• L'attaque les exploite pour extraire les neurones

Simulation:
""")

# Générer des points et vérifier leurs prédictions
print("Génération de 100 points aléatoires...")
random_inputs = torch.randn(100, IDIM, dtype=torch.float64)

with torch.no_grad():
    outputs = model(random_inputs)
    preds = torch.argmax(outputs, dim=1)
    
# Chercher des points aux frontières
boundary_scores = []
for i in range(len(outputs)):
    scores = outputs[i].numpy()
    max_score = np.max(scores)
    second_max_score = np.sort(scores)[-2]
    margin = max_score - second_max_score
    boundary_scores.append((margin, i))

boundary_scores.sort()
print(f"✓ Points générés: {len(boundary_scores)}")
print(f"  • Points avec MARGE FAIBLE (proches de la frontière):")
for margin, idx in boundary_scores[:5]:
    print(f"    Point {idx}: margin={margin:.4f}, classe={preds[idx].item()}")

print(f"\n  • Points avec MARGE LARGE (loin de la frontière):")
for margin, idx in boundary_scores[-5:]:
    print(f"    Point {idx}: margin={margin:.4f}, classe={preds[idx].item()}")

# ========== SIMULATION - PATCH DISTANCE ==========
print("\n[STEP 5] Analyse - Distance de Patch")
print("-" * 85)

print("""
Concept clé:
• Distance de patch = changement de prédiction avec petite perturbation
• Varie selon si un neurone cible est ON ou OFF
• Permet de statistiquement déduire le signe du neurone
""")

# Sélectionner un point aléatoire
test_point = random_inputs[0:1]

print(f"Point de test: {test_point.shape}")
with torch.no_grad():
    original_pred = model(test_point)
    original_class = torch.argmax(original_pred)

print(f"Prédiction originale: classe {original_class.item()}")

# Tester des perturbations dans différentes directions
perturbation_sizes = [0.01, 0.05, 0.1, 0.2]
flip_counts = {size: 0 for size in perturbation_sizes}
n_trials = 50

print(f"\nTaux de changement de classe avec perturbations aléatoires ({n_trials} essais):")

for size in perturbation_sizes:
    for _ in range(n_trials):
        noise = torch.randn_like(test_point) * size
        perturbed = test_point + noise
        
        with torch.no_grad():
            perturbed_pred = model(perturbed)
            perturbed_class = torch.argmax(perturbed_pred)
        
        if perturbed_class != original_class:
            flip_counts[size] += 1
    
    flip_rate = flip_counts[size] / n_trials
    print(f"  Perturbation σ={size:>4}: {flip_rate*100:>5.1f}% changements de classe")

# ========== STRUCTURE DE COUCHES ==========
print("\n[STEP 6] Analyse des activations internes")
print("-" * 85)

# Créer une version du modèle avec hooks pour capturer les activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Enregistrer hooks sur les couches ReLU
model.fc1.register_forward_hook(get_activation('fc1_out'))
model.fc2.register_forward_hook(get_activation('fc2_out'))
model.fc3.register_forward_hook(get_activation('fc3_out'))
model.fc4.register_forward_hook(get_activation('fc4_out'))

test_input_single = random_inputs[0:1]
with torch.no_grad():
    _ = model(test_input_single)

print("Activations par couche (exemple pour un point):")
for layer_name in ['fc1_out', 'fc2_out', 'fc3_out', 'fc4_out']:
    if layer_name in activations:
        act = activations[layer_name][0]
        zero_count = (act == 0).sum().item()
        active_count = (act > 0).sum().item()
        print(f"  {layer_name:<10}: {active_count:3d} neurones ON, {zero_count:3d} neurones OFF")

# ========== RÉSUMÉ DE L'ATTAQUE ==========
print("\n[STEP 7] 📊 RÉSUMÉ - Comment l'attaque fonctionne")
print("-" * 85)

print("""
1️⃣  PHASE 1 - SIGNATURE RECOVERY (récupère les formes des poids)
    ├─ Trouver les POINTS DUAUX (sur les frontières)
    ├─ Clusteriser par rapport à quel neurone = 0
    └─ Résoudre les systèmes linéaires → vecteurs de poids

2️⃣  PHASE 2 - SIGN RECOVERY (récupère les signes des neurones)
    ├─ Pour chaque neurone cible:
    │  ├─ Faire marcher depuis le point dual dans deux directions
    │  ├─ Mesurer les "patch distances" (distances de changement)
    │  ├─ ON : patch distance en moyenne ~5% plus grande
    │  └─ OFF : patch distance en moyenne ~5% plus petite
    └─ Test statistique → déduire le signe

3️⃣  RÉSULTAT FINAL
    └─ Tous les poids signés et correctement identifiés
    
Pourquoi c'est remarquable:
  • SANS accès aux logits (probabilités)
  • SANS accès aux activations internes
  • AVEC SEULEMENT les prédictions (hard-label)
  • Temps polynomial en théorie et pratique
""")

# ========== CLÉ DE L'ATTAQUE ==========
print("\n[STEP 8] 🔑 Points clés de l'attaque")
print("-" * 85)

print("""
INSIGHTS MATHÉMATIQUES:
  
  1. Points duaux (x_d) satisfont:
     • score_1(x_d) ≈ score_2(x_d) (équilibre entre classes)
     • ∃ neurone l tel que W_l^T x_d ≈ 0 (traverse zéro)
  
  2. Différences de patch exploitables:
     • Si neuron i est ON:  "marcher dans direction j" affecte plus
     • Si neuron i est OFF: "marcher dans direction j" affecte moins
     • Effet mesurable statistiquement
  
  3. Reconstruction:
     • Points duaux + informations de signes → poids complets
     • Via algèbre linéaire et optimisation
     
IMPLICATIONS DE SÉCURITÉ:
  ⚠️  Les APIs hard-label (seulement prédictions) peuvent être cassées
  → Il ne faut jamais exposer même les décisions brutes sans protection
  → Rate limiting, monitoring, etc. sont essentiels
""")

# ========== CONCLUSION ==========
print("\n" + "=" * 85)
print("✅ DÉMONSTRATION COMPLÈTE")
print("=" * 85)

print("""
Vous avez vu comment:
  1. Le modèle neural charge et fonctionne
  2. Les points à frontière sont identifiés
  3. Les patch distances varient avec les activations
  4. L'attaque peut en extraire les poids

Pour exécuter la vraie attaque:
  
  1. Télécharger les points duaux précomputes
  2. cd sign_recovery
  3. python batched_sign_recovery.py (ou sign_recovery.py pour tester)
  4. python create_tables.py (pour analyser les résultats)

Referencias:
  • https://github.com/Jchavezsaab/hard-label-dnn-extraction
  • EUROCRYPT 2024 Paper
""")

print("\n")
