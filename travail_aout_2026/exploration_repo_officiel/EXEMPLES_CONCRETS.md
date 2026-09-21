# 💡 EXEMPLES CONCRETS & CAS D'USAGE

## 1. 🔒 Scénario Réel - API ML Compromise

### La Situation
```
Une entreprise expose une API simple de classification:

POST /api/predict
{
  "input": [array de 64 nombres]
}

Response:
{
  "class": 5,           ← SEULEMENT ça!
  "confidence": 0.95    ← Pas fourni (hard-label pur)
}
```

### L'Attaque
L'attaquant utilise cette API pour:

**Semaine 1-2: Trouver des Points Duaux**
```python
# Chercher les points qui changent de prédiction souvent
for i in range(1000000):
    x = random_point()
    pred1 = api.predict(x)
    
    # Affiner jusqu'à la frontière
    while True:
        x_refined = refine_to_boundary(x)
        pred2 = api.predict(x_refined)
        if pred1 != pred2:
            # Point dual trouvé!
            save_dual_point(x_refined)
            break
```

**Coût**: ~1M d'appels API → détectable si monitoring bon

**Semaine 3-4: Grouper les Points**
```python
# Identifier quels neurones = 0
for dual_point in dual_points:
    # Marcher dans les ~256 directions
    for direction in range(256):
        x_test = dual_point + epsilon * direction
        if prediction_changes(x_test):
            # Ce neurone = 0 pour ce point
            cluster[neuron_id].append(dual_point)
```

**Coût**: ~10M d'appels API additionnels

**Semaine 5-6: Extraire les Poids**
```python
# Résoudre les équations linéaires
for neuron in network.neurons:
    cluster = clusters[neuron]
    if len(cluster) > 100:
        # Algèbre linéaire pour extraire W
        weights[neuron] = solve_ls(cluster)
```

**Coût**: Calcul local (pas plus d'API)

**Semaine 7-8: Récupérer les Signes**
```python
# Test statistique sur patch distances
for neuron in network.neurons:
    for trial in range(100):
        x = pick_dual_point(neuron)
        
        # Marcher dans direction aléatoire
        x_plus = walk(x, +direction)
        x_minus = walk(x, -direction)
        
        pred_plus = api.predict(x_plus)
        pred_minus = api.predict(x_minus)
        
        # Compter les changements
        if neuron_active:
            flip_rate_on_avg += count_flips(pred_plus, pred_minus)
        else:
            flip_rate_off_avg += count_flips(pred_plus, pred_minus)
```

**Coût**: ~50M d'appels supplémentaires

### 📊 Résultat
```
SEMAINE 8:
Attaquant a extrait:
  ✅ 165,322 poids exacts
  ✅ Architecture complète
  ✅ Peut reproduire le modèle localement
  ❌ Société ne soupçonne rien (juste du trafic normal)

Coûts:
  • API calls: ~60M (probablement pas flagué)
  • Temps: ~2 mois
  • Effort: 1 ingénieur ML
  • Valeur volée: Potentiellement millions ($)
```

### 🛡️ Défenses Possibles
```python
# Option 1: Rate Limiting
@rate_limit(requests_per_day=100, per_user=True)
def predict(input_data):
    return model(input_data)

# Option 2: Query Auditing
def predict(input_data):
    if looks_suspicious(input_data):
        # Trop de requêtes similaires
        return HTTP_429_TooManyRequests()
    return model(input_data)

# Option 3: Bruit Différentiel
def predict(input_data):
    output = model(input_data)
    # Ajouter du bruit gaussien
    noisy_output = output + gaussian_noise(sigma=0.1)
    return argmax(noisy_output)  # Plus difficile au vol

# Option 4: Défense Robuste
# Entraîner le modèle pour résister aux attaques
model = train_robust(dataset, adversary=HardLabelAttacker)
```

---

## 2. 📈 Exemple Numériques - Petit Modèle

### Modèle Cible
```
Architecture: 64 → 256 → 256 → 256 → 64 → 10

Exemple d'entrée:
  x = [0.1, -0.3, 0.5, ..., 0.2]  (64 nombres)

Prédiction:
  Layer 1: x → z1 = ReLU(W1·x + b1)  [256 valeurs]
  Layer 2: z1 → z2 = ReLU(W2·z1 + b2) [256 valeurs]
  Layer 3: z2 → z3 = ReLU(W3·z2 + b3) [256 valeurs]
  Layer 4: z3 → z4 = ReLU(W4·z3 + b4) [64 valeurs]
  Layer 5: z4 → y = W5·z4 + b5        [10 valeurs - logits]
  
  Décision: argmax(y) = 5
```

### Points Duaux - Propriétés
```
Point dual x_d pour neurone j de couche 1:

Propriété 1 - Sur frontière:
  score_5(x_d) ≈ score_6(x_d)
  À quelques décimales près (p.ex. 0.001 vs 0.002)

Propriété 2 - Neurone zéro:
  W1[j] · x_d + b1[j] ≈ 0
  Exactement zéro après ReLU

Exemple réel:
  x_d = [0.234, -0.567, 0.123, ...]
  
  Avant ReLU: W1[j]·x_d + b1[j] = 1e-15
  Après ReLU: z1[j] = ReLU(1e-15) = 0.0 ✓
  
  Score sortie:
    score_class_5 = 0.4501
    score_class_6 = 0.4499
    Différence: 0.0002 ← équilibré ✓
```

### Clustering - Exemple
```
Supposons 1000 points duaux trouvés
Pour chaque point, identifier quel neurone = 0:

Point 1: Neurone 0 de couche 1 est zéro → group[0].append(pt1)
Point 2: Neurone 0 de couche 1 est zéro → group[0].append(pt2)
Point 3: Neurone 1 de couche 1 est zéro → group[1].append(pt3)
Point 4: Neurone 2 de couche 1 est zéro → group[2].append(pt4)
...
Point 999: Neurone 255 de couche 1 est zéro → group[255].append(pt999)

Résultat:
  group[0] = [pt1, pt2, pt7, ...]  # ~4 points
  group[1] = [pt3, pt15, ...]      # ~3 points
  group[2] = [pt4, pt9, pt18, ...] # ~5 points
  ...
  group[255] = [pt999]              # ~6 points
```

### Récupération des Poids - Mathématiques
```
Pour neurone j:
  Nous avons ensemble de points X_j = [x_1, x_2, ..., x_k]
  Où POUR CHAQUE point x_i en X_j:
    W_j · x_i + b_j = 0  (propriété du point dual)
  
  Donc: W_j · x_i = -b_j
  
  Écrire comme système:
    [x_1^T]     [-b_j]
    [x_2^T]     [-b_j]
    [...]   W_j = [...]
    [x_k^T]     [-b_j]
  
  Résoudre avec least squares:
    W_j = (X_j^T X_j)^-1 X_j^T (-b_j)
  
  Résultat: Vecteur de poids W_j = [-0.234, 0.567, -0.123, ...]
            (magnitudes seulement - signes perdus lors du ReLU!)
```

### Sign Recovery - Patch Distances
```
Neurone cible: Neurone 5 de couche 0

Test:
  1. Prendre un point dual pour ce neurone
  2. Marcher légèrement dans direction aléatoire
  3. Compter: prédiction change-t-elle?

Résultats empiriques (100 essais):

Si neurone 5 est ACTIVÉ (ON):
  │direction1: changement 8 fois sur 10
  │direction2: changement 7 fois sur 10
  │direction3: changement 9 fois sur 10
  └─ Moyenne: 8/10 = 80% flips
  
Si neurone 5 est DÉSACTIVÉ (OFF):
  │direction1: changement 3 fois sur 10
  │direction2: changement 2 fois sur 10
  │direction3: changement 4 fois sur 10
  └─ Moyenne: 3/10 = 30% flips

Différence: 80% - 30% = 50% ← statistiquement significatif ✓

Conclusion: Neurone 5 est probablement ON
```

---

## 3. 🎯 Cas Pratiques - Où Cela S'applique

### ❌ VULNÉRABLE
```
1. APIs de classification simples
   POST /classify {"image": [...]}
   Response: {"class": 5}

2. Systèmes de détection fraude
   POST /is_fraud {"transaction": {...}}
   Response: {"is_fraud": true}

3. Filtres de contenu
   POST /is_spam {"text": "..."}
   Response: {"spam": false}

4. Systèmes de recommandation (si accès seul)
   POST /recommend {"user_id": 123}
   Response: {"item_id": 456}

5. Tout modèle déployé sans défense!
```

### ✅ DÉFENDU
```
1. Avec rate limiting strict
   Max 10 appels/jour/utilisateur
   
2. Avec auditing actif
   Détecte patterns d'attaque
   Bloque les IPs suspectes
   
3. Avec différential privacy
   Bruit ajouté aux réponses
   
4. Non-ReLU architectures
   Sigmoid, tanh → différentes faiblesses
   
5. Ensemble de modèles
   Demande extraction de PLUSIEURS modèles
   Exponentiellement plus d'appels
```

---

## 4. 📊 Comparaison avec Autres Attaques

| Attaque | Setting | Requêtes | Temps | Implication |
|---------|---------|----------|-------|------------|
| Distillation Simple | Access à logits | 1k-10k | Heures | Pas polynomial |
| Membership Inference | Black-box | 100k | Jours | Vie privée |
| **Hard-Label Extraction** | **Hard-label** | **1-10M** | **Semaines** | **POLYNOMIAL!** |
| Privilege Escalation | White-box | 0 | Instantané | Pire cas |

→ **Hard-label est entre membership inference et white-box**

---

## 5. 💻 Coût Réel - Exemple

### Ressources Nécessaires

```
Infrastructure:
  • 1 serveur Linux standard
  • 1 GPU (non requis, mais utile pour calcul)
  • ~100 GB disque (pour 10M points duaux)
  • ~16 GB RAM (pour traitement)

Logiciels:
  • Python (gratuit)
  • NumPy, PyTorch, SciPy (gratuit, open-source)
  • Code du projet (gratuit, GitHub)

Temps Humain:
  • Developpeur ML: ~2-4 semaines full-time
  • OU: étudiant PhD avec tutoriel
  • Moins: si points duaux précomputes fournis

Coût Total:
  • Temps ingé: $50k-100k
  • Infrastructure: $0-2k (cloud)
  • Total: $50k-102k
  
Comparaison:
  ✗ Modèle extractible vaut: $500k-5M
  ✓ Gain: Énorme ROI
```

---

## 6. 🔐 Recommandations Pratiques

### Pour Administrateurs d'APIs ML

```python
# 1. IMPLÉMENTER RATE LIMITING
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"]
)

@app.route('/predict')
@limiter.limit("10 per minute")
def predict():
    return model.predict(request.data)

# 2. AJOUTER BRUIT DIFFÉRENTIEL
import numpy as np

def predict_private(input_x, epsilon=0.5):
    output = model(input_x)
    noise = np.random.laplace(0, 1/epsilon, output.shape)
    noisy_output = output + noise
    return argmax(noisy_output)

# 3. MONITORER LES PATTERNS SUSPECTS
from collections import defaultdict
import hashlib

request_hash = defaultdict(list)

def monitor_request(data):
    h = hashlib.md5(str(data).encode()).hexdigest()
    request_hash[h].append(time.time())
    
    # Si même hash > 10 fois en 1h: suspect!
    if len(request_hash[h]) > 10:
        if time.time() - request_hash[h][0] < 3600:
            log_alert("Possible extraction attempt")
            return False
    return True

# 4. AUTHENTIFIER & AUDITER
@require_auth(roles=['approved_partner'])
@audit_log('model_queries')
def predict(input_x):
    return model.predict(input_x)
```

### Pour Chercheurs ML

```
Options pour sécuriser vos modèles:

1. Certified Defenses
   └─ Garantie mathématique contre TOUTE extraction
   
2. Adversarial Training
   └─ Entraîner le modèle contre les attaquants
   
3. Membership Inference Tests
   └─ Vérifier que le modèle ne révèle pas info privée
   
4. Extraction Games
   └─ Évaluer votre modèle contre extraction automatique
```

---

## 7. 🚨 Cas d'Étude Hypothétique

### Scénario: Startup détectant les fraudes

**Situation Initiale:**
```
Startup a créé modèle de détection fraude
Valeur estimée: $2M
Clients: 50 banques
API: JSON simple
  POST /api/check_transaction
  → Response: {"is_fraud": true/false}
```

**Attaque (Timeline 8 semaines):**
```
Semaine 1-2: Équipe crée 10k points duaux
             Teste API avec patterns aléatoires
             Pas flagué (normal teste/bugfix)

Semaine 3-4: Clustérise les points
             Évalue beaucoup de perturbations
             Observé par monitoring mais "normal" teste

Semaine 5-6: Extrait les poids (calcul local)
             Vérifie sur test set local
             API n'est plus interrogée

Semaine 7-8: Teste extraction vs original API
             Taux d'accord: 99.2% ✓
             Modèle prêt à utiliser

Résultat:
  ✗ Startup a perdu sa propriété intellectuelle
  ✗ Competitors lancent modèles gratuits
  ✗ Revenue baisse de 90% en 6 mois
  ✗ Startup ferme
```

**Avec Défenses:**
```
Semaine 1-2: Attaquant crée points duaux
             Après 100 appels suspect: bloqué
             IP bannée + alerte envoyée
  
Résultat:
  ✓ Attaque détectée rapidement
  ✓ Données extractibles limités
  ✓ Équipe sécurité peut investiguer
  ✓ Startup continue fonctionnement normal
```

---

## 📝 Résumé des Exemples

| Exemple | Principe | Apprendre |
|---------|----------|-----------|
| API Compromise | Cas réel pratique | Défenses critiques |
| Numériques | Mathématiques précises | Fonctionnement précis |
| Cas pratiques | Où ça s'applique | Urgence du problème |
| Comparaison | Autres attaques | Unicité de hard-label |
| Coûts | Ressources réelles | Pourquoi c'est problématique |
| Recommandations | Solutions pratiques | Comment se défendre |
| Cas d'étude | Impact commercial | Stakes réels |

---

**Pour plus d'exemples**, consultez les papiers originaux et le code source sur:
https://github.com/Jchavezsaab/hard-label-dnn-extraction
