# Ressources Deep Learning + Q&A Complet
## Tidiane DIALLO — Thèse sur l'extraction de DNN

> **Comment utiliser ce document** :
> Lisez les ressources dans l'ordre indiqué, puis testez-vous avec le Q&A.
> Si vous ne savez pas répondre → retournez à la ressource correspondante.

---

# PARTIE 1 — RESSOURCES

## Niveau 0 : Mathématiques essentielles (1-2 semaines)

### Algèbre linéaire
| Ressource | Format | Durée | Priorité |
|-----------|--------|-------|----------|
| 3Blue1Brown — Essence of Linear Algebra | YouTube (16 vidéos) | 3h | ★★★ OBLIGATOIRE |
| Khan Academy — Linear Algebra | Web interactif | À la demande | ★★ |

**Ce qu'il faut maîtriser** : vecteurs, matrices, multiplication matricielle,
transposée, valeurs propres, SVD (décomposition en valeurs singulières).

Lien : https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

### Calcul différentiel
| Ressource | Format | Durée | Priorité |
|-----------|--------|-------|----------|
| 3Blue1Brown — Essence of Calculus | YouTube (12 vidéos) | 2h30 | ★★★ OBLIGATOIRE |
| Khan Academy — Multivariable Calculus | Web interactif | À la demande | ★★ |

Lien : https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr

**Ce qu'il faut maîtriser** : dérivée, gradient, règle des chaînes (chain rule),
dérivée partielle, jacobien.

---

## Niveau 1 : Deep Learning — Fondamentaux (3-4 semaines)

### Livre de référence (GRATUIT en ligne)
```
"Deep Learning" — Goodfellow, Bengio, Courville (2016)
https://www.deeplearningbook.org/

Chapitres prioritaires pour votre thèse :
  Chapitre 6  — Deep Feedforward Networks    ★★★
  Section 6.3 — Hidden Units (activations)   ★★★
  Chapitre 7  — Regularization               ★★
  Chapitre 9  — Convolutional Networks       ★★★
```

### Cours vidéo
| Ressource | Format | Durée | Lien |
|-----------|--------|-------|------|
| 3Blue1Brown — Neural Networks (4 vidéos) | YouTube | 1h | https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi |
| Andrej Karpathy — Micrograd (from scratch) | YouTube | 2h30 | https://www.youtube.com/watch?v=VMj-3S1tku0 |
| Andrej Karpathy — makemore (LM from scratch) | YouTube | ~8h total | https://www.youtube.com/watch?v=PaCmpygFfXo |
| Stanford CS231n — CNNs | YouTube + slides | ~20h | http://cs231n.stanford.edu/ |
| fast.ai — Practical Deep Learning | Web interactif | ~40h | https://course.fast.ai/ |

### Code de référence (implémentations from scratch)
```
micrograd (Karpathy) — moteur d'autograd en 100 lignes
https://github.com/karpathy/micrograd

nanoGPT (Karpathy) — transformer from scratch
https://github.com/karpathy/nanoGPT
```

---

## Niveau 2 : Frameworks (PyTorch)

### Installation de PyTorch
```bash
# Via pip (CPU)
pip install torch torchvision torchaudio

# Via pip (GPU CUDA 11.8)
pip install torch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Via pip (GPU CUDA 12.1)
pip install torch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Via Conda (CPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Via Conda (GPU CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Voir : https://pytorch.org/get-started/locally/

### Ressources officielles
| Ressource | Lien |
|-----------|------|
| PyTorch Tutorials (officiels) | https://pytorch.org/tutorials/ |
| PyTorch — 60 Minute Blitz | https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html |
| PyTorch Documentation | https://pytorch.org/docs/stable/ |

### Cours spécialisés
| Ressource | Format | Durée |
|-----------|--------|-------|
| Patrick Loeber — PyTorch Tutorial (YouTube) | YouTube (21 vidéos) | ~5h |
| Aladdin Persson — PyTorch (YouTube) | YouTube | ~10h |

Lien Patrick Loeber : https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

---

## Niveau 3 : Architectures modernes

### Transformers & Attention
| Ressource | Format | Lien |
|-----------|--------|------|
| The Illustrated Transformer (Jay Alammar) | Blog | http://jalammar.github.io/illustrated-transformer/ |
| Attention is All You Need (paper original) | PDF | https://arxiv.org/abs/1706.03762 |
| Andrej Karpathy — GPT from scratch | YouTube (2h) | https://www.youtube.com/watch?v=kCc8FmEb1nY |

### ResNets
| Ressource | Lien |
|-----------|------|
| Deep Residual Learning (He et al. 2015) | https://arxiv.org/abs/1512.03385 |

### Modèles importants à connaître
```
BERT (2018)       — Transformer encodeur, utilise GELU
GPT-2/3/4 (2019+) — Transformer décodeur, utilise GELU
EfficientNet      — CNN, utilise SiLU/Swish
LLaMA (2023)      — LLM moderne, utilise SiLU
```

---

## Niveau 4 : Spécifique à votre thèse

### Papers fondamentaux (voir ANALYSE_PAPERS_CLES.md)
```
1. Carlini et al.          CRYPTO 2020     arxiv: 2003.04884
2. Canales-Martínez et al. EUROCRYPT 2024  eprint: 2023/1526
3. Carlini et al.          EUROCRYPT 2025  eprint: 2024/1580
4. Critique                arXiv 2025      arxiv: 2510.06692
5. MEA-Defender            IEEE S&P 2024   arxiv: 2401.15239
```

### Ressources complémentaires
| Sujet | Ressource |
|-------|-----------|
| Model Stealing | Tramèr et al. 2016 — arxiv: 1609.02943 |
| Adversarial ML | Biggio & Roli — "Wild Patterns" (survey) |
| Cryptanalyse | Cours Cryptography I — Dan Boneh (Coursera) |

---

# PARTIE 2 — QUESTIONS / RÉPONSES

> Testez-vous : lisez la question, répondez mentalement,
> puis découvrez la réponse. Si vous ne savez pas → retournez aux ressources.

---

## BLOC A — Mathématiques de base

---

**Q1. Qu'est-ce qu'un vecteur et pourquoi c'est central dans un réseau de neurones ?**

R : Un vecteur est une liste ordonnée de nombres réels, ex : x = [1.2, 0.5, 3.1].
Dans un réseau de neurones, chaque donnée d'entrée (image, texte tokenisé, etc.)
est représentée comme un vecteur. Les poids d'une couche forment une matrice W.
La sortie d'une couche = W · x + b (multiplication matricielle + biais).
Tout le calcul d'un DNN est une suite de ces multiplications matricielles.

---

**Q2. Qu'est-ce qu'un gradient ? Pourquoi en a-t-on besoin ?**

R : Le gradient est la généralisation de la dérivée à plusieurs variables.
Si f(w₁, w₂, ..., wₙ) est une fonction de perte, le gradient ∇f = [∂f/∂w₁, ..., ∂f/∂wₙ]
indique la direction de montée maximale. Pour entraîner un réseau, on fait
la descente de gradient : w ← w - α·∇f, ce qui réduit la perte à chaque étape.
Sans gradient, impossible de savoir comment modifier les poids pour améliorer le réseau.

---

**Q3. Qu'est-ce que la règle des chaînes (chain rule) ?**

R : Si f(g(x)), alors df/dx = (df/dg) · (dg/dx).
C'est la base de la rétropropagation. Dans un réseau profond :
L = loss(f₃(f₂(f₁(x))))
∂L/∂w₁ = (∂L/∂f₃) · (∂f₃/∂f₂) · (∂f₂/∂f₁) · (∂f₁/∂w₁)
On "propage" le gradient de la sortie vers l'entrée, couche par couche.

---

**Q4. Qu'est-ce qu'une dérivée partielle ?**

R : Pour f(x, y) = x² + y², la dérivée partielle par rapport à x est :
∂f/∂x = 2x  (on traite y comme une constante)
∂f/∂y = 2y  (on traite x comme une constante)
Dans un réseau, on calcule ∂Loss/∂wᵢ pour chaque poids wᵢ séparément,
ce qui donne la direction dans laquelle modifier ce poids spécifique.

---

**Q5. Qu'est-ce que la SVD et pourquoi c'est utile ?**

R : La SVD (décomposition en valeurs singulières) décompose une matrice M :
M = U · Σ · Vᵀ
où U et V sont des matrices orthogonales et Σ est diagonale (valeurs singulières).
Dans le contexte de la thèse : les attaques d'extraction de EUROCRYPT 2024
utilisent la SVD pour résoudre efficacement le système linéaire permettant
de retrouver les poids. C'est ce qui rend l'attaque polynomiale.

---

## BLOC B — Perceptron et MLP

---

**Q6. Qu'est-ce qu'un perceptron ?**

R : C'est le neurone artificiel le plus simple. Il calcule :
ŷ = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b) = activation(W·x + b)
où W = poids (appris), b = biais (appris), activation = fonction non-linéaire.
Seul, un perceptron ne peut résoudre que des problèmes linéairement séparables
(ex: AND, OR) mais pas XOR.

---

**Q7. Pourquoi XOR nécessite-t-il un réseau multicouche ?**

R : XOR n'est pas linéairement séparable : aucune droite ne peut séparer les
points (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 en deux groupes.
Un MLP (réseau multicouche) ajoute une couche cachée qui transforme l'espace
d'entrée en un espace où les classes sont séparables linéairement.
C'est la puissance de la composition de fonctions non-linéaires.

---

**Q8. Qu'est-ce que la rétropropagation (backpropagation) ?**

R : C'est l'algorithme qui calcule les gradients dans un réseau de neurones.
1. Forward pass : calculer la sortie et la perte L
2. Backward pass : calculer ∂L/∂w pour chaque poids w en appliquant
   la règle des chaînes de la couche de sortie vers la couche d'entrée
3. Mise à jour : w ← w - α · ∂L/∂w
PyTorch fait ce calcul automatiquement via son moteur autograd.
Karpathy l'implémente from scratch dans micrograd pour comprendre le mécanisme.

---

**Q9. Qu'est-ce que le forward pass et le backward pass ?**

R :
- Forward pass : passer les données x à travers le réseau pour obtenir ŷ.
  Chaque couche calcule : aₗ = activation(Wₗ · aₗ₋₁ + bₗ)
- Backward pass : calculer les gradients en partant de la perte L vers l'entrée.
  On stocke les activations du forward pass car elles sont nécessaires au backward.
Le forward calcule "ce que le réseau prédit", le backward calcule "comment corriger".

---

**Q10. Qu'est-ce que le learning rate α et comment l'ajuster ?**

R : α contrôle la taille du pas dans la descente de gradient : w ← w - α · ∇L
- α trop grand → le réseau diverge (oscille, perte augmente)
- α trop petit → apprentissage très lent
Valeurs typiques : 0.001 à 0.1 selon l'architecture.
En pratique on utilise des optimiseurs adaptatifs (Adam, AdaGrad) qui ajustent
α automatiquement pour chaque paramètre.

---

## BLOC C — Fonctions d'activation (CŒUR DE LA THÈSE)

---

**Q11. Pourquoi avoir une fonction d'activation ? Que se passe-t-il sans elle ?**

R : Sans activation, un réseau de N couches = une seule couche linéaire
(composition de fonctions linéaires = fonction linéaire).
La non-linéarité permet au réseau d'approximer des fonctions complexes.
Théorème d'approximation universelle : un MLP à une couche cachée avec
activation non-linéaire peut approximer n'importe quelle fonction continue.

---

**Q12. Qu'est-ce que ReLU ? Quelle est sa propriété mathématique fondamentale ?**

R : ReLU(x) = max(0, x)
- Pour x > 0 : ReLU(x) = x  (région linéaire, pente = 1)
- Pour x < 0 : ReLU(x) = 0  (région nulle, pente = 0)
Propriété fondamentale : ReLU est une fonction **linéaire par morceaux** (piecewise linear).
Elle n'est pas différentiable en x = 0 (c'est le "kink").
Sa dérivée : ReLU'(x) = 1 si x > 0, 0 si x < 0, non définie en x = 0.

---

**Q13. Qu'est-ce qu'un "kink" dans le contexte de votre thèse ?**

R : Un kink (point de coude) est un point où ReLU passe de 0 à active.
Dans un réseau ReLU, le kink d'un neurone i de la couche 1 se produit quand :
w₁ᵢ · x + b₁ᵢ = 0  ↔  x = -b₁ᵢ / w₁ᵢ
À ce point, le gradient de la sortie du réseau change brusquement.
C'est exactement ce que l'attaque CRYPTO 2020 exploite : en localisant ces
points par recherche binaire, on obtient de l'information sur w₁ᵢ et b₁ᵢ.

---

**Q14. Qu'est-ce que GELU ? Pourquoi est-il utilisé dans GPT et BERT ?**

R : GELU(x) = x · Φ(x) = x · (1/2)[1 + erf(x/√2)]
où Φ est la CDF (fonction de répartition) de la loi normale standard.
Propriétés :
- Lisse et différentiable partout (pas de kink)
- Légèrement négatif pour x très négatif (contrairement à ReLU)
- Plus proche de la biologie neuronale
Utilisé dans : GPT-2, GPT-3, BERT, tous les LLM modernes.
Pour votre thèse : GELU n'a pas de kinks → l'attaque CRYPTO 2020 échoue directement.

---

**Q15. Qu'est-ce que SiLU / Swish ?**

R : SiLU(x) = Swish(x) = x · σ(x) = x / (1 + e⁻ˣ)
où σ est la fonction sigmoïde.
Propriétés :
- Non-monotone (peut légèrement décroître avant de croître)
- Lisse et différentiable partout
- SiLU'(x) = σ(x) + x · σ(x) · (1 - σ(x))
Utilisé dans : EfficientNet, MobileNetV3, LLaMA.
Même problème pour votre thèse : pas de kinks → attaque directe impossible.

---

**Q16. Quelle est la différence mathématique clé entre ReLU et GELU/SiLU ?**

R : Tout est dans la dérivée seconde :
- ReLU''(x)  = 0 partout sauf en x=0 (impulsion de Dirac, discontinuité)
  → Kinks exploitables par l'attaque cryptanalytique
- GELU''(x)  ≠ 0 partout, fonction continue
  → Pas de discontinuité, gradient varie en douceur
- SiLU''(x)  ≠ 0 partout, fonction continue
  → Même constat
C'est precis´ement ce verrou mathématique que votre thèse doit surmonter.

---

**Q17. Qu'est-ce que la fonction sigmoïde ? Quand l'utiliser ?**

R : σ(x) = 1 / (1 + e⁻ˣ)   ∈ ]0, 1[
- Sortie toujours entre 0 et 1 → interprétable comme probabilité
- σ'(x) = σ(x) · (1 - σ(x))
Problème du vanishing gradient : σ'(x) ≤ 0.25, donc en multipliant
les gradients couche par couche, ils s'approchent de 0 très vite.
Utilisation actuelle : couche de sortie pour classification binaire uniquement.
Remplacée par ReLU/GELU dans les couches cachées.

---

**Q18. Qu'est-ce que Tanh ? Pourquoi est-il meilleur que sigmoïde pour les couches cachées ?**

R : tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)  ∈ ]-1, 1[
Avantage sur sigmoïde : centré en 0 → convergence plus rapide.
tanh'(x) = 1 - tanh²(x)  (max = 1, à comparer à 0.25 pour sigmoïde)
Toujours le vanishing gradient pour les réseaux très profonds.
Utilisé dans : RNN, LSTM (encore aujourd'hui).

---

## BLOC D — Architectures

---

**Q19. Qu'est-ce qu'un CNN ? Pourquoi l'utiliser pour les images ?**

R : CNN = Convolutional Neural Network.
L'opération de convolution applique un filtre (kernel) de petite taille (3×3, 5×5)
sur toute l'image en partageant les poids → bien moins de paramètres qu'un MLP.
Avantages :
- Invariance à la translation (le même filtre détecte une bordure partout)
- Hiérarchie de features : basses couches = bords, hautes couches = formes complexes
- Efficace en mémoire (partage de poids)
Composants : Conv → Activation → Pooling → (répéter) → Couches denses → Softmax

---

**Q20. Qu'est-ce qu'une skip connection (connexion résiduelle) dans ResNet ?**

R : Dans ResNet, chaque bloc calcule : y = F(x) + x
au lieu de : y = F(x)
Le terme +x est la skip connection. Avantages :
- Le gradient peut "court-circuiter" les couches profondes → pas de vanishing gradient
- Le réseau apprend à corriger (résidu) plutôt qu'à tout réapprendre
- Permet d'entraîner des réseaux de 100+ couches
Utilisé dans : ResNet (images), Transformer (NLP) avec son "Add & Norm".

---

**Q21. Qu'est-ce que le mécanisme d'attention ?**

R : L'attention permet à chaque token de "regarder" tous les autres tokens.
Pour une séquence x₁, x₂, ..., xₙ, on calcule pour chaque xᵢ :
- Q = xᵢ · Wq  (requête : "ce que je cherche")
- K = xⱼ · Wk  (clé   : "ce que j'offre")
- V = xⱼ · Wv  (valeur : "ce que je donne si sélectionné")

Attention(Q,K,V) = softmax(Q·Kᵀ / √d) · V

Le résultat : chaque token xᵢ est remplacé par une somme pondérée
des valeurs V, les poids étant la similarité Q·Kᵀ.
Utilisé dans : BERT, GPT, Vision Transformers — qui utilisent tous GELU.

---

**Q22. Qu'est-ce qu'un Transformer ?**

R : Architecture introduite en 2017 ("Attention is All You Need").
Composants clés :
1. Embedding : convertir tokens en vecteurs
2. Positional encoding : ajouter l'information de position
3. Multi-head self-attention : attention en parallèle sur plusieurs "têtes"
4. Feed-forward layer : MLP avec GELU (important pour votre thèse!)
5. Add & Norm (skip connections + normalisation)
Répéter N fois pour avoir N couches.
Tous les LLM modernes (GPT, BERT, LLaMA) utilisent cette architecture
avec GELU ou SiLU dans les couches feed-forward.

---

## BLOC E — Entraînement

---

**Q23. Qu'est-ce que la fonction de perte (loss function) ?**

R : Mesure l'écart entre la prédiction ŷ et la vraie valeur y.
Principales fonctions de perte :
- MSE (Mean Squared Error) : L = (1/n)·Σ(yᵢ - ŷᵢ)²   → régression
- Cross-entropy : L = -Σ yᵢ·log(ŷᵢ)               → classification
- BCELoss : L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]      → binaire
Plus L est petit, meilleure est la prédiction. L'entraînement minimise L.

---

**Q24. Qu'est-ce que l'overfitting et comment le prévenir ?**

R : Overfitting = le modèle "mémorise" les données d'entraînement mais
ne généralise pas sur de nouvelles données.
Symptôme : loss train ↓↓ mais loss validation ↑
Prévention :
- Dropout : désactiver aléatoirement des neurones pendant l'entraînement
- Régularisation L2 (weight decay) : pénaliser les grands poids
- Augmentation de données : créer de nouvelles données par transformation
- Early stopping : arrêter quand la loss validation cesse de diminuer

---

**Q25. Qu'est-ce que la normalisation batch (Batch Normalization) ?**

R : Technique qui normalise les activations de chaque couche à chaque mini-batch.
Pour chaque neurone, on calcule μ et σ sur le batch, puis :
x̂ = (x - μ) / σ  puis  y = γx̂ + β  (γ et β appris)
Avantages :
- Entraînement plus stable et rapide (permet de plus grands learning rates)
- Réduit la sensibilité à l'initialisation
- Effet régularisant (réduit légèrement l'overfitting)

---

**Q26. Qu'est-ce qu'Adam et pourquoi l'utiliser plutôt que SGD ?**

R : Adam (Adaptive Moment Estimation) adapte le learning rate pour chaque paramètre.
Il combine deux idées :
- Momentum : utilise la moyenne des gradients passés (évite les oscillations)
- RMSprop : divise par la moyenne des carrés des gradients (s'adapte à la magnitude)
SGD simple : w ← w - α · ∇L  (même α pour tous les poids)
Adam : α adaptatif per-poids, converge plus vite en pratique.
Paramètre par défaut recommandé : lr=1e-3, β₁=0.9, β₂=0.999.

---

## BLOC F — Sécurité ML et thèse

---

**Q27. Qu'est-ce que le MLaaS et pourquoi pose-t-il un problème de sécurité ?**

R : MLaaS = Machine Learning as a Service.
Des entreprises (Google, AWS, OpenAI) déploient des modèles entraînés comme
services API : l'utilisateur envoie une entrée x, le service retourne f(x).
Les poids du modèle restent secrets (propriété intellectuelle, sécurité).
Problème : un attaquant peut interroger l'API de nombreuses fois et tenter de
reconstruire le modèle secret → attaque d'extraction.

---

**Q28. Quelle est la différence entre white-box, black-box, et hard-label ?**

R :
- White-box   : accès complet au modèle (poids, architecture, gradient)
  → Attaque triviale, peu réaliste en pratique
- Black-box soft-label : on voit seulement les logits/probabilités de sortie
  → f(x) = [0.1, 0.7, 0.2]
  → CRYPTO 2020 et EUROCRYPT 2024 fonctionnent dans ce cadre
- Hard-label : on voit seulement la classe prédite
  → f(x) = "classe 2"
  → Plus réaliste (APIs réelles), EUROCRYPT 2025 résout ce cas pour ReLU

---

**Q29. Qu'est-ce qu'une attaque d'extraction de modèle ?**

R : Objectif : reconstruire f̂ ≈ f (le modèle cible) en observant seulement
les paires (xᵢ, f(xᵢ)) pour des entrées choisies par l'attaquant.
Deux types :
1. Model stealing : entraîner un modèle sur les réponses de l'oracle
   → f̂ imite f fonctionnellement mais pas structurellement
2. Extraction cryptanalytique (votre thèse) : récupérer les poids EXACTS
   → f̂ = f à la précision floating point près

---

**Q30. Comment fonctionne l'attaque CRYPTO 2020 (principe) ?**

R : Principe en 3 étapes :
1. Trouver les points critiques : points où un neurone ReLU commute de 0 à actif.
   On effectue une recherche binaire le long de directions aléatoires.
   À ces points, le gradient du réseau change brusquement.

2. Calculer les gradients de part et d'autre de chaque point critique.
   La différence Δg = g_après - g_avant donne de l'information sur le vecteur
   de poids du neurone concerné.

3. Résoudre le système linéaire : à partir de suffisamment de Δg,
   on peut récupérer les poids W₁, b₁, W₂, b₂ de chaque couche.

Nombre de requêtes : O(2^n) pour CRYPTO 2020, O(poly(n)) pour EUROCRYPT 2024.

---

**Q31. Pourquoi l'attaque CRYPTO 2020 échoue-t-elle sur GELU et SiLU ?**

R : L'attaque repose entièrement sur les kinks de ReLU.
Un kink existe quand : f''(x) = 0 partout SAUF à un point de discontinuité.
Pour ReLU : f''(x) = impulsion de Dirac en 0 → exploitable
Pour GELU : f''(x) = Φ(x) + x·φ(x) → continue, non nulle → PAS de kink
Pour SiLU : f''(x) continue → PAS de kink

Conséquence : si on cherche des kinks sur un réseau GELU,
on trouve du bruit numérique, pas de vrais points critiques.
Le gradient varie, mais doucement — aucun saut brusque à exploiter.

---

**Q32. Qu'est-ce qu'un neurone persistant et pourquoi est-ce un problème ?**

R : Un neurone persistant est un neurone qui reste actif (ou inactif) pour
la quasi-totalité des entrées possibles.
Exemple : si w₁·x + b₁ = 50 pour presque tout x → le neurone est TOUJOURS actif.
Problème pour l'extraction :
- L'attaque cherche des transitions actif/inactif pour trouver les kinks
- Si un neurone ne commute presque jamais, il faut un nombre exponentiel de requêtes
- C'est la critique du Paper 4 (arXiv 2025) contre EUROCRYPT 2025

---

**Q33. Qu'est-ce que le watermarking de modèle ?**

R : Insérer un "tatouage" dans le modèle pour prouver sa propriété intellectuelle.
Si quelqu'un vole le modèle, on peut vérifier si la copie volée contient
le watermark → preuve de vol.
Deux types :
- Backdoor-based : certaines entrées spéciales → sorties prédéfinies
  Problème : supprimé par l'extraction (l'attaquant ne voit pas ces patterns)
- MEA-Defender : watermark intégré dans la distribution normale des données
  → survit à l'extraction (WSR 83% après vol vs 3% pour le backdoor classique)

---

**Q34. Quel est le trade-off entre sécurité et performance dans les défenses ?**

R : Toute défense a un coût en termes de performance du modèle.
Exemples :
- Bruit sur les sorties : perturbe l'extraction mais aussi l'utilisation légitime
- Rate limiting : ralentit l'attaque mais aussi les vrais clients
- Arrondi des sorties : cache de l'information mais peut réduire l'utilité
- Watermarking MEA-Defender : quasi-nul impact sur accuracy (+0.1% max)
L'objectif de votre axe 2 : proposer des défenses avec le meilleur trade-off possible.

---

**Q35. Quel est votre positionnement dans l'état de l'art ?**

R :
```
Carlini 2020  → Extraction ReLU, soft-label, temps exponentiel
Canales 2024  → Extraction ReLU, soft-label, temps polynomial
Carlini 2025  → Extraction ReLU, hard-label, temps polynomial

Vous (2025-2028) :
  Question ouverte 1 : Extraction GELU/SiLU ?
  Question ouverte 2 : Extraction hard-label sur non-ReLU ?
  Question ouverte 3 : Défenses robustes pour ces activations ?
```
Votre contribution sera la première à traiter systématiquement l'extraction
de DNN avec activations modernes (GELU, SiLU) — sujet totalement ouvert.

---

## BLOC G — Questions de niveau thèse

---

**Q36. Pourquoi appelle-t-on ça une attaque "cryptanalytique" ?**

R : L'analogie avec la cryptanalyse est formelle.
Dans la cryptanalyse classique :
- Chiffrement = fonction secrète k : M → C (plaintext → ciphertext)
- Cryptanalyse = retrouver k depuis des paires (M, C)
Dans l'extraction DNN :
- DNN = fonction secrète f_θ : x → y (entrée → sortie)
- Extraction = retrouver θ depuis des paires (x, f_θ(x))
CRYPTO 2020 formalise cette analogie : un réseau ReLU se comporte
comme un "chiffrement" et l'extraction est une "chosen-plaintext attack".

---

**Q37. Qu'est-ce que la complexité polynomiale vs exponentielle dans ce contexte ?**

R :
- Exponentielle O(2^n) : CRYPTO 2020. Pour n=256 neurones → 2^256 opérations.
  C'est le nombre d'atomes dans l'univers observable. Inutilisable en pratique.
- Polynomiale O(poly(n)) : EUROCRYPT 2024. Pour n=256 → 256^k opérations
  avec k une constante (ex: k=3 → 16 millions). Réalisable.
Le passage exponentiel → polynomial est la contribution majeure de 2024.
Pour GELU/SiLU, on ne sait pas encore si une attaque polynomiale existe.

---

**Q38. Qu'est-ce que l'approximation universelle et en quoi est-ce lié à votre thèse ?**

R : Théorème d'approximation universelle (Cybenko, 1989) :
Un MLP à une couche cachée avec activation non-linéaire peut approximer
n'importe quelle fonction continue sur un compact à précision arbitraire.
Lien avec la thèse : si GELU et SiLU sont des activations "aussi expressives"
que ReLU, alors les réseaux GELU/SiLU encodent autant de secrets que les
réseaux ReLU. Mais la structure de l'encodage est différente (pas de kinks)
→ d'où la difficulté de les extraire.

---

**Q39. Comment évaluer la qualité d'une extraction ?**

R : Plusieurs métriques :
1. Erreur sur les poids : ||W_extrait - W_vrai||₂  → 0 = extraction parfaite
2. Fidelité : P(f_extrait(x) = f_vrai(x)) → % d'accord sur des entrées test
3. Précision : accuracy du modèle extrait sur une tâche
4. Erreur en bits : combien de bits des poids sont récupérés correctement
CRYPTO 2020 atteint erreur ~2^-40 (quasi-parfaite) sur les poids.
Pour GELU, on n'a pas encore de résultat comparable.

---

**Q40. Quelle serait la structure d'un premier article de thèse sur GELU ?**

R : Structure classique d'un article de conférence (8-10 pages) :
1. Introduction (1 page)
   - Problème : extraction de DNN, limites des méthodes ReLU
   - Contribution : premier résultat sur GELU
2. Travaux connexes (1 page)
   - CRYPTO 2020, EUROCRYPT 2024/2025, MEA-Defender
3. Préliminaires (1 page)
   - Définitions formelles, modèle d'attaque
4. Analyse de l'obstacle (2 pages)
   - Prouver formellement pourquoi CRYPTO 2020 échoue sur GELU
   - Caractériser la différence mathématique
5. Attaque proposée ou résultat d'impossibilité (3 pages)
   - Nouvelle méthode OU preuve que c'est impossible sous certaines conditions
6. Expériences (1-2 pages)
   - Reproduire sur des réseaux de taille réaliste
7. Conclusion (0.5 page)

---

# RÉCAPITULATIF RAPIDE — À mémoriser

```
Concepts fondamentaux :
  Gradient       → direction de montée de la perte
  Backprop       → chain rule appliquée de sortie vers entrée
  ReLU           → max(0,x), piecewise linear, kinks exploitables
  GELU           → x·Φ(x), lisse, pas de kinks → verrou thèse
  SiLU           → x·σ(x), lisse, pas de kinks → verrou thèse

Architecture :
  MLP            → couches denses + activations
  CNN            → convolutions + partage de poids
  ResNet         → skip connections, gradients stables
  Transformer    → attention + feed-forward GELU

Thèse :
  Attaque        → trouver kinks par recherche binaire
  Verrou         → GELU/SiLU n'ont pas de kinks
  Votre question → Comment extraire un réseau GELU/SiLU ?
  Défense        → watermarking, bruit, rate limiting
```

---

*Document créé pour Tidiane DIALLO — Thèse EDPT 2025-2028*
*Mise à jour : Avril 2026*
