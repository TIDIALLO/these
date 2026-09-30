# 04 — Glossaire FR/EN du domaine

> Les articles sont en anglais. Voici les termes essentiels, leur traduction et une définition courte. Garde-le ouvert pendant tes lectures.

## Concepts d'extraction

| Français | English | Définition courte |
|----------|---------|-------------------|
| Extraction de paramètres | Parameter extraction | Récupérer poids et biais d'un modèle via accès oracle. |
| Vol de modèle | Model stealing / extraction | Idem, terme générique (inclut la copie par apprentissage). |
| Accès oracle | Oracle access | On peut interroger le modèle (entrée → sortie) sans voir l'intérieur. |
| Sortie brute / logits | Raw-output / logits | Vecteur de scores réels avant softmax. |
| Étiquette dure | Hard-label | Seule la classe prédite (`argmax`) est visible. |
| Point critique | Critical point | Entrée où un neurone a pré-activation = 0 (cassure de pente). |
| Point de transition | Transition point | Entrée où la classe prédite change (frontière de décision). |
| Frontière de décision | Decision boundary | Surface séparant deux classes dans l'espace d'entrée. |
| Signature (d'un neurone) | Neuron signature | Vecteur de poids **à un facteur multiplicatif près**. |
| Récupération de signe | Sign recovery | Déterminer le signe du facteur multiplicatif d'un neurone. |
| Pelage de couche | Layer peeling | Extraire la couche 1, la « retirer », puis attaquer la couche 2, etc. |
| Équivalence fonctionnelle | Functional equivalence | Le modèle extrait calcule la **même fonction** que la cible. |
| Haute fidélité | High fidelity | Reproduire exactement la fonction (vs juste la performance). |
| Haute précision | High accuracy | Performer aussi bien sur la tâche (sans forcément même fonction). |

## Concepts de réseaux

| Français | English | Définition courte |
|----------|---------|-------------------|
| Réseau de neurones profond | Deep Neural Network (DNN) | Plusieurs couches de neurones empilées. |
| Perceptron multicouche | Multi-Layer Perceptron (MLP) | DNN entièrement connecté. |
| Entièrement connecté | Fully-connected (FC) / dense | Chaque neurone relié à toutes les entrées de la couche. |
| Couche cachée | Hidden layer | Couche intermédiaire (ni entrée ni sortie). |
| Couche de sortie | Output layer | Dernière couche (souvent sans activation). |
| Poids / biais | Weights / bias | Les paramètres appris `(W, b)`. |
| Pré-activation | Pre-activation | `z = Wx + b` avant la fonction d'activation. |
| Fonction d'activation | Activation function | Non-linéarité `σ` (ReLU, GELU, sigmoïde…). |
| Affine par morceaux | Piecewise linear | Linéaire dans chaque région (cas ReLU). |
| Région linéaire | Linear region | Zone de l'entrée où le réseau ReLU est affine. |
| Contractif (réseau/couche) | Contractive | Le nombre de neurones **décroît** d'une couche à l'autre. |

## Concepts d'entraînement

| Français | English | Définition courte |
|----------|---------|-------------------|
| Fonction de perte | Loss function | Mesure l'erreur de prédiction. |
| Descente de gradient | Gradient descent | Optimisation par pas dans le sens du gradient négatif. |
| Rétropropagation | Backpropagation | Calcul du gradient par dérivation en chaîne. |
| Taux d'apprentissage | Learning rate | Taille du pas d'optimisation `η`. |
| Régularisation | Regularization | Terme ajouté à la perte (ex. défense f.13). |
| Distillation | Distillation | Entraîner un modèle à imiter un autre. |

## Sigles à connaître

- **DNN** Deep Neural Network · **MLP** Multi-Layer Perceptron · **CNN** Convolutional NN
- **MLaaS** Machine Learning as a Service (le contexte de menace réaliste)
- **IACR** International Association for Cryptologic Research (EUROCRYPT, ASIACRYPT, CRYPTO, ToSC)
- **ToSC** Transactions on Symmetric Cryptology · **eprint** archive de prépublications IACR
- **ReLU** Rectified Linear Unit · **PReLU** Parametric ReLU · **GELU** Gaussian Error Linear Unit · **SiLU** Sigmoid Linear Unit (= Swish) · **ELU/SELU** (Scaled) Exponential Linear Unit
