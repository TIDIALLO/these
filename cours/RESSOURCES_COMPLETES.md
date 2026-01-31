# Ressources Complètes - Cours Deep Learning

**Thèse : Attaques d'extraction et défenses pour les DNN au-delà de ReLU**
**Doctorant : Tidiane DIALLO**

---

## 1. Vidéos Recommandées

### 1.1 Fondamentaux (Priorité Haute)

| Ressource | Durée | Description | Lien |
|-----------|-------|-------------|------|
| **3Blue1Brown - Neural Networks** | ~1h | Excellente intuition visuelle sur les réseaux de neurones | [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| **Andrej Karpathy - Micrograd** | 2h30 | Implémentation d'un réseau de neurones from scratch | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| **fast.ai - Lesson 13: Backpropagation** | 1h30 | Backpropagation et MLP en profondeur | [Course](https://course.fast.ai/Lessons/lesson13.html) |

### 1.2 Cours Complets

| Cours | Institution | Description |
|-------|-------------|-------------|
| **Neural Networks: Zero to Hero** | Andrej Karpathy | Cours complet de 0 à GPT | [Site](https://karpathy.ai/zero-to-hero.html) |
| **CS231n: CNNs for Visual Recognition** | Stanford | Réseaux convolutionnels | [Site](http://cs231n.stanford.edu/) |
| **Deep Learning Specialization** | Coursera/deeplearning.ai | 5 cours complets | [Coursera](https://www.coursera.org/specializations/deep-learning) |

---

## 2. Livres et Documentation

### 2.1 Livres Gratuits

| Livre | Auteurs | Chapitres clés | Lien |
|-------|---------|----------------|------|
| **Deep Learning Book** | Goodfellow, Bengio, Courville | Ch.6 (MLP), Ch.9 (CNN) | [deeplearningbook.org](https://www.deeplearningbook.org/) |
| **Neural Networks and Deep Learning** | Michael Nielsen | Tout le livre | [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/) |

### 2.2 Documentation Technique

- **PyTorch Tutorials** : https://pytorch.org/tutorials/
- **TensorFlow Guide** : https://www.tensorflow.org/guide

---

## 3. Tutoriels en Ligne

### 3.1 Perceptron et MLP

| Tutoriel | Source | Description |
|----------|--------|-------------|
| [Multilayer Perceptrons Guide](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning) | DataCamp | Guide complet MLP |
| [The Multilayer Perceptron](https://pabloinsente.github.io/the-multilayer-perceptron) | Pablo Insente | Théorie et implémentation |
| [MLP and Backpropagation Deep Dive](https://medium.com/@sanjay_dutta/multi-layer-perceptron-and-backpropagation-a-deep-dive-8438cc8bcae6) | Medium | Explication détaillée |

### 3.2 Fonctions d'Activation

| Tutoriel | Source | Description |
|----------|--------|-------------|
| [GELU Explained](https://www.ultralytics.com/glossary/gelu-gaussian-error-linear-unit) | Ultralytics | GELU en détail |
| [SiLU Deep Learning Guide](https://www.ultralytics.com/glossary/silu-sigmoid-linear-unit) | Ultralytics | SiLU/Swish en détail |
| [ReLU, GELU, SiLU Comparison](https://medium.com/@varun_mishra/activation-functions-in-focus-understanding-relu-gelu-and-silu-841ed1c6df0c) | Medium | Comparaison des activations |

---

## 4. Papers Fondamentaux

### 4.1 Fonctions d'Activation

| Paper | Année | Auteurs | Lien |
|-------|-------|---------|------|
| **Gaussian Error Linear Units (GELUs)** | 2016 | Hendrycks & Gimpel | [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) |
| **Searching for Activation Functions (Swish)** | 2017 | Ramachandran et al. | [arXiv:1710.05941](https://arxiv.org/abs/1710.05941) |

### 4.2 Attaques d'Extraction (Pour ta thèse)

| Paper | Venue | Description | Lien |
|-------|-------|-------------|------|
| **Cryptanalytic Extraction of Neural Network Models** | CRYPTO 2020 | Attaque sur ReLU | [Paper](https://arxiv.org/abs/2003.04884) |
| **Polynomial Time Cryptanalytic Extraction** | EUROCRYPT 2024 | Extension temps polynomial | À rechercher |
| **Hard-Label Setting** | EUROCRYPT 2025 | Extension hard-label | À rechercher |

### 4.3 Architectures

| Paper | Année | Description |
|-------|-------|-------------|
| **Deep Residual Learning (ResNet)** | 2015 | Skip connections | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| **Attention Is All You Need** | 2017 | Transformers | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |

---

## 5. Code et Repositories

### 5.1 Implémentations de Référence

| Repo | Description | Lien |
|------|-------------|------|
| **micrograd** | Autograd minimaliste par Karpathy | [GitHub](https://github.com/karpathy/micrograd) |
| **nn-zero-to-hero** | Matériaux du cours Karpathy | [GitHub](https://github.com/karpathy/nn-zero-to-hero) |

### 5.2 Ton Projet

```
/root/these-tidiane-diallo/
├── code/
│   ├── attacks/oracle.py      # Oracles pour les attaques
│   └── utils/activations.py   # Fonctions d'activation
├── cours/
│   ├── semaine1/              # Perceptron et MLP
│   └── semaine2/              # Fonctions d'activation
└── docs/
    └── PLAN_ETUDE_12_SEMAINES.md
```

---

## 6. Blogs et Articles

### 6.1 Visualisations

- **The Illustrated Transformer** : http://jalammar.github.io/illustrated-transformer/
- **The Annotated Transformer** : https://nlp.seas.harvard.edu/2018/04/03/attention.html
- **Distill.pub** : https://distill.pub/ (articles interactifs sur le ML)

### 6.2 Explications Techniques

- **Colah's Blog** : https://colah.github.io/ (excellent pour l'intuition)
- **Lilian Weng's Blog** : https://lilianweng.github.io/ (surveys techniques)

---

## 7. Outils

### 7.1 Frameworks

| Outil | Usage | Documentation |
|-------|-------|---------------|
| **PyTorch** | Deep Learning | https://pytorch.org/docs/ |
| **NumPy** | Calcul numérique | https://numpy.org/doc/ |
| **Matplotlib** | Visualisation | https://matplotlib.org/stable/contents.html |
| **SciPy** | Fonctions scientifiques | https://docs.scipy.org/doc/scipy/ |

### 7.2 Environnement

```bash
# Installation recommandée
pip install torch numpy matplotlib scipy jupyter
```

---

## 8. Planning de Lecture Suggéré

### Semaine 1 : Perceptron et MLP
- [ ] Regarder 3Blue1Brown Neural Networks (1h)
- [ ] Lire Deep Learning Book Ch.6 (3h)
- [ ] Regarder Karpathy Micrograd (2h30)
- [ ] Faire les exercices du cours (4h)

### Semaine 2 : Fonctions d'Activation
- [ ] Lire paper GELU (1h)
- [ ] Lire paper Swish/SiLU (1h)
- [ ] Lire Deep Learning Book Section 6.3 (1h)
- [ ] Faire les exercices de visualisation (3h)
- [ ] Comprendre le lien avec les attaques (2h)

---

## 9. Contacts et Communauté

### Forums et Discussions
- **r/MachineLearning** : https://reddit.com/r/MachineLearning
- **PyTorch Forums** : https://discuss.pytorch.org/

### Conférences Importantes
- **NeurIPS** : Neural Information Processing Systems
- **ICML** : International Conference on Machine Learning
- **CRYPTO/EUROCRYPT** : Conférences cryptographie (pour ta thèse)

---

*Ressources compilées le 31 Janvier 2026*
