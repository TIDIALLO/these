# Thèse: Attaques d'extraction et défenses pour les DNN au-delà de ReLU

**Doctorant**: Tidiane DIALLO
**Directeur**: Pr. Abdoul Aziz Ciss (EPT)
**Période**: 2025-2028
**Laboratoire**: CRISIN'2D / Équipe LTISI

## Description

Ce repository contient le code, les expériences et la documentation pour ma thèse de doctorat sur la sécurité des modèles de Deep Learning, avec un focus sur:

1. **Attaques d'extraction** de modèles utilisant des fonctions d'activation au-delà de ReLU (GELU, SiLU/Swish, etc.)
2. **Mécanismes de défense** contre ces attaques
3. **Extension aux architectures CNN**

## Structure du repository

```
these-tidiane-diallo/
├── docs/                          # Documentation
│   ├── RAPPORT_ANALYSE_THESE.md   # Analyse complète du sujet
│   └── PLAN_ETUDE_12_SEMAINES.md  # Plan de remise à niveau
├── papers/                        # Analyses des articles clés
│   └── ANALYSE_PAPERS_CLES.md
├── code/                          # Code source
│   ├── attacks/                   # Implémentations d'attaques
│   ├── defenses/                  # Implémentations de défenses
│   ├── utils/                     # Utilitaires
│   └── experiments/               # Scripts d'expériences
├── experiments/                   # Résultats d'expériences
│   ├── mnist/
│   ├── cifar10/
│   └── gelu_analysis/
├── notes/                         # Notes de travail
│   ├── weekly/                    # Notes hebdomadaires
│   └── papers/                    # Notes de lecture
└── resources/                     # Ressources externes
    └── datasets/                  # Données (non versionnées)
```

## Installation

```bash
# Cloner le repository
git clone https://github.com/TIDIALLO/these-extraction-dnn.git
cd these-extraction-dnn

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Références clés

1. Carlini, N., Jagielski, M., & Mironov, I. (2020). *Cryptanalytic Extraction of Neural Network Models*. CRYPTO 2020.

2. Canales-Martínez et al. (2024). *Polynomial Time Cryptanalytic Extraction of Neural Network Models*. EUROCRYPT 2024.

3. Carlini et al. (2025). *Polynomial Time Cryptanalytic Extraction in Hard-Label Setting*. EUROCRYPT 2025.

## Contact

- Email: tidiane.diallo@ept.sn
- Directeur: aaciss@ept.edu.sn

## License

Ce code est fourni à des fins de recherche académique uniquement.
