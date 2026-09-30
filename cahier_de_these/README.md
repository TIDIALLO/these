# Cahier de thèse

**Sujet :** Attaques d'extraction et défenses pour les réseaux de neurones profonds au-delà de ReLU

**Doctorant :** Tidiane · **Profil :** développeur Python, débutant en deep learning

**Article fondateur du sujet :** Canales-Martínez & Santos, *Extracting Some Layers of Deep Neural Networks in the Hard-Label Setting*, eprint IACR 2025/1118.

---

## À quoi sert ce dossier

C'est ton espace de travail de thèse. Il rassemble : la bibliographie résumée, les idées de recherche, le plan, un cours de deep learning pour démarrer de zéro, et des TP de code pour reproduire l'état de l'art puis aller plus loin.

## Organisation

```
cahier_de_these/
├── README.md                ← tu es ici (carte du dossier)
├── IDEE.md                  ← toutes les idées : ce qui existe + pistes neuves
├── PLAN.md                  ← plan de thèse + planning sur 3 ans
│
├── articles/                ← bibliographie commentée
│   ├── 00_index_bibliographie.md   ← table des matières + classement par thème
│   └── fiche_XX_*.md               ← un résumé structuré (~1 page) par article
│
├── ressources_DNN/          ← cours de deep learning (débutant complet)
│   ├── 01_bases_reseaux_de_neurones.md
│   ├── 02_activations_et_geometrie.md   ← LE chapitre clé pour ta thèse
│   ├── 03_entrainement_et_backprop.md
│   └── 04_glossaire_FR_EN.md
│
└── TP/                      ← travaux pratiques (Python : PyTorch + notes TensorFlow)
    ├── README.md            ← progression des TP + installation
    └── code/                ← scripts .py exécutables et testés
```

## Par où commencer (ordre conseillé)

1. **`ressources_DNN/01` à `03`** — pour avoir les bases DNN si elles ne sont pas solides.
2. **`articles/00_index_bibliographie.md`** puis les fiches dans l'ordre de lecture proposé.
3. **`TP/README.md`** — installer l'environnement, puis dérouler TP1 → TP7.
4. **`IDEE.md`** et **`PLAN.md`** — à relire chaque mois et à enrichir au fil de l'eau.

## Convention de mise à jour

Ce cahier est vivant : ajoute tes propres notes en bas de chaque fiche sous la section
`## Mes notes`. Date tes ajouts (AAAA-MM-JJ).

*Dernière génération : 2026-06-18.*
