# Présentation — Réunion Directeur de Thèse
## Tidiane DIALLO | 14 Avril 2026

**Directeur**: Pr. Abdoul Aziz Ciss  
**Lieu**: EPT — Thiès  
**Durée**: 30-45 min

---

# SLIDE 1 — Page de titre

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Attaques d'extraction et défenses pour les DNN        │
│         au-delà de ReLU                                 │
│                                                         │
│   ────────────────────────────────────────────────      │
│                                                         │
│   Tidiane DIALLO — Doctorant EPT                        │
│   Directeur: Pr. Abdoul Aziz Ciss                       │
│   Labo: CRISIN'2D / LTISI                               │
│                                                         │
│   Réunion d'avancement — Avril 2026                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

# SLIDE 2 — Agenda de la réunion

1. Bilan de la phase d'initiation
2. Compréhension du sujet de thèse
3. Plan d'étude validé (12 semaines)
4. Premières expérimentations
5. Questions et orientations

---

# SLIDE 3 — Bilan Phase Initiation (Janvier → Avril 2026)

## Ce qui a été fait
- Lecture du sujet et de la proposition de thèse
- Identification des 3 articles fondateurs
- Mise en place de l'environnement de travail (repo Git)
- Initiation aux concepts fondamentaux DL

## État des apprentissages
| Domaine | Niveau actuel | Objectif 3 mois |
|---------|---------------|-----------------|
| Deep Learning | Débutant | Intermédiaire |
| PyTorch | Initiation | Maîtrise bases |
| Cryptographie | Débutant | Notions de base |
| Lecture articles | En cours | 3 articles lus |

---

# SLIDE 4 — Compréhension du Problème de Thèse

## La menace: Extraction de modèle DNN

```
Entreprise A                    Attaquant
  ┌─────────┐                  ┌─────────┐
  │  Modèle │  ←── requêtes ── │         │
  │  Secret │  ──── réponses ─→│  Copie  │
  │  (DNN)  │                  │ du DNN  │
  └─────────┘                  └─────────┘
  Investissement:               Coût: 
  millions $                    quelques requêtes
```

## Enjeux
- **Économique**: Voler un modèle propriétaire (ChatGPT, etc.)
- **Sécurité**: Préparer des attaques adversariales
- **Confidentialité**: Extraire des données d'entraînement

---

# SLIDE 5 — État de l'Art: Les 3 Articles Clés

## 1. Carlini et al. (CRYPTO 2020)
- **Résultat**: Extraction exacte de réseaux ReLU en temps polynomial
- **Méthode**: Exploite les "points de coude" (kinks) de ReLU
- **Limitation**: Fonctionne seulement avec ReLU

## 2. Canales-Martínez et al. (EUROCRYPT 2024)
- **Résultat**: Version plus efficace, moins de requêtes
- **Amélioration**: Applicable à des architectures plus larges

## 3. Carlini et al. (EUROCRYPT 2025)
- **Résultat**: Fonctionne en "hard-label" (seulement la classe prédite)
- **Impact**: Plus réaliste — la plupart des API réelles sont hard-label

---

# SLIDE 6 — Positionnement de Ma Thèse

```
        CONNU                          THÈSE (TIDIANE)
   ─────────────                    ─────────────────────
   Attaque ReLU ──────────────────→ Attaque GELU/SiLU?
   (Carlini 2020)                   
                                    
   Attaque soft-label ─────────────→ Attaque hard-label GELU?
   
   Aucune défense ─────────────────→ Mécanismes de défense?
   structurée
```

## Question centrale de la thèse
> "Les attaques cryptanalytiques d'extraction de réseaux de neurones
> peuvent-elles être généralisées aux fonctions d'activation lisses
> modernes (GELU, SiLU), et quelles défenses peut-on concevoir?"

---

# SLIDE 7 — Pourquoi GELU et SiLU sont un vrai challenge

## ReLU vs GELU

```
ReLU: max(0,x)              GELU: x·Φ(x)

  │    /                      │   ___/
  │   /                       │  /
  │  /                        │ /
──┼─/──────────            ───┼/──────────────
  │                           │
  
Point de coude → EXPLOITABLE  Courbe lisse → RÉSISTANTE?
```

## Conséquence pour les attaques
- **ReLU**: On peut trouver exactement où le neurone "s'allume" (binarité)
- **GELU/SiLU**: Pas de point de coude → méthodes classiques ne fonctionnent plus
- **Challenge**: Trouver une nouvelle technique d'approximation/extraction

---

# SLIDE 8 — Plan d'Étude 12 Semaines (Validé)

```
PHASE 1 (S1-S4)          PHASE 2 (S5-S8)          PHASE 3 (S9-S12)
Apr → Mai 2026           Mai → Juin 2026          Juin → Juil 2026
─────────────────        ─────────────────        ─────────────────
✓ Maths pour DL          □ Cryptographie intro    □ Carlini 2020 (code)
✓ Perceptron/MLP         □ Sécurité ML            □ Canales 2024 (étude)
□ CNN                    □ Attaques modèle         □ Première expé
□ PyTorch maîtrisé       □ Défenses existantes    □ Idée contribution
```

---

# SLIDE 9 — Premières Expérimentations

## Code réalisé cette semaine

```python
# Simulation d'une API boîte noire avec GELU
class ModeleVictime(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),   # ← Activation cible
            nn.Linear(8, 1)
        )

# Interrogation: simuler le comportement d'un attaquant
api = APIBoiteNoire(modele)
for x in points_aleatoires:
    reponse = api.query(x)  # Collecter les réponses
```

**Observation**: GELU produit des transitions douces, sans discontinuités.
**Défi confirmé**: Impossible d'utiliser directement la méthode ReLU.

---

# SLIDE 10 — Questions pour le Directeur

1. **Lectures prioritaires**: Y a-t-il d'autres articles que vous recommandez avant Carlini 2020?

2. **Direction de recherche**: Doit-on cibler d'abord GELU (utilisé dans LLMs) ou SiLU (utilisé dans vision)?

3. **Collaboration/accès**: Avez-vous des contacts dans des équipes travaillant sur ce sujet? (Accès à des GPU pour les expériences?)

4. **Calendrier publications**: Quel est le 1er conference/journal visé? (Quand est la deadline?)

5. **Livrables**: Qu'attendez-vous pour notre prochaine réunion (dans 2 semaines)?

---

# SLIDE 11 — Objectifs Prochaines 2 Semaines

## Objectifs (15-28 Avril 2026)

| Semaine | Objectif | Livrable |
|---------|----------|----------|
| S3 (15-21 Avr) | CNN + Architectures modernes | Cours S3 + code |
| S4 (22-28 Avr) | Lecture Carlini 2020 complète | Notes de lecture |
| Prochain meeting | Présenter compréhension Carlini | Slides S2 |

---

# SLIDE 12 — Conclusion

## Ce que j'ai compris de ma thèse

> Ma thèse vise à combler un vide dans la littérature: les attaques 
> d'extraction existent pour ReLU, mais les réseaux modernes utilisent 
> GELU et SiLU. Je dois:
> 1. Concevoir de nouvelles attaques pour ces activations
> 2. Proposer des défenses efficaces
> 3. Valider sur des architectures CNN modernes

## Motivation
Les modèles comme GPT utilisent GELU. Leur sécurité face aux attaques 
d'extraction est encore une question ouverte. Ma thèse y répond.

---

# NOTES PRÉPARATOIRES (pour toi, pas à montrer)

## Vocabulaire à maîtriser avant la réunion
- **DNN**: Deep Neural Network (réseau de neurones profond)
- **MLaaS**: Machine Learning as a Service (OpenAI API, Google AI, etc.)
- **Soft-label**: Le modèle retourne les probabilités pour chaque classe
- **Hard-label**: Le modèle retourne seulement la classe gagnante
- **Kink**: Point de non-différentiabilité dans ReLU
- **Polynomial time**: Temps de calcul qui grandit polynomialement (raisonnable)

## Réponses aux questions possibles du directeur

**Q: "Avez-vous lu les articles?"**
R: "J'ai lu les abstracts, introductions et conclusions des 3 articles. 
   Je suis en train de lire Carlini 2020 en détail."

**Q: "Quel est votre niveau en maths?"**
R: "Je travaille activement les fondamentaux: algèbre linéaire, calcul 
   différentiel. J'ai un plan structuré pour les 12 prochaines semaines."

**Q: "Avez-vous du code?"**
R: "Oui, j'ai implementé un MLP avec PyTorch incluant GELU, et simulé 
   le scénario d'interrogation boîte noire."
