# Fiche 11 — ★ CŒUR DE LA THÈSE ★ — Extraction avec activations non linéaires (NTU, 2026)

**Réf.** *Cryptanalytic Extraction of Deep Neural Networks with Non-Linear Activations*, Cryptology ePrint Archive 2026/253 (équipe SPMS, NTU Singapore ; séminaire déc. 2025).

> Attaque **« universelle »** pour les activations **lisses** (GELU, SiLU, SELU, sigmoïde…). Complète la fiche 10 sur le versant *non* linéaire par morceaux.

## Problème

Les activations modernes (GELU dans les Transformers, SiLU/Swish, SELU, sigmoïde, tanh) sont **lisses et partout différentiables** : **pas de point critique** au sens ReLU (pas de cassure de pente). Les attaques classiques s'effondrent. Peut-on quand même extraire ?

## Idée clé

Beaucoup de ces activations **convergent vers un comportement linéaire** loin de leur petite zone non linéaire centrale. On peut donc :

1. travailler dans les **zones quasi-linéaires adjacentes** (le réseau y est presque affine, comme un ReLU « ouvert ») ;
2. exploiter les **dérivées d'ordre supérieur** pour caractériser la zone non linéaire — ce qui **remplace** la non-différentiabilité du ReLU.

C'est la **première attaque boîte-noire universelle** récupérant **poids ET biais** pour cette classe d'activations.

## Méthode

- Généralise les approches **géométriques** (régions linéaires) au cas lisse via analyse des **zones linéaires adjacentes** + dérivées d'ordre ≥ 2.
- Ne suppose **pas** de non-différentiabilité.

## Résultats

- Récupération réussie pour **GELU, SiLU, SELU, sigmoïde** et d'autres.
- Pour **plusieurs activations, la signature se récupère plus facilement que pour ReLU** (la forme lisse encode plus d'information locale).
- **Identification de l'activation** possible quand elle est inconnue.

## Limites

- Cadre **raw-output** (dérivées de la sortie continue) → **pas hard-label**.
- Coût des dérivées d'ordre supérieur (sensibilité numérique).
- Suppose la convergence vers un régime linéaire (vrai pour GELU/SiLU/SELU, faux pour des activations bornées comme sigmoïde sur tout le domaine — nuances).

---

## Lien avec ta thèse — central

- Avec la fiche 10, ce sont **tes deux piliers « beyond ReLU »**. Ensemble : par morceaux (fiche 10) **et** lisse (fiche 11).
- **Le grand trou** : ces deux articles sont **raw-output**. Le **hard-label pour activations lisses** est **inexploré** → candidat n°1 pour ta contribution majeure.
- Les **dérivées d'ordre supérieur** disparaissent en hard-label (plus de sortie continue) : il faudra inventer des substituts via la **frontière de décision** → vrai défi de recherche, vrai apport potentiel.
- **À reproduire** : TP6 (signature d'un neurone sigmoïde/GELU en raw-output, puis tenter le hard-label et documenter l'échec/les pistes).

## Mes notes
<!-- AAAA-MM-JJ : … -->
