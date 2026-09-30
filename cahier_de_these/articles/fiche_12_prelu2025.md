# Fiche 12 — Extraction des réseaux PReLU (2025)

**Réf.** *Delving into Cryptanalytic Extraction of PReLU Neural Networks*, 2025 (arXiv 2509.16620 ; Springer LNCS 978-981-95-5096-8_18).

> Étude **approfondie d'une seule activation non-ReLU** : PReLU (Parametric ReLU). Bon modèle de « monographie d'activation » pour structurer un chapitre de ta thèse.

## Problème

PReLU = ReLU avec une **pente apprise α** pour la partie négative (σ(x)=x si x≥0, αx sinon). C'est une activation **par morceaux linéaire mais asymétrique** très utilisée. Les attaques ReLU s'y appliquent-elles ? Quelles différences ?

## Idée clé

PReLU garde un **coude** (point critique) comme ReLU, donc la **signature** se récupère par les mêmes outils géométriques. Mais la **partie négative n'est plus nulle** : cela **change la récupération de signe** et ajoute un paramètre (α) à estimer. L'asymétrie peut **aider** ou compliquer selon les cas.

## Méthode

- Récupération de signature via points critiques (comme ReLU).
- Estimation de la **pente négative α** en mesurant les deux régimes de part et d'autre du coude.
- Adaptation de la **récupération de signe** au cas asymétrique.

## Résultats

- Extraction effective des réseaux PReLU.
- Analyse des cas où l'asymétrie **facilite** la récupération de signe (information des deux côtés du coude).

## Limites

- Reste **par morceaux linéaire** (cas « facile » comparé aux activations lisses de la fiche 11).
- Raw-output.

---

## Lien avec ta thèse

- **Modèle de chapitre** : « comment traiter UNE activation donnée de bout en bout ». Tu pourras répliquer cette démarche pour ELU, GELU, etc.
- PReLU est le **pont** entre ReLU (fiches 03–08) et le cas général (fiches 10–11) : commence tes expériences « beyond ReLU » par PReLU (le plus proche du connu).
- **À reproduire** : extension de TP3/TP4 avec activation PReLU (estimer α en plus de la signature).

## Mes notes
<!-- AAAA-MM-JJ : … -->
