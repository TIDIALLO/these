# PLAN.md — Plan de thèse & feuille de route

**Sujet :** Attaques d'extraction et défenses pour les réseaux de neurones profonds au-delà de ReLU.

---

## 1. Problématique (en une page)

Extraire les paramètres d'un réseau de neurones via un simple accès oracle est un problème ancien, devenu un sujet **cryptanalytique** majeur depuis CRYPTO 2020 (f.03). Les progrès récents — récupération de signe polynomiale (f.04), pipeline end-to-end (f.05), passage au **hard-label** réaliste (f.06–08) — partagent **une hypothèse forte : l'activation est ReLU**. Or les réseaux modernes utilisent des activations variées (PReLU, ELU, GELU, SiLU, sigmoïde…). Les tout premiers travaux « au-delà de ReLU » (f.10, f.11, f.12, 2025–2026) restent en **raw-output** et **sans défense**.

**Question centrale.** *Comment les attaques d'extraction et leurs défenses se transforment-elles lorsque l'on quitte ReLU, en particulier dans le cadre hard-label ?*

**Sous-questions.**
1. Peut-on étendre l'extraction **hard-label** aux activations par morceaux non-ReLU, puis aux activations **lisses** ?
2. L'activation **facilite-t-elle ou entrave-t-elle** l'extraction ? Peut-on établir une **taxonomie** théorie+mesure ?
3. Les défenses (la seule existante, f.13, est conçue pour ReLU) **résistent-elles** hors ReLU, et peut-on en concevoir de spécifiques ?

---

## 2. Hypothèses & modèle de menace (à fixer tôt)

- **Accès :** oracle (entrée → label, *hard-label*) ; variante raw-output pour les baselines.
- **Connaissance :** architecture connue (nb couches/neurones) ; activation **connue** (baselines) puis **inconnue** (B4).
- **Objectif :** **haute fidélité / équivalence fonctionnelle** (f.14), pas seulement haute précision.
- **Hors périmètre (sauf extension) :** side-channel (f.15), CNN (f.16) — gardés en réserve.

---

## 3. Structure du manuscrit (cible)

1. **Introduction & motivation** (IP des modèles, MLaaS, enjeux de sécurité).
2. **Préliminaires** : DNN, activations, géométrie des régions linéaires, équivalence fonctionnelle (← `ressources_DNN/`).
3. **État de l'art** : extraction raw-output (f.03–05), hard-label (f.06–09), beyond-ReLU (f.10–12), défenses (f.13). ← bâti sur le tableau de `articles/00_index`.
4. **Contribution 1 — Hard-label au-delà de ReLU** (idées B1, B2, B3).
5. **Contribution 2 — Défenses au-delà de ReLU** (idées B6, B7, B8, B9).
6. **Contribution 3 — Taxonomie & honnêteté des bornes** (idées B3, B10, B11).
7. **Conclusion & perspectives** (CNN, side-channel, B12–B13).

---

## 4. Feuille de route sur 3 ans

### Année 1 — Fondations & reproduction (mois 1–12)

| Mois | Objectif | Livrable | Réf. |
|------|----------|----------|------|
| 1–2 | Bases DNN solides ; environnement Python (PyTorch + TF) | TP1, TP2 faits | `ressources_DNN/`, TP1–2 |
| 2–4 | Reproduire signature + signe (raw-output ReLU) | TP3, TP4 ; baseline mesurée | f.03, f.04, f.05 |
| 4–6 | Reproduire hard-label (transition points) | TP5 ; mini-attaque hard-label | f.06, f.07 |
| 6–8 | Reproduire couche de sortie + contractif (ton article) | reproduction f.08 sur CIFAR-10 | f.08 |
| 8–10 | Reproduire beyond-ReLU raw-output (Leaky/ELU/PReLU) | TP6 ; identification d'activation | f.10, f.12 |
| 10–12 | Reproduire défense « Train to Defend » | TP7 ; rapport « attaque vs défense » | f.13 |
| 12 | **Rapport de 1ʳᵉ année** + état de l'art rédigé | chapitre 3 brouillon | — |

### Année 2 — Première contribution (mois 13–24)

| Mois | Objectif | Livrable |
|------|----------|----------|
| 13–16 | **B1** : hard-label PReLU/Leaky/ELU (coude ≠ 0, estimer α) | algo + expériences |
| 16–20 | **B2** : hard-label activations lisses (courbure de frontière) | prototype + mesures |
| 20–22 | **B3/B10** : taxonomie + complexité mesurée vs profondeur | tableau + analyse |
| 22–24 | **Rédaction & soumission Papier 1** (« Hard-Label Beyond ReLU ») | soumission conf (EUROCRYPT/ToSC/NeurIPS) |

### Année 3 — Défenses, taxonomie, rédaction (mois 25–36)

| Mois | Objectif | Livrable |
|------|----------|----------|
| 25–28 | **B6** : « Train to Defend » hors ReLU ? | étude empirique |
| 28–31 | **B7/B8** : défense spécifique activations lisses + **B9** attaque adaptative | algo + éval ; **Papier 2** |
| 31–33 | Consolidation taxonomie + précision numérique (**B11**) ; **Papier 3** | synthèse |
| 33–36 | **Rédaction du manuscrit** + soutenance | manuscrit, défense |

> Cible publications : **3 papiers** (≈ 1 par contribution), alignés sur EUROCRYPT / ASIACRYPT / ToSC (IACR) et NeurIPS/ICLR (côté ML/défense).

---

## 5. Jalons & critères de succès

- **J1 (M6) :** baseline ReLU reproduite, équivalence fonctionnelle vérifiée. *Succès = erreur de fidélité < 1e-3 sur petit réseau.*
- **J2 (M12) :** beyond-ReLU raw-output reproduit + défense reproduite. Rapport d'année 1.
- **J3 (M24) :** 1ʳᵉ attaque hard-label non-ReLU fonctionnelle + Papier 1 soumis.
- **J4 (M31) :** défense non-ReLU + Papier 2 soumis.
- **J5 (M36) :** manuscrit + soutenance.

---

## 6. Risques & plans B

| Risque | Mitigation |
|--------|------------|
| Hard-label lisse (B2) trop dur | se replier sur B1 (par morceaux) + B3 taxonomie comme contribution principale |
| Reproductions chronophages | réutiliser le code public (Foerster f.05, dépôts Carlini) plutôt que repartir de zéro |
| Instabilité numérique des dérivées (lisses) | travailler en haute précision (float64/mpmath), petits réseaux d'abord |
| Trop d'idées, dispersion | suivre le **Top 3** d'IDEE.md ; une contribution = un papier |
| Défense facilement contournée | en faire une **force** (B9 : montrer la course attaque-défense) |

---

## 7. Méthodo & outils

- **Code :** Python, **PyTorch** (principal) + **TensorFlow/Keras** (pour réutiliser le code historique de Carlini). Voir `TP/README.md`.
- **Repro :** seeds fixés, scripts versionnés (git), journal de manips daté.
- **Métriques systématiques :** (1) nb de requêtes, (2) temps réel, (3) erreur de fidélité (max|Δ| poids et/ou sorties).
- **Écriture :** LaTeX, BibTeX construit depuis `articles/00_index_bibliographie.md`.
- **Veille :** suivre IACR ePrint (catégorie *Attacks and cryptanalysis*), arXiv cs.CR/cs.LG, NeurIPS/ICLR/EUROCRYPT.

---

## 8. Prochaines actions (cette semaine)

1. Lire f.03, f.07, f.08 (les 3 piliers) en notant les équations clés.
2. Installer l'environnement (`TP/README.md`) et faire **TP1** (MLP) + **TP3** (première signature).
3. Fixer ton **modèle de menace** (section 2) avec ton encadrant.
4. Démarrer le **tableau de l'état de l'art** (`articles/00_index`) — le compléter à chaque lecture.

*Dernière mise à jour : 2026-06-18.*
