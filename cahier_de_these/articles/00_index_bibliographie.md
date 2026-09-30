# Index de la bibliographie

Chaque article a une **fiche** (`fiche_XX_*.md`) au format : Référence · Problème · Idée clé · Méthode · Résultats · Limites · **Lien avec ta thèse**.

## Ordre de lecture conseillé

Lis dans cet ordre : tu pars des fondations, tu montes vers le hard-label, puis tu entres dans le cœur « au-delà de ReLU », et tu finis par les défenses.

| # | Lire | Fiche | Pourquoi |
|---|------|-------|----------|
| 1 | Carlini, Jagielski, Mironov 2020 (CRYPTO) | `fiche_03` | Acte de naissance de l'extraction « cryptanalytique ». Indispensable. |
| 2 | Canales-Martínez et al. 2024 (EUROCRYPT) | `fiche_04` | Récupération des signes en temps polynomial. |
| 3 | Foerster et al. 2024 (NeurIPS) | `fiche_05` | Première attaque end-to-end optimisée (« Beyond Slow Signs »). |
| 4 | Chen et al. 2024 (ASIACRYPT) | `fiche_06` | Premier hard-label (temps exponentiel). |
| 5 | Carlini et al. 2025 (EUROCRYPT, best paper) | `fiche_07` | Hard-label en temps polynomial. |
| 6 | **Canales-Martínez & Santos 2025** (ton article) | `fiche_08` | Couche de sortie + couches contractives. |
| 7 | Ito, Miura, Todo 2025 | `fiche_09` | « Est-ce vraiment polynomial ? » + CrossLayer Extraction. |
| 8 | Qi et al. 2026 (ToSC) — **various activations** | `fiche_10` | Cadre pour Leaky ReLU, HardTanh, ELU, Step, PReLU. |
| 9 | Non-Linear Activations 2026 (NTU) | `fiche_11` | GELU, SiLU, SELU, Sigmoid — attaque « universelle ». |
| 10 | PReLU extraction 2025 | `fiche_12` | Cas PReLU en détail (récupération de signe). |
| 11 | **Kurian & Aysu 2025** (NeurIPS) — **défense** | `fiche_13` | Première défense (« Train to Defend »). |
| 12 | Jagielski et al. 2020 (USENIX) | `fiche_14` | Fidélité vs précision, apprentissage-vol. |
| 13 | Side-channel / non-FC 2024 | `fiche_15` | Extraction par canaux auxiliaires, archi non-FC. |
| 14 | CNN extraction 2026 | `fiche_16` | Extension aux réseaux convolutifs. |
| — | Fondations historiques (Baum, Blum-Rivest, Fefferman) | `fiche_01`, `fiche_02` | Contexte théorique (NP-complétude, reconstruction). |

## Classement par thème

**Fondations théoriques (années 1990) :** Baum 1991, Blum & Rivest 1992, Fefferman 1994 → `fiche_01`, `fiche_02`.

**Extraction « raw-output » (accès aux logits) :** Carlini 2020, Canales-Martínez 2024, Foerster 2024, Daniely & Granot 2023 → `fiche_03`, `fiche_04`, `fiche_05`.

**Hard-label (accès au label seul) :** Chen 2024, Carlini 2025, Canales-Martínez & Santos 2025, Ito 2025 → `fiche_06` à `fiche_09`.

**Au-delà de ReLU (cœur de la thèse) :** Qi 2026, Non-Linear 2026, PReLU 2025 → `fiche_10`, `fiche_11`, `fiche_12`.

**Défenses :** Kurian & Aysu 2025 → `fiche_13`. (C'est un champ quasi vierge : opportunité majeure.)

**Variantes d'architecture / canaux auxiliaires :** non-FC side-channel 2024, CNN 2026 → `fiche_15`, `fiche_16`.

## Tableau récapitulatif (à compléter au fil de tes lectures)

| Réf. | Année | Setting | Activation(s) | Complexité temps | Lu ? | Reproduit ? |
|------|-------|---------|---------------|------------------|------|-------------|
| Carlini et al. | 2020 | raw-output | ReLU | poly (signe : exp) | ☐ | ☐ |
| Canales-Martínez et al. | 2024 | raw-output | ReLU | poly | ☐ | ☐ |
| Foerster et al. | 2024 | raw-output | ReLU | poly (×14,8 plus rapide) | ☐ | ☐ |
| Chen et al. | 2024 | hard-label | ReLU | exponentiel | ☐ | ☐ |
| Carlini et al. | 2025 | hard-label | ReLU | polynomial | ☐ | ☐ |
| Canales-Martínez & Santos | 2025 | hard-label | ReLU | polynomial | ☐ | ☐ |
| Ito, Miura, Todo | 2025 | hard-label | ReLU | remet en cause « poly » | ☐ | ☐ |
| Qi et al. | 2026 | raw-output | LReLU, HardTanh, ELU, Step, PReLU | poly | ☐ | ☐ |
| Non-Linear (NTU) | 2026 | raw-output | GELU, SiLU, SELU, Sigmoid | poly | ☐ | ☐ |
| PReLU extraction | 2025 | raw-output | PReLU | poly | ☐ | ☐ |
| Kurian & Aysu (défense) | 2025 | — | ReLU | — | ☐ | ☐ |

> Astuce : garde ce tableau à jour, il deviendra le squelette du chapitre « État de l'art » de ton manuscrit.
