# Travail du semestre 2 (août-septembre 2026) — consolidé le 21/09/2026

Ce dossier rassemble tout le travail qui, jusqu'au 21/09/2026, vivait dispersé et **non versionné** dans `D:\thése\07-08-2026\`, `D:\thése\codes\` et `D:\thése\new\`. Voir `docs/NOTE_AVANCEMENT_20_SEPT_2026_REVISEE.md` pour l'analyse de ce que ce travail démontre réellement.

## Structure

- **`07-08-2026/`** — le cœur du travail : reproduction de l'attaque de couche de sortie (`attaque_couche_sortie.py`), notebook de reproduction multi-attaques (`Reproduction_attaques.ipynb`, `relu1.ipynb`), leur documentation (`08_Documentation_notebook.md`), le dossier de lecture des articles (`06_Dossier_13_articles.md`), et les scripts de vérification ajoutés le 21/09 (`robustesse_rho.py`, `test_echelle_poids.py`) avec leurs sorties (`*.txt`).
- **`new/`** — la révision stratégique du plan de thèse (2 août 2026) suite aux publications concurrentes de début 2026 : `01_Cahier_de_these.md`, `02_Plan_Annee_1_detaille.md` (structure WP1-WP4), `Presentation_bilan_6_mois.pptx`.
- **`codes_notebooks/`** — deuxième copie de `Reproduction_attaques.ipynb` (à vérifier si elle diverge de celle de `07-08-2026/`) et `serie-python.py`.
- **`exploration_repo_officiel/`** — fichiers produits en explorant le dépôt officiel (`../external/hard-label-dnn-extraction/`) : notes de démarrage, synthèse du mécanisme d'attaque, scripts de démo. **`RAPPORT_EXECUTION.txt` est un plantage (UnicodeEncodeError sur un emoji, Windows/PowerShell), pas un résultat** — à corriger et relancer si ces scripts sont encore utiles, sinon à ignorer.

## Le dépôt officiel

Le code de référence (Chávez-Saab, Canales-Martínez et al., EUROCRYPT 2024 / Journal of Cryptology) est en sous-module Git : `../external/hard-label-dnn-extraction/`. Il n'est pas dupliqué dans ce dépôt — après un `git clone` de `these`, lancer :

```bash
git submodule update --init --recursive
```

pour récupérer son contenu (environ 800 Mo, données CIFAR-10 incluses).

## Nettoyage encore à faire

Les emplacements d'origine (`D:\thése\07-08-2026\`, `D:\thése\codes\`, `D:\thése\new\`, `D:\thése\hard-label-dnn-extraction\`) contiennent encore les fichiers d'origine (non supprimés par prudence) et deux environnements virtuels Python (~1,8 Go à eux deux). À supprimer une fois ce dossier vérifié — voir la conversation du 21/09/2026 pour la liste exacte.
