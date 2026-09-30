# Fiche 15 — Extraction hard-label de DNN non entièrement connectés via canaux auxiliaires (2024)

**Réf.** *A Hard-Label Cryptanalytic Extraction of Non-Fully Connected Deep Neural Networks using Side-Channel Attacks*, 2024 (eprint IACR 2024/1870 ; arXiv 2411.10174).

> Combine **hard-label** + **canaux auxiliaires (side-channel)** + architectures **non entièrement connectées**. Élargit le modèle de menace.

## Problème

Les attaques hard-label « pures » supposent des MLP entièrement connectés et n'utilisent que les requêtes. Et si l'attaquant disposait **aussi** d'information physique (temps, consommation, électromagnétique) et ciblait des architectures **non-FC** ?

## Idée clé

Coupler l'analyse hard-label (frontières de décision) avec des **fuites side-channel** pour lever des ambiguïtés que les requêtes seules ne résolvent pas, et adapter la méthode aux topologies **non entièrement connectées** (connexions creuses).

## Méthode / résultats

- Utilise des mesures physiques comme **information complémentaire** au label.
- Étend la faisabilité de l'extraction à des **architectures plus réalistes** que le MLP dense.

## Limites

- Suppose un **accès physique** au dispositif (modèle de menace plus fort).
- Spécifique à certaines plateformes matérielles.

---

## Lien avec ta thèse

- Élargit ta **carte des modèles de menace** : oracle pur vs oracle + side-channel.
- Si tu veux un volet « systèmes/embarqué », c'est la porte d'entrée. Sinon, à citer pour montrer l'étendue du domaine.
- Les **canaux auxiliaires** pourraient **révéler l'activation utilisée** — utile pour l'identification d'activation (fiches 10–11).

## Mes notes
<!-- AAAA-MM-JJ : … -->
