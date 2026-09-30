# Fiche 06 — Chen, Dong, Guo, Shen, Wang, Wang (ASIACRYPT 2024)

**Réf.** Y. Chen, X. Dong, J. Guo, Y. Shen, A. Wang, X. Wang, *Hard-Label Cryptanalytic Extraction of Neural Network Models*, ASIACRYPT 2024, LNCS 15491, p. 207–236. (eprint IACR 2024/1403, arXiv 2409.11646)

> Premier à traiter le **hard-label** : l'attaquant ne voit **que le label final** (« chat »/« voiture »), pas les logits. **Tournant vers le réalisme.**

## Problème

En conditions réelles (MLaaS), l'API renvoie souvent **uniquement la classe prédite**, pas le vecteur de scores. Les attaques précédentes, qui mesurent des dérivées de la sortie continue, ne s'appliquent plus. Peut-on quand même extraire les paramètres ?

## Idée clé

Avec un label seul, on n'observe plus la sortie continue mais on peut observer la **frontière de décision** : les entrées où la classe prédite **change**. Ces **points de transition** jouent le rôle des points critiques et révèlent la géométrie des hyperplans cachés.

## Méthode

- Localiser les **frontières de décision** par recherche binaire le long de segments d'entrée.
- En déduire des contraintes sur les hyperplans des neurones, puis remonter aux paramètres couche par couche.

## Résultats

- **Faisabilité prouvée** du hard-label : nombre de requêtes **polynomial**.
- Fonctionne sur des réseaux à **petit nombre de neurones et de couches**.

## Limites

- **Temps de calcul exponentiel** → ne passe pas à l'échelle (c'est ce que Carlini 2025 corrigera).
- ReLU.

---

## Lien avec ta thèse

- Le **hard-label** est le cadre central de ton article fondateur (Canales-Martínez & Santos 2025) et de ta thèse. Cette fiche en est le point de départ conceptuel.
- Notion de **point de transition** (frontière de décision) ≠ point critique (cassure de pente) : avec des activations lisses, la frontière de décision n'a plus de cassure nette → **question ouverte** pour toi.

## Mes notes
<!-- AAAA-MM-JJ : … -->
