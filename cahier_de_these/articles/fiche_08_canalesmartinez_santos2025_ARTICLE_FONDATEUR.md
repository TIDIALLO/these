# Fiche 08 — ★ ARTICLE FONDATEUR ★ — Canales-Martínez & Santos (2025)

**Réf.** I. A. Canales-Martínez (Technology Innovation Institute), D. Santos (UNICAMP), *Extracting Some Layers of Deep Neural Networks in the Hard-Label Setting*, Cryptology ePrint Archive 2025/1118 (aussi Springer, LNCS 978-3-032-06754-8_15).

> **C'est l'article d'où vient ton sujet.** Il comble le trou laissé par Carlini 2025 (la couche de sortie) et accélère l'attaque pour les architectures **contractives**. À maîtriser à fond.

## Problème

Carlini et al. (EUROCRYPT 2025, fiche 07) récupèrent **toutes les couches sauf la couche de sortie** en hard-label. Pourquoi la couche de sortie résiste-t-elle ? **Parce qu'elle n'a pas de ReLU** : les techniques basées sur les cassures de pente / non-différentiabilité n'y trouvent aucun point critique exploitable. Comment récupérer **cette dernière couche** ?

## Idée clé

Deux contributions :

1. **Récupération de la couche de sortie** par une technique nouvelle qui n'a pas besoin de ReLU sur cette couche — elle exploite la structure linéaire finale et l'information accessible via les labels.
2. **Exploitation des couches contractives** : quand le nombre de neurones **décroît** d'une couche à l'autre (architecture « contractive »), on peut concevoir des méthodes d'extraction **plus efficaces** (temps réel d'exécution réduit).

## Méthode

- Combine la chaîne hard-label de Carlini 2025 (couches cachées) avec un nouveau module pour la **couche de sortie**.
- Tire parti de la **contraction dimensionnelle** : moins de neurones en aval ⇒ contraintes plus faciles à isoler.
- **Complexité asymptotique polynomiale** en temps et en requêtes ; l'attaque complète (Carlini + ce travail) **reste polynomiale**.

## Résultats

- Application réussie à des réseaux entraînés sur **CIFAR-10**.
- **Temps d'exécution réel réduit** pour les architectures à couches contractives.
- Première extraction hard-label couvrant **aussi** la couche de sortie.

## Limites

- La couche de sortie sans ReLU est traitée, mais **toujours dans un cadre où le reste du réseau est ReLU**.
- Le gain dépend de l'**architecture contractive** (hypothèse forte).
- Reste sur des activations classiques → **point de départ, pas point d'arrivée**.

---

## Lien avec ta thèse — le pont vers « au-delà de ReLU »

Ce travail montre, par l'exemple de la **couche de sortie**, que **dès qu'on enlève le ReLU, l'attaque doit être repensée**. Ta thèse généralise ce constat à **tout** le réseau :

- Et si **toutes** les couches utilisaient sigmoïde, tanh, GELU, SiLU, ELU, PReLU… ? (cf. fiches 10–12)
- Les « points critiques » (cassure nette du ReLU) n'existent plus pour les activations **lisses** : il faut des substituts (dérivées d'ordre supérieur, zones quasi-linéaires).
- L'astuce des **couches contractives** est-elle transposable aux activations non-ReLU ? **Question ouverte directe.**

**Plan d'action concret** : (1) reproduire l'extraction de la couche de sortie ; (2) la tester avec une couche cachée non-ReLU ; (3) mesurer où ça casse → matière à ta 1ʳᵉ contribution.

## Mes notes
<!-- AAAA-MM-JJ : … -->
