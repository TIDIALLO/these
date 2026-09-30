# Fiche 01 — Fondations théoriques (années 1990)

> Trois résultats historiques qui posent le problème « peut-on reconstruire un réseau depuis ses sorties ? ». À lire en survol : ils donnent le vocabulaire et les bornes de complexité.

## Baum (1990–1991) — apprentissage par requêtes en temps polynomial

**Réf.** E. B. Baum, *A Polynomial Time Algorithm That Learns Two Hidden Unit Nets*, Neural Computation 2(4), 1990 ; et *Neural net algorithms that learn in polynomial time from examples and queries*, IEEE TNN 2(1), 1991.

**Problème.** Peut-on apprendre les paramètres d'un petit réseau en posant des requêtes choisies (modèle « membership queries ») ?

**Idée clé.** Pour des réseaux à très peu de neurones cachés, oui : on peut concevoir des requêtes qui isolent le comportement de chaque neurone et reconstruire le réseau en temps polynomial.

**Portée / limites.** Restreint à 2 unités cachées. Ne passe pas à l'échelle, mais introduit l'idée fondatrice : **les requêtes actives sont plus puissantes que l'observation passive**.

## Blum & Rivest (1992) — entraîner un réseau est NP-complet

**Réf.** A. L. Blum, R. L. Rivest, *Training a 3-node neural network is NP-complete*, Neural Networks 5(1), 1992.

**Idée clé.** *Entraîner* (trouver des poids qui collent à des données) un réseau à 3 nœuds est NP-complet. Important pour comprendre pourquoi *l'extraction par requêtes* (problème différent) est intéressante : elle contourne la difficulté de l'apprentissage en exploitant un accès oracle.

## Fefferman (1994) — unicité de la reconstruction

**Réf.** C. Fefferman, *Reconstructing a neural net from its output*, Revista Matemática Iberoamericana 10(3), 1994.

**Idée clé.** Si deux réseaux (activations analytiques type tanh) calculent **exactement la même fonction**, alors leurs architectures et poids sont identiques **à des symétries près** (permutation des neurones, changements de signe). C'est le résultat **d'identifiabilité** : il garantit qu'extraire la fonction revient bien, en théorie, à extraire les paramètres.

---

## Lien avec ta thèse

- Fefferman traite des activations **analytiques (tanh, sigmoïde)** — c'est-à-dire exactement le terrain « **au-delà de ReLU** » de ta thèse. Les symétries qu'il décrit conditionnent ce que tu peux espérer récupérer.
- Ces papiers justifient la **notion d'équivalence fonctionnelle** que tu retrouveras partout : on ne récupère jamais les poids « bruts », mais une classe d'équivalence (permutations + signes + échelles).
- À citer dans l'introduction « historique » de ton manuscrit, pas à reproduire en code.

## Mes notes
<!-- AAAA-MM-JJ : … -->
