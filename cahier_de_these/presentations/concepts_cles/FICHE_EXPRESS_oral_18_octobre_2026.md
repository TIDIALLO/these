# Fiche express — à relire juste avant la réunion du 18 octobre 2026

But : les 16 questions que le directeur peut poser aujourd'hui, avec la réponse courte à dire de mémoire. Pour le détail complet, voir `FICHE_NOTES_concepts.md` (même dossier).

---

## Fondamentaux

**1. Pourquoi un réseau ReLU est-il facile à attaquer ?**
Parce qu'il est affine par morceaux : chaque neurone a un coude net en zéro, l'espace d'entrée est découpé en régions, et dans chaque région le réseau est une simple fonction linéaire.

**2. Qu'est-ce qu'un point critique ?**
Une entrée où un neurone caché vaut exactement zéro avant son activation. Visible seulement si on voit la sortie continue du réseau, pas en hard-label.

**3. Qu'est-ce qu'un point de transition ?**
Une entrée où l'étiquette prédite change. Visible même en hard-label, par recherche binaire entre deux points de classes différentes.

**4. Qu'est-ce qu'un point dual ?**
Un point qui est à la fois un point de transition et un point critique pour un neurone donné. C'est le pont entre ce qu'on peut voir en hard-label et l'information qu'on cherche sur les neurones cachés.

**5. Quelle est la différence entre signature et signe ?**
La signature, c'est la direction du vecteur de poids d'un neurone, à un facteur multiplicatif près. Le signe, c'est le signe de ce facteur. On les récupère toujours en deux étapes séparées, parce que ReLU(x) n'est pas égal à moins ReLU(moins x).

**6. Pourquoi le signe est-il plus dur à récupérer que la signature ?**
Parce que ReLU est positivement homogène : ReLU(c·x) = c·ReLU(x) seulement pour c positif. Multiplier par un facteur négatif change la fonction calculée, donc il faut un mécanisme séparé pour trancher le signe.

**7. Pourquoi les activations lisses changent-elles le problème du signe ?**
Parce qu'elles ne sont pas homogènes : GELU, SiLU, tanh n'ont pas la propriété σ(c·x) = c·σ(x). Le problème du signe disparaît presque, mais la normalisation de la signature devient plus difficile à poser.

**8. Raw-output contre hard-label, en une phrase ?**
Raw-output : l'attaquant voit le vecteur de scores complet. Hard-label : il ne voit que la classe gagnante. Le hard-label est le cadre réaliste, parce que la plupart des vraies API ne renvoient que l'étiquette.

**9. Pourquoi la couche de sortie résiste-t-elle aux attaques ReLU classiques ?**
Parce qu'elle n'a pas d'activation : c'est une transformation purement affine, donc aucun coude à exploiter. C'est exactement le trou que mon article de départ comble.

**10. Qu'est-ce que l'équivalence fonctionnelle ?**
Le modèle extrait n'a pas forcément les mêmes poids bruts que l'original, mais il calcule exactement la même fonction, à des permutations de neurones et des facteurs d'échelle près. C'est l'objectif visé, plus fort qu'une simple bonne précision.

---

## Ce qui existe déjà

**11. Que fait exactement le dépôt officiel (Jchavezsaab) ?**
Trois étapes : trouver des points duaux (10 000 par exécution), les regrouper par neurone, puis récupérer les poids et le signe. Les auteurs reconnaissent eux-mêmes utiliser un raccourci boîte blanche pour le regroupement, parce que la version correcte est trop lente en pratique.

**12. Le dépôt couvre-t-il autre chose que ReLU ?**
Oui, partiellement : il contient déjà un script de récupération de signe pour Leaky ReLU, avec une pente négative de 0,1.

**13. Quelle est la seule défense publiée, et comment marche-t-elle ?**
« Train to Defend » (Kurian & Aysu, NeurIPS 2025) : on entraîne le réseau avec un terme qui pousse les neurones d'une même couche à se ressembler. Ça casse le repère dont l'attaque a besoin pour les distinguer, pour moins de 1 % de perte de précision.

**14. Qu'est-ce que j'ai vérifié moi-même, avec quels chiffres ?**
Couche de sortie sur 4 activations : rang 26 sur 26, accord parfait sur 5000 entrées. Hard-label non-ReLU (TP8) : direction retrouvée à 0,98 de similarité cosinus pour ReLU et Leaky ReLU, mais 0,69 pour PReLU et 0 sur 4 pour ELU. Échelle des poids : l'écart entre ReLU et les activations lisses se réduit quand on augmente l'échelle de la première couche.

**15. Quel est le risque principal identifié cette semaine ?**
Qi et al. (ePrint 2026/178, ToSC) revendiquent être les premiers à traiter PReLU en hard-label. Ça recoupe directement ma contribution 1, et je ne l'avais pas vu en lisant seulement le résumé.

**16. Qu'est-ce que je propose de faire cette semaine ?**
Lire en entier les six travaux 2026 identifiés, en priorité celui dont l'oracle n'est pas encore confirmé (2026/253) et celui de Qi et al., puis retrancher l'axe de l'article 1 avec le directeur.

---

*Avant la réunion : relire une fois à voix haute, sans les réponses, en essayant de répondre de mémoire.*
