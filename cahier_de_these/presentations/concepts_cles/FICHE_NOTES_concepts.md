# Fiche de notes — Maîtriser les concepts clés de la thèse

> But de cette fiche : te rendre **fluide** sur tous les termes que tu vas croiser dans les articles et présenter en séminaire. Lis-la une fois en entier, puis garde-la comme référence. Chaque concept = définition simple + intuition + pourquoi c'est important pour TA thèse.

**Phrase à savoir réciter (l'objectif en une phrase) :**
> « J'étudie comment **voler les paramètres** d'un réseau de neurones en l'interrogeant comme une boîte noire, et comment **s'en défendre** — en particulier quand le réseau utilise des **fonctions d'activation autres que ReLU**, et quand l'attaquant ne voit que **l'étiquette finale** (hard-label). »

---

## 1. Le cadre général : extraction de modèle

### 1.1 Qu'est-ce qu'on essaie de faire ?

Un réseau de neurones entraîné = des millions de **paramètres** (poids et biais) qui ont coûté cher (données, calcul, expertise). Ces paramètres sont une **propriété intellectuelle**. Question : **si je peux seulement interroger le réseau** (lui donner des entrées, lire des sorties), **puis-je reconstruire ses paramètres ?**

- **Accès oracle / boîte noire** : on envoie une entrée `x`, on reçoit une sortie. On ne voit **pas** l'intérieur (pas les poids).
- **Extraction de paramètres (parameter extraction)** : reconstruire les poids/biais exacts.
- **Vol de modèle (model stealing)** : terme plus large (inclut aussi « copier la performance » sans copier les poids).

### 1.2 Deux objectifs très différents (à ne pas confondre)

| Objectif | Ce qu'on veut | Analogie |
|----------|---------------|----------|
| **Haute précision (accuracy)** | Un modèle qui **réussit la tâche** aussi bien | « Un élève qui a les mêmes bonnes notes » |
| **Haute fidélité (fidelity)** | Un modèle qui calcule **exactement la même fonction** (mêmes erreurs comprises) | « Un clone parfait, qui se trompe aux mêmes endroits » |

➡️ **Ta thèse vise la HAUTE FIDÉLITÉ** (équivalence fonctionnelle). C'est l'objectif des attaques dites « cryptanalytiques ».

### 1.3 Pourquoi « cryptanalytique » ?

Un réseau de neurones a la même structure qu'un **chiffrement par blocs** :

```
       Chiffrement par blocs          Réseau de neurones
       ----------------------         ------------------
       opération linéaire             couche linéaire (Wx+b)
        avec clé SECRÈTE       <==>    avec poids SECRETS
       suivie d'une S-box             suivie d'une activation
        PUBLIQUE (non linéaire)        PUBLIQUE (ReLU, etc.)
```

Reconstruire les poids ↔ retrouver la clé. D'où l'usage d'outils de **cryptanalyse différentielle** (observer comment la sortie change quand on perturbe l'entrée). C'est l'idée fondatrice de Carlini et al. (CRYPTO 2020).

---

## 2. Ce que voit l'attaquant : raw-output vs hard-label

C'est **LA** distinction centrale de ta thèse. Elle décrit **quelle information** l'oracle renvoie.

### 2.1 Raw-output (sortie brute / logits)

L'oracle renvoie le **vecteur de scores réels** produit par la dernière couche, AVANT toute simplification.

```
entrée x  →  [réseau]  →  logits = [2.7, -0.4, 1.1]   ← on voit TOUT le vecteur
```

- Aussi appelé **logits**.
- C'est l'information **la plus riche** → attaques les plus faciles (Carlini 2020, Canales-Martínez 2024, et les attaques *beyond ReLU* en raw-output : Qi 2026, NTU 2026).
- **Réaliste ?** Moyennement. Beaucoup d'API ne renvoient pas les logits bruts.

### 2.2 Softmax / probabilités (intermédiaire)

Certaines API transforment les logits en **probabilités** (via softmax) : `[0.72, 0.05, 0.23]`. Un peu moins d'info que les logits bruts, mais encore beaucoup.

### 2.3 Hard-label (étiquette dure)

L'oracle ne renvoie **QUE la classe gagnante** (`argmax`), c'est-à-dire un seul mot : « chat », « voiture »…

```
entrée x  →  [réseau]  →  logits = [2.7, -0.4, 1.1]  →  argmax  →  "classe 0"
                          (CACHÉ)                                  ↑ on voit SEULEMENT ça
```

- C'est le cadre **le plus réaliste** (les vrais services MLaaS renvoient souvent juste le label) et **le plus difficile** (on a perdu presque toute l'information continue).
- C'est le cadre de **ton article fondateur** (Canales-Martínez & Santos 2025) et le cœur de ta thèse.

### 2.4 Le mot à retenir

| Setting | Ce que renvoie l'oracle | Difficulté | Réalisme |
|---------|------------------------|-----------|----------|
| Raw-output / logits | vecteur de scores | facile | moyen |
| Softmax | vecteur de probas | facile-moyen | moyen |
| **Hard-label** | **un seul label** | **dur** | **élevé** |

---

## 3. Anatomie d'un réseau (le minimum)

### 3.1 Le neurone = un hyperplan + une activation

```
z = w·x + b        ← combinaison linéaire (w = poids, b = biais)
a = σ(z)           ← activation σ (la non-linéarité)
```

`w·x + b = 0` est l'équation d'un **hyperplan** (une « ligne droite » en dimension ≥ 2). **Retiens : chaque neurone = un hyperplan dans l'espace d'entrée.** Toutes les attaques cherchent à **localiser ces hyperplans**.

### 3.2 Le réseau profond = couches empilées

```
x → couche 1 → couche 2 → … → couche de sortie → logits
```

- Chaque couche : `a = σ(W·a_précédent + b)`.
- **La couche de sortie n'a généralement PAS d'activation** (juste `W·a + b`). C'est important : c'est pourquoi elle « résiste » aux attaques basées sur ReLU, et pourquoi ton article fondateur la traite séparément.
- **Paramètres** = tous les `(W, b)`. **Les extraire = voler le modèle.**

---

## 4. Non-linéarité & fonctions d'activation (le cœur du sujet)

### 4.1 Pourquoi a-t-on besoin de non-linéarité ?

Si on empile uniquement des opérations linéaires, le résultat est… encore linéaire (une matrice × une matrice = une matrice). Un réseau sans activation ne pourrait apprendre que des **droites/plans** → incapable de séparer des données complexes. **L'activation `σ` casse la linéarité** et donne au réseau son pouvoir d'expression.

```
SANS activation :   réseau = une seule fonction linéaire (inutile)
AVEC activation :   réseau = fonction complexe, courbée, expressive
```

### 4.2 Les deux grandes familles (à mémoriser absolument)

C'est la dichotomie qui **structure toute ta thèse**.

#### Famille A — Par morceaux linéaire (« piecewise linear »)

L'activation est faite de **segments de droite** recollés. Il y a des **coudes** (points anguleux).

| Activation | Formule | Allure |
|-----------|---------|--------|
| **ReLU** | `max(0, x)` | `___/` (un coude en 0) |
| **Leaky ReLU** | `x` si x≥0, `0.01x` sinon | `╲__/` (coude en 0, légère pente à gauche) |
| **PReLU** | comme Leaky mais pente α **apprise** | idem, α variable |
| **HardTanh** | `clip(x, -1, 1)` | `_/‾` (deux coudes, bornée) |

➡️ **Propriété clé :** un réseau **entièrement ReLU est AFFINE PAR MORCEAUX** — l'espace d'entrée est découpé en régions, et dans chaque région le réseau est une simple fonction linéaire. **C'est ce qui rend l'attaque facile.**

#### Famille B — Lisse (« smooth »)

L'activation est une **courbe douce**, sans aucun coude, dérivable partout.

| Activation | Formule | Allure |
|-----------|---------|--------|
| **Sigmoïde** | `1/(1+e⁻ˣ)` | `S` doux entre 0 et 1 |
| **tanh** | `tanh(x)` | `S` doux entre -1 et 1 |
| **GELU** | `x·Φ(x)` | ≈ ReLU mais arrondi (Transformers) |
| **SiLU / Swish** | `x·sigmoïde(x)` | ≈ ReLU arrondi |
| **SELU** | ELU mise à l'échelle | auto-normalisante |

➡️ **Propriété clé :** pas de coude → le réseau n'est **plus affine par morceaux**, il est **courbé partout**. **C'est ce qui casse les attaques classiques.**

### 4.3 Le tableau mental ultime

| | Coude ? | Réseau affine par morceaux ? | Attaque classique marche ? |
|---|---------|------------------------------|----------------------------|
| Famille A (ReLU, PReLU…) | **OUI** | OUI | ✅ oui |
| Famille B (sigmoïde, GELU…) | **NON** | NON | ❌ non → **ta recherche** |

---

## 5. Les points qui font fuiter l'information

### 5.1 Point critique (critical point) — cadre raw-output

C'est une entrée `x` où **un neurone vaut exactement zéro** avant activation (`w·x + b = 0`). Pour un ReLU, le neurone bascule « éteint ↔ allumé » à cet endroit → **la pente de la sortie change brutalement**.

```
sortie
  |        /
  |       /     ← CASSURE de pente = point critique
  |______/____________  entrée (le long d'une ligne)
```

**Pourquoi c'est précieux :** mesurer le **saut de pente** au point critique révèle les **poids du neurone** (sa « signature »). C'est la brique de Carlini 2020 (et de ton TP3).

### 5.2 Point de transition (transition point) — cadre hard-label

En hard-label, on ne voit plus la sortie continue → **plus de cassure de pente visible**. Mais on peut détecter les entrées où **la classe prédite change** (« chat » → « voiture »). Ces points sont sur la **frontière de décision**.

```
classe:  A A A A A | B B B B      ← le | = point de transition
                   ↑ on le trouve par RECHERCHE BINAIRE sur un segment
```

**Pourquoi c'est précieux :** les frontières de décision sont faites de morceaux d'hyperplans (pour ReLU) → en collectant beaucoup de points de transition, on reconstruit la géométrie cachée. Brique des attaques hard-label (ton TP5).

### 5.3 Segment / ligne de scan (le « comment » pratique)

On n'explore pas tout l'espace (trop grand). On se déplace le long d'un **segment** (une ligne droite) `x(t) = x₀ + t·d` en faisant varier `t`. Le long de ce segment :

- en raw-output : on cherche les **cassures de pente** (points critiques) ;
- en hard-label : on cherche les **changements de label** (points de transition), typiquement par **recherche binaire** (dichotomie) entre deux extrémités de classes différentes.

C'est le « segment » dont tu parlais : **l'outil d'exploration de base** de toutes ces attaques.

---

## 6. Ce qu'on récupère, étape par étape

L'extraction d'une couche se fait en deux temps :

1. **Signature (neuron signature)** : la **direction** du vecteur de poids, c.-à-d. les poids **à un facteur multiplicatif près**. On l'obtient via les points critiques/transition. *(C'est ce que fait ton TP3 : `|cos| = 1` → direction parfaite.)*
2. **Récupération de signe (sign recovery)** : déterminer le **signe** de ce facteur (+ ou −). Subtil, car `ReLU(w·x+b) ≠ ReLU(-w·x-b)`.
   - Carlini 2020 : signe en temps **exponentiel** (bloquant).
   - Canales-Martínez 2024 : signe en temps **polynomial** (débloque les réseaux profonds).

Puis on **« pèle » (peeling)** la couche : une fois la couche 1 connue, on la « retire » mentalement et on attaque la couche 2, etc.

**Équivalence fonctionnelle** : on ne récupère jamais les poids *bruts* à l'identique, mais une **classe d'équivalence** (à des permutations de neurones, des changements de signe et d'échelle près) qui calcule **la même fonction**. C'est mathématiquement suffisant (résultat d'identifiabilité de Fefferman 1994).

---

## 7. Au-delà de ReLU : ce qui change (le cœur de la contribution)

### 7.1 Famille A non-ReLU (PReLU, Leaky, ELU)

- Il y a **toujours un coude** → on peut **toujours** trouver des points critiques.
- **MAIS** : le coude n'est pas forcément en 0, la pente négative n'est pas nulle (il faut **estimer α**), il peut y avoir **plusieurs coudes** (HardTanh).
- **Bonne surprise** : la pente négative non nulle fait **« fuiter » de l'information des DEUX côtés** du coude → le **signe est souvent plus facile** à récupérer que pour ReLU (ton TP6 le montre : ReLU fuit d'un côté, ELU des deux).

### 7.2 Famille B lisse (sigmoïde, GELU, SiLU)

- **Plus de coude du tout** → plus de point critique → les attaques classiques **échouent**.
- Parade connue (en raw-output) : utiliser les **dérivées d'ordre supérieur** (la courbure remplace la cassure) et les **zones quasi-linéaires** loin du centre (GELU ≈ ReLU au loin). C'est l'attaque NTU 2026.
- **LE GRAND TROU (ta contribution potentielle B2)** : ces dérivées d'ordre supérieur **nécessitent la sortie continue**. En **hard-label**, on ne l'a plus ! → **personne ne sait extraire un réseau lisse en hard-label.** Piste : remplacer la courbure de la sortie par la **courbure de la frontière de décision** (les frontières lisses sont arrondies, pas anguleuses).

---

## 8. Les défenses (l'autre moitié du sujet)

- **Champ quasi vierge** : une seule défense existe (Kurian & Aysu, NeurIPS 2025).
- **Idée « Train to Defend »** : l'attaque marche d'autant mieux que les neurones d'une couche sont **distincts (uniques)**. Donc on **entraîne** le réseau pour rendre les neurones d'une même couche **similaires** (terme de régularisation dans la perte). Résultat : l'attaque ne peut plus les **séparer**. Coût : < 1 % de précision, zéro surcoût à l'inférence. *(Mécanisme illustré dans ton TP7.)*
- **Tes questions ouvertes (B6/B7)** : cette défense tient-elle pour des activations non-ReLU ? Peut-on concevoir une défense **spécifique au cas lisse** (ex. aplatir / égaliser la courbure des frontières) ?

---

## 9. Mini-lexique express (à relire avant un séminaire)

- **Oracle / boîte noire** : on interroge sans voir l'intérieur.
- **Raw-output / logits** : sortie brute (vecteur de scores).
- **Hard-label** : seulement la classe (`argmax`).
- **Hyperplan** : `w·x + b = 0`, la « frontière » d'un neurone.
- **Activation σ** : la non-linéarité (ReLU, GELU…).
- **Par morceaux linéaire** : fait de segments de droite (coudes) → ReLU & co.
- **Lisse** : courbe sans coude → sigmoïde, GELU & co.
- **Point critique** : entrée où un neurone = 0 (cassure de pente, raw-output).
- **Point de transition** : entrée où le label change (frontière de décision, hard-label).
- **Segment / ligne de scan** : `x₀ + t·d`, l'outil d'exploration.
- **Signature** : poids à un facteur près (direction).
- **Récupération de signe** : trouver le signe du facteur.
- **Peeling** : extraire une couche puis la « retirer » pour passer à la suivante.
- **Équivalence fonctionnelle** : calcule la même fonction (à symétries près).
- **Couche contractive** : le nombre de neurones décroît (exploité par ton article fondateur).

---

## 10. Auto-test (réponds sans regarder)

1. Différence entre raw-output et hard-label en une phrase ?
2. Pourquoi un réseau ReLU est-il « affine par morceaux » ?
3. Pourquoi les attaques classiques échouent-elles sur une sigmoïde ?
4. Qu'est-ce qu'un point de transition et comment le trouve-t-on ?
5. Quelle est la différence entre signature et signe ?
6. En une phrase : quel est le **trou de recherche** que ta thèse vise ?
7. Comment fonctionne la défense « Train to Defend » ?

> Si tu sais répondre aux 7, tu maîtrises le socle. Sinon, relis la section correspondante.

*Dernière mise à jour : 2026-06-18.*
