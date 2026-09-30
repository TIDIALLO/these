# 02 — Activations & géométrie (LE chapitre clé de ta thèse)

> Toute l'extraction repose sur la **géométrie** induite par les fonctions d'activation. Ce chapitre explique pourquoi ReLU est « facile », et ce qui change « au-delà de ReLU ». C'est le socle conceptuel de ton sujet.

## 2.1 Le zoo des activations

| Activation | Formule | Forme | Famille |
|-----------|---------|-------|---------|
| **ReLU** | max(0, x) | un coude en 0 | par morceaux linéaire |
| **Leaky ReLU** | x si x≥0, αx sinon (α≈0,01) | coude en 0, pente non nulle | par morceaux linéaire |
| **PReLU** | x si x≥0, αx sinon (**α appris**) | coude en 0, α variable | par morceaux linéaire |
| **HardTanh** | clamp(x, −1, 1) | deux coudes, bornée | par morceaux linéaire |
| **Step** | 0 si x<0, 1 sinon | discontinue | par morceaux (extrême) |
| **ELU** | x si x≥0, α(eˣ−1) sinon | lisse à gauche, linéaire à droite | hybride |
| **Sigmoïde** | 1/(1+e⁻ˣ) | S lisse, bornée [0,1] | lisse |
| **tanh** | tanh(x) | S lisse, bornée [−1,1] | lisse |
| **GELU** | x·Φ(x) | lisse, ≈ReLU au loin | lisse |
| **SiLU/Swish** | x·sigmoid(x) | lisse, ≈ReLU au loin | lisse |
| **SELU** | ELU mise à l'échelle | lisse, auto-normalisante | lisse |

**Deux grandes familles** (cette dichotomie structure ta thèse) :

- **Par morceaux linéaires** (ReLU, Leaky, PReLU, HardTanh, Step) : il existe des **coudes** = points de non-différentiabilité. Les attaques classiques **adorent** ces coudes.
- **Lisses** (sigmoïde, tanh, GELU, SiLU, SELU) : **pas de coude**, partout différentiables. Les attaques classiques **échouent** → il faut de nouveaux outils (f.11).

## 2.2 Pourquoi ReLU est « facile » : l'affine par morceaux

Avec **ReLU**, chaque neurone est soit « éteint » (z<0 → sortie 0) soit « passant » (z>0 → sortie z). Pour une entrée donnée, l'ensemble des états on/off de tous les neurones définit une **région**. Dans chaque région, **le réseau entier est une fonction AFFINE** (linéaire + constante).

L'espace d'entrée est donc découpé en **régions linéaires** (polyèdres). Aux **frontières** entre régions, un neurone passe exactement par `z = 0` : ce sont les **points critiques**.

```
   sortie
     |        /
     |       /        ← pente change brutalement au point critique
     |______/_____________ entrée
            ↑
      point critique (z = 0 pour un neurone)
```

**Pourquoi c'est exploitable :** au point critique, la **pente de la sortie change brutalement**. En mesurant ce saut de pente, on lit **les poids du neurone** (sa « signature »). C'est l'idée de Carlini 2020 (f.03).

## 2.3 Points critiques vs points de transition (raw-output vs hard-label)

- **Point critique** (raw-output) : entrée où un neurone a `z=0`. Visible comme une **cassure de pente** dans la sortie **continue**. Outil des attaques f.03–05.
- **Point de transition** (hard-label) : entrée où la **classe prédite change** (on franchit la frontière de décision). Visible même quand on ne voit que le **label**. Outil des attaques f.06–09.

C'est subtil mais central : en hard-label, on n'a plus la sortie continue, donc plus de « cassure de pente » directe. On travaille sur la **frontière de décision**, qui est elle-même un assemblage de morceaux d'hyperplans (pour ReLU).

## 2.4 Ce qui casse « au-delà de ReLU »

### Cas par morceaux non-ReLU (Leaky, PReLU, ELU, HardTanh)
- Il y a **toujours des coudes**, donc des points critiques → les attaques s'**adaptent** (f.10, f.12).
- Mais le coude n'est pas forcément en 0, la pente négative n'est pas nulle (PReLU : il faut estimer **α**), il peut y avoir **plusieurs coudes** (HardTanh).
- Bonne nouvelle (f.10) : l'**asymétrie** donne souvent **plus** d'information → le **signe** se récupère parfois **plus facilement** que pour ReLU.

### Cas lisse (sigmoïde, GELU, SiLU…)
- **Plus de coude du tout.** Le réseau n'est **plus affine par morceaux** : il est **courbe partout**.
- Les **points critiques disparaissent** → les attaques classiques ne s'appliquent pas.
- Parade (f.11) : utiliser les **dérivées d'ordre supérieur** (la courbure remplace la cassure) et les **zones quasi-linéaires** loin du centre (GELU/SiLU ≈ ReLU au loin).
- **MAIS** ces dérivées d'ordre supérieur **nécessitent la sortie continue** → en **hard-label**, elles disparaissent aussi. **C'est le trou de recherche central de ta thèse** (idée B2).

## 2.5 Le tableau mental à garder

| | Coude ? | Point critique ? | Attaque raw-output | Attaque hard-label |
|---|---|---|---|---|
| ReLU | oui (en 0) | oui | ✅ mûre (f.03–05) | ✅ mûre (f.06–09) |
| PReLU/Leaky/ELU | oui (≠0, asym.) | oui | ✅ (f.10, f.12) | ⬜ **à faire (B1)** |
| Lisses (GELU…) | **non** | **non** | ✅ via dérivées (f.11) | ⬜ **GROS trou (B2)** |

## 2.6 À retenir

1. ReLU ⇒ réseau **affine par morceaux** ⇒ **points critiques** ⇒ extraction facile.
2. Deux familles : **par morceaux** (coudes, exploitables) vs **lisses** (pas de coude, durs).
3. **Raw-output** = points critiques (cassure de pente) ; **hard-label** = points de transition (frontière de décision).
4. Les **dérivées d'ordre supérieur** sauvent le cas lisse en raw-output… mais **pas** en hard-label → ta contribution.

➡️ Suite : `03_entrainement_et_backprop.md`.
