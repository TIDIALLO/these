# Fiche 09 — Ito, Miura, Todo (NTT, 2025)

**Réf.** A. Ito, T. Miura, Y. Todo (NTT Social Informatics Laboratories), *Is the Hard-Label Cryptanalytic Model Extraction Really Polynomial?*, Cryptology ePrint Archive 2025/1868.

> Article **critique** : il remet en cause le « polynomial » de Carlini 2025 et propose une parade (**CrossLayer Extraction**). Très utile pour problématiser ta thèse.

## Problème

Carlini et al. 2025 (fiche 07) affirment une extraction hard-label en **temps polynomial**. Ito et al. montrent que les **hypothèses** sous-jacentes deviennent **de moins en moins réalistes quand la profondeur augmente** : les satisfaire exigerait un **nombre exponentiel de requêtes** vis-à-vis de la profondeur attaquée. Conclusion : l'attaque **n'est pas toujours polynomiale** en pratique.

## Idée clé — CrossLayer Extraction

Au lieu d'extraire directement les paramètres d'un neurone **profond** (coût exponentiel sous les hypothèses réalistes), exploiter les **interactions entre neurones de couches différentes** pour récupérer l'information depuis les couches profondes. On « croise les couches » plutôt que d'attaquer chaque couche isolément.

## Méthode

- Analyse fine des hypothèses de Carlini 2025 et de leur coût réel en requêtes selon la profondeur.
- Nouvelle stratégie d'extraction inter-couches réduisant fortement la **complexité en requêtes**.

## Résultats

- Met en évidence un **écart théorie/pratique** important dans l'état de l'art hard-label.
- Réduit la complexité en requêtes et **atténue** la limitation identifiée.

## Limites

- Reste dans le cadre **ReLU**.
- Analyse encore récente (preprint) — à confronter aux réponses futures de la communauté.

---

## Lien avec ta thèse

- **Excellent matériau de problématisation** : « les bornes polynomiales annoncées tiennent-elles vraiment ? » est une question que tu peux **rejouer pour les activations non-ReLU**.
- L'idée **CrossLayer** (exploiter les interactions inter-couches) est une **technique transférable** : avec des activations lisses, les couches « fuient » différemment de l'information — piste de méthode.
- À citer dès que tu discutes des **complexités réelles** de tes propres attaques (ne pas se contenter de l'asymptotique).

## Mes notes
<!-- AAAA-MM-JJ : … -->
