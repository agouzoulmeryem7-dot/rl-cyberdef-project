# Analyse des deux politiques de référence (semaine 1)

*Source unique : `results/baselines.csv` (600 lignes), régénérable par `python scripts/run_baselines.py`. Aucun chiffre ci-dessous n'est estimé à la main : tous proviennent de ce fichier.*

## 1. Protocole d'évaluation

- **Politiques** : `random` (action uniforme parmi les 37 actions, `RandomPolicy(seed=0)`) et `restore_if_alert` (restaure l'hôte non isolé en alerte dont la moyenne glissante d'alertes est la plus haute ; sinon ne fait rien).
- **Niveaux de bruit** (provisoires) : low (FP 2 %, FN 5 %), medium (5 %, 15 %), high (10 %, 30 %).
- **Épisodes** : 100 par politique et par niveau, soit 6 × 100 = 600 lignes ; graines d'environnement 100000 à 100099 (disjointes des graines d'entraînement 0, 1, 2…). Chaque épisode dure 50 pas (troncature). Les deux politiques voient les mêmes 100 graines à chaque niveau.
- **Statistiques** : moyenne, écart-type empirique (ddof = 1), IC 95 % bootstrap (5000 rééchantillonnages) via `bootstrap_ci`.

## 2. Vérifications effectuées

| Contrôle | Résultat |
|---|---|
| Tests unitaires (`pytest`) | 12 tests passent (le README en annonce 8 : à mettre à jour par la membre 1) |
| Relance de `run_baselines.py` | CSV régénéré **identique octet pour octet** au fichier fourni |
| Lignes / épisodes | 600 ; 100 par (politique, niveau) ; 100 graines distinctes chacune |
| Valeurs manquantes | 0 |
| Niveaux FP/FN | conformes à `NOISE_LEVELS` |

Décision : **le CSV existant est conservé** (correct et reproductible) ; aucun script ni environnement n'a été modifié.

## 3. Résultats (moyenne ± écart-type ; IC 95 % bootstrap entre crochets)

| Politique | Bruit | Retour | Pic infectés | Disponibilité | Restaurations inutiles |
|---|---|---|---|---|---|
| random | (tous) | −249,5 ± 104,5 [−270,1 ; −228,3] | 0,408 ± 0,216 | 0,636 ± 0,077 | 11,8 ± 3,9 |
| restore_if_alert | low | −10,9 ± 11,1 [−13,3 ; −9,4] | 0,092 ± 0,046 | 0,980 ± 0,006 | 10,6 ± 2,9 |
| restore_if_alert | medium | −25,1 ± 40,3 [−33,8 ; −20,3] | 0,112 ± 0,098 | 0,959 ± 0,007 | 22,7 ± 3,8 |
| restore_if_alert | high | −58,7 ± 92,7 [−78,0 ; −42,3] | 0,177 ± 0,205 | 0,938 ± 0,008 | 32,6 ± 7,4 |

Autres métriques : fraction finale d'infectés (random 0,257 ; heuristique 0,000 / 0,007 / 0,049) ; isolations à tort (random 7,7 ; heuristique 0 par construction, elle n'isole jamais).

## 4. Interprétation prudente

**Random vs heuristique.** Aux trois niveaux, la différence appariée de retour (heuristique − random) est positive avec un IC 95 % loin de zéro (≈ +239 [+218 ; +259] en low, +224 en medium, +191 [+165 ; +215] en high), et le pic de compromission est plus bas. L'heuristique est donc nettement meilleure que l'aléatoire **sur cet environnement et ces coûts**. Cela ne dit rien sur sa qualité absolue ni sur d'éventuelles politiques meilleures : aucune borne optimale n'a été calculée.

**Effet du bruit sur l'heuristique.** Le retour se dégrade avec le bruit (différences appariées par graine : low → high ≈ −48 [−67 ; −32]), la disponibilité baisse (0,980 → 0,938) et les restaurations inutiles triplent (10,6 → 32,6). La dispersion croît fortement : 1 % des épisodes dépassent 25 % d'hôtes infectés au pic en low, 4 % en medium, 18 % en high. La moyenne cache donc une queue lourde de mauvais épisodes (retour minimal −473 en high) ; la médiane (−9,6 / −20,0 / −31,6) est bien moins sensible que la moyenne.

**Lecture sécurité–disponibilité.** Avec peu de bruit, l'heuristique paie surtout les restaurations inutiles (même en low, 10,6 par épisode, soit environ une par 5 pas) ; avec beaucoup de bruit, elle cumule sur-défense (plus de fausses alertes) et infections manquées (plus de faux négatifs). Les deux effets sont confondus car FP et FN varient ensemble : on ne peut pas attribuer la dégradation à l'un plutôt qu'à l'autre.

## 5. Limites d'interprétation

1. **La ligne `random` est identique aux trois niveaux de bruit.** Ce n'est pas un bug : la politique ignore les observations et les alertes consomment le même nombre de tirages aléatoires, donc les trajectoires sont les mêmes. Il s'agit d'**une** mesure répétée trois fois, non de trois mesures indépendantes ; elle ne dit rien de l'effet du bruit.
2. **Les IC mesurent la variabilité entre épisodes d'évaluation** pour une politique fixe, pas la variabilité d'entraînement (aucune graine d'apprentissage ici). Ils ne sont pas transposables aux agents DQN.
3. **Un seul réseau, une seule topologie, 12 hôtes, coûts figés** : aucune généralisation à d'autres topologies ou pondérations de coûts.
4. **Niveaux de bruit provisoires**, non issus de données réelles ; ils servent de plan expérimental, pas de modèle de réalité.
5. **Heuristique non optimisée** (un seul choix d'hôte par pas, règle fixe) : « mieux que l'aléatoire » est un seuil bas ; l'intérêt des agents appris se jugera surtout face à elle.
6. **Métriques limitées** : seules la fraction finale et le pic d'infection sont enregistrés (pas la moyenne dans le temps).
7. Les comparaisons entre niveaux de bruit pour l'heuristique reposent sur les mêmes graines d'environnement, mais les trajectoires divergent dès que les alertes changent ; l'appariement est utile, non strict.

## 6. Interface pour le DQN (membre 2)

Évaluer avec `evaluate_policy(policy, cfg, n_episodes=100, seed_base=100000)` (ε = 0), où `policy(obs) -> int` ; comparer à `results/baselines.csv` au même niveau de bruit. Un agent n'a de valeur que s'il bat l'heuristique, pas seulement l'aléatoire.
