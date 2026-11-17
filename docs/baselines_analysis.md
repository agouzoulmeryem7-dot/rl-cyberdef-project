# Analyse des deux politiques de référence (semaine 1)

*Source principale : `results/baselines.csv` (600 lignes), régénérable avec `python scripts/run_baselines.py`. Les résultats présentés ci-dessous proviennent des données enregistrées dans ce fichier.*

## 1. Protocole d'évaluation

Deux politiques de référence sont évaluées :

- **Random** : choisit uniformément une action parmi les 37 actions disponibles, avec `RandomPolicy(seed=0)`.
- **Restore-if-alert** : restaure l'hôte non isolé présentant la moyenne glissante d'alertes la plus élevée ; si aucun hôte ne satisfait la condition, l'agent ne fait rien.

Trois niveaux de bruit sont considérés. Ces valeurs sont des paramètres expérimentaux provisoires et ne proviennent pas de mesures réelles :

| Niveau | Faux positifs (FP) | Faux négatifs (FN) |
|---|---:|---:|
| Low | 2 % | 5 % |
| Medium | 5 % | 15 % |
| High | 10 % | 30 % |

Le protocole comprend 100 épisodes par politique et par niveau de bruit, soit 6 configurations × 100 épisodes = 600 épisodes.

Les graines d'environnement utilisées pour l'évaluation vont de 100000 à 100099. Elles sont distinctes des graines d'entraînement utilisées pour le DQN (0, 1, 2, etc.).

Pour chaque niveau de bruit, les deux politiques sont évaluées avec les mêmes graines d'environnement afin de faciliter la comparaison. Chaque épisode comporte 50 pas et se termine par une troncature à l'horizon prévu.

Les statistiques calculées comprennent :
- La moyenne et l'écart-type empirique (ddof = 1).
- Les intervalles de confiance à 95 % obtenus par bootstrap avec 5000 rééchantillonnages.
- Les différences appariées entre politiques lorsque les graines communes permettent cette comparaison.

## 2. Vérifications effectuées

Plusieurs contrôles ont été réalisés afin de vérifier la cohérence et la reproductibilité de l'évaluation.

| Contrôle | Résultat |
|---|---|
| Tests de l'environnement | 12 tests passent |
| Tests complets après intégration du DQN | 35 tests passent |
| Exécution de `run_baselines.py` | Le CSV est régénéré avec succès |
| Nombre de lignes | 600 |
| Nombre d'épisodes par politique et niveau | 100 |
| Graines d'évaluation | 100 graines distinctes |
| Valeurs manquantes | 0 |
| Paramètres FP/FN | Conformes aux niveaux définis dans `NOISE_LEVELS` |

La régénération du fichier permet de vérifier que le protocole peut être exécuté à nouveau. Les résultats numériques ont été contrôlés à partir du CSV généré.

Le fichier `results/baselines.csv` constitue la source des résultats présentés dans cette analyse.

## 3. Résultats obtenus

Les résultats sont présentés sous la forme moyenne ± écart-type, avec l'intervalle de confiance bootstrap à 95 % entre crochets.

| Politique | Niveau de bruit | Retour cumulé | Pic d'infectés | Disponibilité | Restaurations inutiles |
|---|---|---:|---:|---:|---:|
| Random | Tous | −249,5 ± 104,5 [−270,1 ; −228,3] | 0,408 ± 0,216 | 0,636 ± 0,077 | 11,8 ± 3,9 |
| Restore-if-alert | Low | −10,9 ± 11,1 [−13,3 ; −9,4] | 0,092 ± 0,046 | 0,980 ± 0,006 | 10,6 ± 2,9 |
| Restore-if-alert | Medium | −25,1 ± 40,3 [−33,8 ; −20,3] | 0,112 ± 0,098 | 0,959 ± 0,007 | 22,7 ± 3,8 |
| Restore-if-alert | High | −58,7 ± 92,7 [−78,0 ; −42,3] | 0,177 ± 0,205 | 0,938 ± 0,008 | 32,6 ± 7,4 |

Autres métriques observées :

- Fraction finale d'hôtes infectés :
  - Random : 0,257.
  - Restore-if-alert : 0,000 en low, 0,007 en medium et 0,049 en high.
- Nombre moyen d'isolations à tort :
  - Random : 7,7.
  - Restore-if-alert : 0, par construction, car cette politique n'effectue aucune isolation.
- Médiane du retour de Restore-if-alert :
  - Low : −9,6.
  - Medium : −20,0.
  - High : −31,6.

## 4. Interprétation des résultats

### 4.1. Comparaison entre Random et Restore-if-alert

Aux trois niveaux de bruit, la politique Restore-if-alert obtient un retour nettement supérieur à celui de la politique aléatoire.

Les différences appariées de retour (heuristique − random) sont positives et leurs intervalles de confiance à 95 % sont éloignés de zéro :

- Low : environ +239 [ +218 ; +259 ].
- Medium : environ +224.
- High : environ +191 [ +165 ; +215 ].

Le pic de compromission est également plus faible avec l'heuristique.

Ces résultats indiquent que, dans l'environnement simulé et avec les coûts actuellement définis, une règle simple basée sur les alertes est nettement plus efficace qu'un choix aléatoire.

Cependant, cette observation ne signifie pas que l'heuristique est optimale. Elle constitue un point de comparaison de référence, et non une borne de performance maximale. L'objectif de la suite du projet est notamment de déterminer si les agents appris peuvent améliorer ses résultats.

### 4.2. Effet de l'augmentation du bruit sur l'heuristique

Les performances de Restore-if-alert se dégradent lorsque le niveau de bruit augmente.

Entre les niveaux low et high :

- Le retour moyen passe de −10,9 à −58,7.
- La disponibilité moyenne diminue de 0,980 à 0,938.
- Le nombre moyen de restaurations inutiles augmente de 10,6 à 32,6.
- La fraction finale d'hôtes infectés augmente de 0,000 à 0,049.

La dispersion des résultats augmente également fortement.

La proportion d'épisodes dont le pic d'infection dépasse 25 % des hôtes est de :
- 1 % en low.
- 4 % en medium.
- 18 % en high.

Le retour minimal observé en high atteint −473.

La moyenne ne décrit donc pas entièrement le comportement de la politique : certains épisodes présentent des résultats particulièrement défavorables. La médiane est moins sensible à ces épisodes extrêmes.

### 4.3. Compromis entre sécurité et disponibilité

Les résultats mettent en évidence un compromis entre la limitation de la propagation de l'infection et le maintien de la disponibilité des hôtes.

Avec un niveau de bruit faible, l'heuristique limite fortement les infections, mais effectue déjà plusieurs restaurations inutiles.

Avec un niveau de bruit élevé, les fausses alertes peuvent conduire à des interventions inutiles, tandis que les faux négatifs peuvent empêcher la détection de certaines infections.

La politique doit donc faire face à deux difficultés :
- Éviter les interventions injustifiées qui réduisent la disponibilité.
- Réagir suffisamment aux infections pour limiter leur propagation.

Les deux effets ne peuvent pas être séparés dans cette première expérience, car les taux de FP et de FN augmentent simultanément.

## 5. Limites de l'analyse

Les résultats doivent être interprétés en tenant compte des limites suivantes.

1. **Comportement identique de Random selon le niveau de bruit.** Les résultats de Random sont identiques aux trois niveaux. Cette politique ne dépend pas des observations : elle ignore donc les alertes. Dans l'implémentation actuelle, la consommation des tirages aléatoires conduit aux mêmes trajectoires. Il s'agit ainsi d'une répétition du même comportement de référence, et non de trois mesures indépendantes de l'effet du bruit.

2. **Variabilité limitée aux épisodes d'évaluation.** Les intervalles de confiance présentés mesurent la variabilité entre les épisodes pour une politique fixe. Ils ne représentent pas la variabilité entre plusieurs entraînements d'un agent appris et ne doivent pas être interprétés comme des intervalles de confiance du DQN.

3. **Environnement unique.** L'étude utilise une topologie fixe de 12 hôtes et des coûts de récompense déterminés. Les résultats ne permettent pas de généraliser directement les conclusions à d'autres topologies ou à d'autres pondérations des coûts.

4. **Niveaux de bruit provisoires.** Les valeurs FP/FN sont des paramètres expérimentaux choisis pour étudier différents niveaux de difficulté. Elles ne constituent pas un modèle statistique validé de réseaux réels.

5. **Heuristique simple.** Restore-if-alert applique une règle fixe et ne réalise qu'un choix d'hôte par pas. Elle n'a pas été optimisée. Être meilleur que Random représente donc un premier point de référence, mais pas une preuve de performance optimale.

6. **Métriques d'infection limitées.** La fraction finale d'hôtes infectés et le pic d'infection sont enregistrés. La moyenne temporelle de l'infection n'est pas encore utilisée dans cette analyse.

7. **Limite de l'appariement entre niveaux de bruit.** Les comparaisons entre niveaux utilisent les mêmes graines d'environnement. Toutefois, les trajectoires peuvent diverger dès que les alertes changent. L'appariement facilite la comparaison, mais ne garantit pas que les épisodes restent strictement identiques.

8. **Absence de conclusion sur les agents appris.** Les résultats des deux politiques de référence ne permettent pas de conclure sur la supériorité du DQN ou de ses variantes. Une comparaison séparée est nécessaire avec un protocole d'entraînement et d'évaluation contrôlé.

## 6. Conséquences pour la suite du projet

Cette première analyse fournit deux références utiles :

- Random représente une politique sans stratégie de défense.
- Restore-if-alert représente une heuristique simple qui exploite les alertes disponibles.

L'évaluation des agents appris doit donc aller au-delà d'une simple comparaison avec Random. Un agent DQN ne pourra être considéré comme une amélioration intéressante que si ses résultats sont également comparés à ceux de Restore-if-alert.

Pour assurer une comparaison aussi équitable que possible, les agents seront évalués avec :
- Le même environnement et le même niveau de bruit pour une comparaison donnée.
- Le même nombre d'épisodes d'évaluation.
- Les mêmes graines d'évaluation lorsque cela est possible.
- Une évaluation sans exploration, avec ε = 0.
- Des métriques communes : retour, infections, disponibilité et interventions inutiles.

Les résultats seront présentés sans exclure les épisodes défavorables.

## 7. Interface d'évaluation du DQN

L'interface fournie pour l'évaluation des politiques est :

`evaluate_policy(policy, cfg, n_episodes=100, seed_base=100000)`

La politique doit respecter l'interface :

`policy(obs) -> int`

L'évaluation du DQN doit être effectuée sans exploration (ε = 0) et au même niveau de bruit que les politiques de référence.

Les résultats seront ensuite comparés à ceux enregistrés dans `results/baselines.csv`.

**Conclusion :** les expériences de référence montrent qu'une heuristique simple peut déjà obtenir de bonnes performances dans cet environnement. Cette observation rend la comparaison des variantes du DQN particulièrement importante : l'objectif n'est pas seulement d'apprendre à faire mieux que le hasard, mais de déterminer si les améliorations algorithmiques apportent un gain mesurable par rapport à une stratégie de défense simple.