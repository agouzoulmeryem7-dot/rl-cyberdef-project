# Améliorations de DQN pour la cyberdéfense sous alertes bruitées : quelles briques comptent, et à quel niveau de bruit ?

*Note de cadrage — Semaine 1 — Module Apprentissage par renforcement*

## 1. Problématique

Un agent de cyberdéfense doit protéger un réseau simulé de 12 hôtes contre la propagation d'une infection.

À chaque pas de temps, il peut choisir entre plusieurs actions : isoler un hôte, restaurer un hôte, appliquer un patch ou ne rien faire. Cependant, l'agent ne dispose pas d'un accès direct à l'état réel du réseau. Il prend ses décisions à partir d'observations contenant des alertes potentiellement bruitées.

Du point de vue du défenseur, cette situation présente les caractéristiques d'un problème partiellement observable (POMDP) : les observations ne donnent pas nécessairement une information complète et exacte sur l'état réel du réseau.

L'article de référence de Shen et al. (2024), publié dans l'IEEE Internet of Things Journal, étudie différentes améliorations de DQN pour des stratégies de suppression de propagation de malware dans des environnements IoT. Il compare notamment Double DQN, Dueling DQN et leur combinaison.

Notre étude s'inspire de cette famille de travaux, mais ne constitue pas une reproduction de l'article. Nous utilisons notre propre environnement simulé, notre protocole expérimental et nos métriques d'évaluation.

## 2. Question de recherche

Parmi les améliorations suivantes de DQN :

- Double DQN ;
- Dueling DQN ;
- Prioritized Experience Replay (PER) ;

lesquelles apportent une amélioration mesurable à un DQN de base dans une tâche de cyberdéfense simulée, seules ou en combinaison ?

**Variation étudiée par le groupe :** les effets de ces améliorations dépendent-ils du niveau de bruit des alertes ?

Trois niveaux de bruit sont retenus à titre expérimental :

| Niveau | Faux positifs (FP) | Faux négatifs (FN) |
|---|---:|---:|
| Low | 2 % | 5 % |
| Medium | 5 % | 15 % |
| High | 10 % | 30 % |

Ces valeurs sont des paramètres provisoires choisis pour construire différents scénarios expérimentaux. Elles ne proviennent pas de statistiques réelles.

Les taux de faux positifs et de faux négatifs varient simultanément dans ce protocole. Leurs effets individuels ne pourront donc pas être séparés dans cette étude.

Le compromis entre sécurité et disponibilité constitue une dimension complémentaire de l'évaluation, et non une deuxième variation indépendante.

## 3. Démarche expérimentale du groupe

Notre travail suit une démarche expérimentale progressive :

1. Comprendre et valider l'environnement de cyberdéfense fourni.
2. Évaluer deux politiques de référence : une politique aléatoire et une heuristique basée sur les alertes.
3. Implémenter et évaluer un DQN de base.
4. Introduire progressivement les améliorations Double DQN, Dueling DQN et PER.
5. Comparer les différentes configurations avec un protocole commun.
6. Analyser les résultats en fonction du niveau de bruit et des métriques de sécurité et de disponibilité.

Les comparaisons seront réalisées avec un budget d'entraînement identique, des graines contrôlées et des métriques communes.

L'objectif n'est pas de supposer qu'une méthode améliorée sera nécessairement meilleure, mais de mesurer son apport réel, d'identifier les situations dans lesquelles elle est utile et d'analyser ses éventuelles limites.

## 4. Environnement expérimental

L'environnement est défini dans `cyberdef/env.py` et utilise l'interface Gymnasium.

Il représente un réseau simulé de 12 hôtes organisé selon une topologie fixe en anneau avec des connexions supplémentaires.

Un épisode comporte au maximum 50 pas de temps.

### 4.1. État réel et observation

L'état réel du réseau comprend des informations relatives à chaque hôte, notamment :
- Son état d'infection.
- Son état d'isolation.
- Son état de patch.

Ces informations ne sont pas toutes directement accessibles à l'agent.

L'observation fournie à l'agent est un vecteur de 48 valeurs de type `float32`, normalisées dans l'intervalle [0, 1].

Elle comprend notamment :
- Des indicateurs d'alertes bruitées.
- Des moyennes glissantes d'alertes.
- Des indicateurs d'isolation.
- Des indicateurs de patch.

L'agent doit donc prendre ses décisions à partir d'informations imparfaites.

### 4.2. Espace d'actions

L'espace d'actions contient 37 actions discrètes :

- Une action pour ne rien faire.
- 12 actions d'isolation, une par hôte.
- 12 actions de restauration, une par hôte.
- 12 actions de patch, une par hôte.

### 4.3. Fonction de récompense

La récompense pénalise principalement :
- Le nombre d'hôtes infectés.
- L'indisponibilité des hôtes.
- Les coûts associés aux actions de défense.

La forme générale utilisée est :

`Reward = -(1 × nombre d'hôtes infectés + 0,3 × nombre d'hôtes indisponibles + coût de l'action)`

Cette formulation vise à représenter le compromis entre la limitation des infections et le maintien de la disponibilité du réseau.

## 5. Algorithmes étudiés

### 5.1. Politiques de référence

Deux politiques sont utilisées comme points de comparaison :

- **Random :** sélectionne aléatoirement une action.
- **Restore-if-alert :** restaure l'hôte non isolé présentant la moyenne glissante d'alertes la plus élevée, lorsque la condition de la règle est satisfaite.

Ces politiques permettent de situer les performances des agents appris par rapport à des stratégies simples.

### 5.2. DQN de base

Le DQN constitue l'algorithme de référence parmi les agents appris.

Son implémentation utilise :
- Un réseau neuronal pour approximer les valeurs Q.
- Un réseau cible pour stabiliser l'apprentissage.
- Un replay buffer pour stocker les transitions.
- Une stratégie ε-greedy pour l'exploration.
- Une fonction de perte Huber et l'optimiseur Adam.

### 5.3. Améliorations étudiées

Trois améliorations seront activées indépendamment ou combinées :

- **Double DQN :** vise à réduire la surestimation des valeurs Q en séparant la sélection de l'action de son évaluation.
- **Dueling DQN :** sépare l'estimation de la valeur de l'état et l'avantage associé à chaque action.
- **Prioritized Experience Replay (PER) :** favorise l'échantillonnage des transitions considérées comme plus importantes pour l'apprentissage, avec des poids de correction d'importance.

Avec trois améliorations activables indépendamment, huit configurations sont possibles :

| Configuration | Double | Dueling | PER |
|---|---|---|---|
| 1. DQN | Non | Non | Non |
| 2. Double DQN | Oui | Non | Non |
| 3. Dueling DQN | Non | Oui | Non |
| 4. DQN + PER | Non | Non | Oui |
| 5. Double + Dueling | Oui | Oui | Non |
| 6. Double + PER | Oui | Non | Oui |
| 7. Dueling + PER | Non | Oui | Oui |
| 8. Double + Dueling + PER | Oui | Oui | Oui |

Cette organisation permet d'étudier les effets individuels des améliorations ainsi que leurs interactions éventuelles.

## 6. Métriques d'évaluation

L'évaluation ne repose pas uniquement sur le retour cumulé. Plusieurs indicateurs complémentaires sont utilisés :

| Métrique | Objectif |
|---|---|
| Retour cumulé | Mesurer la performance globale selon la fonction de récompense |
| Fraction finale d'hôtes infectés | Mesurer l'état du réseau à la fin de l'épisode |
| Pic de compromission | Mesurer le niveau maximal d'infection atteint |
| Disponibilité moyenne | Évaluer le maintien des services et des hôtes |
| Hôtes sains indisponibles | Quantifier les pertes de disponibilité injustifiées |
| Isolations à tort | Mesurer les interventions d'isolation injustifiées |
| Restaurations inutiles | Mesurer les restaurations effectuées sans nécessité réelle |

Ces métriques permettent de distinguer une politique qui limite efficacement les infections d'une politique qui obtient un bon retour au prix d'interventions excessives.

Les agents appris seront évalués sans exploration, avec ε = 0.

## 7. Protocole expérimental prévu

Afin de rendre les comparaisons aussi équitables et reproductibles que possible, les règles suivantes sont retenues :

- Utiliser le même environnement pour les configurations comparées.
- Conserver les mêmes hyperparamètres et le même budget d'entraînement pour les huit configurations, sauf modification explicitement justifiée.
- Prévoir au minimum cinq graines d'entraînement indépendantes par configuration pour la campagne comparative complète.
- Séparer les graines d'entraînement des graines d'évaluation.
- Évaluer les politiques avec le même nombre d'épisodes et les mêmes graines d'évaluation lorsque cela est possible.
- Calculer des intervalles de confiance bootstrap à 95 %.
- Comparer les performances globales et les résultats selon les niveaux de bruit.
- Conserver les résultats bruts au format CSV ou JSON afin de permettre leur vérification.

Le plan d'étude du bruit reste à préciser collectivement avant la semaine 2 : il faudra déterminer si les agents sont entraînés et évalués au même niveau de bruit ou si des scénarios d'évaluation avec un niveau de bruit différent de celui de l'entraînement sont également utilisés.

Ce choix sera documenté avant la réalisation des expériences correspondantes.

## 8. Reproductibilité et transparence

La reproductibilité constitue un objectif important du projet.

Les éléments suivants seront conservés :

- Les graines d'entraînement utilisées pour les essais préliminaires (0, 1, 2 et 3 pour le DQN de base).
- Les graines d'évaluation initiale (à partir de 100000), utilisées pendant les expérimentations préliminaires et la sélection des paramètres.
- De nouvelles graines d'évaluation (à partir de 200000) pour l'évaluation finale indépendante, après fixation des hyperparamètres et sélection du modèle.
- Les fichiers de résultats bruts.
- Les paramètres d'entraînement.
- Les scripts permettant de relancer les expériences.
- Les tests unitaires et les vérifications de l'environnement.
- Les courbes d'apprentissage et les tableaux de comparaison.

Les résultats défavorables ou inattendus seront également conservés et analysés. Aucune configuration ne sera considérée comme supérieure sans comparaison expérimentale.

## 9. État d'avancement à la fin de la semaine 1

À la fin de la première semaine, les travaux suivants ont été réalisés :

- L'environnement de cyberdéfense a été intégré et validé par des tests.
- Les deux politiques de référence, Random et Restore-if-alert, ont été évaluées sur trois niveaux de bruit.
- Les résultats des politiques de référence ont été enregistrés dans `results/baselines.csv` et analysés.
- Le DQN de base a été implémenté et entraîné sur quatre graines (0, 1, 2 et 3).
- Les premières courbes d'apprentissage et les résultats préliminaires du DQN ont été produits.

Les comparaisons complètes entre les huit configurations, l'étude approfondie de la sensibilité au bruit et l'analyse des effets de chaque amélioration restent à réaliser.

Les résultats obtenus pendant cette première semaine sont donc considérés comme des résultats préliminaires et ne constituent pas encore les conclusions définitives de l'étude.

## 10. Répartition des contributions

La répartition du travail est organisée comme suit :

- **Membre 1 :** développement, validation et documentation de l'environnement de cyberdéfense.
- **Membre 2 :** implémentation du DQN de base, entraînement, évaluation et production des courbes d'apprentissage.
- **Membre 3 :** recherche bibliographique, cadrage scientifique, définition de la variation expérimentale et analyse des politiques de référence.

Chaque membre est responsable de la vérification, de la compréhension et de la présentation de ses contributions. Les choix expérimentaux et les résultats sont discutés collectivement avant leur intégration dans le rapport final.

## Référence bibliographique

Shen, Y., Shepherd, C., Ahmed, C. M., Yu, S., & Li, T. (2024). *Comparative DQN-Improved Algorithms for Stochastic Games-Based Automated Edge Intelligence-Enabled IoT Malware Spread-Suppression Strategies*. IEEE Internet of Things Journal, 11(12), 22550–22561.

Cette publication constitue une source d'inspiration méthodologique pour l'étude des améliorations de DQN. Le présent projet adapte cette problématique à un environnement de cyberdéfense distinct et à un protocole expérimental propre au groupe.