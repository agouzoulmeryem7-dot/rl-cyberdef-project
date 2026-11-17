# Apprentissage par renforcement appliqué à la cyberdéfense

Mini-projet — Master AIDC, FST Béni Mellal.

Ce projet étudie l'utilisation de l'apprentissage par renforcement pour la défense automatisée d'un réseau informatique dans un environnement simulé. Les expériences sont réalisées dans un cadre strictement défensif.

**Variation étudiée par le groupe :** sensibilité au bruit des alertes de sécurité (faux positifs et faux négatifs), en évaluant conjointement la sécurité du réseau et la disponibilité des services.

## 1. État d'avancement du projet

| Composante | État |
|---|---|
| Environnement de cyberdéfense simulé | Implémenté |
| Politiques de référence aléatoire et heuristique | Implémentées et évaluées |
| DQN de base | Implémenté et entraîné |
| Étude initiale multi-graines du DQN | 4 graines d'entraînement (0, 1, 2, 3) |
| Double DQN | À implémenter |
| Dueling DQN | À implémenter |
| Prioritized Experience Replay (PER) | À implémenter |
| Étude comparative complète et analyse du bruit | À réaliser |

Les résultats initiaux du DQN restent préliminaires. À ce stade, le DQN de base n'a pas démontré de meilleures performances que la politique heuristique de référence.

## 2. Installation

Version de Python recommandée : 3.12.

Créer et activer un environnement virtuel :

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances du projet :

```bash
pip install -r requirements.txt
pip install torch pandas matplotlib
```

## 3. Tests et évaluation des politiques de référence

Exécuter l'ensemble des tests :

```bash
python -m pytest -q
```

La version intégrée actuelle du projet passe les 35 tests.

Générer les résultats des politiques de référence :

```bash
python scripts/run_baselines.py
```

Cette commande génère le fichier `results/baselines.csv`.

Deux politiques de référence sont utilisées :

- `RandomPolicy` : sélectionne les actions de manière aléatoire.
- `RestoreIfAlert` : applique une politique heuristique de restauration lorsqu'une alerte est détectée.

## 4. Description de l'environnement

L'environnement simule un réseau composé de :

- 12 hôtes ;
- 37 actions discrètes ;
- 48 caractéristiques d'observation ;
- un horizon maximal de 50 étapes par épisode.

Les principales actions de défense sont :

- **Isoler :** déconnecte un hôte du réseau jusqu'à sa restauration.
- **Restaurer :** nettoie un hôte infecté, supprime son isolement et le remet en service. L'hôte reste indisponible pendant une étape.
- **Patcher :** réduit la probabilité d'infection à l'aide du paramètre `patch_factor`, sans garantir une immunité totale.
- **Ne rien faire :** ne déclenche aucune action défensive.

La disponibilité à une étape est calculée comme suit :

`1 - (nombre d'hôtes isolés ou en cours de restauration / nombre total d'hôtes)`.

L'environnement prend en charge trois niveaux de bruit des alertes : `low`, `medium` et `high`. Les taux de faux positifs et de faux négatifs varient conjointement dans le protocole expérimental actuel. Leurs valeurs restent provisoires jusqu'à la validation définitive du protocole.

## 5. Métriques d'évaluation

Les politiques sont évaluées sans exploration (`epsilon = 0` pour le DQN).

Les principales métriques enregistrées sont :

- `return` : récompense cumulée au cours d'un épisode ;
- `final_infected_frac` : proportion finale d'hôtes infectés ;
- `peak_infected_frac` : proportion maximale d'hôtes infectés pendant l'épisode ;
- `availability` : disponibilité des services ;
- `healthy_unavailable_mean` : nombre moyen d'hôtes sains indisponibles ;
- `healthy_isolations` : nombre d'isolations injustifiées d'hôtes sains ;
- `healthy_restorations` : nombre de restaurations effectuées sur des hôtes sains.

Ces métriques permettent d'étudier le compromis entre la sécurité du réseau et la disponibilité des services.

## 6. Conventions expérimentales

- Les paramètres de récompense et la configuration de l'environnement sont définis dans `EnvConfig`, dans `cyberdef/env.py`.
- Plusieurs graines d'entraînement sont utilisées afin d'étudier la variabilité des résultats entre différentes exécutions.
- Les premières évaluations ont utilisé des graines d'environnement à partir de `100000`.
- Une évaluation finale indépendante devra utiliser de nouvelles graines à partir de `200000`, après la fixation des hyperparamètres et la sélection du modèle.
- L'évaluation du DQN utilise une politique gloutonne (`epsilon = 0`).
- L'environnement termine actuellement les épisodes par troncature à l'horizon configuré ; `terminated` reste toujours égal à `False`.

## 7. Démarche expérimentale

1. **Environnement et politiques de référence :** implémentation et validation de l'environnement simulé, puis évaluation des politiques aléatoire et heuristique.
2. **DQN de base :** entraînement du DQN initial, analyse de sa courbe d'apprentissage et étude préliminaire sur plusieurs graines.
3. **Améliorations du DQN :** implémentation et comparaison de Double DQN, Dueling DQN et Prioritized Experience Replay.
4. **Analyse expérimentale :** réalisation des études d'ablation, comparaison des configurations sur plusieurs graines et analyse de l'influence du bruit des alertes sur la sécurité et la disponibilité.

## 8. Reproductibilité

Les résultats expérimentaux sont enregistrés dans des fichiers CSV et JSON dans le répertoire `results/`. Les configurations d'entraînement, les graines utilisées et les protocoles d'évaluation doivent être documentés afin de permettre la reproduction et la comparaison équitable des expériences.

Les conclusions présentées doivent refléter fidèlement les résultats obtenus, y compris les résultats négatifs ou non concluants.