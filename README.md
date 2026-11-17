# RL appliqué à la cyberdéfense : DQN, Double, Dueling, PER

Mini-projet RL (Master IA). Cadre strictement défensif et simulé.
Variation du groupe : sensibilité au bruit d'alerte (FP/FN), évaluée en sécurité et disponibilité.

## Installation
    pip install -r requirements.txt

## Étape 1 (actuelle) : environnement + références
    python -m pytest -q                       # 8 tests doivent passer
    python scripts/run_baselines.py           # écrit results/baselines.csv

## Règles de l'environnement (v2)
- Isoler : hôte déconnecté, ne propage ni ne reçoit ; indisponible jusqu'à sa restauration.
- Restaurer : hôte nettoyé s'il est infecté, isolement levé, remis en service ; indisponible UN pas (sain ou infecté). Restaurer un hôte sain est compté (healthy_restorations).
- Patcher : proba d'infection multipliée par `patch_factor` (0,2 par défaut), pas d'immunité totale.
- Disponibilité d'un pas = 1 - (#hôtes isolés ou en restauration)/N.
- Niveaux FP/FN (low/medium/high) : valeurs PROVISOIRES, FP et FN varient ensemble ; à figer avant la campagne.

## Métriques d'évaluation (par épisode, ε = 0)
return ; final_infected_frac ; peak_infected_frac (sécurité) ; availability ; healthy_unavailable_mean ;
healthy_isolations ; healthy_restorations (disponibilité / sur-défense).

## Conventions (à ne pas changer après les premiers résultats)
- Coûts de récompense et paramètres : `EnvConfig` dans `cyberdef/env.py`.
- Graines d'entraînement : 0, 1, 2, ... ; graines d'évaluation : 100000+ (disjointes).
- Évaluation avec ε = 0.
- Fin d'épisode = troncature (terminated toujours False).

## Étapes suivantes
2. DQN de base + Double / Dueling / PER activables, courbe d'apprentissage.
3. Tests des variantes, ablation 2^3 x graines, balayage FP/FN, analyse.

## Déclaration IA
Ce code a été produit avec l'aide d'un assistant d'IA (Claude), relu et testé par le groupe.
