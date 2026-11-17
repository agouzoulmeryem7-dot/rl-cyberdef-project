# HANDOFF — Membre 3 (cadrage, littérature, variation, références) — Semaine 1

## 1. Documents créés
| Fichier | Contenu |
|---|---|
| `docs/note_de_cadrage.md` | Note de cadrage d'une page (question, variation, environnement, métriques, protocole, répartition, planning, déclaration IA) |
| `docs/baselines_analysis.md` | Protocole, résultats réels, dispersions, effets du bruit, limites |
| `notebooks/01_baselines_analysis.ipynb` | Lit `results/baselines.csv`, contrôles, statistiques, graphiques, interprétation (exécuté, sorties incluses) |
| `HANDOFF_MEMBER3.md` | Ce fichier |

Aucun fichier Python (`env.py`, `baselines.py`, `evaluate.py`, `run_baselines.py`, tests) ni `README.md` n'a été modifié. `results/baselines.csv` est **conservé tel quel**.

## 2. Articles analysés
- **Lu en texte intégral** : Shen, Shepherd, Ahmed, Yu, Li, *IEEE IoT-J* 11(12):22550–22561, 2024 (DOI 10.1109/JIOT.2024.3381281, version auteur ORCA).
- **`literature_review.md`** : repris tel quel ; seul l'article principal y est lu en intégralité, les autres références sont vérifiées sur résumés/métadonnées. Hammar & Stadler 2020, Hessel 2018, Henderson 2018 et Sutton & Barto restent **non vérifiés** : ne pas les citer avant contrôle.

Points relevés dans l'article, à garder en tête pour le rapport (sans en faire des accusations) :
- Environnement `idsgame` adapté, malware à politique aléatoire ; états/actions/récompenses non détaillés → notre projet **n'est pas une reproduction**.
- Pas de DQN de base, pas de PER, pas de graines ni d'IC annoncés (comparaisons en « x fois sur 40 »).
- Les écarts entre algorithmes sont faibles sur les figures (le texte parle lui-même de « tiny advantage »).
- Le texte désigne γ = 0,85 comme optimal (section B) mais compare les algorithmes avec γ = 0,99 (section C).
- Les pseudo-codes (Algorithmes 1–3) sont schématiques : la ligne de cible terminale est suivie d'une ligne qui recalcule `y` sans condition, et l'Algorithme 2 (D2QMS) n'utilise pas de réseau cible séparé. À ne pas copier ; notre implémentation doit gérer correctement le terminal/troncature.
- Le texte dit « Fuzzy Neural Network » pour le réseau ; les autres éléments (2 couches cachées, ReLU, Huber, Adam) sont cohérents avec la revue.

## 3. Variation retenue
Sensibilité au bruit des alertes : low (FP 2 %, FN 5 %), medium (5 %, 15 %), high (10 %, 30 %). Valeurs **provisoires**, FP et FN varient ensemble (leurs effets sont confondus). Le compromis sécurité–disponibilité est une dimension d'évaluation, pas une seconde variation.

## 4. Baselines exécutées et résultats réels
`random` et `restore_if_alert`, 3 niveaux, 100 épisodes chacun (graines 100000–100099, ε = 0). Moyenne du retour [IC 95 % bootstrap] :

| Politique | low | medium | high |
|---|---|---|---|
| random | −249,5 [−270,1 ; −228,3] | idem | idem |
| restore_if_alert | −10,9 [−13,3 ; −9,4] | −25,1 [−33,8 ; −20,3] | −58,7 [−78,0 ; −42,3] |

Vérifications : 12 tests passent ; relance du script → CSV **identique octet pour octet** ; 600 lignes, 0 valeur manquante, 100 épisodes et 100 graines par groupe.
À retenir : l'heuristique est nettement meilleure que l'aléatoire aux trois niveaux ; elle se dégrade avec le bruit (retour, disponibilité, restaurations inutiles ×3, queue de mauvais épisodes plus épaisse). La ligne `random` est **identique** aux trois niveaux (ce n'est pas un bug, cf. `baselines_analysis.md` §5).

## 5. Commandes de reproduction
```bash
cd rl_cyberdef
pip install -r requirements.txt          # numpy, gymnasium, pytest
pip install pandas matplotlib jupyter    # pour le notebook
python -m pytest -q                      # 12 tests passent
python scripts/run_baselines.py          # réécrit results/baselines.csv (identique)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_baselines_analysis.ipynb
```

## 6. Coordination
- **Membre 1 (environnement)** : (a) confirmer que `EnvConfig` et les coûts sont figés ; (b) le README annonce « 8 tests » alors que 12 passent : à mettre à jour ; (c) confirmer les niveaux FP/FN avant la campagne ; (d) `pandas` et `matplotlib` sont utilisés par le notebook mais pas listés dans `requirements.txt`.
- **Membre 2 (DQN)** : évaluer avec `evaluate_policy(policy, cfg, n_episodes=100, seed_base=100000)` à ε = 0 ; comparer à `results/baselines.csv` au **même niveau de bruit** ; viser l'heuristique, pas seulement l'aléatoire ; graines d'entraînement 0, 1, 2… (disjointes).

## 7. Questions ouvertes
1. Balayage du bruit : entraîner et évaluer au même niveau (8 configs × 3 niveaux × ≥ 5 graines = ≥ 120 entraînements) ou ablation 2³ à `medium` seulement puis balayage restreint ? **À trancher avant le début de la semaine 2** et à figer.
2. Faut-il enregistrer aussi la fraction moyenne d'infectés dans le temps (absente du CSV) ? Demande à la membre 1.
3. Séparer FP et FN dans une analyse annexe (grille FP × FN) ou assumer la confusion ?
4. Budget de pas d'entraînement et hyperparamètres communs (membre 2).
5. Noms des membres et répartition définitive à inscrire dans la note de cadrage.
6. Références non vérifiées (Hammar & Stadler, Hessel, Henderson, Sutton & Barto) à contrôler avant le rapport.

## 8. Proposition de mise à jour du README (à valider par la membre 1)
- Remplacer « 8 tests doivent passer » par « 12 tests doivent passer ».
- Ajouter `pandas`, `matplotlib`, `jupyter` aux dépendances de l'étape 1 (notebook).
- Ajouter une ligne : `jupyter nbconvert --to notebook --execute --inplace notebooks/01_baselines_analysis.ipynb`.
- Mentionner que `random` est identique aux trois niveaux de bruit.
