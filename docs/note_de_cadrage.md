# Améliorations de DQN pour la cyberdéfense sous alertes bruitées : quelles briques comptent, et à quel niveau de bruit ?

*Note de cadrage — semaine 1 — Module Apprentissage par renforcement*

**Problématique.** Un agent protège un réseau simulé de 12 hôtes contre la propagation d'une infection ; à chaque pas il peut isoler, restaurer ou patcher un hôte, mais ne voit que des alertes bruitées (POMDP du point de vue du défenseur). L'article de référence (Shen *et al.*, IEEE IoT-J, 2024) compare Double, Dueling et leur combinaison, sans DQN de base ni PER, et rapporte ses comparaisons en « x fois sur 40 » sans graines ni intervalles de confiance explicites. Notre étude n'est **pas** une reproduction : l'environnement, le protocole et l'analyse sont les nôtres.

**Question de recherche.** Parmi Double DQN, Dueling et le replay priorisé (PER), lesquels améliorent réellement DQN sur cette tâche, seuls ou combinés ? **Variation du groupe :** ces effets dépendent-ils du niveau de bruit des alertes ? Trois niveaux **provisoires** (choix expérimentaux, non issus de statistiques réelles) : faible (FP 2 %, FN 5 %), moyen (5 %, 15 %), élevé (10 %, 30 %). FP et FN varient ensemble : leurs effets ne seront donc pas séparables (limite assumée). Le compromis sécurité–disponibilité est une dimension d'évaluation, non une seconde variation.

**Environnement** (`cyberdef/env.py`, validé par la membre 1). Gymnasium, topologie fixe (anneau + cordes), 50 pas, troncature sans état terminal. État caché : infecté / isolé / patché par hôte. 37 actions (rien ; isoler, restaurer, patcher × 12 hôtes). Observation (48 valeurs) : alertes bruitées, moyenne glissante, indicateurs isolé et patché. Récompense : −(1·infectés + 0,3·indisponibles + coût d'action). Tableau complet dans le rapport.

**Algorithmes.** Références : politique aléatoire ; heuristique « restaurer si alerte ». Appris : DQN de base (réseau cible, rejeu, ε-greedy) et trois briques activables indépendamment (Double, Dueling, PER avec poids d'importance), soit 2³ = 8 configurations.

**Métriques** (évaluation à ε = 0, graines d'évaluation 100000+ disjointes de l'entraînement) : retour cumulé ; fraction d'hôtes infectés en fin d'épisode ; pic de compromission ; disponibilité moyenne ; hôtes sains indisponibles ; isolations à tort ; restaurations inutiles.

**Protocole prévu (à figer avant tout résultat d'apprentissage).** Mêmes hyperparamètres et même budget de pas pour les 8 configurations ; ≥ 5 graines d'entraînement par configuration ; intervalles de confiance bootstrap à 95 % ; effets par composant. Le plan de balayage du bruit (entraînement et évaluation au même niveau ou non) reste à décider collectivement avant la semaine 2 et sera documenté sans modification ultérieure.

**Reproductibilité.** Graines d'entraînement 0, 1, 2… ; graines d'évaluation 100000+ ; résultats bruts en CSV ; README et commandes de relance ; tests `pytest`.

**Répartition (semaine 1).** Membre 1 : environnement et tests. Membre 2 : DQN de base et courbe d'apprentissage. Membre 3 : littérature, cadrage, variation, mesure et analyse des références. Journal des contributions tenu par chacun.

**Planning.** S1 : cadrage, références, DQN de base (*réalisé à ce stade : références mesurées sur 100 épisodes × 3 niveaux*). S1–2 : Double, Dueling, PER vérifiés un par un. S2 : campagne 2³ × ≥ 5 graines + variation, résultats bruts. S3 : analyse, rapport, soutenance ; article en S3–4.

**Déclaration IA.** Ce travail est réalisé avec l'aide d'un assistant d'IA (Claude) pour le code, la lecture de l'article et la rédaction ; tout contenu est relu, testé et doit pouvoir être expliqué par chaque membre en soutenance.
