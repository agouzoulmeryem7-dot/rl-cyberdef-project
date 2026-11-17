"""Évaluation FINALE proposée : modèle sauvegardé + 2 références sur de NOUVELLES graines (ε = 0).

À lancer UNE SEULE FOIS, après fixation des hyperparamètres (aucun réglage ensuite sur ces graines).
Ne ré-entraîne rien : recharge results/dqn_model.pt.
Sorties (dans --out-dir) : dqn_eval_new.csv et baselines_new.csv (même format que dqn_eval.csv /
baselines.csv), directement utilisables par :
  python scripts/compare_policies.py --baselines <out>/baselines_new.csv --dqn <out>/dqn_eval_new.csv --out <out>/comparison_new.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cyberdef import CyberDefenseEnv, EnvConfig, NOISE_LEVELS
from cyberdef.baselines import RandomPolicy, RestoreIfAlert
from cyberdef.dqn import DQNAgent
from cyberdef.evaluate import evaluate_policy

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="results/dqn_model.pt")
ap.add_argument("--noise", choices=list(NOISE_LEVELS), default="medium")
ap.add_argument("--seed-base", type=int, default=200_000)
ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--out-dir", default="results/final_eval")
ap.add_argument("--train-seed", type=int, default=0, help="graine d'entraînement du modèle (métadonnée)")
args = ap.parse_args()
# 100000..100099 ont servi au réglage et à l'évaluation initiale : interdites pour un test final.
if args.seed_base < 200_000:
    sys.exit("seed-base doit être >= 200000 (les graines 100000+ ont servi au réglage des hyperparamètres)")

cfg = EnvConfig().with_noise(args.noise)
agent = DQNAgent.from_checkpoint(args.model)
n_actions = CyberDefenseEnv(cfg).action_space.n
os.makedirs(args.out_dir, exist_ok=True)


def dump(path, policies):
    rows = [{"policy": n, "noise": args.noise, **r} for n, p in policies
            for r in evaluate_policy(p, cfg, args.episodes, seed_base=args.seed_base)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("écrit", path, len(rows), "lignes")


dump(os.path.join(args.out_dir, "dqn_eval_new.csv"), [("dqn", agent.greedy_policy())])
dump(os.path.join(args.out_dir, "baselines_new.csv"),
     [("random", RandomPolicy(n_actions, seed=0)), ("restore_if_alert", RestoreIfAlert(cfg.n_hosts))])
