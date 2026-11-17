"""Mesure les deux références aux 3 niveaux de bruit. Sortie : results/baselines.csv"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cyberdef import CyberDefenseEnv, EnvConfig, NOISE_LEVELS
from cyberdef.baselines import RandomPolicy, RestoreIfAlert
from cyberdef.evaluate import bootstrap_ci, evaluate_policy

METRICS = ["return", "final_infected_frac", "peak_infected_frac", "availability",
           "healthy_unavailable_mean", "healthy_isolations", "healthy_restorations"]

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--out", default="results/baselines.csv")
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
all_rows = []
for level in NOISE_LEVELS:
    cfg = EnvConfig().with_noise(level)
    n_actions = CyberDefenseEnv(cfg).action_space.n
    for name, policy in [("random", RandomPolicy(n_actions, seed=0)),
                         ("restore_if_alert", RestoreIfAlert(cfg.n_hosts))]:
        rows = evaluate_policy(policy, cfg, args.episodes)
        all_rows += [{"policy": name, "noise": level, **r} for r in rows]
        print(f"\n[{level}] {name}")
        for m in METRICS:
            mean, lo, hi = bootstrap_ci([r[m] for r in rows])
            print(f"  {m:24s} {mean:8.3f}  IC95% [{lo:.3f}, {hi:.3f}]")

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)
print(f"\nÉcrit : {args.out}")
