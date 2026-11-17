"""Compare random / DQN / restore_if_alert sur les MÊMES épisodes d'évaluation (ε=0).

Méthode IDENTIQUE pour les trois politiques : cyberdef.evaluate.bootstrap_ci
(bootstrap percentile sur les épisodes, 5000 rééchantillonnages, alpha=0.05, seed=0).
Lecture seule de baselines.csv et dqn_eval.csv. Sortie : results/dqn_comparison.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from cyberdef.evaluate import bootstrap_ci

METRICS = ["return", "final_infected_frac", "peak_infected_frac", "availability",
           "healthy_unavailable_mean", "healthy_isolations", "healthy_restorations"]

ap = argparse.ArgumentParser()
ap.add_argument("--noise", default="medium")
ap.add_argument("--baselines", default="results/baselines.csv")
ap.add_argument("--dqn", default="results/dqn_eval.csv")
ap.add_argument("--out", default="results/dqn_comparison.csv")
args = ap.parse_args()

b = pd.read_csv(args.baselines)
b = b[b["noise"] == args.noise]
d = pd.read_csv(args.dqn)
d = d[d["noise"] == args.noise]
data = {"random": b[b.policy == "random"], "dqn": d, "restore_if_alert": b[b.policy == "restore_if_alert"]}
data = {k: v.sort_values("env_seed").reset_index(drop=True) for k, v in data.items()}
seeds = [tuple(v["env_seed"]) for v in data.values()]
assert seeds[0] == seeds[1] == seeds[2], "les trois politiques doivent partager les mêmes graines"
print(f"bruit={args.noise} | {len(seeds[0])} épisodes | graines {seeds[0][0]}..{seeds[0][-1]}")

rows = []
print("\n== Moyennes et IC95% bootstrap (même méthode pour les 3 politiques) ==")
for name, df in data.items():
    for m in METRICS:
        mean, lo, hi = bootstrap_ci(df[m].values)
        rows.append({"kind": "policy", "name": name, "metric": m, "mean": mean, "ci_low": lo, "ci_high": hi,
                     "n": len(df)})
        if m == "return":
            print(f"{name:18s} return {mean:8.2f}  IC95% [{lo:8.2f}, {hi:8.2f}]  médiane {df[m].median():8.2f}  "
                  f"écart-type {df[m].std():7.2f}  épisodes < -100 : {(df[m] < -100).sum()}")

print("\n== Différences appariées (même graine d'environnement), IC95% bootstrap sur les différences ==")
r = {k: v["return"].values for k, v in data.items()}
for a, c in [("dqn", "random"), ("dqn", "restore_if_alert")]:
    diff = r[a] - r[c]
    mean, lo, hi = bootstrap_ci(diff)
    rows.append({"kind": "paired_diff", "name": f"{a} - {c}", "metric": "return", "mean": mean,
                 "ci_low": lo, "ci_high": hi, "n": len(diff)})
    print(f"{a} - {c:17s} {mean:8.2f}  IC95% [{lo:8.2f}, {hi:8.2f}]  | {a} meilleur sur {(diff > 0).sum()}/{len(diff)} épisodes")

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nÉcrit : {args.out}")
