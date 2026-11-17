"""Agrège plusieurs entraînements DQN (graines d'entraînement différentes, mêmes hyperparamètres).

Entrées (lecture seule) : results/dqn_training.csv + results/dqn_eval.csv (seed 0) et
results/multiseed/seed_*/dqn_training.csv + dqn_eval.csv ; results/baselines.csv pour les références.
Sorties : results/dqn_multiseed_summary.csv, results/dqn_learning_curve_multiseed.png
Aucune donnée n'est inventée : tout vient des CSV ci-dessus.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from cyberdef.evaluate import bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--main-dir", default="results")
ap.add_argument("--multi-glob", default="results/multiseed/seed_*")
ap.add_argument("--baselines", default="results/baselines.csv")
ap.add_argument("--noise", default="medium")
ap.add_argument("--out-csv", default="results/dqn_multiseed_summary.csv")
ap.add_argument("--out-png", default="results/dqn_learning_curve_multiseed.png")
args = ap.parse_args()

dirs = [args.main_dir] + sorted(d for d in glob.glob(args.multi_glob) if os.path.isdir(d))
runs = []
for d in dirs:
    t, e = os.path.join(d, "dqn_training.csv"), os.path.join(d, "dqn_eval.csv")
    if os.path.exists(t) and os.path.exists(e):
        tr, ev = pd.read_csv(t), pd.read_csv(e)
        runs.append((int(tr["seed"].iloc[0]), tr, ev))
runs.sort(key=lambda r: r[0])
assert len(runs) >= 2, "il faut au moins 2 graines"
n_ep = min(len(r[1]) for r in runs)
window = int(runs[0][1]["hp_window"].iloc[0])

# ---------------------------------------------------------------- tableau par graine
rows = []
for seed, tr, ev in runs:
    mean, lo, hi = bootstrap_ci(ev["return"].values)
    last = tr["reward"].iloc[-500:].mean()
    rows.append({"train_seed": seed, "eval_return_mean": mean, "eval_ci_low": lo, "eval_ci_high": hi,
                 "eval_median": float(ev["return"].median()), "eval_episodes_below_-100": int((ev["return"] < -100).sum()),
                 "train_last500_mean_reward": float(last), "n_eval_episodes": len(ev)})
df = pd.DataFrame(rows)
b = pd.read_csv(args.baselines)
b = b[b["noise"] == args.noise]
base = {p: g["return"].mean() for p, g in b.groupby("policy")}

print(f"{len(runs)} graines d'entraînement : {list(df.train_seed)} (mêmes hyperparamètres, éval. ε=0, graines 100000+)")
print(df.round(2).to_string(index=False))
m, s = df["eval_return_mean"].mean(), df["eval_return_mean"].std(ddof=1)
print(f"\nMoyenne sur graines d'entraînement : {m:.2f}   écart-type entre graines : {s:.2f}   "
      f"min {df['eval_return_mean'].min():.2f}   max {df['eval_return_mean'].max():.2f}")
for k, v in base.items():
    print(f"  référence {k}: {v:.2f}")
print(f"  graines dont la moyenne dépasse restore_if_alert : "
      f"{(df['eval_return_mean'] > base.get('restore_if_alert', np.inf)).sum()}/{len(df)}")
summary = df.copy()
summary.loc[len(summary)] = {"train_seed": "mean_over_seeds", "eval_return_mean": m, "eval_ci_low": np.nan,
                             "eval_ci_high": np.nan, "eval_median": df["eval_median"].mean(),
                             "eval_episodes_below_-100": df["eval_episodes_below_-100"].mean(),
                             "train_last500_mean_reward": df["train_last500_mean_reward"].mean(),
                             "n_eval_episodes": df["n_eval_episodes"].iloc[0]}
summary.loc[len(summary)] = {**{c: np.nan for c in summary.columns}, "train_seed": "std_over_seeds", "eval_return_mean": s}
summary.to_csv(args.out_csv, index=False)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

curves = np.stack([pd.Series(tr["reward"].values[:n_ep]).rolling(window, min_periods=1).mean().values
                   for _, tr, _ in runs])
x = np.arange(1, n_ep + 1)
fig, ax = plt.subplots(figsize=(10, 5.5))
for (seed, _, _), c in zip(runs, curves):
    ax.plot(x, c, lw=0.8, alpha=0.45, label=f"seed {seed}")
ax.plot(x, curves.mean(axis=0), color="#c0392b", lw=2.4, label=f"Moyenne des {len(runs)} graines")
ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color="#c0392b", alpha=0.12, label="min–max entre graines")
for (k, v), st in zip(base.items(), ["--", ":"]):
    ax.axhline(v, color="#444", ls=st, lw=1.3, label=f"{k} : {v:.1f} (éval. ε=0)")
ax.set_xlabel("Épisode d'entraînement")
ax.set_ylabel(f"Récompense brute, moyenne glissante ({window} ép.)")
ax.set_title(f"DQN de base — {len(runs)} graines d'entraînement, bruit {args.noise}")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(args.out_png, dpi=150)
print(f"\nÉcrit : {args.out_csv}, {args.out_png}")
