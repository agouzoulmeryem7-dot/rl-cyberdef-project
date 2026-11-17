"""Entraîne le DQN de base, puis l'ÉVALUE séparément (ε = 0, graines >= 100000).

Sorties (dans --out-dir, par défaut results/) :
  dqn_training.csv        une ligne par épisode d'ENTRAÎNEMENT (+ hyperparamètres hp_*)
  dqn_eval.csv            une ligne par épisode d'ÉVALUATION (même format que baselines.csv, policy=dqn)
  dqn_hyperparams.json    hyperparamètres + infos d'exécution
  dqn_learning_curve.png  figure tracée à partir de dqn_training.csv
  dqn_model.pt            poids du Q-network entraîné

Exemples :
  python scripts/train_dqn.py --episodes 100 --out-dir results/validation   # validation courte
  python scripts/train_dqn.py                                                # entraînement complet
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from cyberdef import CyberDefenseEnv, EnvConfig, NOISE_LEVELS
from cyberdef.dqn import DQNConfig, plot_learning_curve, train_dqn, write_training_csv
from cyberdef.evaluate import EVAL_SEED_BASE, bootstrap_ci, evaluate_policy

METRICS = ["return", "final_infected_frac", "peak_infected_frac", "availability",
           "healthy_unavailable_mean", "healthy_isolations", "healthy_restorations"]


def baseline_means(path, noise):
    """Moyennes des deux références (lecture seule de baselines.csv), ou {} si absent."""
    if not os.path.exists(path):
        return {}
    import pandas as pd
    d = pd.read_csv(path)
    d = d[d["noise"] == noise]
    return {p: float(g["return"].mean()) for p, g in d.groupby("policy")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", choices=list(NOISE_LEVELS), default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--baselines", default="results/baselines.csv")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eps-decay-steps", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--reward-scale", type=float, default=None)
    ap.add_argument("--target-update-every", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-starts", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = DQNConfig(n_episodes=args.episodes)
    if args.lr is not None:
        cfg.lr = args.lr
    for name in ["gamma", "reward_scale", "target_update_every", "batch_size", "learning_starts"]:
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if args.eps_decay_steps is not None:
        cfg.eps_decay_steps = args.eps_decay_steps
    elif args.episodes < 3000:    # validation courte : décroissance proportionnelle
        cfg.eps_decay_steps = max(1, int(0.4 * args.episodes * 50))
    env_cfg = EnvConfig().with_noise(args.noise)
    assert args.seed + args.episodes < EVAL_SEED_BASE, "graines d'entraînement et d'évaluation se chevauchent"

    # ------------------------------------------------------------ ENTRAÎNEMENT
    print(f"Entraînement : bruit={args.noise} seed={args.seed} épisodes={args.episodes}")
    t0 = time.time()
    agent, rows = train_dqn(CyberDefenseEnv(env_cfg), cfg, seed=args.seed, log_every=args.log_every)
    train_time = time.time() - t0
    extra = {"noise": args.noise, "fp_rate": env_cfg.fp_rate, "fn_rate": env_cfg.fn_rate}
    train_csv = os.path.join(args.out_dir, "dqn_training.csv")
    write_training_csv(rows, train_csv, cfg, extra)
    agent.save(os.path.join(args.out_dir, "dqn_model.pt"))
    with open(os.path.join(args.out_dir, "dqn_hyperparams.json"), "w") as f:
        json.dump({"dqn": cfg.to_dict(), "env": {k: v for k, v in env_cfg.__dict__.items()},
                   "noise": args.noise, "seed": args.seed, "train_seconds": round(train_time, 1),
                   "torch": torch.__version__, "loss": "Huber (smooth_l1)", "optimizer": "Adam"},
                  f, indent=2)
    print(f"Entraînement terminé en {train_time:.0f}s -> {train_csv}")

    # ------------------------------------------------------------ ÉVALUATION (ε = 0, graines disjointes)
    eval_rows = evaluate_policy(agent.greedy_policy(), env_cfg, args.eval_episodes, seed_base=EVAL_SEED_BASE)
    eval_csv = os.path.join(args.out_dir, "dqn_eval.csv")
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "noise", "train_seed", *eval_rows[0].keys()])
        w.writeheader()
        w.writerows({"policy": "dqn", "noise": args.noise, "train_seed": args.seed, **r} for r in eval_rows)
    print(f"\nÉVALUATION DQN (ε=0, {args.eval_episodes} épisodes, graines {EVAL_SEED_BASE}+) -> {eval_csv}")
    for m in METRICS:
        mean, lo, hi = bootstrap_ci([r[m] for r in eval_rows])
        print(f"  {m:24s} {mean:8.3f}  IC95% [{lo:.3f}, {hi:.3f}]")
    base = baseline_means(args.baselines, args.noise)
    for name, val in base.items():
        print(f"  (référence {name}, return moyen : {val:.2f})")

    # ------------------------------------------------------------ COURBE (depuis le CSV)
    png = os.path.join(args.out_dir, "dqn_learning_curve.png")
    plot_learning_curve(train_csv, png, title=f"DQN de base — bruit {args.noise}, seed {args.seed}",
                        baselines={k: v for k, v in base.items()})
    print(f"Courbe -> {png}")


if __name__ == "__main__":
    main()
