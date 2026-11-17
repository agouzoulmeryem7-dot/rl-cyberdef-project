"""Évaluation d'une politique (ε = 0) sur des graines d'environnement distinctes de l'entraînement."""
import numpy as np

from .env import CyberDefenseEnv

TRAIN_SEED_BASE = 0          # graines d'entraînement : 0, 1, 2, ...
EVAL_SEED_BASE = 100_000     # graines d'évaluation : disjointes


def evaluate_policy(policy, cfg, n_episodes: int = 100, seed_base: int = EVAL_SEED_BASE):
    env = CyberDefenseEnv(cfg)
    rows = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + k)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        rows.append({"env_seed": seed_base + k, **info["episode_metrics"]})
    return rows


def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
