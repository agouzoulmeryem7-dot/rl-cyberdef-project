"""DQN de base (Mnih et al., 2015) pour CyberDefenseEnv.

Composants : Q-Network, Replay Buffer (replay_buffer.py), Target Network,
epsilon-greedy, cible de Bellman, perte de Huber, Adam, mise à jour périodique
du réseau cible, boucle d'entraînement. Pas de Double / Dueling / PER (semaines suivantes).

Contrat avec l'environnement (vérifié dans env.py) :
  observation Box(48,) float32 dans [0,1] ; 37 actions Discrete, toutes valides (pas de masque) ;
  reward = -coût (négatif) ; terminated toujours False ; truncated True au pas 50.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import ReplayBuffer


# --------------------------------------------------------------------------- hyperparamètres
@dataclass
class DQNConfig:
    hidden_sizes: tuple = (128, 128)   # 2 couches cachées ReLU (comme Shen et al. 2024)
    gamma: float = 0.9                 # facteur d'actualisation (0.99 instable ici : voir HANDOFF)
    lr: float = 3e-4                   # pas d'apprentissage Adam
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000       # transitions à accumuler avant d'apprendre
    target_update_every: int = 200     # recopie du réseau cible tous les N pas d'environnement
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 40_000      # décroissance linéaire sur N pas d'environnement
    grad_clip: float = 10.0
    reward_scale: float = 0.1          # les récompenses brutes (~ -10 par pas) sont multipliées par ce facteur
    n_episodes: int = 3_000
    window: int = 50                   # fenêtre de la moyenne glissante

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d


# --------------------------------------------------------------------------- réseau
class QNetwork(nn.Module):
    """MLP : observation (obs_dim) -> une valeur Q par action (n_actions)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))   # sortie linéaire : Q peut être négatif
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    frac = min(max(step / max(decay_steps, 1), 0.0), 1.0)
    return eps_start + frac * (eps_end - eps_start)


# --------------------------------------------------------------------------- chargement sûr
def _load_checkpoint(path: str, device) -> dict:
    """torch.load en mode sûr (weights_only=True). Autorise en plus le scalaire NumPy, présent dans
    le fichier results/dqn_model.pt produit avant la correction de save() (obs_dim était un np.int64)."""
    core = getattr(np, "_core", None) or np.core
    with torch.serialization.safe_globals([core.multiarray.scalar, np.dtype, type(np.dtype("int64"))]):
        return torch.load(path, map_location=device, weights_only=True)


# --------------------------------------------------------------------------- agent
class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig | None = None,
                 seed: int = 0, device: str = "cpu"):
        self.cfg = cfg or DQNConfig()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(device)
        self.q_net = QNetwork(obs_dim, n_actions, self.cfg.hidden_sizes).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, self.cfg.hidden_sizes).to(self.device)
        self.sync_target()
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)          # le réseau cible n'est jamais entraîné par gradient
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, seed=seed)
        self.rng = np.random.default_rng(seed + 12345)

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    @torch.no_grad()
    def select_action(self, obs, epsilon: float = 0.0) -> int:
        """epsilon-greedy. epsilon=0 -> politique gloutonne (évaluation)."""
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        return int(self.q_net(x).argmax(dim=1).item())

    def greedy_policy(self) -> Callable:
        """Politique (obs -> action) avec epsilon = 0, compatible avec evaluate_policy()."""
        return lambda obs: self.select_action(obs, epsilon=0.0)

    def bellman_target(self, rewards, next_states, dones) -> torch.Tensor:
        """y = r + gamma * (1 - done) * max_a' Q_cible(s', a')   (r déjà mis à l'échelle)."""
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            return rewards + self.cfg.gamma * (1.0 - dones) * next_q

    def compute_loss(self, batch) -> torch.Tensor:
        states, actions, rewards, next_states, dones = (t.to(self.device) for t in batch)
        q_sa = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)   # Q(s, a) de l'action jouée
        target = self.bellman_target(rewards * self.cfg.reward_scale, next_states, dones)
        return F.smooth_l1_loss(q_sa, target)                                  # perte de Huber

    def update(self) -> float | None:
        """Une étape de gradient sur un mini-lot. None si le buffer est encore trop petit."""
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.learning_starts):
            return None
        loss = self.compute_loss(self.buffer.sample(self.cfg.batch_size))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.q_net.state_dict(), "config": self.cfg.to_dict(),
                    "obs_dim": int(self.obs_dim), "n_actions": int(self.n_actions)}, path)

    def load_weights(self, path: str) -> None:
        """Recharge les poids sauvegardés par save() dans cet agent (réseau principal + cible).

        Vérifie que les dimensions du fichier correspondent à l'agent. Ne recharge ni le buffer
        ni l'état de l'optimiseur : le modèle sert à l'évaluation / à la reprise de l'inférence.
        """
        ckpt = _load_checkpoint(path, self.device)
        if (ckpt["obs_dim"], ckpt["n_actions"]) != (self.obs_dim, self.n_actions):
            raise ValueError(f"Dimensions incompatibles : fichier ({ckpt['obs_dim']}, {ckpt['n_actions']}) "
                             f"!= agent ({self.obs_dim}, {self.n_actions})")
        self.q_net.load_state_dict(ckpt["state_dict"])
        self.sync_target()

    @classmethod
    def from_checkpoint(cls, path: str, seed: int = 0, device: str = "cpu") -> "DQNAgent":
        """Reconstruit un agent (architecture lue dans le fichier) et charge ses poids."""
        ckpt = _load_checkpoint(path, device)
        d = dict(ckpt["config"])
        d["hidden_sizes"] = tuple(d["hidden_sizes"])      # JSON/liste -> tuple
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], DQNConfig(**d), seed=seed, device=device)
        agent.load_weights(path)
        return agent


# --------------------------------------------------------------------------- boucle d'entraînement
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dqn(env, cfg: DQNConfig, seed: int = 0, n_episodes: int | None = None,
              log_every: int = 0, device: str = "cpu"):
    """Entraîne un DQN. Retourne (agent, lignes) ; une ligne par épisode d'ENTRAÎNEMENT.

    Graines d'environnement : seed + numéro d'épisode (0, 1, 2, ... pour seed=0), donc
    disjointes des graines d'évaluation (>= 100000) tant que seed + n_episodes < 100000.
    """
    set_global_seed(seed)
    n_episodes = cfg.n_episodes if n_episodes is None else n_episodes
    obs_dim = int(np.prod(env.observation_space.shape))
    agent = DQNAgent(obs_dim, env.action_space.n, cfg, seed=seed, device=device)
    rows, returns, step = [], [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return, ep_len, losses, done = 0.0, 0, [], False
        eps_ep_start = linear_epsilon(step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
        while not done:
            eps = linear_epsilon(step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
            action = agent.select_action(obs, eps)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            # on stocke `terminated` (pas `truncated`) : la fin à l'horizon n'est pas une vraie fin
            agent.buffer.push(obs, action, reward, next_obs, terminated)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            step += 1
            if step % cfg.target_update_every == 0:
                agent.sync_target()
            obs, ep_return, ep_len = next_obs, ep_return + reward, ep_len + 1
            done = terminated or truncated
        returns.append(ep_return)
        rows.append({
            "episode": ep + 1,
            "env_seed": seed + ep,
            "reward": ep_return,
            "moving_avg_reward": float(np.mean(returns[-cfg.window:])),
            "epsilon": eps,                       # epsilon au dernier pas de l'épisode
            "epsilon_episode_start": eps_ep_start,
            "length": ep_len,
            "total_steps": step,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "seed": seed,
        })
        if log_every and (ep + 1) % log_every == 0:
            r = rows[-1]
            print(f"ep {r['episode']:5d} | reward {r['reward']:8.1f} | moy{cfg.window} "
                  f"{r['moving_avg_reward']:8.1f} | eps {r['epsilon']:.3f} | loss {r['mean_loss']:.4f}",
                  flush=True)
    return agent, rows


# --------------------------------------------------------------------------- enregistrement
def write_training_csv(rows, path: str, cfg: DQNConfig, extra: dict | None = None) -> None:
    """CSV d'entraînement : une ligne/épisode + hyperparamètres (colonnes hp_*, constantes)."""
    hp = {f"hp_{k}": (json.dumps(v) if isinstance(v, list) else v) for k, v in cfg.to_dict().items()}
    hp.update({f"hp_{k}": v for k, v in (extra or {}).items()})
    full = [{**r, **hp} for r in rows]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(full[0].keys()))
        w.writeheader()
        w.writerows(full)


def plot_learning_curve(csv_path: str, png_path: str, title: str = "DQN de base : courbe d'apprentissage",
                        baselines: dict | None = None) -> None:
    """Figure construite UNIQUEMENT à partir du CSV d'entraînement (aucune donnée fictive).
    `baselines` : dict {nom: valeur} de lignes horizontales de référence (issues de baselines.csv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    d = pd.read_csv(csv_path)
    window = int(d["hp_window"].iloc[0]) if "hp_window" in d else 50
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(d["episode"], d["reward"], color="#9bb7d4", alpha=0.45, lw=0.8,
            label="Récompense par épisode (entraînement, ε>0)")
    ax.plot(d["episode"], d["moving_avg_reward"], color="#c0392b", lw=2.2,
            label=f"Moyenne glissante ({window} épisodes)")
    for (name, val), style in zip((baselines or {}).items(), ["--", ":"]):
        ax.axhline(val, color="#444", ls=style, lw=1.3, label=f"{name} : {val:.1f} (éval. ε=0)")
    ax.set_xlabel("Épisode d'entraînement")
    ax.set_ylabel("Récompense cumulée par épisode (brute)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(d["episode"], d["epsilon"], color="#2e7d32", lw=1.2, alpha=0.8, label="ε")
    ax2.set_ylabel("ε (exploration)")
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
