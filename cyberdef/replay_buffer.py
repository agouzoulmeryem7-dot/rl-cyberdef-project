"""Replay Buffer circulaire (rejeu uniforme) pour le DQN de base.

Idée : on mémorise les transitions (s, a, r, s', done) vécues par l'agent et on
en tire des mini-lots ALÉATOIRES pour apprendre. Cela casse la corrélation entre
pas consécutifs d'un même épisode et permet de réutiliser chaque expérience.

Implémentation : tableaux NumPy pré-alloués, écrasement du plus ancien quand plein.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity doit être > 0")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        # done = terminated (vraie fin). Une troncature N'EST PAS un done : on amorce la cible.
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0          # prochaine case à écrire
        self.size = 0         # nombre de transitions stockées
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def push(self, state, action, reward, next_state, done) -> None:
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Tirage uniforme sans remise. Retourne 5 tenseurs torch (s, a, r, s', done)."""
        if batch_size > self.size:
            raise ValueError(f"batch_size={batch_size} > taille du buffer={self.size}")
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )
