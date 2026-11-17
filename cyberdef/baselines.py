"""Deux références : politique aléatoire et heuristique « restaurer si alerte »."""
import numpy as np

from .env import RESTORE, action_index


class RandomPolicy:
    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs) -> int:
        return int(self.rng.integers(self.n_actions))


class RestoreIfAlert:
    """Restaure l'hôte non isolé en alerte dont la moyenne glissante d'alertes est la plus haute."""

    def __init__(self, n_hosts: int):
        self.n = n_hosts

    def __call__(self, obs) -> int:
        n = self.n
        alerts, ema, isolated = obs[:n], obs[n:2 * n], obs[2 * n:3 * n]
        cand = (alerts > 0.5) & (isolated < 0.5)
        if not cand.any():
            return 0
        host = int(np.argmax(np.where(cand, ema, -1.0)))
        return action_index(RESTORE, host, n)
