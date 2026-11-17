"""Environnement Gymnasium de cyberdéfense (simulé, strictement défensif).

Un agent protège un réseau de N hôtes contre la propagation d'une infection.
Il n'observe que des alertes bruitées (faux positifs FP / faux négatifs FN).

ÉTAT CACHÉ (par hôte) : infected, isolated, patched (booléens).

ACTIONS (1 par pas) : 0 = ne rien faire ; 1 + k*N + i, k = 0 isoler, 1 restaurer, 2 patcher l'hôte i.
  - ISOLER   : l'hôte est déconnecté du réseau : il ne propage ni ne reçoit d'infection,
               mais n'est plus en service. L'isolement dure jusqu'à une restauration de cet hôte.
  - RESTAURER: l'hôte est nettoyé s'il est infecté, l'isolement est levé, puis il est remis en service.
               Pendant le pas de la restauration, l'hôte est INDISPONIBLE (redémarrage), qu'il soit
               infecté ou sain. Restaurer un hôte sain = restauration à tort (comptée).
  - PATCHER  : l'hôte reçoit une protection permanente : sa probabilité d'infection est multipliée
               par patch_factor (< 1). Ce n'est PAS une immunité totale. Sans effet sur un hôte déjà infecté.

DISPONIBILITÉ : un hôte est indisponible au pas t s'il est isolé OU s'il vient d'être restauré à ce pas.

OBSERVATION (4N valeurs dans [0,1]) : alertes bruitées, moyenne glissante des alertes,
  indicateur isolé, indicateur patché (les deux derniers sont connus du défenseur).

RÉCOMPENSE : -(c_infected * #infectés + c_unavailable * #indisponibles + coût de l'action).
  coût de l'action : c_restore (restauration) ou c_patch (patch), 0 sinon.
  Une restauration coûte donc c_restore + c_unavailable (pour l'hôte indisponible ce pas).

FIN D'ÉPISODE : troncature à l'horizon (terminated est toujours False -> on amorce la cible).
Les coefficients de EnvConfig sont FIGÉS avant les expériences.

ORDRE DANS UN PAS : action -> propagation -> récompense -> observation bruitée.
Un hôte restauré à ce pas est hors service : il ne propage pas et ne peut pas être infecté ce pas.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# NIVEAUX DE BRUIT PROVISOIRES (taux de faux positifs, taux de faux négatifs).
# FP = P(alerte | hôte sain) ; FN = P(pas d'alerte | hôte infecté).
# FP et FN varient ensemble. Valeurs de départ choisies par le groupe, NON issues d'une source :
# à confirmer et à figer avant la campagne d'expériences.
NOISE_LEVELS = {"low": (0.02, 0.05), "medium": (0.05, 0.15), "high": (0.10, 0.30)}

ISOLATE, RESTORE, PATCH = 0, 1, 2


@dataclass(frozen=True)
class EnvConfig:
    n_hosts: int = 12
    horizon: int = 50
    beta: float = 0.25            # proba d'infection par voisin infecté actif et par pas
    patch_factor: float = 0.2     # multiplicateur de la proba d'infection d'un hôte patché
    n_initial_infected: int = 1
    fp_rate: float = 0.05         # P(alerte | hôte sain)
    fn_rate: float = 0.15         # P(pas d'alerte | hôte infecté)
    c_infected: float = 1.0       # par hôte infecté et par pas
    c_unavailable: float = 0.3    # par hôte indisponible (isolé ou en restauration) et par pas
    c_restore: float = 0.5        # coût d'action d'une restauration
    c_patch: float = 0.2          # coût d'action d'un patch
    ema_decay: float = 0.5
    topo_seed: int = 0            # topologie identique pour toutes les expériences

    def with_noise(self, level: str) -> "EnvConfig":
        fp, fn = NOISE_LEVELS[level]
        return EnvConfig(**{**self.__dict__, "fp_rate": fp, "fn_rate": fn})


def make_topology(n: int, seed: int) -> np.ndarray:
    """Anneau (connexe) + n//2 cordes aléatoires. Matrice d'adjacence booléenne."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = True
    for _ in range(n // 2):
        i, j = rng.choice(n, 2, replace=False)
        adj[i, j] = adj[j, i] = True
    return adj


def action_index(kind: int, host: int, n_hosts: int) -> int:
    return 1 + kind * n_hosts + host


class CyberDefenseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        n = self.cfg.n_hosts
        self.adj = make_topology(n, self.cfg.topo_seed)
        self.action_space = spaces.Discrete(1 + 3 * n)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4 * n,), dtype=np.float32)

    def _observe(self) -> np.ndarray:
        c = self.cfg
        p_alert = np.where(self.infected, 1.0 - c.fn_rate, c.fp_rate)
        self.alerts = (self.np_random.random(c.n_hosts) < p_alert).astype(np.float32)
        self.ema = c.ema_decay * self.ema + (1.0 - c.ema_decay) * self.alerts
        return np.concatenate(
            [self.alerts, self.ema, self.isolated.astype(np.float32), self.patched.astype(np.float32)]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        n = c.n_hosts
        self.infected = np.zeros(n, dtype=bool)
        self.infected[self.np_random.choice(n, c.n_initial_infected, replace=False)] = True
        self.isolated = np.zeros(n, dtype=bool)
        self.patched = np.zeros(n, dtype=bool)
        self.down_now = np.zeros(n, dtype=bool)   # restaurés au pas courant (hors service)
        self.ema = np.zeros(n, dtype=np.float32)
        self.t = 0
        self._ret = 0.0
        self._peak = self.infected.mean()
        self._avail_sum = 0.0
        self._healthy_unav_sum = 0.0
        self._healthy_isolations = 0
        self._healthy_restorations = 0
        return self._observe(), {}

    def step(self, action: int):
        c = self.cfg
        n = c.n_hosts
        action_cost = 0.0
        self.down_now = np.zeros(n, dtype=bool)
        wrong_down = np.zeros(n, dtype=bool)      # hôtes sains restaurés à tort à ce pas
        if action != 0:
            kind, host = divmod(int(action) - 1, n)
            if kind == ISOLATE:
                if not self.infected[host] and not self.isolated[host]:
                    self._healthy_isolations += 1   # isolation à tort (hôte sain)
                self.isolated[host] = True
            elif kind == RESTORE:
                if not self.infected[host]:
                    self._healthy_restorations += 1  # restauration à tort (hôte sain)
                    wrong_down[host] = True
                self.infected[host] = False
                self.isolated[host] = False
                self.down_now[host] = True           # indisponible pendant ce pas
                action_cost = c.c_restore
            else:
                self.patched[host] = True
                action_cost = c.c_patch

        # propagation : seuls les hôtes en ligne (ni isolés ni en restauration) propagent / sont infectés
        online = ~self.isolated & ~self.down_now
        active = self.infected & online
        susceptible = ~self.infected & online
        counts = self.adj[:, active].sum(axis=1)
        p_inf = 1.0 - (1.0 - c.beta) ** counts
        p_inf = np.where(self.patched, c.patch_factor * p_inf, p_inf)
        newly = susceptible & (self.np_random.random(n) < p_inf)
        self.infected |= newly

        unavailable = self.isolated | self.down_now
        n_inf, n_unav = int(self.infected.sum()), int(unavailable.sum())
        n_healthy_unav = int(((self.isolated & ~self.infected) | wrong_down).sum())
        reward = -(c.c_infected * n_inf + c.c_unavailable * n_unav + action_cost)

        self.t += 1
        self._ret += reward
        self._peak = max(self._peak, n_inf / n)
        self._avail_sum += 1.0 - n_unav / n
        self._healthy_unav_sum += n_healthy_unav

        truncated = self.t >= c.horizon
        info = {"n_infected": n_inf, "n_unavailable": n_unav, "n_healthy_unavailable": n_healthy_unav}
        if truncated:
            info["episode_metrics"] = {
                "return": self._ret,
                "final_infected_frac": n_inf / n,
                "peak_infected_frac": self._peak,
                "availability": self._avail_sum / self.t,
                "healthy_unavailable_mean": self._healthy_unav_sum / self.t,
                "healthy_isolations": self._healthy_isolations,
                "healthy_restorations": self._healthy_restorations,
            }
        return self._observe(), float(reward), False, truncated, info
