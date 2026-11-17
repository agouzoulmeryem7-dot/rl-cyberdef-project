import numpy as np
from cyberdef import CyberDefenseEnv, EnvConfig, action_index
from cyberdef.baselines import RandomPolicy, RestoreIfAlert
from cyberdef.env import ISOLATE, RESTORE, PATCH

N = 12


def _neighbor_of_infected(env):
    src = int(np.argmax(env.infected))
    return src, int(np.argmax(env.adj[src]))


def test_spaces_and_truncation():
    env = CyberDefenseEnv(EnvConfig(horizon=5))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (48,) and env.action_space.n == 37
    for _ in range(5):
        obs, r, term, trunc, info = env.step(0)
        assert not term
    assert trunc and "episode_metrics" in info


def test_reset_is_deterministic():
    a = CyberDefenseEnv(); b = CyberDefenseEnv()
    oa, _ = a.reset(seed=3); ob, _ = b.reset(seed=3)
    assert np.array_equal(oa, ob)


def test_no_noise_alerts_equal_infection():
    env = CyberDefenseEnv(EnvConfig(fp_rate=0.0, fn_rate=0.0))
    obs, _ = env.reset(seed=1)
    assert np.array_equal(obs[:N].astype(bool), env.infected)


def test_no_spread_when_beta_zero():
    env = CyberDefenseEnv(EnvConfig(beta=0.0))
    env.reset(seed=2)
    for _ in range(10):
        _, _, _, _, info = env.step(0)
    assert info["n_infected"] == 1


def test_isolate_then_restore_infected_host():
    env = CyberDefenseEnv(EnvConfig(beta=0.0))
    env.reset(seed=2)
    h = int(np.argmax(env.infected))
    env.step(action_index(ISOLATE, h, N))
    assert env.isolated[h] and env.infected[h]
    env.step(action_index(RESTORE, h, N))
    assert not env.isolated[h] and not env.infected[h]
    assert env._healthy_restorations == 0        # restauration justifiée


def test_isolating_healthy_host_is_counted_and_unavailable():
    env = CyberDefenseEnv(EnvConfig(beta=0.0))
    env.reset(seed=2)
    h = int(np.argmin(env.infected))
    _, r, _, _, info = env.step(action_index(ISOLATE, h, N))
    assert info["n_healthy_unavailable"] == 1 and env._healthy_isolations == 1
    assert info["n_unavailable"] == 1


def test_isolated_infected_host_does_not_spread():
    env = CyberDefenseEnv(EnvConfig(beta=1.0))
    env.reset(seed=4)
    h = int(np.argmax(env.infected))
    env.step(action_index(ISOLATE, h, N))
    assert int(env.infected.sum()) == 1


def test_restoring_healthy_host_costs_one_step_of_unavailability():
    cfg = EnvConfig(beta=0.0)
    env = CyberDefenseEnv(cfg)
    env.reset(seed=2)
    h = int(np.argmin(env.infected))
    _, r1, _, _, i1 = env.step(action_index(RESTORE, h, N))
    assert i1["n_unavailable"] == 1 and i1["n_healthy_unavailable"] == 1
    assert env._healthy_restorations == 1
    expected = -(cfg.c_infected * 1 + cfg.c_unavailable * 1 + cfg.c_restore)
    assert abs(r1 - expected) < 1e-9
    _, _, _, _, i2 = env.step(0)                 # disponible de nouveau au pas suivant
    assert i2["n_unavailable"] == 0 and not env.isolated[h]


def test_restoring_infected_host_cleans_it_and_is_down_one_step():
    env = CyberDefenseEnv(EnvConfig(beta=0.0))
    env.reset(seed=2)
    h = int(np.argmax(env.infected))
    _, _, _, _, info = env.step(action_index(RESTORE, h, N))
    assert not env.infected[h]
    assert info["n_unavailable"] == 1 and info["n_healthy_unavailable"] == 0
    assert env._healthy_restorations == 0


def test_host_in_restoration_cannot_be_infected_that_step():
    env = CyberDefenseEnv(EnvConfig(beta=1.0))
    env.reset(seed=5)
    src, nb = _neighbor_of_infected(env)
    env.step(action_index(RESTORE, nb, N))
    assert not env.infected[nb]                  # hors service : protégé ce pas
    env.step(0)
    assert env.infected[nb]                      # puis exposé normalement (beta=1)


def test_patch_reduces_infection_risk():
    env = CyberDefenseEnv(EnvConfig(beta=1.0, patch_factor=0.0))
    env.reset(seed=5)
    src, nb = _neighbor_of_infected(env)
    env.step(action_index(PATCH, nb, N))
    assert env.patched[nb] and not env.infected[nb]
    for _ in range(5):
        env.step(0)
    assert not env.infected[nb]                  # facteur 0 : plus aucun risque

    env2 = CyberDefenseEnv(EnvConfig(beta=1.0, patch_factor=1.0))
    env2.reset(seed=5)
    src, nb = _neighbor_of_infected(env2)
    env2.step(action_index(PATCH, nb, N))
    assert env2.infected[nb]                     # facteur 1 : aucun effet


def test_baselines_run_and_metrics_present():
    env = CyberDefenseEnv()
    for pol in (RandomPolicy(37), RestoreIfAlert(12)):
        obs, _ = env.reset(seed=0)
        for _ in range(50):
            obs, _, _, trunc, info = env.step(pol(obs))
        assert trunc
        m = info["episode_metrics"]
        assert 0.0 <= m["availability"] <= 1.0
        assert set(m) == {"return", "final_infected_frac", "peak_infected_frac", "availability",
                          "healthy_unavailable_mean", "healthy_isolations", "healthy_restorations"}
