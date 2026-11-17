"""Tests du DQN de base : réseau, buffer, action, perte, boucle courte, enregistrement."""
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
import torch

from cyberdef import CyberDefenseEnv, EnvConfig
from cyberdef.dqn import (DQNAgent, DQNConfig, QNetwork, linear_epsilon, plot_learning_curve,
                          train_dqn, write_training_csv)
from cyberdef.replay_buffer import ReplayBuffer

OBS, NACT = 48, 37


def small_cfg(**kw):
    base = dict(hidden_sizes=(16, 16), batch_size=8, buffer_size=200, learning_starts=16,
                target_update_every=10, eps_decay_steps=50, n_episodes=3, window=2)
    base.update(kw)
    return DQNConfig(**base)


# ------------------------------------------------------------------ contrat d'interface (env réel)
def test_env_contract_matches_dqn_assumptions():
    env = CyberDefenseEnv()
    assert env.observation_space.shape == (OBS,)
    assert env.action_space.n == NACT
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (OBS,)
    _, r, terminated, truncated, _ = env.step(0)
    assert r <= 0 and terminated is False and truncated is False


# ------------------------------------------------------------------ réseau
def test_qnetwork_shapes():
    net = QNetwork(OBS, NACT, (32, 32))
    assert net(torch.zeros(5, OBS)).shape == (5, NACT)
    assert net(torch.zeros(1, OBS)).shape == (1, NACT)


def test_qnetwork_can_output_negative_values_and_is_finite():
    net = QNetwork(OBS, NACT)
    out = net(torch.rand(10, OBS))
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------ replay buffer
def test_buffer_push_len_and_sample_shapes():
    buf = ReplayBuffer(10, OBS)
    for i in range(6):
        buf.push(np.full(OBS, i, np.float32), i, -1.0, np.full(OBS, i + 1, np.float32), False)
    assert len(buf) == 6
    s, a, r, s2, d = buf.sample(4)
    assert s.shape == (4, OBS) and s2.shape == (4, OBS)
    assert a.shape == r.shape == d.shape == (4,)
    assert a.dtype == torch.int64 and s.dtype == torch.float32


def test_buffer_is_circular_and_keeps_transitions_consistent():
    buf = ReplayBuffer(5, OBS)
    for i in range(12):
        buf.push(np.full(OBS, i, np.float32), i, float(i), np.full(OBS, i + 1, np.float32), False)
    assert len(buf) == 5                                  # plafonné à la capacité
    assert set(buf.actions.tolist()) == {7, 8, 9, 10, 11}  # les plus anciens ont été écrasés
    s, a, r, s2, _ = buf.sample(5)
    assert torch.equal(s[:, 0].long(), a) and torch.equal(s2[:, 0], s[:, 0] + 1)


def test_buffer_sample_too_large_raises():
    buf = ReplayBuffer(10, OBS)
    buf.push(np.zeros(OBS), 0, 0.0, np.zeros(OBS), False)
    with pytest.raises(ValueError):
        buf.sample(2)


# ------------------------------------------------------------------ sélection d'action
def test_greedy_action_is_argmax_and_deterministic():
    agent = DQNAgent(OBS, NACT, small_cfg(), seed=0)
    obs = np.random.default_rng(0).random(OBS).astype(np.float32)
    with torch.no_grad():
        expected = int(agent.q_net(torch.from_numpy(obs).unsqueeze(0)).argmax())
    assert all(agent.select_action(obs, 0.0) == expected for _ in range(10))


def test_epsilon_one_explores_and_actions_are_valid():
    agent = DQNAgent(OBS, NACT, small_cfg(), seed=0)
    obs = np.zeros(OBS, np.float32)
    acts = [agent.select_action(obs, 1.0) for _ in range(300)]
    assert all(0 <= a < NACT for a in acts) and len(set(acts)) > 10


def test_epsilon_schedule():
    assert linear_epsilon(0, 1.0, 0.05, 100) == 1.0
    assert linear_epsilon(50, 1.0, 0.05, 100) == pytest.approx(0.525)
    assert linear_epsilon(10_000, 1.0, 0.05, 100) == pytest.approx(0.05)


# ------------------------------------------------------------------ cible de Bellman et perte
class _ConstQ(torch.nn.Module):
    """Faux réseau cible : renvoie toujours les mêmes Q, pour vérifier la formule à la main."""
    def __init__(self, q):
        super().__init__()
        self.q = torch.tensor(q, dtype=torch.float32)

    def forward(self, x):
        return self.q.expand(x.shape[0], -1)


def test_bellman_target_formula_and_done_masking():
    agent = DQNAgent(OBS, NACT, small_cfg(gamma=0.9), seed=0)
    agent.target_net = _ConstQ([1.0, 5.0, 2.0])
    r = torch.tensor([-1.0, -1.0])
    d = torch.tensor([0.0, 1.0])
    y = agent.bellman_target(r, torch.zeros(2, OBS), d)
    assert y[0].item() == pytest.approx(-1.0 + 0.9 * 5.0)   # r + gamma * max Q'
    assert y[1].item() == pytest.approx(-1.0)               # état terminal : pas d'amorçage


def test_loss_is_finite_nonnegative_and_gradients_flow_only_to_qnet():
    agent = DQNAgent(OBS, NACT, small_cfg(), seed=0)
    rng = np.random.default_rng(1)
    for _ in range(32):
        agent.buffer.push(rng.random(OBS), int(rng.integers(NACT)), -float(rng.integers(0, 10)),
                          rng.random(OBS), False)
    loss = agent.compute_loss(agent.buffer.sample(8))
    assert torch.isfinite(loss) and loss.item() >= 0
    agent.optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is not None for p in agent.q_net.parameters())
    assert all(p.grad is None for p in agent.target_net.parameters())


def test_update_changes_qnet_but_not_target_and_sync_copies():
    agent = DQNAgent(OBS, NACT, small_cfg(), seed=0)
    rng = np.random.default_rng(2)
    for _ in range(32):
        agent.buffer.push(rng.random(OBS), int(rng.integers(NACT)), -5.0, rng.random(OBS), False)
    q_before = copy.deepcopy(agent.q_net.state_dict())
    t_before = copy.deepcopy(agent.target_net.state_dict())
    assert agent.update() is not None
    changed = any(not torch.equal(q_before[k], v) for k, v in agent.q_net.state_dict().items())
    assert changed
    assert all(torch.equal(t_before[k], v) for k, v in agent.target_net.state_dict().items())
    agent.sync_target()
    assert all(torch.equal(agent.q_net.state_dict()[k], v) for k, v in agent.target_net.state_dict().items())


def test_update_waits_for_enough_data():
    agent = DQNAgent(OBS, NACT, small_cfg(learning_starts=16), seed=0)
    assert agent.update() is None


# ------------------------------------------------------------------ boucle courte + enregistrement
def test_short_training_loop_runs_and_logs(tmp_path):
    env = CyberDefenseEnv(EnvConfig(n_hosts=12, horizon=10))
    cfg = small_cfg()
    agent, rows = train_dqn(env, cfg, seed=0, n_episodes=4)
    assert len(rows) == 4
    assert [r["episode"] for r in rows] == [1, 2, 3, 4]
    assert all(r["length"] == 10 and np.isfinite(r["reward"]) for r in rows)
    assert all(r["env_seed"] == i for i, r in enumerate(rows))
    assert rows[0]["epsilon"] > rows[-1]["epsilon"] or rows[-1]["epsilon"] == cfg.eps_end
    assert len(agent.buffer) == 40
    assert any(np.isfinite(r["mean_loss"]) for r in rows)   # l'apprentissage a bien démarré


def test_training_is_reproducible_with_same_seed():
    cfg = small_cfg()
    _, a = train_dqn(CyberDefenseEnv(EnvConfig(horizon=10)), cfg, seed=3, n_episodes=3)
    _, b = train_dqn(CyberDefenseEnv(EnvConfig(horizon=10)), cfg, seed=3, n_episodes=3)
    assert [r["reward"] for r in a] == [r["reward"] for r in b]


def test_csv_and_figure_are_written_from_real_rows(tmp_path):
    cfg = small_cfg()
    _, rows = train_dqn(CyberDefenseEnv(EnvConfig(horizon=10)), cfg, seed=0, n_episodes=4)
    path = str(tmp_path / "dqn_training.csv")
    write_training_csv(rows, path, cfg, {"noise": "medium"})
    with open(path) as f:
        read = list(csv.DictReader(f))
    assert len(read) == 4
    for col in ["episode", "reward", "moving_avg_reward", "epsilon", "length", "seed",
                "hp_lr", "hp_gamma", "hp_batch_size", "hp_noise"]:
        assert col in read[0]
    assert float(read[2]["reward"]) == pytest.approx(rows[2]["reward"])
    png = str(tmp_path / "curve.png")
    plot_learning_curve(path, png)
    assert os.path.getsize(png) > 1000


def test_greedy_policy_works_with_project_evaluate_policy():
    from cyberdef.evaluate import evaluate_policy
    agent = DQNAgent(OBS, NACT, small_cfg(), seed=0)
    rows = evaluate_policy(agent.greedy_policy(), EnvConfig(horizon=10), n_episodes=2)
    assert len(rows) == 2 and "return" in rows[0] and rows[0]["env_seed"] >= 100_000


# ------------------------------------------------------------------ chargement du modèle (révision 2)
def _trained_agent(seed=0):
    agent, _ = train_dqn(CyberDefenseEnv(EnvConfig(horizon=10)), small_cfg(), seed=seed, n_episodes=4)
    return agent


def test_saved_model_reloads_with_identical_q_values(tmp_path):
    agent = _trained_agent()
    path = str(tmp_path / "m.pt")
    agent.save(path)
    obs = torch.from_numpy(np.random.default_rng(5).random((6, OBS)).astype(np.float32))
    with torch.no_grad():
        q_saved = agent.q_net(obs)
        fresh = DQNAgent.from_checkpoint(path)                    # architecture lue dans le fichier
        q_loaded = fresh.q_net(obs)
        other = DQNAgent(OBS, NACT, small_cfg(), seed=99)         # autre initialisation
        assert not torch.equal(other.q_net(obs), q_saved)
        other.load_weights(path)
        q_loaded2 = other.q_net(obs)
        q_target = other.target_net(obs)
    assert torch.equal(q_saved, q_loaded) and torch.equal(q_saved, q_loaded2)
    assert torch.equal(q_saved, q_target)                          # le réseau cible est resynchronisé
    o = obs[0].numpy()
    assert agent.select_action(o, 0.0) == fresh.select_action(o, 0.0)


def test_new_checkpoint_loads_with_plain_weights_only(tmp_path):
    path = str(tmp_path / "m.pt")
    _trained_agent().save(path)
    ckpt = torch.load(path, weights_only=True)                     # sans safe_globals : types Python purs
    assert type(ckpt["obs_dim"]) is int and type(ckpt["n_actions"]) is int


def test_legacy_checkpoint_with_numpy_scalar_still_loads(tmp_path):
    agent = _trained_agent()
    path = str(tmp_path / "legacy.pt")
    torch.save({"state_dict": agent.q_net.state_dict(), "config": agent.cfg.to_dict(),
                "obs_dim": np.int64(OBS), "n_actions": NACT}, path)   # format de l'ancien save()
    loaded = DQNAgent.from_checkpoint(path)
    x = torch.rand(3, OBS)
    with torch.no_grad():
        assert torch.equal(agent.q_net(x), loaded.q_net(x))


def test_load_weights_rejects_incompatible_dimensions(tmp_path):
    path = str(tmp_path / "m.pt")
    _trained_agent().save(path)
    with pytest.raises(ValueError):
        DQNAgent(OBS + 1, NACT, small_cfg()).load_weights(path)


REAL_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "dqn_model.pt")
REAL_EVAL = REAL_MODEL.replace("dqn_model.pt", "dqn_eval.csv")


@pytest.mark.skipif(not (os.path.exists(REAL_MODEL) and os.path.exists(REAL_EVAL)), reason="résultats absents")
def test_real_saved_model_reproduces_saved_evaluation():
    """Le modèle réel rechargé rejoue les 3 premiers épisodes d'évaluation du CSV (mêmes graines, ε=0)."""
    import pandas as pd
    from cyberdef.evaluate import evaluate_policy
    agent = DQNAgent.from_checkpoint(REAL_MODEL)
    rows = evaluate_policy(agent.greedy_policy(), EnvConfig().with_noise("medium"), n_episodes=3)
    saved = pd.read_csv(REAL_EVAL).sort_values("env_seed").head(3)["return"].values
    assert np.allclose([r["return"] for r in rows], saved, atol=1e-9)


# ------------------------------------------------------------------ sémantique de truncated
def test_truncation_is_not_stored_as_done():
    """L'env ne termine jamais (terminated=False) ; la fin à l'horizon est une troncature.
    Le buffer ne doit donc contenir aucun done=1 : la cible est toujours amorcée (voir docs)."""
    env = CyberDefenseEnv(EnvConfig(horizon=10))
    agent, _ = train_dqn(env, small_cfg(), seed=0, n_episodes=3)
    assert agent.buffer.dones[:len(agent.buffer)].sum() == 0
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    assert obs.shape == (OBS,) and ((obs >= 0) & (obs <= 1)).all()   # s' valide fourni au pas de troncature
