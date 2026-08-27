import numpy as np
import torch

from src.coordination.hcc import HierarchicalCoordinationCritic


def _critic_config(**overrides):
    config = {
        "device": "cpu",
        "hidden_dim": 16,
        "hidden_layers": 1,
        "learning_rate": 1e-3,
        "gamma": 0.9,
        "tau": 0.05,
        "replay_size": 64,
        "batch_size": 4,
        "training_per_episode": 1,
        "energy_scale_joules": 100.0,
        "beta_final": 0.2,
        "warmup_transitions": 0,
        "beta_anneal_transitions": 1,
    }
    config.update(overrides)
    return config


def test_coordination_critic_learns_finite_two_head_costs():
    critic = HierarchicalCoordinationCritic(4, 3, _critic_config(), seed=7)
    for index in range(8):
        state = np.full(4, index / 10.0, dtype=np.float32)
        next_state = np.full(4, (index + 1) / 10.0, dtype=np.float32)
        critic.replay.store(
            state,
            index % 3,
            0.4 + 0.05 * index,
            0.2 + 0.1 * index,
            float(index % 4 == 0),
            next_state,
            (index + 1) % 3,
            0.45 + 0.05 * index,
            index == 7,
        )
    critic.observed_transitions = critic.replay.size

    metrics = critic.learn()

    assert metrics
    assert critic.guidance_active
    assert critic.beta == 0.2
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["energy_q_mean"] >= 0.0
    assert metrics["deadline_q_mean"] >= 0.0


def test_coordination_candidate_values_cover_discrete_continuous_joint_action():
    critic = HierarchicalCoordinationCritic(4, 3, _critic_config(), seed=11)
    result = critic.candidate_coordination_values(
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([0.5, 0.9], dtype=np.float32),
        deadline_lambda=2.0,
    )

    assert set(result) == {"values", "energy_costs", "deadline_costs"}
    assert all(values.shape == (2,) for values in result.values())
    assert np.all(result["energy_costs"] >= 0.0)
    assert np.all(result["deadline_costs"] >= 0.0)
    assert np.all(result["values"] <= 0.0)


def test_coordination_warmup_keeps_baseline_in_control():
    critic = HierarchicalCoordinationCritic(
        4,
        3,
        _critic_config(warmup_transitions=10, beta_anneal_transitions=10),
        seed=13,
    )
    for index in range(10):
        critic.replay.store(
            np.zeros(4, dtype=np.float32),
            0,
            1.0,
            0.0,
            0.0,
            np.zeros(4, dtype=np.float32),
            0,
            1.0,
            index == 9,
        )
    critic.observed_transitions = critic.replay.size

    assert critic.beta == 0.0
    assert not critic.guidance_active

    critic.replay.store(
        np.zeros(4, dtype=np.float32),
        0,
        1.0,
        0.0,
        0.0,
        np.zeros(4, dtype=np.float32),
        0,
        1.0,
        True,
    )
    critic.observed_transitions = critic.replay.size
    critic.learn_updates = 1
    assert np.isclose(critic.beta, 0.02)
    assert critic.guidance_active


def test_coordination_checkpoint_keeps_guidance_without_replay():
    critic = HierarchicalCoordinationCritic(4, 3, _critic_config(), seed=17)
    critic.observed_transitions = 20
    critic.learn_updates = 7
    payload = critic.state_dict(include_replay=False)

    restored = HierarchicalCoordinationCritic(4, 3, _critic_config(), seed=19)
    restored.load_state_dict(payload)

    assert restored.replay.size == 0
    assert restored.observed_transitions == 20
    assert restored.learn_updates == 7
    assert restored.guidance_active
    assert np.isclose(restored.beta, critic.beta)


def test_coordination_initialization_preserves_existing_policy_rng():
    torch.manual_seed(1234)
    expected = torch.rand(5)
    torch.manual_seed(1234)

    HierarchicalCoordinationCritic(4, 3, _critic_config(), seed=23)
    actual = torch.rand(5)

    assert torch.equal(actual, expected)
