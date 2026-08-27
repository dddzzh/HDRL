import numpy as np
import torch

from src.baselines.hirl import HIRLAgent, HIRLConfig
from src.baselines.hirl.replay import FailureAwareReplay


def small_config(**overrides):
    values = {
        "hidden_dim": 16,
        "batch_size": 4,
        "td3_replay_size": 32,
        "ddqn_replay_size": 32,
        "target_update_frequency": 2,
        "device": "cpu",
    }
    values.update(overrides)
    return HIRLConfig.from_mapping(values)


def test_hirl_preserves_shared_state_and_server_rho_action():
    agent = HIRLAgent(7, 3, small_config())
    state = np.linspace(-0.2, 0.3, 7, dtype=np.float32)
    decision = agent.select_action(
        state, np.array([True, False, True]), load=0.5, training=False
    )
    assert decision["state"].shape == (7,)
    assert decision["ddqn_state"].shape == (8,)
    assert decision["server_index"] in {0, 2}
    assert agent.config.min_rho <= decision["rho"] <= agent.config.max_rho


def test_hirl_updates_td3_and_ddqn_with_failure_aware_replay():
    agent = HIRLAgent(5, 2, small_config())
    mask = np.ones(2, dtype=bool)
    for index in range(6):
        state = np.full(5, index / 10.0, dtype=np.float32)
        next_state = np.full(5, (index + 1) / 10.0, dtype=np.float32)
        decision = agent.select_action(state, mask, load=0.2, training=True)
        agent.remember(
            decision, upper_reward=0.5, lower_reward=-0.25,
            next_state=next_state, done=index == 5, failure=index % 2 == 0,
        )
    metrics = agent.update()
    assert metrics is not None
    assert np.isfinite(metrics["td3_critic_loss"])
    assert np.isfinite(metrics["ddqn_loss"])
    expected_lr = agent.config.ddqn_learning_rate * (
        1.0 + agent.config.load_learning_rate_alpha * agent.current_load
    )
    assert metrics["ddqn_learning_rate"] == expected_lr


def test_failure_indicator_increases_priority_for_equal_td_error():
    replay = FailureAwareReplay(4, 2, (1,), seed=1)
    for failure in (False, True):
        replay.add([0, 0], [0], 0, [0, 0], False, failure)
    replay.update_priorities([0, 1], [0.5, 0.5], failure_priority=1.0)
    assert replay.priorities[1] == 2.0 * replay.priorities[0]


def test_hirl_checkpoint_round_trip_without_replay():
    agent = HIRLAgent(4, 2, small_config())
    payload = agent.checkpoint(episode=3, best_episode_reward=1.25, include_replay=False)
    restored = HIRLAgent(4, 2, small_config())
    restored.load_checkpoint(payload)
    for left, right in zip(agent.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(left, right)
