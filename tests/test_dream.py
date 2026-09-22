from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.baselines.dream.dream_action import DREAM_RHO_LEVELS, action_dimension, decode_action, encode_action
from src.baselines.dream.dream_agent import DReaMAgent, DReaMReplayBuffer
from src.baselines.dream.dream_config import DReaMConfig
from src.baselines.dream.dream_fault_tolerance import FaultToleranceTechnique, get_additional_replica_count
from src.baselines.dream.dream_network import DReaMQNetwork
from src.baselines.dream.dream_reward import SharedWorkflowReward
from src.baselines.dream.dream_scheduler import DReaMScheduler
from src.baselines.dream.dream_state import SharedEdgeCloudStateBuilder
from src.shared.hierarchical_state_reward import (
    FIXED_LAMBDA_UPPER_REWARD_VERSION,
)


def small_config(**updates):
    values = {"batch_size": 4, "replay_capacity": 8, "target_update_frequency": 50, "device": "cpu"}; values.update(updates)
    return DReaMConfig(**values)


def test_paper_defaults_and_network_shape_and_batchnorm():
    config = DReaMConfig(); assert config.batch_size == 1000 and config.replay_capacity == 16800
    net = DReaMQNetwork(11, 35); net.eval(); assert net(torch.randn(1, 11)).shape == (1, 35)
    net.train(); assert net(torch.randn(4, 11)).shape == (4, 35)


def test_all_joint_actions_round_trip_and_rho():
    for ft in (False, True):
        for action_id in range(action_dimension(5, ft)):
            d = decode_action(action_id, 5, ft)
            assert d.rho in DREAM_RHO_LEVELS
            assert encode_action(d.server_id, d.dvfs_index, 5, ft, d.ft_index) == action_id
            if not ft: assert d.ft_index is None and d.ft_technique is None


def test_target_initial_and_gradient_step_update():
    agent = DReaMAgent(6, 14, small_config(target_update_frequency=2))
    before = {k: v.clone() for k, v in agent.target_net.state_dict().items()}
    assert all(torch.equal(v, agent.online_net.state_dict()[k]) for k, v in before.items())
    with torch.no_grad(): next(agent.online_net.parameters()).add_(1.0)
    for _ in range(2): assert not agent.after_environment_step()
    assert all(torch.equal(v, agent.target_net.state_dict()[k]) for k, v in before.items())
    for i in range(4):
        agent.replay_buffer.add(np.ones(6) * i, i % 14, 1.0, np.ones(6), False)
    assert not agent.update()["target_updated"]
    assert agent.update()["target_updated"]
    assert all(torch.equal(v, agent.online_net.state_dict()[k]) for k, v in agent.target_net.state_dict().items())


def test_epsilon_floor_replay_capacity_and_update_smoke():
    agent = DReaMAgent(6, 14, small_config(epsilon_decay=0.5))
    for _ in range(100): agent.after_environment_step()
    assert agent.epsilon == pytest.approx(0.01)
    for i in range(12): agent.replay_buffer.add(np.ones(6) * i, i % 14, 1.0, np.ones(6), False)
    assert len(agent.replay_buffer) == 8
    result = agent.update(); assert np.isfinite(result["loss"]) and agent.gradient_steps == 1
    assert result["grad_norm_after_clip"] <= agent.config.max_grad_norm
    assert np.isfinite(result["q_abs_max"])
    assert np.isfinite(result["target_abs_max"])
    assert np.isfinite(result["parameter_abs_max"])


def test_divergence_guard_rejects_oversized_parameters():
    agent = DReaMAgent(6, 14, small_config(max_abs_parameter=2.0))
    with torch.no_grad():
        next(agent.online_net.parameters()).fill_(3.0)
    with pytest.raises(FloatingPointError, match="safety limit"):
        agent.assert_healthy("test")


def test_replay_rejects_nonfinite_transition():
    buffer = DReaMReplayBuffer(8, 6)
    bad_state = np.ones(6, dtype=np.float32)
    bad_state[0] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite state"):
        buffer.add(bad_state, 0, 1.0, np.zeros(6, dtype=np.float32), False)


def test_ft_interfaces_closed_and_replica_counts():
    assert get_additional_replica_count(FaultToleranceTechnique.PB) == 1
    assert get_additional_replica_count(FaultToleranceTechnique.TMR) == 2
    assert get_additional_replica_count(FaultToleranceTechnique.NMR) == 4
    with pytest.raises(RuntimeError, match="state builder"): DReaMScheduler(object())
    with pytest.raises(RuntimeError, match="reward function"): DReaMScheduler(object(), state_builder=object())


def test_evaluation_has_no_epsilon_exploration():
    agent = DReaMAgent(6, 14, small_config())
    actions = {agent.select_action(np.zeros(6, np.float32), training=False) for _ in range(20)}
    assert len(actions) == 1


def test_dream_uses_the_shared_347_dimensional_upper_state():
    builder = SharedEdgeCloudStateBuilder({"max_ready_tasks": 32, "max_servers": 26})
    assert builder.state_dim == 347


def test_dream_reward_matches_common_fixed_lambda_formula():
    class StubObjective:
        reward_mode = "deadline_incremental_only"
        fixed_deadline_lambda = 10.0
        high_energy_weight = 3.0

        def compute_step_feedback(self, *args, **kwargs):
            return {
                "normalized_makespan_increment": 0.2,
                "high_deadline_violation_delta": 0.4,
                "primary_energy_ref": 100.0,
                "primary_energy_actual": 70.0,
                "task_energy_scale": 100.0,
                "workflow_task_count": 10.0,
                "reliability_violation": 0.2,
                "reliability_violation_delta": 0.2,
                "reliability_crossed": 1.0,
            }

    reward, info = SharedWorkflowReward(
        StubObjective(), fixed_reliability_lambda=10.0
    ).compute_reward(
        env=None,
        task=None,
        workflow=None,
        decision=SimpleNamespace(server_id=2, rho=0.8, dvfs_index=4),
        transition_info={
            "pre_state": {
                "decision_context": {
                    "primary_server_mask": np.ones(3, dtype=bool),
                    "backup_pair_mask": np.zeros((3, 3), dtype=bool),
                }
            },
            "env_action": {"workflow_id": "wf", "task_id": 1},
            "environment_reward": -70.0,
            "environment_info": {},
        },
    )

    # 3 * ((100 - 70) / 100) / 10 - 10 * 0.2 - 10 * 0.2
    assert reward == pytest.approx(-3.91)
    assert info["source"] == (
        f"{FIXED_LAMBDA_UPPER_REWARD_VERSION}+workflow_reliability_v2"
    )
    assert info["components"]["energy_saving"] == pytest.approx(30.0)
    assert info["components"]["energy_weight"] == pytest.approx(3.0)
    assert info["components"]["reliability_violation_delta"] == pytest.approx(0.2)
    assert info["feedback"]["high_reward"] == pytest.approx(reward)
