from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.baselines.dqn_edge.agent import DQNEdgeAgent
from src.baselines.dqn_edge.config import DQNEdgeConfig
from src.baselines.dqn_edge.network import ContextAttention, ObjectiveQNetwork
from src.baselines.dqn_edge.reward import (
    DQNEdgeCommonReward,
    DQNEdgeNormalizedPaperReward,
    DQNEdgePaperReward,
)
from src.baselines.dqn_edge.state import DQNEdgeCommonStateBuilder, DQNEdgeStateBuilder
from src.simulator.workflow import DependencyEdge, Task, Workflow


def test_paper_defaults_and_network_shapes():
    config = DQNEdgeConfig()
    assert config.learning_rate == pytest.approx(5e-6)
    assert config.gamma == pytest.approx(0.9)
    assert config.batch_size == 128 and config.replay_capacity == 10_000
    assert ObjectiveQNetwork(9, 3, 64)(torch.zeros(2, 9)).shape == (2, 3)
    weights = ContextAttention()(torch.zeros(2, 4))
    assert weights.shape == (2, 2)
    assert torch.allclose(weights.sum(1), torch.ones(2))


def test_upward_rank_prioritizes_critical_chain():
    workflow = Workflow(
        "wf",
        (Task(0, 1e9), Task(1, 8e9), Task(2, 1e9)),
        (DependencyEdge(0, 1, 1.0), DependencyEdge(0, 2, 1.0)),
        0.0,
    )
    builder = DQNEdgeStateBuilder(2)
    servers = [{"max_freq_hz": 1e9}, {"max_freq_hz": 2e9}]
    ranks = builder.rank_tasks(workflow, servers, {"bandwidth_mbps": 1000.0}, beta=1.0)
    assert ranks[1] > ranks[2]
    assert 0 in builder._critical_cache[id(workflow)]
    assert 1 in builder._critical_cache[id(workflow)]


def test_equation_28_reward_is_negative_latency_and_energy():
    reward = DQNEdgePaperReward()
    time_reward, energy_reward, target, info = reward.compute(
        {"task_id": 3},
        {"arrival_time": 10.0, "task_energy": {3: 4.0}},
        {"event": {"final_aft": 25.0, "total_energy": 4.0}},
    )
    assert time_reward == pytest.approx(-15.0)
    assert energy_reward == pytest.approx(-4.0)
    assert target.sum() == pytest.approx(1.0)
    assert info["source"] == "wen_2026_dqn_edge_equation_28"


def test_paper_reward_keeps_two_objectives_separate():
    reward = DQNEdgePaperReward()
    time_reward, energy_reward, _, _ = reward.compute(
        {"task_id": 1},
        {"arrival_time": 4.0, "task_energy": {1: 9.0}},
        {"event": {"final_aft": 10.0, "total_energy": 9.0}},
    )
    assert time_reward == pytest.approx(-6.0)
    assert energy_reward == pytest.approx(-9.0)


def test_normalized_paper_reward_preserves_eq28_with_dimensionless_scales():
    reward = DQNEdgeNormalizedPaperReward()
    time_reward, energy_reward, target, info = reward.compute(
        {"task_id": 3, "relative_deadline": 30.0},
        {"arrival_time": 10.0, "relative_deadline": 30.0, "task_energy": {3: 4.0}},
        {"event": {"final_aft": 25.0, "total_energy": 4.0}},
        state={"decision_context": {"task_energy_scale": 8.0}},
    )
    assert time_reward == pytest.approx(-0.5)
    assert energy_reward == pytest.approx(-0.5)
    assert target.sum() == pytest.approx(1.0)
    assert info["raw_time_cost"] == pytest.approx(15.0)
    assert info["raw_energy_cost"] == pytest.approx(4.0)
    assert info["source"] == "wen_2026_dqn_edge_equation_28_normalized"


def test_dual_dqn_attention_update_and_epsilon_decay():
    config = DQNEdgeConfig(batch_size=4, replay_capacity=8, device="cpu")
    agent = DQNEdgeAgent(7, 3, config)
    for index in range(4):
        state = np.full(7, index, np.float32)
        agent.replay_buffer.add(state, index % 3, -1.0, -2.0, state + 1, False,
                                np.asarray([0.1, 0.2], np.float32),
                                np.asarray([0.6, 0.4], np.float32))
    result = agent.update()
    assert all(np.isfinite(result[key]) for key in ("time_loss", "energy_loss", "attention_loss"))
    before = agent.epsilon
    agent.after_environment_step()
    assert agent.epsilon == pytest.approx(before - config.epsilon_decay_per_step)


def test_state_dimension_matches_equation_27_resource_interpretation():
    builder = DQNEdgeStateBuilder(5)
    assert builder.state_dim == 13


def test_common_state_matches_current_upper_dimension():
    builder = DQNEdgeCommonStateBuilder(26, {"max_ready_tasks": 32})
    assert builder.state_dim == 347


def test_common_reward_keeps_normalized_time_and_energy_objectives_separate():
    reward = DQNEdgeCommonReward(
        {
            "deadline_violation_weight": 1.5,
            "high_energy_weight": 1.0,
        }
    )
    assert reward.objective.reward_mode == "deadline_incremental_only"
    time_reward, energy_reward, components = reward.split_feedback(
        {
            "normalized_makespan_increment": 0.2,
            "high_deadline_violation_delta": 0.4,
            "task_energy_scale": 100.0,
            "primary_energy_actual": 70.0,
            "workflow_task_count": 10.0,
        }
    )
    assert time_reward == pytest.approx(-0.8)
    assert energy_reward == pytest.approx(0.03)
    assert components["reward"] == pytest.approx(-0.77)


def test_rank_cache_is_cleared_between_episodes():
    builder = DQNEdgeStateBuilder(2)
    builder._rank_cache[123] = {0: 1.0}
    builder._critical_cache[123] = {0}
    builder.reset_episode()
    assert builder._rank_cache == {}
    assert builder._critical_cache == {}
