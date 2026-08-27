from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.D3QN.hierarchical_d3qn_agent import HighLevelD3QNAgent
from src.coordination.c3c import (
    CounterfactualCrossLevelCritic,
    compute_c3c_step_components,
)
from src.coordination.c3c_agent import C3CHierarchicalAgent
from src.sac.hierarchical_low_sac import LowLevelDVFSContinuousSACAgent


def _parameter_vector(module):
    return torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in module.parameters()]
    )


def test_c3c_reward_decomposition_is_exact_and_continuous():
    info = {
        "task_energy_scale": 600.0,
        "workflow_task_count": 24,
        "high_placement_energy_rho1": 500.0,
        "total_energy_actual": 330.0,
        "low_sac_energy_reference": 500.0,
        "low_sac_energy_actual": 330.0,
        "deadline_limit": 100.0,
        "low_sac_workflow_partial_makespan_before": 20.0,
        "low_sac_reference_partial_makespan_after": 23.0,
        "low_sac_workflow_partial_makespan_after": 30.0,
        "low_sac_deadline_constraint_crossed": 0.0,
    }
    row = compute_c3c_step_components(
        info, energy_weight=1.0, deadline_lambda=2.0
    )
    assert abs(row["decomposition_error"]) < 1e-12
    assert np.isclose(row["base_energy_reward"], 100.0 / (24.0 * 600.0))
    assert np.isclose(row["residual_energy_reward"], 170.0 / (24.0 * 600.0))
    assert np.isclose(row["base_deadline_cost"], 0.03)
    assert np.isclose(row["residual_deadline_cost"], 0.07)
    assert np.isclose(row["realized_reward"], 270.0 / 14400.0 - 0.2)


def test_c3c_sub_deadline_decomposition_matches_base_agents():
    info = {
        "high_reward_profile": "sub_deadline_lagrangian_v1",
        "task_energy_scale": 600.0,
        "workflow_task_count": 24,
        "high_placement_energy_rho1": 500.0,
        "total_energy_actual": 330.0,
        "low_sac_energy_reference": 500.0,
        "low_sac_energy_actual": 330.0,
        "low_sac_reference_sub_deadline_cost": 0.05,
        "low_sac_actual_sub_deadline_cost": 0.08,
        "low_sac_deadline_constraint_crossed": 0.0,
    }
    row = compute_c3c_step_components(
        info, energy_weight=1.0, deadline_lambda=2.0
    )
    assert abs(row["decomposition_error"]) < 1e-12
    assert np.isclose(row["base_energy_reward"], 100.0 / (24.0 * 600.0))
    assert np.isclose(row["residual_energy_reward"], 170.0 / (24.0 * 600.0))
    assert np.isclose(row["base_deadline_cost"], 0.05)
    assert np.isclose(row["residual_deadline_cost"], 0.03)
    assert np.isclose(
        row["realized_reward"], 270.0 / (24.0 * 600.0) - 2.0 * 0.08
    )


def test_c3c_updates_high_low_and_coordination_networks():
    state = {"max_ready_tasks": 4, "max_servers": 2}
    high = HighLevelD3QNAgent(
        {
            "state": state,
            "device": "cpu",
            "batch_size": 8,
            "replay_size": 64,
            "training_per_episode": 1,
            "epsilon_decay": 0.0,
        },
        seed=11,
    )
    low = LowLevelDVFSContinuousSACAgent(
        {
            "state": state,
            "device": "cpu",
            "rho_min": 0.2,
            "batch_size": 8,
            "replay_size": 64,
            "training_per_episode": 1,
            "automatic_entropy_tuning": False,
            "reward": {"mode": "fixed_deadline_penalty"},
        },
        seed=12,
    )
    critic = CounterfactualCrossLevelCritic(
        high_state_dim=high.encoder.high_state_dim,
        action_dim=2,
        config={
            "device": "cpu",
            "batch_size": 8,
            "replay_size": 64,
            "training_per_episode": 1,
            "warmup_transitions": 0,
            "beta_anneal_transitions": 1,
            "beta_final": 0.25,
            "low_guidance_enabled": True,
            "low_guidance_weight_final": 0.02,
            "low_guidance_updates": 1,
        },
        seed=13,
    )
    critic.attach_low_actor(low.actor)
    rng = np.random.default_rng(14)
    for index in range(16):
        candidate_bases = rng.normal(
            0.0, 0.2, size=(2, critic.layout.base_dim)
        ).astype(np.float32)
        critic.replay.store(
            {
                "high_state": rng.normal(
                    0.0, 0.2, size=high.encoder.high_state_dim
                ).astype(np.float32),
                "action": index % 2,
                "base_energy_reward": 0.01 + 0.001 * index,
                "base_deadline_cost": 0.01 * (index % 3),
                "selected_base": candidate_bases[index % 2],
                "hardware_min": 0.2,
                "rho": 0.4 + 0.02 * (index % 8),
                "residual_energy_reward": -0.01 + 0.001 * index,
                "residual_deadline_cost": 0.02 * (index % 2),
                "crossed": float(index % 7 == 0),
                "next_high_state": rng.normal(
                    0.0, 0.2, size=high.encoder.high_state_dim
                ).astype(np.float32),
                "next_candidate_bases": rng.normal(
                    0.0, 0.2, size=(2, critic.layout.base_dim)
                ).astype(np.float32),
                "next_hardware_mins": np.asarray([0.2, 0.2], dtype=np.float32),
                "next_valid_servers": np.asarray([True, True]),
                "done": bool(index == 15),
            }
        )
    critic.observed_transitions = critic.replay.size

    high_before = _parameter_vector(high.agent_R.q_eval)
    low_before = _parameter_vector(low.actor)
    critic_before = _parameter_vector(critic.critic_1)

    critic_metrics = critic.update_critics(high, low, deadline_lambda=2.0)
    high_metrics = critic.update_high(high, low, deadline_lambda=2.0)
    # Remove twin disagreement for this focused actor-gradient assertion.  A
    # very large test lambda then guarantees a harmful residual and a nonzero
    # ReLU safety penalty.
    critic.critic_2.load_state_dict(critic.critic_1.state_dict())
    low_metrics = critic.update_low_safety(low, deadline_lambda=100.0)

    assert critic_metrics["updates"] == 1.0
    assert high_metrics["updates"] == 1.0
    assert low_metrics["updates"] == 1.0
    assert not torch.equal(critic_before, _parameter_vector(critic.critic_1))
    assert not torch.equal(high_before, _parameter_vector(high.agent_R.q_eval))
    assert not torch.equal(low_before, _parameter_vector(low.actor))
    assert np.isfinite(critic_metrics["critic_1_loss"])
    assert np.isfinite(high_metrics["loss"])
    assert np.isfinite(low_metrics["loss"])


def test_c3c_checkpoint_round_trip():
    config = {
        "low_algorithm": "sac",
        "shared": {
            "device": "cpu",
            "state": {"max_ready_tasks": 4, "max_servers": 2},
        },
        "high": {
            "batch_size": 8,
            "replay_size": 32,
            "training_per_episode": 1,
            "backup_selection": "disabled",
        },
        "low": {
            "algorithm": "sac",
            "batch_size": 8,
            "replay_size": 32,
            "training_per_episode": 1,
            "automatic_entropy_tuning": False,
            "reward": {"mode": "fixed_deadline_penalty"},
        },
        "c3c": {
            "enabled": True,
            "batch_size": 8,
            "replay_size": 32,
            "training_per_episode": 1,
            "constraint_gate_enabled": True,
            "constraint_gate_activate_success": 1.0,
            "constraint_gate_deactivate_success": 0.75,
            "constraint_gate_window_workflows": 4,
            "constraint_gate_min_workflows": 4,
            "constraint_gate_anneal_workflows": 8,
        },
    }
    first = C3CHierarchicalAgent(config, seed=21)
    first.update_constraint_gate([0.0, 0.0, 0.0, 0.0])
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "c3c.pt"
        first.save_checkpoint(checkpoint, episode=7, config=config)
        second = C3CHierarchicalAgent(config, seed=99)
        payload = second.load_checkpoint(checkpoint)
    assert payload["episode"] == 7
    assert torch.equal(
        _parameter_vector(first.high_agent.agent_R.q_eval),
        _parameter_vector(second.high_agent.agent_R.q_eval),
    )
    assert torch.equal(
        _parameter_vector(first.low_agent.actor),
        _parameter_vector(second.low_agent.actor),
    )
    assert torch.equal(
        _parameter_vector(first.c3c.critic_1),
        _parameter_vector(second.c3c.critic_1),
    )
    assert first.constraint_gate_status() == second.constraint_gate_status()


def test_c3c_constraint_gate_uses_hysteresis_and_annealing():
    config = {
        "low_algorithm": "sac",
        "shared": {
            "device": "cpu",
            "state": {"max_ready_tasks": 4, "max_servers": 2},
        },
        "high": {"backup_selection": "disabled"},
        "low": {
            "algorithm": "sac",
            "automatic_entropy_tuning": False,
            "reward": {"mode": "fixed_deadline_penalty"},
        },
        "c3c": {
            "enabled": True,
            "constraint_gate_enabled": True,
            "constraint_gate_activate_success": 1.0,
            "constraint_gate_deactivate_success": 0.75,
            "constraint_gate_window_workflows": 4,
            "constraint_gate_min_workflows": 4,
            "constraint_gate_anneal_workflows": 8,
        },
    }
    agent = C3CHierarchicalAgent(config, seed=31)
    assert agent.constraint_gate_factor == 0.0
    agent.update_constraint_gate([0.0, 0.0])
    assert agent.constraint_gate_factor == 0.0
    activated = agent.update_constraint_gate([0.0, 0.0])
    assert activated["c3c_constraint_gate_active"] == 1.0
    assert np.isclose(activated["c3c_constraint_gate_factor"], 0.25)
    deactivated = agent.update_constraint_gate([1.0, 1.0])
    assert deactivated["c3c_constraint_gate_active"] == 0.0
    assert deactivated["c3c_constraint_gate_factor"] == 0.0


def test_c3c_q_scale_alignment_bounds_coordination_correction():
    config = {
        "low_algorithm": "sac",
        "shared": {
            "device": "cpu",
            "state": {"max_ready_tasks": 4, "max_servers": 2},
        },
        "high": {"backup_selection": "disabled"},
        "low": {
            "algorithm": "sac",
            "automatic_entropy_tuning": False,
            "reward": {"mode": "fixed_deadline_penalty"},
        },
        "c3c": {
            "enabled": True,
            "q_scale_alignment_enabled": True,
            "correction_q_std_clip": 0.5,
        },
    }
    agent = C3CHierarchicalAgent(config, seed=32)
    correction, metrics = agent._coordinated_correction(
        np.asarray([0.0, 4.0]),
        {
            "advantage": np.asarray([0.0, 2.0]),
            "beta_effective": np.asarray([0.25, 0.25]),
            "correction": np.asarray([10.0, 10.0]),
        },
        np.asarray([True, True]),
        gate_factor=1.0,
    )
    assert np.allclose(correction, [-0.5, 0.5])
    assert metrics["c3c_correction_abs_mean"] <= 0.5 * metrics["c3c_base_q_std"]


def test_always_on_c3c_uses_relative_safe_correction():
    config = {
        "low_algorithm": "sac",
        "shared": {
            "device": "cpu",
            "state": {"max_ready_tasks": 4, "max_servers": 2},
        },
        "high": {"backup_selection": "disabled"},
        "low": {
            "algorithm": "sac",
            "automatic_entropy_tuning": False,
            "reward": {"mode": "fixed_deadline_penalty"},
        },
        "c3c": {
            "enabled": True,
            "constraint_gate_enabled": False,
            "q_scale_alignment_enabled": True,
            "correction_q_std_clip": 0.25,
            "relative_to_d3qn_action": True,
            "extra_deadline_risk_weight": 1.0,
            "low_guidance_lambda_source": "low",
        },
    }
    agent = C3CHierarchicalAgent(config, seed=33)
    assert agent.constraint_gate_factor == 1.0

    base_q = np.asarray([2.0, 1.0])
    valid = np.asarray([True, True])
    risky_correction, risky_metrics = agent._coordinated_correction(
        base_q,
        {
            "energy": np.asarray([0.0, 1.0]),
            "deadline": np.asarray([0.0, 1.0]),
            "advantage": np.asarray([0.0, -1.0]),
            "beta_effective": np.asarray([0.1, 0.1]),
            "correction": np.asarray([0.0, 0.0]),
        },
        valid,
        gate_factor=1.0,
    )
    assert np.isclose(risky_correction[0], 0.0)
    assert risky_correction[1] < 0.0
    assert risky_metrics["c3c_extra_deadline_risk_mean"] > 0.0

    efficient_correction, _ = agent._coordinated_correction(
        base_q,
        {
            "energy": np.asarray([0.0, 1.0]),
            "deadline": np.asarray([0.0, 0.0]),
            "advantage": np.asarray([0.0, 1.0]),
            "beta_effective": np.asarray([0.1, 0.1]),
            "correction": np.asarray([0.0, 0.0]),
        },
        valid,
        gate_factor=1.0,
    )
    assert np.isclose(efficient_correction[0], 0.0)
    assert efficient_correction[1] > 0.0


def test_always_on_c3c_low_safety_uses_low_lambda():
    config = {
        "low_algorithm": "sac",
        "shared": {
            "device": "cpu",
            "state": {"max_ready_tasks": 4, "max_servers": 2},
        },
        "high": {"backup_selection": "disabled"},
        "low": {
            "algorithm": "sac",
            "automatic_entropy_tuning": False,
            "reward": {"mode": "fixed_deadline_penalty"},
        },
        "c3c": {
            "enabled": True,
            "constraint_gate_enabled": False,
            "low_guidance_lambda_source": "low",
        },
    }
    agent = C3CHierarchicalAgent(config, seed=34)
    captured = {}

    def fake_update_low_safety(low_agent, deadline_lambda):
        captured["low_agent"] = low_agent
        captured["deadline_lambda"] = float(deadline_lambda)
        return {"updates": 0.0, "weight": 0.0}

    agent.c3c.update_low_safety = fake_update_low_safety
    metrics = agent.update_low_safety()
    assert captured["low_agent"] is agent.low_agent
    assert np.isclose(captured["deadline_lambda"], agent.low_agent.deadline_lambda)
    assert metrics["deadline_lambda_is_low"] == 1.0
