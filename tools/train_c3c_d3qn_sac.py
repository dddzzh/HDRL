from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CPU_THREADS = os.environ.get("DRLEC_CPU_THREADS", "1")
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = _CPU_THREADS
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.D3QN.reward import HierarchicalObjective
from src.coordination.c3c_agent import C3CHierarchicalAgent
from src.device_utils import configure_torch_runtime, describe_device
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from tools.hierarchical_common import (
    _assert_metrics_finite,
    append_history_row,
    average_metrics,
    build_env,
    build_resources,
    build_workflows,
    configure_agent_state,
    initialize_history_files,
    load_json,
    metric_rank,
    validate_training_config,
    write_history_files,
)
from tools.train_hierarchical_d3qn import (
    _completed_workflow_deadline_usage,
    collect_d3qn_episode,
)


def parse_args(default_config="configs/hierarchical_d3qn_sac_c3c.json"):
    parser = argparse.ArgumentParser(
        description="Train the independent C3C D3QN-SAC joint learner."
    )
    parser.add_argument(
        "--config", default=default_config
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def _parameter_checksum(module):
    with torch.no_grad():
        return float(
            sum(
                parameter.detach().float().double().pow(2).sum().item()
                for parameter in module.parameters()
            )
        )


def _apply_smoke_overrides(config):
    config = copy.deepcopy(config)
    training = config.setdefault("training", {})
    training["total_episodes"] = min(int(training.get("total_episodes", 2)), 2)
    training["eval_interval"] = 1
    training["eval_episodes"] = 1
    training["save_every"] = 1
    for section, default_batch in (("high", 8), ("low", 8)):
        block = config.setdefault("agent", {}).setdefault(section, {})
        block["batch_size"] = min(int(block.get("batch_size", default_batch)), 8)
        block["replay_size"] = min(int(block.get("replay_size", 1000)), 1000)
        block["training_per_episode"] = 2
    c3c = config.setdefault("agent", {}).setdefault("c3c", {})
    c3c["batch_size"] = min(int(c3c.get("batch_size", 8)), 8)
    c3c["replay_size"] = min(int(c3c.get("replay_size", 1000)), 1000)
    c3c["training_per_episode"] = 2
    c3c["warmup_transitions"] = 0
    c3c["beta_anneal_transitions"] = 1
    c3c["low_guidance_updates"] = 1
    return config


def _augment_metrics(
    result,
    reward_metrics,
    high_metrics=None,
    low_metrics=None,
    c3c_metrics=None,
    safe_metrics=None,
    parameter_deltas=None,
    profile_name="c3c_rho1_residual_v1",
):
    metrics = dict(result["metrics"])
    high_metrics = dict(high_metrics or {})
    low_metrics = dict(low_metrics or {})
    c3c_metrics = dict(c3c_metrics or {})
    safe_metrics = dict(safe_metrics or {})
    parameter_deltas = dict(parameter_deltas or {})
    metrics.update(
        {
            "algorithm": "c3c_d3qn_sac",
            "reward_profile": str(profile_name),
            "mean_high_reward": float(reward_metrics.get("base_reward_mean", 0.0)),
            "total_high_reward": float(
                reward_metrics.get("base_reward_mean", 0.0)
                * result["high_buffer"].size
            ),
            "c3c_joint_reward_mean": float(
                reward_metrics.get("realized_reward_mean", 0.0)
            ),
            "c3c_residual_reward_mean": float(
                reward_metrics.get("residual_reward_mean", 0.0)
            ),
            "c3c_decomposition_error_abs_max": float(
                reward_metrics.get("decomposition_error_abs_max", 0.0)
            ),
            "c3c_base_energy_reward_mean": float(
                reward_metrics.get("base_energy_reward_mean", 0.0)
            ),
            "c3c_base_energy_reward_std": float(
                reward_metrics.get("base_energy_reward_std", 0.0)
            ),
            "c3c_residual_energy_reward_mean": float(
                reward_metrics.get("residual_energy_reward_mean", 0.0)
            ),
            "c3c_residual_energy_reward_std": float(
                reward_metrics.get("residual_energy_reward_std", 0.0)
            ),
            "c3c_residual_energy_unique_ratio": float(
                reward_metrics.get("residual_energy_reward_unique_ratio", 0.0)
            ),
            "c3c_residual_deadline_cost_mean": float(
                reward_metrics.get("residual_deadline_cost_mean", 0.0)
            ),
            "c3c_residual_deadline_cost_std": float(
                reward_metrics.get("residual_deadline_cost_std", 0.0)
            ),
            "c3c_residual_deadline_zero_rate": float(
                reward_metrics.get("residual_deadline_cost_zero_rate", 0.0)
            ),
            "high_updates": float(high_metrics.get("updates", 0.0)),
            "high_loss": float(high_metrics.get("loss", 0.0)),
            "high_grad_norm": float(
                high_metrics.get("grad_norm_before_clip", 0.0)
            ),
            "low_updates": float(low_metrics.get("updates", 0.0)),
            "low_actor_loss": float(low_metrics.get("actor_loss", 0.0)),
            "low_critic_loss": float(low_metrics.get("critic_1_loss", 0.0)),
            "low_actor_grad_norm": float(
                low_metrics.get("actor_grad_norm_before_clip", 0.0)
            ),
            "low_training_reward_mean": float(
                low_metrics.get(
                    "training_reward_mean",
                    low_metrics.get("reward_mean", 0.0),
                )
            ),
            "low_base_reward_mean": float(
                low_metrics.get("base_reward_mean", 0.0)
            ),
            "low_lagrangian_penalty_mean": float(
                low_metrics.get("lagrangian_penalty_mean", 0.0)
            ),
            "low_deadline_constraint_replay_mean": float(
                low_metrics.get("deadline_constraint_cost_mean", 0.0)
            ),
            "c3c_updates": float(c3c_metrics.get("updates", 0.0)),
            "c3c_loss": float(c3c_metrics.get("critic_1_loss", 0.0)),
            "c3c_grad_norm": float(
                c3c_metrics.get("grad_norm_before_clip", 0.0)
            ),
            "c3c_beta": float(c3c_metrics.get("beta", 0.0)),
            "c3c_replay_size": float(c3c_metrics.get("replay_size", 0.0)),
            "c3c_energy_q_mean": float(
                c3c_metrics.get("energy_q_mean", 0.0)
            ),
            "c3c_deadline_q_mean": float(
                c3c_metrics.get("deadline_q_mean", 0.0)
            ),
            "c3c_energy_td_abs_mean": float(
                c3c_metrics.get("energy_td_abs_mean", 0.0)
            ),
            "c3c_deadline_td_abs_mean": float(
                c3c_metrics.get("deadline_td_abs_mean", 0.0)
            ),
            "c3c_low_safety_updates": float(safe_metrics.get("updates", 0.0)),
            "c3c_low_safety_loss": float(safe_metrics.get("loss", 0.0)),
            "c3c_low_safety_weight": float(safe_metrics.get("weight", 0.0)),
            "c3c_low_safety_harmful_rate": float(
                safe_metrics.get("harmful_rate", 0.0)
            ),
            "c3c_low_safety_confidence_mean": float(
                safe_metrics.get("confidence_mean", 0.0)
            ),
            "c3c_low_safety_lambda": float(
                safe_metrics.get("deadline_lambda", 0.0)
            ),
            "c3c_low_safety_lambda_is_low": float(
                safe_metrics.get("deadline_lambda_is_low", 0.0)
            ),
            "c3c_selection_gate_factor_mean": float(
                reward_metrics.get("selection_constraint_gate_factor_mean", 0.0)
            ),
            "c3c_selection_base_q_std_mean": float(
                reward_metrics.get("selection_base_q_std_mean", 0.0)
            ),
            "c3c_selection_advantage_std_mean": float(
                reward_metrics.get("selection_advantage_std_mean", 0.0)
            ),
            "c3c_selection_correction_abs_mean": float(
                reward_metrics.get("selection_correction_abs_mean", 0.0)
            ),
            "c3c_selection_correction_to_q_std_mean": float(
                reward_metrics.get("selection_correction_to_q_std_mean", 0.0)
            ),
            "c3c_selection_base_action_correction_mean": float(
                reward_metrics.get("selection_base_action_correction_mean", 0.0)
            ),
            "c3c_selection_extra_deadline_risk_mean": float(
                reward_metrics.get("selection_extra_deadline_risk_mean", 0.0)
            ),
            "c3c_selection_relative_to_d3qn_action_mean": float(
                reward_metrics.get("selection_relative_to_d3qn_action_mean", 0.0)
            ),
            "c3c_upward_rank_feature_abs_mean": float(
                reward_metrics.get("selection_upward_rank_feature_abs_mean", 0.0)
            ),
            **{key: float(value) for key, value in parameter_deltas.items()},
        }
    )
    _assert_metrics_finite(metrics)
    return metrics


def evaluate(config, resources, agent, objective, *, seed, episodes, smoke_test):
    old_epsilon = float(agent.high_agent.epsilon)
    agent.high_agent.epsilon = 0.0
    rows = []
    try:
        for index in range(int(episodes)):
            episode_seed = int(seed) + index
            workflows = build_workflows(
                config, resources, episode_seed, smoke_test=smoke_test
            )
            env = build_env(
                workflows,
                resources,
                dict(config.get("simulation", {})),
                episode_seed,
            )
            result = collect_d3qn_episode(
                env,
                agent,
                objective,
                HighRolloutBuffer(),
                LowRolloutBuffer(),
                training=False,
                high_deterministic=True,
                low_deterministic=True,
                fixed_low_rho_one=False,
                high_algorithm="d3qn",
            )
            reward_metrics = agent.recompose_and_remember_episode(
                result["high_buffer"],
                energy_weight=objective.high_energy_weight,
                store=False,
            )
            rows.append(
                {
                    **_augment_metrics(
                    result,
                    reward_metrics,
                    c3c_metrics={
                        "updates": 0.0,
                        "beta": agent.c3c.beta,
                        "replay_size": agent.c3c.replay.size,
                    },
                    profile_name=agent.reward_profile,
                    ),
                    **agent.constraint_gate_status(),
                }
            )
    finally:
        agent.high_agent.epsilon = old_epsilon
    return average_metrics(rows)


def _print_row(label, episode, row):
    print(
        f"[{label}-c3c] ep={episode:4d} "
        f"deadline={100.0 * float(row.get('deadline_success_rate', 0.0)):.2f}% "
        f"energy={float(row.get('total_energy', 0.0)) / 1000.0:.3f}kJ "
        f"rho={float(row.get('rho_mean', 0.0)):.4f} "
        f"updates(H/L/C)={int(row.get('high_updates', 0))}/"
        f"{int(row.get('low_updates', 0))}/{int(row.get('c3c_updates', 0))} "
        f"beta={float(row.get('c3c_beta', 0.0)):.4f} "
        f"gate={float(row.get('c3c_constraint_gate_factor', row.get('c3c_selection_gate_factor_mean', 1.0))):.3f} "
        f"corrQ={float(row.get('c3c_selection_correction_to_q_std_mean', 0.0)):.4f} "
        f"changed={100.0 * float(row.get('coordination_changed_greedy_rate', 0.0)):.2f}% "
        f"rankU={float(row.get('c3c_upward_rank_feature_abs_mean', 0.0)):.4g} "
        f"rH={float(row.get('mean_high_reward', 0.0)):.5g} "
        f"rL={float(row.get('low_training_reward_mean', 0.0)):.5g} "
        f"rC={float(row.get('c3c_residual_reward_mean', 0.0)):.5g} "
        f"rE_std={float(row.get('c3c_residual_energy_reward_std', 0.0)):.4g} "
        f"rD_zero={float(row.get('c3c_residual_deadline_zero_rate', 0.0)):.3f} "
        f"decomp={float(row.get('c3c_decomposition_error_abs_max', 0.0)):.2e}"
    )


def _validate_training_diagnostics(row, *, replay_batch_size):
    if float(row.get("c3c_decomposition_error_abs_max", 0.0)) > 1e-5:
        raise RuntimeError("C3C reward decomposition failed during training.")
    replay_ready = float(row.get("c3c_replay_size", 0.0)) >= int(
        replay_batch_size
    )
    for name in ("high", "low", "c3c"):
        updates = float(row.get(f"{name}_updates", 0.0))
        delta = abs(float(row.get(f"{name}_parameter_checksum_delta", 0.0)))
        if replay_ready and updates <= 0.0:
            raise RuntimeError(f"{name} network did not update although replay is ready.")
        if updates > 0.0 and delta <= 1e-12:
            raise RuntimeError(f"{name} optimizer reported updates but parameters did not change.")
    warnings = []
    if float(row.get("c3c_residual_energy_reward_std", 0.0)) <= 1e-8:
        warnings.append("energy residual has near-zero variance")
    if float(row.get("c3c_residual_energy_unique_ratio", 1.0)) < 0.05:
        warnings.append("energy residual appears excessively discrete")
    if float(row.get("c3c_residual_deadline_zero_rate", 0.0)) > 0.99:
        warnings.append("Deadline residual is more than 99% zero")
    if warnings:
        print("[c3c-reward-warning] " + "; ".join(warnings))


def main(default_config="configs/hierarchical_d3qn_sac_c3c.json"):
    args = parse_args(default_config=default_config)
    config = load_json(args.config)
    if args.smoke_test:
        config = _apply_smoke_overrides(config)
    training = config.setdefault("training", {})
    validate_training_config(training)
    seed = int(args.seed if args.seed is not None else training.get("seed", 42))
    total_episodes = int(
        args.episodes
        if args.episodes is not None
        else training.get("total_episodes", 1500)
    )
    resources = build_resources(config, seed)
    agent_config = configure_agent_state(config, resources, smoke_test=args.smoke_test)
    agent_config.setdefault("c3c", {})["profile_name"] = str(
        config.get("objective", {}).get(
            "c3c_reward_profile", "c3c_rho1_residual_v1"
        )
    )
    if args.smoke_test:
        # configure_agent_state runs after the JSON-level overrides.
        agent_config["high"]["batch_size"] = 8
        agent_config["low"]["batch_size"] = 8
        agent_config["c3c"]["batch_size"] = 8
    objective = HierarchicalObjective(config.get("objective", {}))
    if objective.reward_profile not in {
        "ep740_lagrangian_v1",
        "sub_deadline_lagrangian_v1",
    }:
        raise ValueError(
            "The independent C3C collector requires objective.reward_profile "
            "to be 'ep740_lagrangian_v1' or "
            "'sub_deadline_lagrangian_v1'."
        )
    if objective.reward_profile == "sub_deadline_lagrangian_v1" and not bool(
        config.get("simulation", {}).get("sub_deadline", {}).get(
            "enabled", False
        )
    ):
        raise ValueError(
            "sub_deadline_lagrangian_v1 requires "
            "simulation.sub_deadline.enabled=true"
        )
    agent = C3CHierarchicalAgent(agent_config, seed=seed)
    if args.resume_checkpoint:
        agent.load_checkpoint(args.resume_checkpoint)

    runtime = configure_torch_runtime(training.get("torch_cpu_threads", 1))
    print(
        "[device-c3c] "
        f"high={describe_device(agent.high_agent.agent_R.device)} "
        f"low={describe_device(agent.low_agent.device)} "
        f"c3c={describe_device(agent.c3c.device)} "
        "training=joint_three_networks "
        f"reward={agent.reward_profile} "
        f"torch_threads={runtime['torch_cpu_threads']}"
    )

    checkpoint_root = Path(
        training.get("checkpoint_dir", "results/hierarchical_d3qn_sac_c3c")
    )
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root
    run_name = training.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = checkpoint_root / str(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    initialize_history_files(output_dir, "train_history")
    initialize_history_files(output_dir, "eval_history")

    eval_interval = int(training.get("eval_interval", 20))
    eval_episodes = int(training.get("eval_episodes", 20))
    eval_seed = int(training.get("eval_seed", seed + 100000))
    save_every = int(training.get("save_every", 20))
    deadline_target = float(training.get("checkpoint_deadline_target", 0.99))
    include_replay = bool(training.get("save_replay_buffer", False))
    history = []
    eval_history = []
    best_rank = None

    for episode in range(1, total_episodes + 1):
        episode_seed = seed + episode
        workflows = build_workflows(
            config, resources, episode_seed, smoke_test=args.smoke_test
        )
        env = build_env(
            workflows,
            resources,
            dict(config.get("simulation", {})),
            episode_seed,
        )
        result = collect_d3qn_episode(
            env,
            agent,
            objective,
            HighRolloutBuffer(),
            LowRolloutBuffer(),
            training=True,
            high_deterministic=False,
            low_deterministic=False,
            fixed_low_rho_one=False,
            high_algorithm="d3qn",
        )
        reward_metrics = agent.recompose_and_remember_episode(
            result["high_buffer"],
            energy_weight=objective.high_energy_weight,
            store=True,
        )

        before = {
            "high": _parameter_checksum(agent.high_agent.agent_R.q_eval),
            "low": _parameter_checksum(agent.low_agent.actor),
            "c3c": _parameter_checksum(agent.c3c.critic_1),
        }
        # Deliberate joint update order: coordination model first, placement
        # learner second, SAC base learner third, safe C3C actor feedback last.
        c3c_metrics = agent.update_c3c()
        high_metrics = agent.update_high()
        low_metrics = agent.low_agent.update(result["low_buffer"])
        safe_metrics = agent.update_low_safety()
        after = {
            "high": _parameter_checksum(agent.high_agent.agent_R.q_eval),
            "low": _parameter_checksum(agent.low_agent.actor),
            "c3c": _parameter_checksum(agent.c3c.critic_1),
        }
        parameter_deltas = {
            f"{name}_parameter_checksum_delta": after[name] - before[name]
            for name in before
        }

        deadline_usage = np.asarray(
            _completed_workflow_deadline_usage(result["workflow_results"]),
            dtype=np.float64,
        )
        misses = (deadline_usage > 1.0).astype(np.float64)
        # The common collector returns binary workflow misses for the legacy
        # profile and per-workflow summed local costs for the sub-Deadline
        # profile.  Feeding those exact values to both dual controllers keeps
        # C3C and the no-C3C experiment semantically identical.
        low_dual_costs = np.asarray(
            result["low_deadline_dual_costs"], dtype=np.float64
        )
        high_dual_costs = np.asarray(
            result["high_deadline_dual_costs"], dtype=np.float64
        )
        low_dual = agent.low_agent.update_deadline_lagrangian(
            low_dual_costs, episode=episode - 1
        )
        high_dual = agent.high_agent.update_deadline_lagrangian(
            high_dual_costs, episode=episode - 1
        )
        objective.set_deadline_lambda(agent.high_agent.deadline_lambda)
        gate_before = agent.constraint_gate_status()
        gate_after = agent.update_constraint_gate(misses)

        row = _augment_metrics(
            result,
            reward_metrics,
            high_metrics,
            low_metrics,
            c3c_metrics,
            safe_metrics,
            parameter_deltas,
            profile_name=agent.reward_profile,
        )
        row.update(
            {
                "episode": episode,
                "episode_seed": episode_seed,
                "high_lambda": float(agent.high_agent.deadline_lambda),
                "low_lambda": float(agent.low_agent.deadline_lambda),
                "high_dual_updated": float(high_dual.get("dual_updated", 0.0)),
                "low_dual_updated": float(
                    low_dual.get("low_deadline_dual_updated", 0.0)
                ),
                "c3c_constraint_gate_factor_used": float(
                    gate_before["c3c_constraint_gate_factor"]
                ),
                **gate_after,
            }
        )
        _validate_training_diagnostics(
            row, replay_batch_size=agent.c3c.replay.batch_size
        )
        history.append(row)
        append_history_row(output_dir, "train_history", row)
        _print_row("train", episode, row)

        if episode % eval_interval == 0 or episode == total_episodes:
            eval_row = evaluate(
                config,
                resources,
                agent,
                objective,
                seed=eval_seed,
                episodes=eval_episodes,
                smoke_test=args.smoke_test,
            )
            eval_row["episode"] = episode
            eval_history.append(eval_row)
            append_history_row(output_dir, "eval_history", eval_row)
            _print_row("eval", episode, eval_row)
            rank = metric_rank(eval_row, deadline_target=deadline_target)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                agent.save_checkpoint(
                    output_dir / "best_c3c_checkpoint.pt",
                    objective_state=objective.state_dict(),
                    episode=episode,
                    config=config,
                    include_replay=False,
                )

        if episode % save_every == 0 or episode == total_episodes:
            agent.save_checkpoint(
                output_dir / "last_c3c_checkpoint.pt",
                objective_state=objective.state_dict(),
                episode=episode,
                config=config,
                include_replay=include_replay,
            )

    write_history_files(output_dir, "train_history", history)
    write_history_files(output_dir, "eval_history", eval_history)
    print(f"output_dir={output_dir}")
    print(f"final_checkpoint={output_dir / 'last_c3c_checkpoint.pt'}")


if __name__ == "__main__":
    main()
