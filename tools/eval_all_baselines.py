from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PY310_SITE = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_PY310_SITE.is_dir() and str(LOCAL_PY310_SITE) not in sys.path:
    sys.path.insert(0, str(LOCAL_PY310_SITE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.D3QN.hierarchical_d3qn_agent import HierarchicalD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.baselines.oo_actor_critic import OOActorCriticAgent
from src.baselines.dream import DReaMAgent, DReaMScheduler
from src.baselines.dream.dream_action import action_dimension
from src.baselines.dream.dream_config import DReaMConfig
from src.baselines.dream.dream_reward import SharedWorkflowReward
from src.baselines.dream.dream_state import SharedEdgeCloudStateBuilder
from src.baselines.dqn_edge import DQNEdgeConfig
from src.ppo.hierarchical_state import (
    HierarchicalStateEncoder as LegacyHierarchicalStateEncoder,
)
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from src.shared.hierarchical_state_reward import build_shared_state_encoder
from src.simulator.deadline import DEADLINE_PROBABILITIES
from tools.hierarchical_common import (
    DEADLINE_TYPES,
    apply_deadline_type,
    build_env,
    build_resources,
    build_workflows,
    config_deadline_type,
    configure_agent_state,
    load_json,
    satisfaction_statistics,
)
from tools.train_hierarchical_d3qn import collect_d3qn_episode
from tools.train_oo_actor_critic import collect_oo_episode
from tools.train_dream import run_episode as run_dream_episode
from tools.train_dqn_edge import (
    build_dqn_edge_runtime,
    run_episode as run_dqn_edge_episode,
)
from tools.evaluation_seed_schedule import resolve_seed_schedule


DEFAULT_D3QN_CHECKPOINT = (
    "results/hierarchical_d3qn_sac/20260806_111214/best_d3qn_checkpoint.pt"
)
DEFAULT_OO_CONFIG = "results/oo_actor_critic/20260806_081046/resolved_config.json"
DEFAULT_OO_CHECKPOINT = (
    "results/oo_actor_critic/20260806_081046/best_oo_actor_critic_checkpoint.pt"
)
DEFAULT_DREAM_CONFIG = "configs/dream_dqn.json"
DEFAULT_DREAM_CHECKPOINT = "results/dream_dqn/best_checkpoint.pt"
DEFAULT_DQN_EDGE_CONFIG = "configs/dqn_edge_small_20edge_6cloud.json"
DEFAULT_DQN_EDGE_CHECKPOINT = "results/dqn_edge/best_checkpoint.pt"
DEFAULT_MODES = (
    "d3qn_sac",
    "oo_actor_critic",
    "dream_dqn",
    "dqn_edge",
    "heft_sac",
    "fcfs_sac",
    "fastest_sac",
    "random_sac",
)
SUPPORTED_MODES = (*DEFAULT_MODES, "heft_rho1")

DATASET_SCALE_NAMES = {
    "small": "small",
    "medium": "middle",
    "large": "large",
}
MODE_TO_HIGH_ALGORITHM = {
    "d3qn_sac": "d3qn",
    "heft_sac": "heft",
    "heft_rho1": "heft",
    "fcfs_sac": "fcfs",
    "fastest_sac": "fastest",
    "random_sac": "random",
}
REPORT_METRICS = (
    "total_energy",
    "energy_per_workflow",
    "energy_per_task",
    "average_response_time",
    "episode_makespan",
    "deadline_success_rate",
    "reliability_success_rate",
    "avg_workflow_reliability",
    "rho_mean",
    "low_sac_total_energy_saving_ratio",
    "steps",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate D3QN+SAC, OO actor-critic, HEFT+SAC, FCFS+SAC, "
            "Fastest+SAC and Random+SAC on paired Pegasus workflow seeds."
        )
    )
    parser.add_argument("--config", default="configs/v2_small_20edge_6cloud_dvfs_v3_future_esp.json")
    parser.add_argument("--d3qn-checkpoint", default=DEFAULT_D3QN_CHECKPOINT)
    parser.add_argument("--oo-config", default=DEFAULT_OO_CONFIG)
    parser.add_argument("--oo-checkpoint", default=DEFAULT_OO_CHECKPOINT)
    parser.add_argument("--dream-config", default=DEFAULT_DREAM_CONFIG)
    parser.add_argument("--dream-checkpoint", default=DEFAULT_DREAM_CHECKPOINT)
    parser.add_argument("--dqn-edge-config", default=DEFAULT_DQN_EDGE_CONFIG)
    parser.add_argument("--dqn-edge-checkpoint", default=DEFAULT_DQN_EDGE_CHECKPOINT)
    parser.add_argument("--scales", default="small,medium")
    parser.add_argument(
        "--deadline-types",
        default=None,
        help=(
            "Comma-separated Deadline regimes to evaluate every mode under, "
            "e.g. 'tight,medium,loose' (P(2x deadline) = 0.8/0.5/0.2). "
            "Defaults to the config value only. Every regime reuses the same "
            "checkpoints and the same seeds, and for a given seed the arrival "
            "times and task graphs are identical, so the regimes form a paired "
            "generalization test of one unified model."
        ),
    )
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--seed-start", type=int, default=342000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed-schedule", default=None,
                        help="JSON containing ordered, unique real seeds for each scale/deadline.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent seed processes (one D3QN+SAC or DReaM mode).")
    parser.add_argument(
        "--ready-task-order",
        choices=("upward_rank", "smallest_workload"),
        default=None,
        help="Override the common environment ready-task dispatch order.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_csv_values(value, allowed, label):
    result = []
    for raw in str(value).split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported {label} {item!r}; expected one of {sorted(allowed)}.")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError(f"At least one {label} is required.")
    return result


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    t_critical = 2.093 if count == 20 else 1.96
    ci95 = float(t_critical * std / math.sqrt(count)) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_half_width": ci95,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def load_d3qn(checkpoint_path, seed):
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    checkpoint_format = payload.get("checkpoint_format")
    if checkpoint_format not in {
        "hierarchical_d3qn_primary_v3",
        "hierarchical_reliability_aware_d3qn_v1",
    }:
        raise RuntimeError(
            "D3QN comparison requires a primary-only or reliability-aware "
            f"D3QN checkpoint; got {checkpoint_format!r}."
        )
    checkpoint_config = copy.deepcopy(payload.get("config") or {})
    agent_cfg = copy.deepcopy(checkpoint_config.get("agent") or {})
    agent_cfg["use_low_agent"] = True
    if checkpoint_format == "hierarchical_reliability_aware_d3qn_v1":
        from tools.train_reliability_aware_hierarchical_d3qn import (
            ReliabilityAwareHierarchicalD3QNAgent,
        )

        agent = ReliabilityAwareHierarchicalD3QNAgent(agent_cfg, seed=seed)
    else:
        agent = HierarchicalD3QNAgent(agent_cfg, seed=seed)
    agent.load_checkpoint(checkpoint_path)
    agent.high_agent.epsilon = 0.0
    objective = HierarchicalObjective(checkpoint_config.get("objective", {}))
    return payload, checkpoint_config, agent, objective


def load_oo(config, checkpoint_path, resources, seed):
    checkpoint_payload = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    checkpoint_config = copy.deepcopy(checkpoint_payload.get("config") or config)
    configured_agent = configure_agent_state(
        checkpoint_config, resources, smoke_test=False
    )
    shared = dict(configured_agent.get("shared", {}))
    oo_cfg = {**shared, **dict(configured_agent.get("oo", {}))}
    state_cfg = dict(shared.get("state", {}))
    checkpoint_state_dim = int(checkpoint_payload.get("state_dim", 0))
    encoder = build_shared_state_encoder(state_cfg)
    encoder_name = "shared_current_high_state"
    if encoder.high_state_dim != checkpoint_state_dim:
        legacy_encoder = LegacyHierarchicalStateEncoder(state_cfg)
        if legacy_encoder.high_state_dim != checkpoint_state_dim:
            raise RuntimeError(
                "OO checkpoint state dimension is incompatible with both current and "
                f"legacy encoders: checkpoint={checkpoint_state_dim}, "
                f"current={encoder.high_state_dim}, legacy={legacy_encoder.high_state_dim}."
            )
        encoder = legacy_encoder
        encoder_name = "legacy_3x21_state"
    agent = OOActorCriticAgent(
        state_dim=encoder.high_state_dim,
        num_servers=len(resources.get("servers", [])),
        rho_levels=oo_cfg.get("rho_levels", (0.2, 0.4, 0.6, 0.8, 1.0)),
        config=oo_cfg,
        seed=seed,
    )
    payload = agent.load_checkpoint(checkpoint_path, load_replay=False)
    objective = HierarchicalObjective(checkpoint_config.get("objective", {}))
    deadline_weight = float(
        oo_cfg.get(
            "deadline_cost_reward_weight",
            0.0 if objective.reward_mode == "deadline_incremental_only" else 1.0,
        )
    )
    return payload, encoder, agent, objective, deadline_weight, encoder_name


def add_normalized_metrics(metrics, num_workflows):
    result = dict(metrics)
    energy = float(result.get("total_energy", 0.0) or 0.0)
    result["energy_per_workflow"] = energy / max(1, int(num_workflows))
    result["energy_per_task"] = energy / max(1.0, float(result.get("steps", 0.0)))
    return result


def run_d3qn_family(
    config, resources, agent, objective, seed, high_algorithm,
    fixed_low_rho_one=False,
):
    workflows = build_workflows(config, resources, seed, smoke_test=False)
    env = build_env(workflows, resources, dict(config.get("simulation", {})), seed)
    result = collect_d3qn_episode(
        env,
        agent,
        objective,
        HighRolloutBuffer(),
        LowRolloutBuffer(),
        training=False,
        high_deterministic=True,
        low_deterministic=True,
        fixed_low_rho_one=fixed_low_rho_one,
        high_algorithm=high_algorithm,
    )
    return result["metrics"]


def _initialize_d3qn_evaluation(checkpoint, config, resources, seed):
    global _d3qn_evaluation_runtime
    from src.device_utils import configure_torch_runtime
    configure_torch_runtime(1)
    _, _, agent, objective = load_d3qn(checkpoint, seed)
    _d3qn_evaluation_runtime = (config, resources, agent, objective)


def _evaluate_d3qn_seed(seed):
    config, resources, agent, objective = _d3qn_evaluation_runtime
    with redirect_stdout(io.StringIO()):
        return run_d3qn_family(config, resources, agent, objective, seed, "d3qn")


def parallel_d3qn_results(checkpoint, config, resources, seeds, workers, model_seed):
    seeds = list(seeds)
    if not seeds:
        return
    with ProcessPoolExecutor(
            max_workers=min(workers, len(seeds)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_d3qn_evaluation,
            initargs=(checkpoint, config, resources, model_seed)) as pool:
        yield from zip(seeds, pool.map(_evaluate_d3qn_seed, seeds))


def run_oo(config, resources, encoder, agent, objective, deadline_weight, seed):
    workflows = build_workflows(config, resources, seed, smoke_test=False)
    env = build_env(workflows, resources, dict(config.get("simulation", {})), seed)
    result = collect_oo_episode(
        env,
        encoder,
        agent,
        objective,
        deterministic=True,
        deadline_cost_reward_weight=deadline_weight,
    )
    return result["metrics"]


def load_dream(config, checkpoint_path, resources, seed):
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_cfg = dict(config.get("agent", {}).get("shared", {}).get("state", {}))
    state_cfg["max_servers"] = len(resources.get("servers", []))
    builder = SharedEdgeCloudStateBuilder(state_cfg)
    values = dict(config.get("agent", {}).get("dream", {}))
    values.update(seed=seed, device="cuda" if torch.cuda.is_available() else "cpu")
    agent = DReaMAgent(
        builder.state_dim, action_dimension(len(resources["servers"]), False),
        DReaMConfig.from_mapping(values),
    )
    agent.load_checkpoint(payload)
    objective = HierarchicalObjective(
        config.get("objective", {"reward_mode": "deadline_incremental_only"})
    )
    return payload, DReaMScheduler(
        agent, builder, SharedWorkflowReward(objective), False
    )


def run_dream(config, resources, scheduler, seed):
    workflows = build_workflows(config, resources, seed, smoke_test=False)
    env = build_env(
        workflows, resources,
        {**config.get("simulation", {}), "enable_backup_duplication": False}, seed,
    )
    result = run_dream_episode(env, scheduler, training=False)
    return {
        "total_energy": result["total_energy"],
        "average_response_time": result["avg_workflow_response_time"],
        "episode_makespan": result["episode_makespan"],
        "deadline_success_rate": result["deadline_success_rate"],
        "reliability_success_rate": result["reliability_success_rate"],
        "avg_workflow_reliability": result["avg_workflow_reliability"],
        "rho_mean": result["rho_mean"],
        "steps": result["steps"],
    }


def _initialize_dream_evaluation(checkpoint, training_config, config, resources, seed):
    global _dream_evaluation_runtime
    from src.device_utils import configure_torch_runtime
    configure_torch_runtime(1)
    _, scheduler = load_dream(training_config, checkpoint, resources, seed)
    _dream_evaluation_runtime = (config, resources, scheduler)


def _evaluate_dream_seed(seed):
    config, resources, scheduler = _dream_evaluation_runtime
    with redirect_stdout(io.StringIO()):
        return run_dream(config, resources, scheduler, seed)


def parallel_dream_results(checkpoint, training_config, config, resources, seeds, workers, model_seed):
    seeds = list(seeds)
    if not seeds:
        return
    with ProcessPoolExecutor(
            max_workers=min(workers, len(seeds)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_dream_evaluation,
            initargs=(checkpoint, training_config, config, resources, model_seed)) as pool:
        yield from zip(seeds, pool.map(_evaluate_dream_seed, seeds))


def load_dqn_edge(config, checkpoint_path, resources, seed):
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    values = dict(payload.get("config", config.get("agent", {}).get("dqn_edge", {})))
    values.update(seed=seed, device="cuda" if torch.cuda.is_available() else "cpu")
    agent_config = DQNEdgeConfig.from_mapping(values)
    paper_state_dim = 3 + 2 * len(resources["servers"])
    variant = payload.get("variant")
    if variant is None:
        variant = (
            "paper"
            if int(payload.get("state_dim", paper_state_dim)) == paper_state_dim
            else config.get("agent", {}).get("variant", "common_reward")
        )
    runtime = build_dqn_edge_runtime(
        config,
        resources,
        agent_config,
        variant=variant,
        state_config=payload.get("state_config"),
        objective_config=payload.get("objective_config"),
    )
    runtime["agent"].load_checkpoint(payload)
    return payload, runtime


def run_dqn_edge(config, resources, runtime, seed):
    workflows = build_workflows(config, resources, seed, smoke_test=False)
    env = build_env(
        workflows,
        resources,
        {
            **config.get("simulation", {}),
            "enable_backup_duplication": False,
            "default_rho": 1.0,
        },
        seed,
    )
    result = run_dqn_edge_episode(env, runtime["scheduler"], training=False)
    return {
        "total_energy": result["total_energy"],
        "average_response_time": result["avg_workflow_response_time"],
        "episode_makespan": result["episode_makespan"],
        "deadline_success_rate": result["deadline_success_rate"],
        "reliability_success_rate": result["reliability_success_rate"],
        "avg_workflow_reliability": result["avg_workflow_reliability"],
        "rho_mean": 1.0,
        "steps": result["steps"],
        "dqn_edge_time_weight": result.get("mean_time_weight"),
        "dqn_edge_energy_weight": result.get("mean_energy_weight"),
    }


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scale"], row["deadline_type"], row["mode"]), []).append(row)
    summaries = []
    for (scale, deadline_type, mode), group in sorted(grouped.items()):
        metrics = {}
        for metric in REPORT_METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            if values:
                metrics[metric] = describe(values)
        summaries.append(
            {
                "scale": scale,
                "deadline_type": deadline_type,
                "mode": mode,
                "num_workflows": int(group[0]["num_workflows"]),
                "seed_start": int(min(row["eval_seed"] for row in group)),
                "seed_end": int(max(row["eval_seed"] for row in group)),
                "metrics": metrics,
                # ESP over the per-seed success rates the mean above hides.
                "deadline_satisfaction": satisfaction_statistics(
                    row["deadline_success_rate"] for row in group
                ),
                "reliability_satisfaction": satisfaction_statistics(
                    row["reliability_success_rate"] for row in group
                ),
                "joint_esp_0p90": float(
                    np.mean(
                        [
                            float(row["deadline_success_rate"]) >= 0.90 - 1e-9
                            and float(row["reliability_success_rate"]) >= 1.0 - 1e-9
                            for row in group
                        ]
                    )
                ),
            }
        )
    return summaries


def paired_energy_deltas(rows):
    lookup = {
        (row["scale"], row["deadline_type"], row["mode"], row["eval_seed"]): row
        for row in rows
    }
    result = {}
    keys = sorted({(row["scale"], row["deadline_type"]) for row in rows})
    modes = sorted({row["mode"] for row in rows})
    for scale, deadline_type in keys:
        seeds = sorted({
            row["eval_seed"] for row in rows
            if row["scale"] == scale and row["deadline_type"] == deadline_type
        })
        for mode in modes:
            deltas = []
            for seed in seeds:
                ours = lookup.get((scale, deadline_type, "d3qn_sac", seed))
                other = lookup.get((scale, deadline_type, mode, seed))
                if ours is None or other is None:
                    continue
                denominator = max(float(ours["total_energy"]), 1e-8)
                deltas.append(100.0 * (float(other["total_energy"]) - denominator) / denominator)
            if deltas:
                result[(scale, deadline_type, mode)] = describe(deltas)
    return result


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_rows(output_dir, rows):
    rows_path = output_dir / "comparison_rows.jsonl"
    temporary_path = rows_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
    temporary_path.replace(rows_path)
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: value
                for key, value in row.items()
                if value is None or isinstance(value, (str, int, float, bool))
            }
        )
    fields = sorted({key for row in scalar_rows for key in row})
    with (output_dir / "comparison_rows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def write_summary(output_dir, summaries, paired_deltas):
    payload = []
    for row in summaries:
        result = copy.deepcopy(row)
        result["paired_energy_delta_vs_d3qn_percent"] = paired_deltas.get(
            (row["scale"], row["deadline_type"], row["mode"])
        )
        payload.append(result)
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, ensure_ascii=False, indent=2)

    csv_rows = []
    for row in payload:
        base = {
            "scale": row["scale"],
            "deadline_type": row["deadline_type"],
            "mode": row["mode"],
            "num_workflows": row["num_workflows"],
            "seed_start": row["seed_start"],
            "seed_end": row["seed_end"],
        }
        for metric, stats in row["metrics"].items():
            csv_rows.append({**base, "metric": metric, **stats})
        satisfaction = row.get("deadline_satisfaction") or {}
        for name, value in sorted(satisfaction.get("esp", {}).items()):
            csv_rows.append({**base, "metric": f"deadline_esp@{name}",
                             "n": satisfaction["n"], "mean": value})
        if satisfaction:
            csv_rows.append({**base, "metric": "deadline_cvar25",
                             "n": satisfaction["n"], "mean": satisfaction["cvar"],
                             "min": satisfaction["min"]})
        reliability = row.get("reliability_satisfaction") or {}
        if reliability:
            csv_rows.append({**base, "metric": "reliability_esp@1.00",
                             "n": reliability["n"],
                             "mean": reliability.get("esp", {}).get("1.00")})
        csv_rows.append({**base, "metric": "joint_esp@0.90",
                         "n": satisfaction.get("n"),
                         "mean": row.get("joint_esp_0p90")})
    fields = [
        "scale",
        "deadline_type",
        "mode",
        "num_workflows",
        "seed_start",
        "seed_end",
        "metric",
        "n",
        "mean",
        "std",
        "ci95_half_width",
        "min",
        "max",
    ]
    with (output_dir / "comparison_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    labels = {
        "d3qn_sac": "D3QN+SAC (ours)",
        "oo_actor_critic": "OO Actor-Critic",
        "dream_dqn": "Adapted DReaM-DQN",
        "dqn_edge": "Adapted DQN-Edge",
        "heft_sac": "HEFT+SAC",
        "heft_rho1": r"HEFT+$\rho$=1",
        "fcfs_sac": "FCFS+SAC",
        "fastest_sac": "Fastest+SAC",
        "random_sac": "Random+SAC",
    }
    lines = [
        "# Baseline comparison across evaluation seeds",
        "",
        "Energy is the complete-system total, averaged over evaluation seeds.",
        "Positive delta means more energy than D3QN+SAC.",
        "`Deadline` is the mean per-run success rate; `ESP@g` is the fraction of",
        "runs whose own success rate reaches g; `CVaR25` is the mean of the worst",
        "quarter of runs. A high mean with a low CVaR25 is an unstable method.",
        "",
    ]
    for scale, deadline_type in sorted(
        {(row["scale"], row["deadline_type"]) for row in payload}
    ):
        probability = DEADLINE_PROBABILITIES[deadline_type]
        lines.extend(
            [
                f"## {scale} / deadline={deadline_type} "
                f"(P(2x)={probability:.1f}, P(3x)={1.0 - probability:.1f})",
                "",
                "| Method | Energy (kJ) | Delta vs ours | Deadline | ESP@0.90 "
                "| CVaR25 | min | Reliability | Rel-ESP@1.00 | Joint@0.90 "
                "| Response (s) | rho |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        scale_rows = [
            row for row in payload
            if row["scale"] == scale and row["deadline_type"] == deadline_type
        ]
        scale_rows.sort(key=lambda row: row["metrics"]["total_energy"]["mean"])
        for row in scale_rows:
            metrics = row["metrics"]
            delta = row.get("paired_energy_delta_vs_d3qn_percent")
            delta_text = "--" if delta is None else f"{delta['mean']:+.2f}%"
            rho = metrics.get("rho_mean", {}).get("mean")
            rho_text = "--" if rho is None else f"{rho:.3f}"
            satisfaction = row.get("deadline_satisfaction") or {}
            reliability = row.get("reliability_satisfaction") or {}
            esp = satisfaction.get("esp", {})
            lines.append(
                f"| {labels.get(row['mode'], row['mode'])} "
                f"| {metrics['total_energy']['mean'] / 1000.0:.2f} "
                f"| {delta_text} "
                f"| {metrics['deadline_success_rate']['mean']:.2%} "
                f"| {esp.get('0.90', float('nan')):.2f} "
                f"| {satisfaction.get('cvar', float('nan')):.3f} "
                f"| {satisfaction.get('min', float('nan')):.3f} "
                f"| {metrics['reliability_success_rate']['mean']:.2%} "
                f"| {reliability.get('esp', {}).get('1.00', float('nan')):.2f} "
                f"| {row.get('joint_esp_0p90', float('nan')):.2f} "
                f"| {metrics['average_response_time']['mean']:.2f} "
                f"| {rho_text} |"
            )
        lines.append("")
    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args):
    config = load_json(args.config)
    if args.ready_task_order is not None:
        config.setdefault("simulation", {})["ready_task_order"] = args.ready_task_order
    scales = parse_csv_values(args.scales, {"small", "medium", "large"}, "scale")
    modes = parse_csv_values(args.modes, set(SUPPORTED_MODES), "mode")
    workers = int(getattr(args, "workers", 1))
    if workers < 1 or (workers > 1 and modes not in (["d3qn_sac"], ["dream_dqn"])):
        raise ValueError("workers must be positive; parallel evaluation requires one d3qn_sac or dream_dqn mode")
    oo_config = load_json(args.oo_config) if "oo_actor_critic" in modes else {}
    dream_config = load_json(args.dream_config) if "dream_dqn" in modes else {}
    dqn_edge_config = load_json(args.dqn_edge_config) if "dqn_edge" in modes else {}
    if args.deadline_types is None:
        deadline_types = [config_deadline_type(config)]
    else:
        deadline_types = parse_csv_values(
            args.deadline_types, set(DEADLINE_TYPES), "deadline type"
        )
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.smoke_test:
        args.num_seeds = 1

    schedule_path = getattr(args, "seed_schedule", None)
    seed_schedule = resolve_seed_schedule(
        resolve_path(schedule_path) if schedule_path else None,
        scales, deadline_types, int(args.seed_start), int(args.num_seeds))
    if schedule_path and args.output_dir:
        previous_rows = resolve_path(args.output_dir) / "comparison_rows.jsonl"
        if previous_rows.exists() and previous_rows.stat().st_size:
            raise ValueError("Explicit-seed evaluation must start fresh; sequential policy state is not restored")

    server_seed = int(config.get("training", {}).get("seed", 42))
    with redirect_stdout(io.StringIO()):
        resources = build_resources(config, server_seed)
    d3qn_payload = d3qn_config = d3qn_agent = d3qn_objective = None
    needs_d3qn_family = any(mode in MODE_TO_HIGH_ALGORITHM for mode in modes)
    if needs_d3qn_family:
        d3qn_payload, d3qn_config, d3qn_agent, d3qn_objective = load_d3qn(
            resolve_path(args.d3qn_checkpoint), args.seed_start
        )
        checkpoint_simulation = dict(d3qn_config.get("simulation", {}))
        if "sub_deadline" in checkpoint_simulation:
            config.setdefault("simulation", {})["sub_deadline"] = copy.deepcopy(
                checkpoint_simulation["sub_deadline"]
            )
    oo_payload = None
    oo_encoder = oo_agent = oo_objective = oo_deadline_weight = oo_encoder_name = None
    if "oo_actor_critic" in modes:
        (
            oo_payload,
            oo_encoder,
            oo_agent,
            oo_objective,
            oo_deadline_weight,
            oo_encoder_name,
        ) = load_oo(oo_config, resolve_path(args.oo_checkpoint), resources, args.seed_start)
    dream_payload = dream_scheduler = None
    if "dream_dqn" in modes:
        dream_payload, dream_scheduler = load_dream(
            dream_config, resolve_path(args.dream_checkpoint), resources, args.seed_start
        )
    dqn_edge_payload = dqn_edge_runtime = None
    if "dqn_edge" in modes:
        dqn_edge_payload, dqn_edge_runtime = load_dqn_edge(
            dqn_edge_config,
            resolve_path(args.dqn_edge_checkpoint),
            resources,
            args.seed_start,
        )
    if args.output_dir:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = (
            PROJECT_ROOT
            / "results"
            / "all_baseline_comparison"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(resolve_path(args.config)),
        "d3qn_checkpoint": str(resolve_path(args.d3qn_checkpoint)),
        "d3qn_checkpoint_episode": int((d3qn_payload or {}).get("episode", 0)),
        "d3qn_checkpoint_reward_config": (d3qn_config or {}).get("objective", {}),
        "oo_config": str(resolve_path(args.oo_config)),
        "oo_checkpoint": str(resolve_path(args.oo_checkpoint)),
        "oo_checkpoint_episode": int((oo_payload or {}).get("episode", 0)),
        "oo_state_encoder": oo_encoder_name,
        "dream_config": str(resolve_path(args.dream_config)),
        "dream_checkpoint": str(resolve_path(args.dream_checkpoint)),
        "dream_checkpoint_episode": int((dream_payload or {}).get("episode", 0)),
        "dqn_edge_config": str(resolve_path(args.dqn_edge_config)),
        "dqn_edge_checkpoint": str(resolve_path(args.dqn_edge_checkpoint)),
        "dqn_edge_checkpoint_episode": int((dqn_edge_payload or {}).get("episode", 0)),
        "dqn_edge_variant": (dqn_edge_runtime or {}).get("variant"),
        "server_seed": server_seed,
        "workflow_seed_start": int(args.seed_start),
        "workflow_seed_end": int(args.seed_start + args.num_seeds - 1),
        "num_seeds_per_scale": int(args.num_seeds),
        "scales": scales,
        "deadline_types": deadline_types,
        "config_deadline_type": config_deadline_type(config),
        "modes": modes,
        "workers": workers,
        "num_workflows": int(config.get("generator", {}).get("num_workflows", 20)),
        "ready_task_order": str(
            config.get("simulation", {}).get("ready_task_order", "smallest_workload")
        ),
        "heuristic_dvfs": "same deterministic SAC loaded from the D3QN checkpoint",
    }
    if schedule_path:
        manifest.update(evaluation_seed_schedule=seed_schedule,
                        evaluation_seed_schedule_sha256=hashlib.sha256(
                            resolve_path(schedule_path).read_bytes()).hexdigest(),
                        model_initialization_seed=int(args.seed_start))
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    rows_path = output_dir / "comparison_rows.jsonl"
    if rows_path.exists():
        seed_end = int(args.seed_start + args.num_seeds)
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Rows written before the regime sweep existed carry no
            # deadline_type; they were produced under the config's own regime.
            row.setdefault("deadline_type", config_deadline_type(config))
            if (
                row.get("scale") in scales
                and row.get("mode") in modes
                and row.get("deadline_type") in deadline_types
                and int(row.get("eval_seed", -1)) in seed_schedule[row["scale"]][row["deadline_type"]]
            ):
                rows.append(row)
        if rows:
            print(f"[resume] loaded={len(rows)} completed evaluations")
    completed = {
        (str(row["scale"]), str(row["deadline_type"]), str(row["mode"]),
         int(row["eval_seed"]))
        for row in rows
    }
    num_workflows = int(config.get("generator", {}).get("num_workflows", 20))
    for scale in scales:
        base_scenario = copy.deepcopy(config)
        base_scenario.setdefault("generator", {})["dataset_scale"] = DATASET_SCALE_NAMES[scale]
        base_scenario.setdefault("generator", {})["num_workflows"] = num_workflows
        for deadline_type in deadline_types:
            # Only simulation.deadline_type differs between regimes, so a given
            # seed yields the same arrivals and task graphs in all of them.
            scenario = apply_deadline_type(base_scenario, deadline_type)
            print(
                f"[scale] {scale} deadline={deadline_type} workflows={num_workflows} "
                f"seeds={args.seed_start}-{args.seed_start + args.num_seeds - 1}"
            )
            seeds = seed_schedule[scale][deadline_type]
            if schedule_path:
                print(f"[seed-schedule] scale={scale} deadline={deadline_type} ordered_seeds={seeds}", flush=True)
            if workers > 1:
                parallel_mode = modes[0]
                pending = [seed for seed in seeds
                           if (scale, deadline_type, parallel_mode, seed) not in completed]
                print(f"[eval-parallel] workers={min(workers, len(pending))} "
                      f"pending={len(pending)} frozen_model=True", flush=True)
                if parallel_mode == "dream_dqn":
                    results = parallel_dream_results(
                        resolve_path(args.dream_checkpoint), dream_config, scenario,
                        resources, pending, workers, args.seed_start)
                else:
                    results = parallel_d3qn_results(
                        resolve_path(args.d3qn_checkpoint), scenario, resources,
                        pending, workers, args.seed_start)
            else:
                results = ((seed, None) for seed in seeds)
            for eval_seed, parallel_metrics in results:
                for mode in modes:
                    if (scale, deadline_type, mode, eval_seed) in completed:
                        continue
                    with redirect_stdout(io.StringIO()):
                        if parallel_metrics is not None:
                            metrics = parallel_metrics
                        elif mode == "oo_actor_critic":
                            metrics = run_oo(
                                scenario,
                                resources,
                                oo_encoder,
                                oo_agent,
                                oo_objective,
                                oo_deadline_weight,
                                eval_seed,
                            )
                        elif mode == "dream_dqn":
                            metrics = run_dream(
                                scenario, resources, dream_scheduler, eval_seed
                            )
                        elif mode == "dqn_edge":
                            metrics = run_dqn_edge(
                                scenario, resources, dqn_edge_runtime, eval_seed
                            )
                        else:
                            metrics = run_d3qn_family(
                                scenario,
                                resources,
                                d3qn_agent,
                                d3qn_objective,
                                eval_seed,
                                MODE_TO_HIGH_ALGORITHM[mode],
                                fixed_low_rho_one=(mode == "heft_rho1"),
                            )
                    metrics = add_normalized_metrics(metrics, num_workflows)
                    row = {
                        "scale": scale,
                        "deadline_type": deadline_type,
                        "mode": mode,
                        "eval_seed": eval_seed,
                        "num_workflows": num_workflows,
                        **metrics,
                    }
                    rows.append(row)
                    write_rows(output_dir, rows)
                    print(
                        f"[eval] scale={scale:<6} deadline={deadline_type:<6} "
                        f"seed={eval_seed} mode={mode:<16} "
                        f"energy_kJ={float(metrics['total_energy']) / 1000.0:8.2f} "
                        f"deadline={float(metrics['deadline_success_rate']):6.2%} "
                        f"reliability={float(metrics['reliability_success_rate']):6.2%}"
                    )

    summaries = summarize_rows(rows)
    paired_deltas = paired_energy_deltas(rows)
    write_summary(output_dir, summaries, paired_deltas)
    print(f"output_dir={output_dir}")
    return output_dir, rows, summaries


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
