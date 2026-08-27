from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import sys
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
from src.baselines.hirl import HIRLAgent, HIRLConfig, HIRLScheduler
from src.ppo.hierarchical_state import (
    HierarchicalStateEncoder as LegacyHierarchicalStateEncoder,
)
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from src.shared.hierarchical_state_reward import build_shared_state_encoder
from tools.hierarchical_common import (
    build_env,
    build_resources,
    build_workflows,
    configure_agent_state,
    load_json,
)
from tools.train_hierarchical_d3qn import collect_d3qn_episode
from tools.train_oo_actor_critic import collect_oo_episode
from tools.train_dream import run_episode as run_dream_episode
from tools.train_dqn_edge import (
    build_dqn_edge_runtime,
    run_episode as run_dqn_edge_episode,
)
from tools.train_hirl import run_episode as run_hirl_episode


DEFAULT_D3QN_CHECKPOINT = (
    "results/hierarchical_d3qn_sac/20260806_111214/best_d3qn_checkpoint.pt"
)
DEFAULT_OO_CONFIG = "results/oo_actor_critic/20260806_081046/resolved_config.json"
DEFAULT_OO_CHECKPOINT = (
    "results/oo_actor_critic/20260806_081046/best_oo_actor_critic_checkpoint.pt"
)
DEFAULT_DREAM_CONFIG = "configs/dream_dqn.json"
DEFAULT_DREAM_CHECKPOINT = "results/dream_dqn/best_checkpoint.pt"
DEFAULT_DQN_EDGE_CONFIG = "configs/dqn_edge.json"
DEFAULT_DQN_EDGE_CHECKPOINT = "results/dqn_edge/best_checkpoint.pt"
DEFAULT_HIRL_CONFIG = "configs/hirl_td3_ddqn.json"
DEFAULT_HIRL_CHECKPOINT = "results/hirl_td3_ddqn/best_checkpoint.pt"
DEFAULT_MODES = (
    "d3qn_sac",
    "oo_actor_critic",
    "dream_dqn",
    "dqn_edge",
    "hirl_td3_ddqn",
    "heft_sac",
    "fcfs_sac",
    "fastest_sac",
    "random_sac",
)

DATASET_SCALE_NAMES = {
    "small": "small",
    "medium": "middle",
    "large": "large",
}
MODE_TO_HIGH_ALGORITHM = {
    "d3qn_sac": "d3qn",
    "heft_sac": "heft",
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
    parser.add_argument("--config", default="configs/v2_small_20edge_6cloud.json")
    parser.add_argument("--d3qn-checkpoint", default=DEFAULT_D3QN_CHECKPOINT)
    parser.add_argument("--oo-config", default=DEFAULT_OO_CONFIG)
    parser.add_argument("--oo-checkpoint", default=DEFAULT_OO_CHECKPOINT)
    parser.add_argument("--dream-config", default=DEFAULT_DREAM_CONFIG)
    parser.add_argument("--dream-checkpoint", default=DEFAULT_DREAM_CHECKPOINT)
    parser.add_argument("--dqn-edge-config", default=DEFAULT_DQN_EDGE_CONFIG)
    parser.add_argument("--dqn-edge-checkpoint", default=DEFAULT_DQN_EDGE_CHECKPOINT)
    parser.add_argument("--hirl-config", default=DEFAULT_HIRL_CONFIG)
    parser.add_argument("--hirl-checkpoint", default=DEFAULT_HIRL_CHECKPOINT)
    parser.add_argument("--scales", default="small,medium")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--seed-start", type=int, default=342000)
    parser.add_argument("--num-seeds", type=int, default=20)
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
    if payload.get("checkpoint_format") != "hierarchical_d3qn_primary_v3":
        raise RuntimeError("D3QN comparison requires a primary-only D3QN v3 checkpoint.")
    checkpoint_config = copy.deepcopy(payload.get("config") or {})
    agent_cfg = copy.deepcopy(checkpoint_config.get("agent") or {})
    agent_cfg["use_low_agent"] = True
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


def run_d3qn_family(config, resources, agent, objective, seed, high_algorithm):
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
        fixed_low_rho_one=False,
        high_algorithm=high_algorithm,
    )
    return result["metrics"]


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


def load_hirl(config, checkpoint_path, resources, seed):
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    checkpoint_config = copy.deepcopy(payload.get("experiment_config") or config)
    state_cfg = dict(checkpoint_config.get("agent", {}).get("shared", {}).get("state", {}))
    state_cfg["max_servers"] = len(resources.get("servers", []))
    builder = SharedEdgeCloudStateBuilder(state_cfg)
    values = dict(checkpoint_config.get("agent", {}).get("hirl", {}))
    values.update(seed=seed, device="cuda" if torch.cuda.is_available() else "cpu")
    agent = HIRLAgent(
        builder.state_dim,
        len(resources["servers"]),
        HIRLConfig.from_mapping(values),
    )
    agent.load_checkpoint(payload, load_replay=False)
    return payload, checkpoint_config, HIRLScheduler(agent, builder)


def run_hirl(config, resources, scheduler, seed):
    workflows = build_workflows(config, resources, seed, smoke_test=False)
    env = build_env(
        workflows, resources,
        {**config.get("simulation", {}), "enable_backup_duplication": False}, seed,
    )
    result = run_hirl_episode(env, scheduler, training=False)
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


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scale"], row["mode"]), []).append(row)
    summaries = []
    for (scale, mode), group in sorted(grouped.items()):
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
                "mode": mode,
                "num_workflows": int(group[0]["num_workflows"]),
                "seed_start": int(min(row["eval_seed"] for row in group)),
                "seed_end": int(max(row["eval_seed"] for row in group)),
                "metrics": metrics,
            }
        )
    return summaries


def paired_energy_deltas(rows):
    lookup = {(row["scale"], row["mode"], row["eval_seed"]): row for row in rows}
    result = {}
    scales = sorted({row["scale"] for row in rows})
    modes = sorted({row["mode"] for row in rows})
    for scale in scales:
        for mode in modes:
            deltas = []
            for seed in sorted({row["eval_seed"] for row in rows if row["scale"] == scale}):
                ours = lookup.get((scale, "d3qn_sac", seed))
                other = lookup.get((scale, mode, seed))
                if ours is None or other is None:
                    continue
                denominator = max(float(ours["total_energy"]), 1e-8)
                deltas.append(100.0 * (float(other["total_energy"]) - denominator) / denominator)
            if deltas:
                result[(scale, mode)] = describe(deltas)
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
    with (output_dir / "comparison_rows.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
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
            (row["scale"], row["mode"])
        )
        payload.append(result)
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, ensure_ascii=False, indent=2)

    csv_rows = []
    for row in payload:
        base = {
            "scale": row["scale"],
            "mode": row["mode"],
            "num_workflows": row["num_workflows"],
            "seed_start": row["seed_start"],
            "seed_end": row["seed_end"],
        }
        for metric, stats in row["metrics"].items():
            csv_rows.append({**base, "metric": metric, **stats})
    fields = [
        "scale",
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
        "hirl_td3_ddqn": "Adapted HIRL (TD3+DDQN)",
        "heft_sac": "HEFT+SAC",
        "fcfs_sac": "FCFS+SAC",
        "fastest_sac": "Fastest+SAC",
        "random_sac": "Random+SAC",
    }
    lines = [
        "# Small/Medium 20-seed comparison",
        "",
        "Energy is the complete-system total, averaged over evaluation seeds.",
        "Positive delta means more energy than D3QN+SAC.",
        "",
    ]
    for scale in sorted({row["scale"] for row in payload}):
        lines.extend(
            [
                f"## {scale}",
                "",
                "| Method | Energy (kJ) | Delta vs ours | Deadline | Reliability | Response (s) | rho |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        scale_rows = [row for row in payload if row["scale"] == scale]
        scale_rows.sort(key=lambda row: row["metrics"]["total_energy"]["mean"])
        for row in scale_rows:
            metrics = row["metrics"]
            delta = row.get("paired_energy_delta_vs_d3qn_percent")
            delta_text = "--" if delta is None else f"{delta['mean']:+.2f}%"
            rho = metrics.get("rho_mean", {}).get("mean")
            rho_text = "--" if rho is None else f"{rho:.3f}"
            lines.append(
                f"| {labels.get(row['mode'], row['mode'])} "
                f"| {metrics['total_energy']['mean'] / 1000.0:.2f} "
                f"| {delta_text} "
                f"| {metrics['deadline_success_rate']['mean']:.2%} "
                f"| {metrics['reliability_success_rate']['mean']:.2%} "
                f"| {metrics['average_response_time']['mean']:.2f} "
                f"| {rho_text} |"
            )
        lines.append("")
    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args):
    config = load_json(args.config)
    if args.ready_task_order is not None:
        config.setdefault("simulation", {})["ready_task_order"] = args.ready_task_order
    oo_config = load_json(args.oo_config)
    dream_config = load_json(args.dream_config)
    dqn_edge_config = load_json(args.dqn_edge_config)
    hirl_config = load_json(args.hirl_config)
    scales = parse_csv_values(args.scales, {"small", "medium", "large"}, "scale")
    modes = parse_csv_values(args.modes, set(DEFAULT_MODES), "mode")
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.smoke_test:
        args.num_seeds = 1

    server_seed = int(config.get("training", {}).get("seed", 42))
    with redirect_stdout(io.StringIO()):
        resources = build_resources(config, server_seed)
    d3qn_payload = d3qn_config = d3qn_agent = d3qn_objective = None
    needs_d3qn_family = any(mode in MODE_TO_HIGH_ALGORITHM for mode in modes)
    if needs_d3qn_family:
        d3qn_payload, d3qn_config, d3qn_agent, d3qn_objective = load_d3qn(
            resolve_path(args.d3qn_checkpoint), args.seed_start
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
    hirl_payload = hirl_checkpoint_config = hirl_scheduler = None
    if "hirl_td3_ddqn" in modes:
        hirl_payload, hirl_checkpoint_config, hirl_scheduler = load_hirl(
            hirl_config, resolve_path(args.hirl_checkpoint), resources, args.seed_start
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
        "hirl_config": str(resolve_path(args.hirl_config)),
        "hirl_checkpoint": str(resolve_path(args.hirl_checkpoint)),
        "hirl_checkpoint_episode": int((hirl_payload or {}).get("episode", 0)),
        "hirl_adaptation": (hirl_checkpoint_config or {}).get("adaptation", {}),
        "server_seed": server_seed,
        "workflow_seed_start": int(args.seed_start),
        "workflow_seed_end": int(args.seed_start + args.num_seeds - 1),
        "num_seeds_per_scale": int(args.num_seeds),
        "scales": scales,
        "modes": modes,
        "num_workflows": int(config.get("generator", {}).get("num_workflows", 20)),
        "ready_task_order": str(
            config.get("simulation", {}).get("ready_task_order", "smallest_workload")
        ),
        "heuristic_dvfs": "same deterministic SAC loaded from the D3QN checkpoint",
    }
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
            if (
                row.get("scale") in scales
                and row.get("mode") in modes
                and int(args.seed_start) <= int(row.get("eval_seed", -1)) < seed_end
            ):
                rows.append(row)
        if rows:
            print(f"[resume] loaded={len(rows)} completed evaluations")
    completed = {
        (str(row["scale"]), str(row["mode"]), int(row["eval_seed"])) for row in rows
    }
    num_workflows = int(config.get("generator", {}).get("num_workflows", 20))
    for scale in scales:
        scenario = copy.deepcopy(config)
        scenario.setdefault("generator", {})["dataset_scale"] = DATASET_SCALE_NAMES[scale]
        scenario.setdefault("generator", {})["num_workflows"] = num_workflows
        print(
            f"[scale] {scale} workflows={num_workflows} "
            f"seeds={args.seed_start}-{args.seed_start + args.num_seeds - 1}"
        )
        for offset in range(int(args.num_seeds)):
            eval_seed = int(args.seed_start + offset)
            for mode in modes:
                if (scale, mode, eval_seed) in completed:
                    continue
                with redirect_stdout(io.StringIO()):
                    if mode == "oo_actor_critic":
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
                    elif mode == "hirl_td3_ddqn":
                        metrics = run_hirl(
                            scenario, resources, hirl_scheduler, eval_seed
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
                        )
                metrics = add_normalized_metrics(metrics, num_workflows)
                row = {
                    "scale": scale,
                    "mode": mode,
                    "eval_seed": eval_seed,
                    "num_workflows": num_workflows,
                    **metrics,
                }
                rows.append(row)
                write_rows(output_dir, rows)
                print(
                    f"[eval] scale={scale:<6} seed={eval_seed} mode={mode:<16} "
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
