"""Read-only diagnostics for the triple-head D3QN upper level and the low DVFS layer.

The tool never trains and never writes a checkpoint.  It answers four questions
raised in ``docs/CLAUDE_HANDOFF_CURRENT_ALGORITHM_ISSUES_20260827.md``:

1. Action-dimension diagnostics for the three heads (issue 3.6): per-decision
   ``Std_a(Q_E)``, ``Std_a(Q_D)``, ``Std_a(Q_R)``, per-head greedy actions and
   the rate at which each head changes the final greedy action.
2. The physically available DVFS headroom.  ``compute_energy`` implements
   ``E(rho) = (L / f_max) * (P_static / rho + P_dyn * rho^2)`` which has an
   interior minimum at ``rho* = (P_static / (2 * P_dyn))^(1/3)``.  The
   ``analytic`` rho mode plays that per-server optimum (clipped to the
   environment's own feasibility bound) so the SAC policy can be compared
   against the reachable optimum rather than against ``rho = 1``.
3. Per-seed failure attribution for the hard seeds (issue 3.4): which
   workflows miss, on which servers, with how much queue waiting.
4. Makespan naming (issue 3.9): all four makespan-like quantities are reported
   side by side.

Usage::

    py tools/diagnose_triple_head_dvfs.py \
        --high-checkpoint results/.../checkpoint_ep000300_high.pt \
        --rho-mode rho1 --output results/diagnostics/xxx.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from tools.hierarchical_common import (
    build_env,
    build_resources,
    build_workflows,
    configure_agent_state,
)
from tools.train_hierarchical_d3qn import collect_d3qn_episode

EPS = 1e-12
RHO_MODES = ("rho1", "sac", "analytic")


def energy_optimal_rho(static_power, dynamic_power):
    """Interior minimiser of ``P_static / rho + P_dyn * rho^2`` on ``(0, 1]``."""
    static_power = float(static_power)
    dynamic_power = float(dynamic_power)
    if dynamic_power <= 0.0:
        return 1.0
    return float(min(1.0, (static_power / (2.0 * dynamic_power)) ** (1.0 / 3.0)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument(
        "--sac-checkpoint",
        default=None,
        help="Optional separate checkpoint whose low SAC weights are loaded.",
    )
    parser.add_argument("--rho-mode", choices=RHO_MODES, default="rho1")
    parser.add_argument(
        "--uniform-rho",
        type=float,
        default=None,
        help=(
            "Play a fixed rho on every task (clipped to the environment's own "
            "rho_hardware_min). Maps out the joint Deadline/reliability feasible "
            "frequency frontier without any training."
        ),
    )
    parser.add_argument(
        "--analytic-rho-scale",
        type=float,
        default=1.0,
        help="Multiply the analytic rho* before the feasibility clip (<=1 is more aggressive).",
    )
    parser.add_argument(
        "--analytic-rho-blend",
        type=float,
        default=None,
        help=(
            "Selective DVFS: play rho = 1 - blend * (1 - rho*_j), so servers where "
            "DVFS cannot save energy (rho* = 1, e.g. edge_lowend) stay at rho = 1 "
            "and only the DVFS-favourable servers are slowed. blend=1 is the full "
            "per-server optimum, blend=0 is rho=1 everywhere."
        ),
    )
    parser.add_argument(
        "--deadline-lambda",
        type=float,
        default=None,
        help="Override lambda_D at action-selection time only (no training).",
    )
    parser.add_argument(
        "--reliability-lambda",
        type=float,
        default=None,
        help="Override lambda_R at action-selection time only (no training).",
    )
    parser.add_argument("--seed-start", type=int, default=242000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--per-task-seeds",
        default="",
        help="Comma-separated seeds for which a full per-task trace is written.",
    )
    return parser.parse_args()


def describe(values):
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_half_width": 0.0,
                "min": 0.0, "max": 0.0}
    count = int(array.size)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    t_critical = 2.093 if count == 20 else 1.96
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_half_width": float(t_critical * std / math.sqrt(count)) if count > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_agent(high_checkpoint, sac_checkpoint, use_low_agent, seed):
    payload = torch.load(str(high_checkpoint), map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload.get("config") or {})
    if not config:
        raise RuntimeError("Checkpoint does not carry its training configuration.")
    resources = build_resources(config, int(config.get("training", {}).get("seed", 42)))
    agent_config = configure_agent_state(config, resources, smoke_test=False)
    agent_config["use_low_agent"] = bool(use_low_agent)
    saved_high = dict(payload.get("dual_head_state_dict") or {})
    if "reliability_q_eval" not in saved_high:
        raise RuntimeError("This diagnostic targets the three-head checkpoint format.")
    from tools.train_reliability_aware_hierarchical_d3qn import (
        ReliabilityAwareHierarchicalD3QNAgent,
    )

    agent = ReliabilityAwareHierarchicalD3QNAgent(agent_config, seed=seed)
    agent.load_checkpoint(high_checkpoint)
    if use_low_agent and sac_checkpoint is not None:
        low_payload = torch.load(
            str(sac_checkpoint), map_location="cpu", weights_only=False
        )
        low_state = low_payload.get("low_model_state_dict")
        if low_state is None:
            raise RuntimeError(f"{sac_checkpoint} has no low_model_state_dict.")
        agent.low_agent.load_state_dict(copy.deepcopy(low_state))
        # Reload the high heads: the low checkpoint may carry stale high weights.
        agent.load_checkpoint(high_checkpoint)
    agent.high_agent.epsilon = 0.0
    return agent, config, resources, payload


def install_head_probe(agent, records):
    """Wrap ``select_assignment`` and record per-decision head diagnostics."""
    high = agent.high_agent
    original = high.select_assignment

    def probed(env_state, **kwargs):
        decision = original(env_state, **kwargs)
        lambda_d = float(high.deadline_lambda)
        lambda_r = float(high.lambda_reliability)
        combined, energy, deadline, reliability = (
            high.agent_R.combined_action_q_values(
                decision["state"], lambda_d, lambda_r
            )
        )
        valid = np.asarray(decision["primary_server_mask"], dtype=bool)
        if valid.sum() <= 0:
            return decision
        neg_inf = -np.inf

        def masked(values):
            return np.where(valid, values, neg_inf)

        energy_only = int(np.argmax(masked(energy)))
        # Deadline and reliability heads carry costs: lower is better.
        deadline_only = int(np.argmax(masked(-deadline)))
        reliability_only = int(np.argmax(masked(-reliability)))
        energy_deadline = int(np.argmax(masked(energy - lambda_d * deadline)))
        final = int(np.argmax(masked(combined)))
        records.append(
            {
                "workflow_id": str(decision["workflow_id"]),
                "task_id": int(decision["task_id"]),
                "chosen_server_id": int(decision["primary_server_id"]),
                "valid_actions": int(valid.sum()),
                "std_energy": float(np.std(energy[valid])),
                "std_deadline": float(np.std(deadline[valid])),
                "std_reliability": float(np.std(reliability[valid])),
                "range_energy": float(np.ptp(energy[valid])),
                "range_deadline": float(np.ptp(deadline[valid])),
                "range_reliability": float(np.ptp(reliability[valid])),
                "mean_energy": float(np.mean(energy[valid])),
                "mean_deadline": float(np.mean(deadline[valid])),
                "mean_reliability": float(np.mean(reliability[valid])),
                "lambda_deadline": lambda_d,
                "lambda_reliability": lambda_r,
                "deadline_to_energy_scale": float(
                    lambda_d * np.std(deadline[valid]) / max(np.std(energy[valid]), 1e-8)
                ),
                "reliability_to_energy_scale": float(
                    lambda_r
                    * np.std(reliability[valid])
                    / max(np.std(energy[valid]), 1e-8)
                ),
                "greedy_energy_only": energy_only,
                "greedy_deadline_only": deadline_only,
                "greedy_reliability_only": reliability_only,
                "greedy_energy_deadline": energy_deadline,
                "greedy_final": final,
                "deadline_head_changed_action": int(energy_only != energy_deadline),
                "reliability_head_changed_action": int(energy_deadline != final),
                "any_constraint_head_changed_action": int(energy_only != final),
            }
        )
        return decision

    high.select_assignment = probed
    return original


def install_rho_mode(agent, rho_mode, analytic_scale, uniform_rho=None, analytic_blend=None):
    if agent.low_agent is None:
        return
    original = agent.low_agent.select_frequency

    if uniform_rho is not None:
        target = float(uniform_rho)
        if not 0.0 < target <= 1.0:
            raise ValueError("--uniform-rho must be in (0, 1].")

        def uniform_select(high_observation, task_index, server_index,
                           deterministic=False, safety_metrics=None):
            decision = dict(
                original(high_observation, task_index, server_index,
                         deterministic=deterministic, safety_metrics=safety_metrics)
            )
            metrics = dict(safety_metrics or {})
            lower = float(metrics.get("rho_hardware_min", agent.low_agent.rho_min))
            rho = float(min(1.0, max(lower, target)))
            decision["rho"] = rho
            decision["action_rho"] = rho
            decision["action_u"] = rho
            return decision

        agent.low_agent.select_frequency = uniform_select
        return

    if rho_mode != "analytic":
        return
    blend = None if analytic_blend is None else float(analytic_blend)
    if blend is not None and not 0.0 <= blend <= 1.0:
        raise ValueError("--analytic-rho-blend must be in [0, 1].")

    def analytic_select(high_observation, task_index, server_index,
                        deterministic=False, safety_metrics=None):
        decision = dict(
            original(
                high_observation,
                task_index,
                server_index,
                deterministic=deterministic,
                safety_metrics=safety_metrics,
            )
        )
        server = high_observation.servers[int(server_index)]
        rho_star = energy_optimal_rho(
            server.get("static_power", 0.0), server.get("dynamic_power", 0.0)
        )
        if blend is None:
            rho_star = rho_star * float(analytic_scale)
        else:
            # Selective DVFS: servers with rho* = 1 (DVFS never pays there) stay
            # at rho = 1 for every blend value.
            rho_star = 1.0 - blend * (1.0 - rho_star)
        metrics = dict(safety_metrics or {})
        lower = max(
            float(metrics.get("rho_hardware_min", agent.low_agent.rho_min)),
            float(metrics.get("rho_safe", agent.low_agent.rho_min)),
        )
        rho = float(min(1.0, max(lower, rho_star)))
        decision["rho"] = rho
        decision["action_rho"] = rho
        decision["action_u"] = rho
        return decision

    agent.low_agent.select_frequency = analytic_select


def summarize_records(records):
    if not records:
        return {}
    keys = (
        "std_energy", "std_deadline", "std_reliability",
        "range_energy", "range_deadline", "range_reliability",
        "mean_energy", "mean_deadline", "mean_reliability",
        "deadline_to_energy_scale", "reliability_to_energy_scale",
        "deadline_head_changed_action", "reliability_head_changed_action",
        "any_constraint_head_changed_action",
    )
    out = {f"{key}_mean": float(np.mean([r[key] for r in records])) for key in keys}
    out["decisions"] = int(len(records))
    out["lambda_deadline"] = float(records[0]["lambda_deadline"])
    out["lambda_reliability"] = float(records[0]["lambda_reliability"])
    return out


def episode_task_stats(event_log):
    primary = [e for e in event_log if e.get("replica_role") == "primary"]
    by_type = defaultdict(lambda: {"tasks": 0, "energy": 0.0, "rho_sum": 0.0,
                                   "exec": 0.0, "wait": 0.0})
    for event in primary:
        row = by_type[str(event.get("server_type"))]
        row["tasks"] += 1
        row["energy"] += float(event.get("energy", 0.0))
        row["rho_sum"] += float(event.get("rho", 1.0))
        row["exec"] += float(event.get("exec_time", 0.0))
        row["wait"] += float(event.get("task_waiting_time", 0.0))
    result = {}
    for name, row in by_type.items():
        result[name] = {
            "tasks": row["tasks"],
            "energy": row["energy"],
            "energy_share": 0.0,
            "rho_mean": row["rho_sum"] / max(row["tasks"], 1),
            "exec_time_total": row["exec"],
            "wait_time_total": row["wait"],
        }
    total = sum(r["energy"] for r in result.values())
    for row in result.values():
        row["energy_share"] = row["energy"] / max(total, EPS)
    return result


def missed_workflow_detail(workflow_results, event_log):
    events_by_workflow = defaultdict(list)
    for event in event_log:
        if event.get("replica_role") == "primary":
            events_by_workflow[str(event.get("workflow_id"))].append(event)
    rows = []
    for workflow in workflow_results:
        if bool(workflow.get("deadline_satisfied", True)):
            continue
        workflow_id = str(workflow["workflow_id"])
        events = events_by_workflow.get(workflow_id, [])
        wait_total = sum(float(e.get("task_waiting_time", 0.0)) for e in events)
        exec_total = sum(float(e.get("exec_time", 0.0)) for e in events)
        comm_total = sum(float(e.get("task_communication_time", 0.0)) for e in events)
        server_counts = defaultdict(int)
        for event in events:
            server_counts[str(event.get("server_type"))] += 1
        worst = max(
            events, key=lambda e: float(e.get("task_waiting_time", 0.0)), default=None
        )
        rows.append({
            "workflow_id": workflow_id,
            "task_count": len(events),
            "response_time": float(workflow.get("response_time") or 0.0),
            "relative_deadline": float(workflow.get("relative_deadline") or 0.0),
            "c_over_d": float(workflow.get("response_time") or 0.0)
            / max(float(workflow.get("relative_deadline") or 0.0), EPS),
            "wait_time_total": wait_total,
            "exec_time_total": exec_total,
            "communication_time_total": comm_total,
            "wait_share_of_response": wait_total
            / max(float(workflow.get("response_time") or 0.0), EPS),
            "server_type_counts": dict(server_counts),
            "worst_wait_task": None if worst is None else {
                "task_id": int(worst.get("task_id", -1)),
                "server_id": int(worst.get("server_id", -1)),
                "server_type": str(worst.get("server_type")),
                "task_waiting_time": float(worst.get("task_waiting_time", 0.0)),
                "exec_time": float(worst.get("exec_time", 0.0)),
                "rho": float(worst.get("rho", 1.0)),
            },
        })
    return rows


def main():
    args = parse_args()
    torch.set_num_threads(1)
    use_low = args.rho_mode in {"sac", "analytic"} or args.uniform_rho is not None
    per_task_seeds = {
        int(value) for value in args.per_task_seeds.split(",") if value.strip()
    }
    with redirect_stdout(StringIO()):
        agent, config, resources, payload = build_agent(
            args.high_checkpoint, args.sac_checkpoint, use_low, args.seed_start
        )
        install_rho_mode(agent, args.rho_mode, args.analytic_rho_scale, args.uniform_rho, args.analytic_rho_blend)
    if args.deadline_lambda is not None:
        # Evaluation-only override: lambda_D enters action selection through
        # Q_E - lambda_D * Q_D and nothing is trained here.
        agent.high_agent.deadline_lagrangian.lambda_deadline = float(args.deadline_lambda)
    if args.reliability_lambda is not None:
        agent.high_agent.reliability_lagrangian.lambda_reliability = float(
            args.reliability_lambda
        )
    objective = HierarchicalObjective(config.get("objective", {}))

    rows = []
    all_head_records = []
    per_task_traces = {}
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        head_records = []
        original_select = install_head_probe(agent, head_records)
        try:
            with redirect_stdout(StringIO()):
                workflows = build_workflows(config, resources, seed, smoke_test=False)
                env = build_env(workflows, resources, dict(config.get("simulation", {})), seed)
                result = collect_d3qn_episode(
                    env, agent, objective, HighRolloutBuffer(), LowRolloutBuffer(),
                    training=False, high_deterministic=True, low_deterministic=True,
                    fixed_low_rho_one=(args.rho_mode == "rho1" and args.uniform_rho is None),
                    high_algorithm="d3qn",
                )
        finally:
            agent.high_agent.select_assignment = original_select
        metrics = result["metrics"]
        event_log = result["event_log"]
        workflow_results = result["workflow_results"]
        by_type = episode_task_stats(event_log)
        rho_values = [
            float(e.get("rho", 1.0)) for e in event_log
            if e.get("replica_role") == "primary"
        ]
        row = {
            "eval_seed": seed,
            "deadline_success_rate": float(metrics["deadline_success_rate"]),
            "reliability_success_rate": float(metrics["reliability_success_rate"]),
            "avg_workflow_reliability": float(metrics["avg_workflow_reliability"]),
            "total_energy": float(metrics["total_energy"]),
            "average_workflow_response_time": float(
                metrics["average_workflow_response_time"]
            ),
            "sum_of_workflow_response_times": float(metrics["episode_makespan"]),
            "avg_workflow_makespan": float(metrics["avg_workflow_makespan"]),
            "max_workflow_makespan": float(metrics["max_workflow_makespan"]),
            "global_schedule_makespan": float(metrics["global_makespan"]),
            "rho_mean": float(np.mean(rho_values)) if rho_values else 1.0,
            "rho_min_observed": float(np.min(rho_values)) if rho_values else 1.0,
            "rho_p05": float(np.percentile(rho_values, 5)) if rho_values else 1.0,
            "missed_workflows": missed_workflow_detail(workflow_results, event_log),
            "server_type_stats": by_type,
            **summarize_records(head_records),
        }
        rows.append(row)
        all_head_records.extend(head_records)
        if seed in per_task_seeds:
            per_task_traces[str(seed)] = [
                {
                    "workflow_id": str(e.get("workflow_id")),
                    "task_id": int(e.get("task_id", -1)),
                    "server_id": int(e.get("server_id", -1)),
                    "server_type": str(e.get("server_type")),
                    "rho": float(e.get("rho", 1.0)),
                    "energy": float(e.get("energy", 0.0)),
                    "exec_time": float(e.get("exec_time", 0.0)),
                    "task_waiting_time": float(e.get("task_waiting_time", 0.0)),
                    "start_time": float(e.get("start_time", 0.0)),
                    "finish_time": float(e.get("finish_time", 0.0)),
                }
                for e in event_log if e.get("replica_role") == "primary"
            ]
        print(
            f"[diag:{args.rho_mode}] seed={seed} "
            f"deadline={row['deadline_success_rate']:.2%} "
            f"E={row['total_energy'] / 1000.0:.2f}kJ "
            f"rho={row['rho_mean']:.3f} "
            f"stdQE={row.get('std_energy_mean', 0.0):.4f} "
            f"stdQD={row.get('std_deadline_mean', 0.0):.4f} "
            f"stdQR={row.get('std_reliability_mean', 0.0):.6f} "
            f"D-changed={row.get('deadline_head_changed_action_mean', 0.0):.3f} "
            f"R-changed={row.get('reliability_head_changed_action_mean', 0.0):.3f}",
            flush=True,
        )

    scalar_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "eval_seed"
    ]
    summary = {key: describe([row[key] for row in rows]) for key in scalar_keys}
    aggregate_by_type = defaultdict(lambda: {"tasks": 0, "energy": 0.0, "rho_sum": 0.0})
    for row in rows:
        for name, stats in row["server_type_stats"].items():
            agg = aggregate_by_type[name]
            agg["tasks"] += stats["tasks"]
            agg["energy"] += stats["energy"]
            agg["rho_sum"] += stats["rho_mean"] * stats["tasks"]
    total_energy = sum(v["energy"] for v in aggregate_by_type.values())
    server_type_summary = {
        name: {
            "tasks": v["tasks"],
            "energy_kJ": v["energy"] / 1000.0,
            "energy_share": v["energy"] / max(total_energy, EPS),
            "rho_mean": v["rho_sum"] / max(v["tasks"], 1),
            "energy_optimal_rho": None,
        }
        for name, v in aggregate_by_type.items()
    }
    for server in resources.get("servers", []):
        name = str(server.get("server_type", server.get("type")))
        if name in server_type_summary and server_type_summary[name]["energy_optimal_rho"] is None:
            server_type_summary[name]["energy_optimal_rho"] = energy_optimal_rho(
                server.get("static_power", 0.0), server.get("dynamic_power", 0.0)
            )

    report = {
        "high_checkpoint": str(Path(args.high_checkpoint).resolve()),
        "sac_checkpoint": None if args.sac_checkpoint is None else str(
            Path(args.sac_checkpoint).resolve()
        ),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "rho_mode": args.rho_mode,
        "analytic_rho_scale": float(args.analytic_rho_scale),
        "uniform_rho": args.uniform_rho,
        "analytic_rho_blend": args.analytic_rho_blend,
        "deadline_lambda_override": args.deadline_lambda,
        "reliability_lambda_override": args.reliability_lambda,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.num_seeds - 1,
        "summary": summary,
        "server_type_summary": server_type_summary,
        "head_diagnostics_all_decisions": summarize_records(all_head_records),
        "rows": rows,
        "per_task_traces": per_task_traces,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[summary:{args.rho_mode}] "
        f"deadline={summary['deadline_success_rate']['mean']:.2%} "
        f"energy_kJ={summary['total_energy']['mean'] / 1000.0:.2f} "
        f"response={summary['average_workflow_response_time']['mean']:.2f} "
        f"rho={summary['rho_mean']['mean']:.4f}",
        flush=True,
    )
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
