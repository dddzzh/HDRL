"""Run the D1-D6 Deadline diagnostics without changing training behavior.

D1 compares HEFT/rho=1 with the checkpoint's upper policy/rho=1.  HEFT is a
reference heuristic, not a proof of feasibility.  D2-D5 inspect the checkpoint
policy's decisions with HEFT-scaled task sub-Deadlines enabled.  D6 measures
the action-to-action Q spread that actually affects argmax selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        help="Optional evaluation config; defaults to the config stored in the checkpoint.",
    )
    parser.add_argument("--seed-start", type=int, default=642000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--esp-threshold", type=float, default=0.92)
    parser.add_argument("--target-sub-share", type=float, default=0.20)
    parser.add_argument("--output")
    return parser.parse_args()


def sub_deadline_lateness(finish_time, sub_deadline_absolute, task_count, deadline):
    """Linear task tardiness used by the proposed auxiliary credit."""
    return max(0.0, float(finish_time) - float(sub_deadline_absolute)) / (
        max(float(task_count), 1.0) * max(float(deadline), EPS)
    )


def beta_for_target_share(base_total, sub_total, target_share):
    """Solve beta*sub / (base + beta*sub) = target_share."""
    target_share = float(target_share)
    if not 0.0 < target_share < 1.0:
        raise ValueError("target_sub_share must be in (0, 1).")
    if float(sub_total) <= EPS:
        return None
    return float(
        target_share
        / (1.0 - target_share)
        * float(base_total)
        / float(sub_total)
    )


def _describe(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _outcome_summary(rows, threshold):
    if not rows:
        raise ValueError("outcome rows must not be empty")
    return {
        "seeds": len(rows),
        "mean_deadline_success_rate": float(
            np.mean([row["deadline_success_rate"] for row in rows])
        ),
        "esp": float(
            np.mean([row["deadline_success_rate"] + EPS >= threshold for row in rows])
        ),
        "mean_reliability_success_rate": float(
            np.mean([row["reliability_success_rate"] for row in rows])
        ),
        "mean_energy_joules": float(np.mean([row["total_energy"] for row in rows])),
    }


def _episode_outcome(seed, metrics):
    return {
        "eval_seed": int(seed),
        "deadline_success_rate": float(metrics["deadline_success_rate"]),
        "reliability_success_rate": float(metrics["reliability_success_rate"]),
        "total_energy": float(metrics["total_energy"]),
    }


def _q_spread(agent, high_decision):
    values = agent.high_agent.agent_R.action_q_values(high_decision["state"])
    energy = np.asarray(values[0], dtype=np.float64)
    deadline = np.asarray(values[1], dtype=np.float64)
    valid = np.asarray(high_decision["primary_server_mask"], dtype=bool)
    energy = energy[valid]
    deadline = deadline[valid]
    energy_span = float(np.ptp(energy)) if energy.size else 0.0
    deadline_span = float(np.ptp(deadline)) if deadline.size else 0.0
    deadline_lambda = float(getattr(agent.high_agent, "deadline_lambda", 0.0))
    return {
        "energy_span": energy_span,
        "deadline_span": deadline_span,
        "energy_std": float(np.std(energy)) if energy.size else 0.0,
        "deadline_std": float(np.std(deadline)) if deadline.size else 0.0,
        "deadline_lambda": deadline_lambda,
        "constraint_pressure": deadline_lambda
        * deadline_span
        / max(energy_span, EPS),
    }


def _decision_record(env, state, high_decision, agent):
    workflow_id = str(high_decision["workflow_id"])
    task_id = int(high_decision["task_id"])
    workflow_state = env.workflow_by_id[workflow_id]
    workflow = workflow_state["workflow"]
    task = env._get_task(workflow, task_id)
    context = state["decision_context"]
    mask = np.asarray(high_decision["primary_server_mask"], dtype=bool)
    observation = high_decision["observation"]
    candidates = []
    for index, server in enumerate(observation.servers):
        if index >= mask.size or not mask[index]:
            continue
        plan = env.estimate_replica_plan(
            task=task,
            server=env._get_server(int(server["id"])),
            rho=1.0,
            workflow_state=workflow_state,
            queue_type="execution",
            replica_role="primary",
        )
        candidates.append(
            {
                "server_id": int(server["id"]),
                "finish_time": float(plan["aft"]),
                "queue_delay": max(
                    0.0, float(plan["ast"]) - float(plan["base_ready_time"])
                ),
            }
        )
    selected_id = int(high_decision["primary_server_id"])
    selected = next(row for row in candidates if row["server_id"] == selected_id)
    selected_metrics = context["server_safety_metrics"][str(selected_id)]
    sub_deadline_absolute = selected_metrics.get("task_sub_deadline_absolute")
    if sub_deadline_absolute is None:
        raise RuntimeError("D1-D6 diagnostics require task sub-Deadlines.")
    deadline = float(selected_metrics["workflow_relative_deadline"])
    task_count = float(selected_metrics["workflow_task_count"])
    free_candidates = sum(row["queue_delay"] <= EPS for row in candidates)
    selected_sub_late = selected["finish_time"] > float(sub_deadline_absolute) + EPS
    avoidable_sub_late = selected_sub_late and any(
        row["finish_time"] <= float(sub_deadline_absolute) + EPS
        for row in candidates
    )
    return {
        "workflow_id": workflow_id,
        "task_id": task_id,
        "selected_contended": selected["queue_delay"] > EPS,
        "free_candidate_count": int(free_candidates),
        "contention_avoidable": bool(
            selected["queue_delay"] > EPS and free_candidates > 0
        ),
        "selected_predicted_sub_late": bool(selected_sub_late),
        "predicted_sub_late_avoidable": bool(avoidable_sub_late),
        "sub_deadline_normalized": sub_deadline_lateness(
            selected["finish_time"], sub_deadline_absolute, task_count, deadline
        ),
        **_q_spread(agent, high_decision),
    }


def _actual_sub_deadline_counts(env, workflow_results):
    on_time = {
        str(row["workflow_id"])
        for row in workflow_results
        if float(row["response_time"])
        <= float(row["relative_deadline"]) + EPS
    }
    tasks = late = 0
    for event in env.event_log:
        if str(event.get("replica_role", "primary")) != "primary":
            continue
        workflow_id = str(event["workflow_id"])
        if workflow_id not in on_time:
            continue
        state = env.workflow_by_id[workflow_id]
        sub_deadline = state["task_sub_deadlines_absolute"][int(event["task_id"])]
        tasks += 1
        late += int(float(event["finish_time"]) > float(sub_deadline) + EPS)
    return len(on_time), tasks, late


def _run_episode(config, resources, agent, objective, seed, high_algorithm, records=None):
    from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
    from tools.hierarchical_common import build_env, build_workflows
    from tools.train_hierarchical_d3qn import collect_d3qn_episode

    with redirect_stdout(StringIO()):
        workflows = build_workflows(config, resources, seed, smoke_test=False)
        env = build_env(workflows, resources, dict(config["simulation"]), seed)
    original_select = agent.select_action
    if records is not None:
        def select_with_diagnostics(state, *args, **kwargs):
            decision = original_select(state, *args, **kwargs)
            records.append(_decision_record(env, state, decision[0], agent))
            return decision

        agent.select_action = select_with_diagnostics
    high_buffer = HighRolloutBuffer()
    try:
        with redirect_stdout(StringIO()):
            result = collect_d3qn_episode(
                env,
                agent,
                objective,
                high_buffer,
                LowRolloutBuffer(),
                training=False,
                high_deterministic=True,
                low_deterministic=True,
                fixed_low_rho_one=True,
                high_algorithm=high_algorithm,
            )
    finally:
        agent.select_action = original_select
    return env, result, high_buffer


def main():
    args = parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in this Python environment; run this tool "
            "with the same interpreter used for training."
        ) from exc

    from src.D3QN.reward import HierarchicalObjective
    from tools.hierarchical_common import build_resources, configure_agent_state
    from tools.train_coupled_hierarchical_d3qn_sac import select_agent_class

    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        config = copy.deepcopy(payload.get("config") or {})
    if not config:
        raise RuntimeError("No config supplied and checkpoint has no embedded config.")
    config.setdefault("simulation", {})["sub_deadline"] = {
        "enabled": True,
        "method": "heft_aft_scaled_v1",
    }
    resources = build_resources(
        config, int(config.get("training", {}).get("seed", 42))
    )
    agent_config = configure_agent_state(config, resources, smoke_test=False)
    agent_config["use_low_agent"] = False
    agent = select_agent_class(config)(agent_config, seed=args.seed_start)
    agent.load_checkpoint(checkpoint)
    agent.high_agent.epsilon = 0.0
    objective = HierarchicalObjective(config.get("objective", {}))

    policy_rows = []
    heft_rows = []
    decision_rows = []
    base_total = 0.0
    on_time_workflows = on_time_tasks = on_time_sub_late = 0
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        episode_records = []
        env, result, high_buffer = _run_episode(
            config, resources, agent, objective, seed, "d3qn", episode_records
        )
        if len(episode_records) != high_buffer.size:
            raise RuntimeError("Decision diagnostics and rollout buffer are misaligned.")
        for index, record in enumerate(episode_records):
            base = max(0.0, float(high_buffer.deadline_costs[index])) / max(
                float(high_buffer.deadline_limits[index]), EPS
            )
            record["normalized_makespan_increment"] = base
            record["eval_seed"] = int(seed)
            base_total += base
        decision_rows.extend(episode_records)
        counts = _actual_sub_deadline_counts(env, result["workflow_results"])
        on_time_workflows += counts[0]
        on_time_tasks += counts[1]
        on_time_sub_late += counts[2]
        policy_rows.append(_episode_outcome(seed, result["metrics"]))

        _, heft_result, _ = _run_episode(
            config, resources, agent, objective, seed, "heft"
        )
        heft_rows.append(_episode_outcome(seed, heft_result["metrics"]))
        print(
            f"[D1-D6] seed={seed} "
            f"D3QN/rho1={policy_rows[-1]['deadline_success_rate']:.2%} "
            f"HEFT/rho1={heft_rows[-1]['deadline_success_rate']:.2%}",
            flush=True,
        )

    sub_total = float(sum(row["sub_deadline_normalized"] for row in decision_rows))
    beta_star = beta_for_target_share(
        base_total, sub_total, args.target_sub_share
    )
    selected_contended = sum(row["selected_contended"] for row in decision_rows)
    predicted_sub_late = sum(
        row["selected_predicted_sub_late"] for row in decision_rows
    )
    workflows_per_seed = int(config.get("generator", {}).get("num_workflows", 0))
    effective_threshold = (
        math.ceil(args.esp_threshold * workflows_per_seed - EPS)
        / workflows_per_seed
        if workflows_per_seed > 0
        else args.esp_threshold
    )
    report = {
        "protocol": {
            "checkpoint": str(checkpoint),
            "checkpoint_episode": int(payload.get("episode", 0)),
            "seeds": [args.seed_start, args.seed_start + args.num_seeds - 1],
            "rho": 1.0,
            "esp_threshold_requested": float(args.esp_threshold),
            "esp_threshold_effective": float(effective_threshold),
            "target_sub_deadline_share": float(args.target_sub_share),
            "note": "HEFT/rho1 is a reference heuristic, not a feasibility proof.",
        },
        "D1_reference_feasibility": {
            "d3qn_rho1": _outcome_summary(policy_rows, args.esp_threshold),
            "heft_rho1": _outcome_summary(heft_rows, args.esp_threshold),
        },
        "D2_contention_avoidability": {
            "decisions": len(decision_rows),
            "selected_contended": int(selected_contended),
            "selected_contended_fraction": selected_contended
            / max(len(decision_rows), 1),
            "avoidable_selected_contention": int(
                sum(row["contention_avoidable"] for row in decision_rows)
            ),
            "avoidable_fraction_given_contention": sum(
                row["contention_avoidable"] for row in decision_rows
            )
            / max(selected_contended, 1),
            "free_candidate_count": _describe(
                [row["free_candidate_count"] for row in decision_rows]
            ),
        },
        "D3_sub_deadline_false_positive": {
            "on_time_workflows": int(on_time_workflows),
            "tasks_in_on_time_workflows": int(on_time_tasks),
            "sub_deadline_late_tasks_in_on_time_workflows": int(on_time_sub_late),
            "fraction": on_time_sub_late / max(on_time_tasks, 1),
        },
        "D4_sub_deadline_avoidability": {
            "selected_predicted_sub_deadline_late": int(predicted_sub_late),
            "avoidable_with_another_server": int(
                sum(row["predicted_sub_late_avoidable"] for row in decision_rows)
            ),
            "avoidable_fraction": sum(
                row["predicted_sub_late_avoidable"] for row in decision_rows
            )
            / max(predicted_sub_late, 1),
        },
        "D5_sub_deadline_scale": {
            "normalized_makespan_increment_total": float(base_total),
            "sub_deadline_normalized_total_at_beta1": sub_total,
            "sub_share_at_beta1": sub_total / max(base_total + sub_total, EPS),
            "beta_for_target_share": beta_star,
            "suggested_sweep": (
                [0.0]
                if beta_star is None
                else [0.0, 0.5 * beta_star, beta_star, 2.0 * beta_star]
            ),
        },
        "D6_q_action_scale": {
            "energy_action_span": _describe(
                [row["energy_span"] for row in decision_rows]
            ),
            "deadline_action_span": _describe(
                [row["deadline_span"] for row in decision_rows]
            ),
            "energy_action_std": _describe(
                [row["energy_std"] for row in decision_rows]
            ),
            "deadline_action_std": _describe(
                [row["deadline_std"] for row in decision_rows]
            ),
            "constraint_pressure": _describe(
                [row["constraint_pressure"] for row in decision_rows]
            ),
        },
        "rows": {
            "d3qn_rho1": policy_rows,
            "heft_rho1": heft_rows,
        },
    }
    output = (
        Path(args.output).resolve()
        if args.output
        else checkpoint.parent
        / f"deadline_D1_D6_{args.seed_start}_{args.num_seeds}seed.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
