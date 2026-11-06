"""Isolated Deadline-credit ablations over the unchanged coupled trainer.

Only this entry point installs a temporary credit hook.  Existing entry points,
networks, task-level credit allocation, dual updates and selection are reused.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import production first: it configures the local dependency path and CPU threads.
from tools import train_hierarchical_d3qn as production
from tools import train_coupled_hierarchical_d3qn_sac as coupled
from tools.eval_alibaba_sl111 import digest, identity

VARIANTS = ("cf_self", "cf_uniform", "raw_self")
_FULL_CREDIT = production._workflow_counterfactual_low_deadline_credit


def deadline_credit(workflow_results, low_transitions, *, variant):
    """Return (source credit G, victim dual cost d, details) in the old format."""
    if variant not in (*VARIANTS, "full"):
        raise ValueError(f"Unknown Deadline-credit variant: {variant!r}")
    ids = [str(row["workflow_id"]) for row in workflow_results]
    if len(ids) != len(set(ids)):
        raise ValueError("Workflow IDs must be unique within a trajectory.")
    for row in workflow_results:
        response = float(row["response_time"])
        deadline = float(row["relative_deadline"])
        if not math.isfinite(response) or response < 0 or not math.isfinite(deadline) or deadline <= 0:
            raise ValueError("Response times must be finite/nonnegative; Deadlines finite/positive.")
    model = production._fixed_schedule_counterfactual_model(workflow_results)
    if model is None:
        raise ValueError("Credit ablations require complete task_plans; legacy fallback is forbidden.")
    full_credit, incremental_costs, full_details = _FULL_CREDIT(workflow_results, low_transitions)
    if variant == "full":
        return full_credit, incremental_costs, full_details

    costs = list(incremental_costs)
    if variant == "cf_uniform":
        delays = dict.fromkeys(ids, 0.0)
        for transition in low_transitions:
            workflow_id = str(transition["workflow_id"])
            delay = float(transition["reward_components"].get("time_increase_seconds", 0.0))
            if workflow_id not in delays or not math.isfinite(delay) or delay < 0:
                raise ValueError("Invalid source workflow or execution-time increase.")
            delays[workflow_id] += delay
        active = {key for key, value in delays.items() if value > 1e-8}
        total = math.fsum(costs)
        if total > 0 and not active:
            raise ValueError("Positive incremental cost without any slowed source workflow.")
        credit = [total / len(active) if key in active else 0.0 for key in ids]
    elif variant == "cf_self":
        credit = list(costs)
    else:
        credit = [min(1.0, float(full_details[key]["actual_violation"])) for key in ids]
        costs = list(credit)
    if variant != "raw_self" and not math.isclose(math.fsum(credit), math.fsum(costs), abs_tol=1e-10):
        raise RuntimeError("Deadline-credit allocation did not conserve victim cost.")

    # Metadata must describe the selected allocation, not Full's source allocation.
    # The existing task-level distributor and replay writer consume this format.
    all_full = production._replay_fixed_schedule(model, ids)
    recoverable_miss = any(
        float(row["response_time"]) > float(row["relative_deadline"]) + 1e-8
        and all_full[str(row["workflow_id"])] <= float(row["relative_deadline"]) + 1e-8
        for row in workflow_results
    )
    total = math.fsum(costs)
    details = {}
    for index, (key, row) in enumerate(zip(ids, workflow_results)):
        value = credit[index]
        if variant == "cf_uniform":
            own = value * costs[index] / total if total else 0.0
            affected = sum(cost > 0 for cost in costs) if value else 0
            crossing = bool(value > 0 and recoverable_miss)
        else:
            own = value
            affected = int(value > 0)
            crossing = bool(value > 0 and float(row["response_time"]) > float(row["relative_deadline"]) + 1e-8)
            if variant == "cf_self":
                crossing = crossing and all_full[key] <= float(row["relative_deadline"]) + 1e-8
        details[key] = {
            **full_details[key],
            # Preserve legacy aliases, but name both distinct replay references.
            "counterfactual_reference": "self_workflow_restored_to_full_frequency",
            "self_restored_response_time": full_details[key]["counterfactual_response_time"],
            "self_restored_violation": full_details[key]["counterfactual_violation"],
            "all_full_frequency_response_time": float(all_full[key]),
            "all_full_frequency_violation": max(
                0.0,
                (float(all_full[key]) - full_details[key]["deadline"]) / full_details[key]["deadline"],
            ),
            "deadline_cost_basis": (
                "raw_actual_lateness" if variant == "raw_self"
                else "all_full_frequency_incremental_lateness"
            ),
            "credit_variant": variant,
            "responsible": float(value > 0),
            "responsible_cost": value,
            "self_responsible_cost": own,
            "external_responsible_cost": max(0.0, value - own),
            "affected_workflow_count": affected,
            "responsible_crossing": float(crossing),
        }
    return credit, costs, details


@contextmanager
def installed_credit(variant, diagnostic_sink=None, input_sink=None):
    """Scope hooks to this serial training invocation, including error exits."""
    original_credit = production._workflow_counterfactual_low_deadline_credit
    original_collect = production.collect_d3qn_episode
    original_build = production.build_workflows
    collecting_training = True

    def credit_hook(results, transitions):
        # Frozen evaluation stores no SAC transitions and does not learn from G.
        # Keep its old serial/parallel metrics and diagnostics exactly unchanged.
        if not collecting_training:
            return original_credit(results, transitions)
        credit, costs, details = deadline_credit(results, transitions, variant=variant)
        if collecting_training and diagnostic_sink is not None:
            diagnostic_sink({
                "variant": variant,
                "workflow_ids": [str(row["workflow_id"]) for row in results],
                "source_credit": credit,
                "victim_dual_cost": costs,
                "details": details,
            })
        return credit, costs, details

    def collect_hook(*args, **kwargs):
        nonlocal collecting_training
        previous = collecting_training
        collecting_training = bool(kwargs.get("training", True))
        try:
            return original_collect(*args, **kwargs)
        finally:
            collecting_training = previous

    def build_hook(config, resources, seed, smoke_test=False):
        workflows = original_build(config, resources, seed, smoke_test=smoke_test)
        if input_sink is not None:
            input_sink({"seed": int(seed), **identity(workflows)})
        return workflows

    production._workflow_counterfactual_low_deadline_credit = credit_hook
    production.collect_d3qn_episode = collect_hook
    if input_sink is not None:
        production.build_workflows = build_hook
    try:
        yield
    finally:
        production._workflow_counterfactual_low_deadline_credit = original_credit
        production.collect_d3qn_episode = original_collect
        production.build_workflows = original_build


def validate_config(config):
    variant = config.get("deadline_credit_ablation", {}).get("variant")
    if variant not in VARIANTS:
        raise ValueError(f"deadline_credit_ablation.variant must be one of {VARIANTS}.")
    agent = config["agent"]
    training = config["training"]
    production.validate_training_config(training)
    if agent.get("low_dual_attribution") != "workflow_counterfactual":
        raise ValueError("Keep the existing workflow_counterfactual collection path.")
    if agent.get("high_algorithm") != "d3qn" or agent.get("low_algorithm") != "sac":
        raise ValueError("These ablations require the full D3QN + SAC model.")
    if not agent.get("use_low_agent", True) or not agent.get("high", {}).get("deadline_head_enabled", True):
        raise ValueError("Do not disable the SAC or upper Deadline Head.")
    if agent.get("low", {}).get("replay_priority", {}).get("enabled", False):
        raise ValueError("Responsibility-prioritized sampling is not a factor in this ablation.")
    if any(training.get(key) for key in ("freeze_high", "freeze_low", "initial_high_checkpoint", "initial_full_checkpoint")):
        raise ValueError("Credit ablations start fresh and jointly train both levels.")
    if any(int(training.get(key, 0)) != 0 for key in ("episode_index_offset", "high_active_episode_offset", "low_active_episode_offset")):
        raise ValueError("Continuation offsets are not supported by this fresh-training entry point.")
    return variant


def run_training(config, *, episodes_override=None, seed_override=None, smoke_test=False):
    config = copy.deepcopy(config)
    variant = validate_config(config)
    training = config["training"]
    seed = int(training["seed"] if seed_override is None else seed_override)
    default_episodes = 3 if smoke_test else training["total_episodes"]
    episodes = int(default_episodes if episodes_override is None else episodes_override)
    if seed < 0 or episodes < 1:
        raise ValueError("Seed must be nonnegative and episodes must be positive.")
    training.update(seed=seed, total_episodes=episodes)
    training["run_name"] = f"seed{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    result_root = Path(training["checkpoint_dir"])
    if not result_root.is_absolute():
        result_root = ROOT / result_root
    output_dir = result_root / training["run_name"]
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = (
        "tools/train_deadline_credit_ablation.py",
        "tools/train_coupled_hierarchical_d3qn_sac.py",
        "tools/train_hierarchical_d3qn.py",
        "tools/hierarchical_common.py",
        "tools/eval_alibaba_sl111.py",
        "src/sac/hierarchical_low_sac.py",
        "src/sac/deadline_lagrangian.py",
        "src/datasets/legacy_workflow_generator.py",
        "configs/v3_sm_74_legacy_train.json",
        "configs/resources_edge_cloud_dvfs_v3.json",
    )
    manifest = {
        "variant": variant, "status": "running", "smoke_test": bool(smoke_test),
        "seed": seed, "requested_episodes": episodes, "config": config,
        "episode_workload_seeds": list(range(seed + 1, seed + episodes + 1)),
        "validation_seed_starts": production.checkpoint_validation_seeds(training, seed + 100000),
        "source_sha256": {path: digest(ROOT / path) for path in sources},
        "reliability_feedback": "unchanged coupled.run_training / install_low_dual_attribution",
    }
    manifest_path = output_dir / "credit_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "launch_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[credit-ablation] variant={variant} episodes={episodes} seed={seed} output_dir={output_dir}", flush=True)
    with (output_dir / "deadline_credit_history.jsonl").open("x", encoding="utf-8") as handle, \
            (output_dir / "workload_inputs.jsonl").open("x", encoding="utf-8") as input_handle:
        def record(row):
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        def record_input(row):
            input_handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            input_handle.flush()
        try:
            with installed_credit(variant, record, record_input):
                result = coupled.run_training(config, episodes_override=episodes, smoke_test=smoke_test)
        except BaseException as error:
            manifest.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            raise
    manifest.update(status="completed", validation_feasible=result["best_metrics"] is not None)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check", action="store_true", help="Validate config only; do not create a run or train.")
    parser.add_argument("--smoke-test", action="store_true", help="Existing reduced-resource smoke path; not a scientific result.")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    config = production.load_json(args.config)
    variant = validate_config(config)
    if args.check:
        print(f"[credit-ablation-check] variant={variant} episodes={args.episodes or config['training']['total_episodes']} seed={config['training']['seed'] if args.seed is None else args.seed} valid=True training_started=False")
        return
    result = run_training(config, episodes_override=args.episodes, seed_override=args.seed, smoke_test=args.smoke_test)
    print(f"output_dir={result['output_dir']}", flush=True)


if __name__ == "__main__":
    main()
