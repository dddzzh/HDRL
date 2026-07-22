from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schedulers.registry import create_scheduler
from src.simulator.metrics import compute_summary
from src.simulator.resource import load_resources
from src.ppo.dual_agent import HighLevelPPOAgent
from tools import eval_hierarchical_ppo as hppo_eval
from tools import train_hierarchical_ppo as train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation experiments comparing trained hierarchical PPO with baselines."
    )
    parser.add_argument("--checkpoint", required=True, help="Hierarchical PPO checkpoint .pt path.")
    parser.add_argument("--config", default=None, help="Optional config override; default reads checkpoint run config.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["hppo", "hppo_high_only", "fcfs", "heft"],
        choices=["hppo", "hppo_high_only", "fcfs", "heft"],
        help="Algorithms to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base evaluation seed.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Number of eval episodes.")
    parser.add_argument("--num-workflows", type=int, default=None, help="Workflows per eval episode.")
    parser.add_argument(
        "--dataset",
        choices=["pegasus", "alibaba_json", "alibaba_pool"],
        default=None,
        help="Workflow source override.",
    )
    parser.add_argument("--task-count", type=int, default=None, help="Pegasus task count override.")
    parser.add_argument("--max-ready-tasks", type=int, default=None, help="PPO visible ready-task capacity.")
    parser.add_argument("--device", default=None, help="cpu | cuda | auto.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="results/scheduler_experiments")
    parser.add_argument("--trace-decisions", action="store_true")
    parser.add_argument("--trace-decision-limit", type=int, default=None)
    parser.add_argument("--baseline-rho", type=float, default=1.0, help="Fixed rho used by FCFS/HEFT baselines.")
    parser.add_argument(
        "--disable-low-agent",
        action="store_true",
        help="Evaluate hppo with only the trained high-level server policy and a fixed rho.",
    )
    parser.add_argument(
        "--high-only-rho",
        type=float,
        default=None,
        help="Fixed rho used when low agent is disabled; default uses simulation.default_rho.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _eval_args_from_experiment(args):
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        config=args.config,
        seed=args.seed,
        device=args.device,
        eval_episodes=args.eval_episodes,
        num_workflows=args.num_workflows,
        dataset=args.dataset,
        task_count=args.task_count,
        max_ready_tasks=args.max_ready_tasks,
        output_dir=args.output_dir,
        deterministic=args.deterministic,
        trace_decisions=args.trace_decisions,
        trace_decision_limit=args.trace_decision_limit,
    )


def prepare_context(args):
    eval_args = _eval_args_from_experiment(args)
    checkpoint = hppo_eval.resolve_path(eval_args.checkpoint)
    high_checkpoint, low_checkpoint = resolve_high_low_checkpoints(checkpoint)
    config, config_source = hppo_eval.load_eval_config(eval_args, high_checkpoint)
    training_cfg = config.get("training", {})
    seed = args.seed if args.seed is not None else int(training_cfg.get("seed", 42))
    eval_episodes = (
        args.eval_episodes
        if args.eval_episodes is not None
        else int(training_cfg.get("eval_episodes", 5))
    )
    resources = load_resources(
        str(PROJECT_ROOT / config["resources"]["path"])
        if not Path(config["resources"]["path"]).is_absolute()
        else config["resources"]["path"],
        num_edge=config["resources"]["num_edge"],
        num_cloud=config["resources"]["num_cloud"],
        seed=seed,
    )
    workflow_source = train._prepare_eval_workflow_source(config, eval_args, resources, seed)
    sim_config = config.get("simulation", {})
    return {
        "args": args,
        "eval_args": eval_args,
        "config": config,
        "config_source": config_source,
        "seed": seed,
        "eval_episodes": eval_episodes,
        "resources": resources,
        "workflow_source": workflow_source,
        "sim_config": sim_config,
        "high_checkpoint": high_checkpoint,
        "low_checkpoint": low_checkpoint,
    }


def resolve_high_low_checkpoints(checkpoint: Path) -> tuple[Path, Path | None]:
    if checkpoint.is_dir():
        raise ValueError(f"Please pass a checkpoint .pt file, not a run directory: {checkpoint}")
    if checkpoint.suffix != ".pt":
        raise ValueError(f"checkpoint must be a .pt file: {checkpoint}")

    name = checkpoint.name
    if name.endswith("_high.pt"):
        high = checkpoint
        low = checkpoint.with_name(name.replace("_high.pt", "_low.pt"))
    elif name.endswith("_low.pt"):
        low = checkpoint
        high = checkpoint.with_name(name.replace("_low.pt", "_high.pt"))
    else:
        high = checkpoint.with_name(f"{checkpoint.stem}_high{checkpoint.suffix}")
        low = checkpoint.with_name(f"{checkpoint.stem}_low{checkpoint.suffix}")
    if not high.exists():
        raise FileNotFoundError(f"high-level checkpoint not found: {high}")
    return high, low if low.exists() else None


class HighOnlyHierarchicalAgent:
    """Use the trained high-level server policy and disable the low DVFS policy."""

    def __init__(self, high_agent, fixed_rho):
        self.high_agent = high_agent
        self.fixed_rho = float(fixed_rho)

    def select_action(self, state, deterministic=True):
        high_decision = self.high_agent.select_assignment(state, deterministic=deterministic)
        env_action = {
            "workflow_id": high_decision["workflow_id"],
            "task_id": high_decision["task_id"],
            "server_id": high_decision["primary_server_id"],
            "rho": self.fixed_rho,
            "backup": True,
            "backup_server_id": high_decision["backup_server_id"],
        }
        return high_decision, {"rho": self.fixed_rho, "disabled": True}, env_action


def build_high_only_agent(context):
    args = context["eval_args"]
    config = context["config"]
    resources = context["resources"]
    workflow_source = context["workflow_source"]
    high_checkpoint = context["high_checkpoint"]
    agent_cfg = hppo_eval.build_agent_config(
        config,
        args,
        resources,
        workflow_source,
        high_checkpoint,
        high_checkpoint,
    )
    high_cfg = {**dict(agent_cfg.get("shared", {})), **dict(agent_cfg.get("high", {}))}
    high_cfg["checkpoint"] = str(high_checkpoint)
    fixed_rho = (
        context["args"].high_only_rho
        if context["args"].high_only_rho is not None
        else float(context["sim_config"].get("default_rho", 1.0))
    )
    return HighOnlyHierarchicalAgent(HighLevelPPOAgent(high_cfg, seed=context["seed"]), fixed_rho)


def evaluate_scheduler(name, context, scheduler_params=None):
    rows = []
    config = context["config"]
    resources = context["resources"]
    source = context["workflow_source"]
    sim_config = context["sim_config"]
    eval_args = context["eval_args"]
    scheduler_params = dict(scheduler_params or {})

    for eval_index in range(context["eval_episodes"]):
        eval_seed = context["seed"] + eval_index
        workflows = train._workflows_for_episode(source, config, eval_args, resources, eval_seed)
        env = train.build_env(workflows, resources, copy.deepcopy(sim_config), seed=eval_seed)
        scheduler = create_scheduler(name, params=scheduler_params, seed=eval_seed)
        result = env.run(scheduler)
        summary = compute_summary(result["workflow_results"], result["event_log"], env.servers)
        row = {
            "eval_index": eval_index + 1,
            "seed": eval_seed,
            "algorithm": name,
            "total_energy": summary.get("total_energy"),
            "makespan": summary.get("avg_workflow_makespan"),
            "total_finished_workflows": summary.get("total_finished_workflows"),
            "deadline_violation": train.deadline_violation(summary),
            "deadline_success_count": summary.get("deadline_success_count"),
            "deadline_total_count": summary.get("deadline_total_count"),
            "deadline_success_rate": summary.get("deadline_success_rate"),
            "avg_workflow_reliability": summary.get("avg_workflow_reliability"),
            "reliability_violation": train.reliability_violation(summary),
            "reliability_success_count": summary.get("reliability_success_count"),
            "reliability_total_count": summary.get("reliability_total_count"),
            "reliability_success_rate": summary.get("reliability_success_rate"),
            "avg_server_utilization": summary.get("avg_server_utilization"),
            "backup_replica_count": summary.get("backup_replica_count"),
            "backup_energy": summary.get("backup_energy"),
        }
        rows.append(row)
        print(
            f"[{name}] index={row['eval_index']} seed={eval_seed} "
            f"energy={train.fmt_metric(row['total_energy'])}J "
            f"finished={train.fmt_metric(row['total_finished_workflows'])} "
            f"deadline_rate={train.fmt_metric(row['deadline_success_rate'])}"
        )
    summary = train._average_eval_rows(rows)
    summary["algorithm"] = name
    return {"summary": summary, "episodes": rows}


def evaluate_high_only_hppo(context, algorithm_name="hppo_high_only"):
    args = context["eval_args"]
    config = context["config"]
    resources = context["resources"]
    source = context["workflow_source"]
    sim_config = context["sim_config"]
    training_cfg = config.get("training", {})
    reward_config = {
        **training_cfg,
        "lambda_R": training_cfg.get("lambda_R", 0.0),
    }
    agent = build_high_only_agent(context)
    print(
        f"[{algorithm_name}] evaluating high-level policy only; "
        f"low agent disabled, fixed_rho={train.fmt_metric(agent.fixed_rho)}"
    )
    rows, summary = hppo_eval.evaluate_episodes(
        agent,
        source,
        config,
        args,
        resources,
        copy.deepcopy(sim_config),
        reward_config,
        context["seed"],
        context["eval_episodes"],
    )
    summary["algorithm"] = algorithm_name
    return {"summary": summary, "episodes": rows}


def main():
    args = parse_args()
    context = prepare_context(args)
    output_dir = resolve_path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if "hppo" in args.algorithms:
        if args.disable_low_agent:
            results["hppo"] = evaluate_high_only_hppo(context, algorithm_name="hppo")
        else:
            if context["low_checkpoint"] is None:
                raise FileNotFoundError(
                    "Full hppo evaluation requires a matching low-level checkpoint. "
                    "Use --disable-low-agent or algorithm hppo_high_only to evaluate high-only."
                )
            print("[hppo] evaluating trained hierarchical PPO")
            hppo_payload = hppo_eval.evaluate_checkpoint(
                context["eval_args"],
                write_output=False,
                verbose=True,
            )
            results["hppo"] = {
                "summary": {"algorithm": "hppo", **hppo_payload["summary"]},
                "episodes": hppo_payload["episodes"],
            }

    if "hppo_high_only" in args.algorithms:
        results["hppo_high_only"] = evaluate_high_only_hppo(context)

    baseline_params = {"rho": args.baseline_rho}
    for name in args.algorithms:
        if name in {"hppo", "hppo_high_only"}:
            continue
        print(f"[{name}] evaluating baseline")
        params = baseline_params
        results[name] = evaluate_scheduler(name, context, params)

    summary_rows = [payload["summary"] for payload in results.values()]
    payload = {
        "config": context["config_source"],
        "dataset": context["workflow_source"]["description"],
        "seed": context["seed"],
        "eval_episodes": context["eval_episodes"],
        "algorithms": args.algorithms,
        "summaries": summary_rows,
        "results": results,
    }
    output_path = output_dir / "comparison_results.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"comparison results saved: {output_path}")
    print("summary:")
    for row in summary_rows:
        print(
            f"  {row['algorithm']}: energy={train.fmt_metric(row.get('total_energy'))}J "
            f"deadline_rate={train.fmt_metric(row.get('deadline_success_rate'))} "
            f"reliability={train.fmt_metric(row.get('avg_workflow_reliability'))}"
        )


if __name__ == "__main__":
    main()
