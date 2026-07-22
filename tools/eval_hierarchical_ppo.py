from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.device_utils import describe_runtime_devices
from src.ppo.dual_agent import HierarchicalPPOAgent
from src.simulator.resource import load_resources
from tools import train_hierarchical_ppo as train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained hierarchical PPO checkpoint without running training."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="训练时使用的配置文件路径；不传时从 checkpoint 或 run 目录自动读取。",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "checkpoint .pt 路径。传 *_high.pt 会自动匹配 *_low.pt；"
            "传 *_low.pt 会自动匹配 *_high.pt；传 base 名称如 hierarchical_ppo_ep320.pt "
            "会自动匹配 hierarchical_ppo_ep320_high.pt / _low.pt。"
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="评估随机种子。")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto。")
    parser.add_argument("--eval-episodes", type=int, default=None, help="评估 episode 数。")
    parser.add_argument("--num-workflows", type=int, default=None, help="每个 episode 的 workflow 数。")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["pegasus", "alibaba_json", "alibaba_pool"],
        help="评估使用的 workflow 来源。",
    )
    parser.add_argument("--task-count", type=int, default=None, help="Pegasus 固定 task 数。")
    parser.add_argument(
        "--max-ready-tasks",
        type=int,
        default=None,
        help="评估时 PPO 可见 ready task 数；默认沿用 config，设 0 表示完整展开。",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hierarchical_ppo_eval",
        help="评估结果输出目录。",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否用确定性动作评估，默认开启。",
    )
    parser.add_argument(
        "--trace-decisions",
        action="store_true",
        help="打印 env.step 的调度轨迹。",
    )
    parser.add_argument(
        "--trace-decision-limit",
        type=int,
        default=None,
        help="最多打印多少步调度轨迹。",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_checkpoint_pair(checkpoint: Path) -> tuple[Path, Path]:
    if checkpoint.is_dir():
        raise ValueError(
            "Please pass a checkpoint .pt file, not a run directory. Example: "
            f"{checkpoint / 'hierarchical_ppo_ep320_high.pt'}"
        )
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
    if not high.exists() or not low.exists():
        raise FileNotFoundError(f"checkpoint pair not found: high={high}, low={low}")
    return high, low


def load_checkpoint_extra(checkpoint: Path) -> dict:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to read checkpoint metadata") from exc
    payload = torch.load(checkpoint, map_location="cpu")
    return dict(payload.get("extra", {}) or {})


def load_eval_config(args, high_checkpoint: Path) -> tuple[dict, str]:
    if args.config is not None:
        config_path = resolve_path(args.config)
        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file), str(config_path)

    effective_config = high_checkpoint.parent / "effective_config.json"
    if effective_config.exists():
        with effective_config.open("r", encoding="utf-8") as file:
            return json.load(file), str(effective_config)

    extra = load_checkpoint_extra(high_checkpoint)
    if isinstance(extra.get("config"), dict):
        return copy.deepcopy(extra["config"]), f"{high_checkpoint}:extra.config"

    default_path = PROJECT_ROOT / "configs" / "hierarchical_ppo_default.json"
    with default_path.open("r", encoding="utf-8") as file:
        return json.load(file), str(default_path)


def build_agent_config(config, args, resources, workflow_source, high_checkpoint, low_checkpoint):
    agent_config = copy.deepcopy(
        config.get("agent", config.get("scheduler", {}).get("params", {}).get("agent", {}))
    )
    if "shared" not in agent_config:
        state_cfg = agent_config.get("state", config.get("scheduler", {}).get("params", {}).get("state", {}))
        shared = dict(agent_config)
        shared["state"] = dict(state_cfg)
        agent_config = {"shared": shared}
    if args.device is not None:
        agent_config.setdefault("shared", {})["device"] = args.device
    agent_config.setdefault("shared", {}).setdefault("device", "cpu")
    agent_config["shared"].setdefault("state", {})
    agent_config["shared"]["state"]["max_ready_tasks"] = train.infer_ready_task_capacity(
        config,
        args,
        workflow_source,
    )
    agent_config["shared"]["state"]["max_servers"] = train.infer_server_count(resources)
    sim_config = dict(config.get("simulation", {}))
    if "low_deadline_safety_margin_ratio" in training_cfg:
        sim_config["low_deadline_safety_margin_ratio"] = float(
            training_cfg["low_deadline_safety_margin_ratio"]
        )
    agent_config["shared"]["state"]["time_horizon"] = float(
        sim_config.get(
            "time_horizon",
            agent_config["shared"]["state"].get("time_horizon", 3600.0),
        )
    )
    agent_config.setdefault("low", {}).setdefault("rho_min", float(sim_config.get("rho_min", 0.1)))
    agent_config.setdefault("low", {})["reward_gamma"] = 0.0
    agent_config.setdefault("low", {})["gamma"] = 0.0
    agent_config.setdefault("low", {})["use_p3o"] = False
    agent_config.setdefault("low", {})["server_balanced_minibatch"] = True
    agent_config.setdefault("high", {})["checkpoint"] = str(high_checkpoint)
    agent_config.setdefault("low", {})["checkpoint"] = str(low_checkpoint)
    return agent_config


def evaluate_episodes(agent, source, config, args, resources, sim_config, reward_config, seed, eval_episodes):
    rows = []
    trace_config = {
        "enabled": bool(args.trace_decisions),
        "limit": args.trace_decision_limit,
    }
    for eval_index in range(eval_episodes):
        eval_seed = seed + eval_index
        workflows = train._workflows_for_episode(source, config, args, resources, eval_seed)
        env = train.build_env(workflows, resources, sim_config, seed=eval_seed)
        env.sim_config["trace_scheduling"] = trace_config["enabled"]
        env.sim_config["trace_limit"] = trace_config["limit"]
        env.in_run = trace_config["enabled"]
        result = train.run_evaluation_episode(
            env,
            agent,
            reward_config,
            deterministic=args.deterministic,
        )
        summary = result["summary"]
        row = {
            "eval_index": eval_index + 1,
            "seed": eval_seed,
            "steps": result["steps"],
            "done": result["done"],
            "total_reward": result.get("total_reward", result.get("total_high_reward", 0.0) + result.get("total_low_reward", 0.0)),
            "total_high_reward": result.get("total_high_reward"),
            "total_low_reward": result.get("total_low_reward"),
            "total_low_energy_reward": result.get("total_low_energy_reward"),
            "total_low_local_deadline_penalty": result.get("total_low_local_deadline_penalty"),
            "total_low_workflow_deadline_penalty": result.get("total_low_workflow_deadline_penalty"),
            "total_low_infeasible_penalty": result.get("total_low_infeasible_penalty"),
            "total_primary_energy_ref_rho1": result.get("total_primary_energy_ref_rho1"),
            "total_energy": summary.get("total_energy", result["total_energy"]),
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
            f"[eval] index={row['eval_index']} seed={eval_seed} "
            f"reward={train.fmt_metric(row['total_reward'])} "
            f"energy={train.fmt_metric(row['total_energy'])}J "
            f"finished={train.fmt_metric(row['total_finished_workflows'])} "
            f"deadline_rate={train.fmt_metric(row['deadline_success_rate'])} "
            f"reliability={train.fmt_metric(row['avg_workflow_reliability'])} "
            f"backup={train.fmt_metric(row['backup_replica_count'])}"
        )
    return rows, train._average_eval_rows(rows)


def evaluate_checkpoint(args, *, write_output=True, verbose=True):
    checkpoint = resolve_path(args.checkpoint)
    high_checkpoint, low_checkpoint = resolve_checkpoint_pair(checkpoint)
    config, config_source = load_eval_config(args, high_checkpoint)
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
    workflow_source = train._prepare_eval_workflow_source(config, args, resources, seed)
    sim_config = config.get("simulation", {})
    reward_config = {
        **training_cfg,
        "lambda_R": training_cfg.get("lambda_R", 0.0),
    }

    agent_config = build_agent_config(
        config,
        args,
        resources,
        workflow_source,
        high_checkpoint,
        low_checkpoint,
    )
    agent = HierarchicalPPOAgent(agent_config, seed=seed)

    if verbose:
        print(describe_runtime_devices(agent.high_agent.device))
        print(f"config source: {config_source}")
        print(f"checkpoint high: {high_checkpoint}")
        print(f"checkpoint low:  {low_checkpoint}")
        print(
            "Hierarchical PPO eval-only: "
            f"episodes={eval_episodes}, dataset={workflow_source['description']}, "
            f"max_ready_tasks={agent_config['shared']['state']['max_ready_tasks']}, "
            f"max_servers={agent_config['shared']['state']['max_servers']}, "
            f"deterministic={args.deterministic}"
        )

    rows, summary = evaluate_episodes(
        agent,
        workflow_source,
        config,
        args,
        resources,
        sim_config,
        reward_config,
        seed,
        eval_episodes,
    )
    if verbose:
        train.log_eval_metrics("final", summary)

    payload = {
        "config": str(config_source),
        "checkpoint_high": str(high_checkpoint),
        "checkpoint_low": str(low_checkpoint),
        "dataset": workflow_source["description"],
        "max_ready_tasks": agent_config["shared"]["state"]["max_ready_tasks"],
        "max_servers": agent_config["shared"]["state"]["max_servers"],
        "summary": summary,
        "episodes": rows,
    }
    if write_output:
        output_dir = resolve_path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "eval_history.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        payload["output_path"] = str(output_path)
        if verbose:
            print(f"eval history saved: {output_path}")
    return payload


def main():
    args = parse_args()
    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
