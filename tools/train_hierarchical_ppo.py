from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.workflow_factory import WorkflowSampler, create_workflows
from src.ppo.dual_agent import HierarchicalPPOAgent
from src.ppo.hierarchical_reward import (
    EnergyRewardNormalizer,
    P3O_COST_EPS,
    compute_allowed_log_risk,
    compute_deadline_cost_increment,
    compute_energy_decomposed_rewards,
    compute_high_level_time_reward,
    compute_low_energy_saving_reward,
    compute_low_local_deadline_penalty,
    compute_partial_workflow_log_risk,
    compute_partial_workflow_makespan,
    compute_reliability_cost_increment,
    compute_time_high_energy_low_rewards,
    extract_task_energy,
)
from src.ppo.rollout_buffer import PPORolloutBuffer
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.metrics import compute_summary
from src.simulator.resource import load_resources
from src.device_utils import describe_runtime_devices

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def parse_args():
    def parse_bool(value):
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

    parser = argparse.ArgumentParser(
        description=(
            "Train hierarchical PPO scheduler. PPO hyperparameters and network "
            "sizes are configured in configs/hierarchical_ppo_default.json."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/hierarchical_ppo_default.json",
        help="实验配置文件路径；PPO 超参数、网络结构、数据集和资源配置主要在这里改。",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="训练 episode 数；兼容旧参数，等价于 --train-episodes。",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=None,
        help="训练 episode 数；优先级高于 config.training.episodes。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；控制 workflow 采样、资源生成和 PPO 初始化。",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=None,
        help="每累计多少条 transition 触发一次 high/low PPO update。",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="每多少个 episode 保存一次 checkpoint；设为 0 表示只保存 final。",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="每多少个 episode 做一次 deterministic evaluation；设为 0 表示不评估。",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="每次 evaluation 跑多少个 episode 后取平均。",
    )
    parser.add_argument(
        "--num-workflows",
        type=int,
        default=None,
        help="每个 episode 的 workflow 数；覆盖 dataset.num_workflows 或 generator.num_workflows。",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=None,
        help=(
            "Pegasus workflow 的固定任务数；例如 10、20、50、200、1000。"
        ),
    )
    parser.add_argument(
        "--max-ready-tasks",
        type=int,
        default=None,
        help=(
            "PPO 每一步最多暴露给高层策略的 ready task 数。默认使用 config.agent.shared.state.max_ready_tasks；"
            "设为 0 表示按 episode 总 task 数展开，不建议大规模训练使用。"
        ),
    )
    parser.add_argument(
        "--debug-high-state",
        action="store_true",
        help="Print one high-level task-server feature debug snapshot at startup.",
    )
    parser.add_argument(
        "--trace-decisions",
        action="store_true",
        help="打印每一步调度决策以及 env.step 后的执行效果。",
    )
    parser.add_argument(
        "--trace-decision-limit",
        type=int,
        default=None,
        help="每个 episode 最多打印多少步决策；不设置表示全部打印。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["pegasus", "alibaba_json", "alibaba_pool"],
        help=(
            "Workflow source: Pegasus generator, a fixed Alibaba JSON, or alibaba_pool "
            "(sample N workflows per episode from a pool)"
        ),
    )
    parser.add_argument(
        "--arrival-process",
        "--arrival_process",
        dest="arrival_process",
        choices=["poisson", "multi_device_poisson", "device_poisson", "senior_poisson"],
        default=None,
        help="Workflow arrival process. multi_device_poisson matches the reference createTimeTable logic.",
    )
    parser.add_argument(
        "--arrival-rate",
        "--arrival_rate",
        dest="arrival_rate",
        type=float,
        default=None,
        help="Poisson lambda per second. In multi_device_poisson this is per mobile device.",
    )
    parser.add_argument(
        "--num-mobile-devices",
        "--num_mobile_devices",
        dest="num_mobile_devices",
        type=int,
        default=None,
        help="Number of independent mobile-device arrival streams for multi_device_poisson.",
    )
    parser.add_argument("--use-p3o", "--use_p3o", dest="use_p3o", type=parse_bool, nargs="?", const=True, default=None)
    parser.add_argument("--kappa-deadline", "--kappa_deadline", dest="kappa_deadline", type=float, default=None)
    parser.add_argument("--kappa-reliability", "--kappa_reliability", dest="kappa_reliability", type=float, default=None)
    parser.add_argument("--Rreq", "--rreq", "--reliability-req", "--reliability_req", dest="rreq", type=float, default=None)
    parser.add_argument("--deadline-cost-limit", "--deadline_cost_limit", dest="deadline_cost_limit", type=float, default=None)
    parser.add_argument("--reliability-cost-limit", "--reliability_cost_limit", dest="reliability_cost_limit", type=float, default=None)
    parser.add_argument("--cost-value-coef", "--cost_value_coef", dest="cost_value_coef", type=float, default=None)
    return parser.parse_args()


def infer_server_count(resources):
    servers = resources.get("servers", [])
    if not isinstance(servers, list) or not servers:
        raise ValueError("resources must contain a non-empty 'servers' list before creating PPO agents")
    return len(servers)


def workflow_task_count(workflow):
    if hasattr(workflow, "tasks"):
        return len(workflow.tasks)
    return len(workflow.get("tasks", []))


def configured_ready_task_capacity(config, args):
    if args.max_ready_tasks is not None:
        value = int(args.max_ready_tasks)
    else:
        agent_cfg = config.get("agent", {})
        shared_cfg = agent_cfg.get("shared", {})
        state_cfg = shared_cfg.get("state", agent_cfg.get("state", {}))
        value = int(state_cfg.get("max_ready_tasks", 32))
    return None if value <= 0 else max(1, value)


def infer_ready_task_capacity(config, args, workflow_source):
    configured_capacity = configured_ready_task_capacity(config, args)
    if workflow_source["type"] == "pegasus":
        gen_cfg = _generator_config(config, args)
        num_workflows = int(gen_cfg.get("num_workflows", 1))
        max_tasks = int(gen_cfg.get("max_tasks", gen_cfg.get("num_tasks", 1)))
        total_capacity = max(1, num_workflows * max_tasks)
        if configured_capacity is None:
            return total_capacity
        return min(configured_capacity, total_capacity)

    if workflow_source["type"] == "alibaba_json":
        total_capacity = max(
            1,
            sum(workflow_task_count(workflow) for workflow in workflow_source["fixed_workflows"]),
        )
        if configured_capacity is None:
            return total_capacity
        return min(configured_capacity, total_capacity)

    if workflow_source["type"] == "alibaba_pool":
        pool_max_tasks = max(
            workflow_task_count(workflow)
            for workflow in workflow_source["sampler"].pool
        )
        total_capacity = max(1, int(workflow_source["sample_count"]) * pool_max_tasks)
        if configured_capacity is None:
            return total_capacity
        return min(configured_capacity, total_capacity)

    raise ValueError(f"Unknown dataset type: {workflow_source['type']}")


def build_env(workflows, resources, sim_config, seed: int):
    return EdgeCloudWorkflowEnv(
        workflows=workflows,
        resources=resources,
        sim_config=sim_config,
        seed=seed,
    )


def _resolve_dataset_type(config, args):
    if args.dataset is not None:
        return args.dataset
    if _has_pegasus_scale_override(args):
        return "pegasus"
    return str(config.get("dataset", {}).get("type", "pegasus"))


def _has_pegasus_scale_override(args) -> bool:
    return getattr(args, "task_count", None) is not None


def _apply_exact_range(gen_cfg, exact_key, min_key, max_key, exact_value):
    value = int(exact_value)
    gen_cfg[exact_key] = value
    gen_cfg[min_key] = value
    gen_cfg[max_key] = value


def _apply_pegasus_scale_overrides(gen_cfg, args):
    if "num_tasks" not in gen_cfg and "min_tasks" not in gen_cfg and "max_tasks" not in gen_cfg:
        gen_cfg["num_tasks"] = 10

    if getattr(args, "task_count", None) is not None:
        _apply_exact_range(gen_cfg, "num_tasks", "min_tasks", "max_tasks", args.task_count)

    min_tasks = int(gen_cfg.get("min_tasks", gen_cfg.get("num_tasks", 1)))
    max_tasks = int(gen_cfg.get("max_tasks", min_tasks))
    if min_tasks <= 0 or max_tasks < min_tasks:
        raise ValueError(f"invalid Pegasus task range: min_tasks={min_tasks}, max_tasks={max_tasks}")
    return gen_cfg


def _apply_arrival_overrides(arrival_cfg, args):
    arrival_cfg["arrival_process"] = "multi_device_poisson"
    if getattr(args, "arrival_rate", None) is not None:
        arrival_cfg["arrival_rate"] = float(args.arrival_rate)
    arrival_cfg.pop("mean_interarrival", None)
    if getattr(args, "num_mobile_devices", None) is not None:
        arrival_cfg["num_mobile_devices"] = int(args.num_mobile_devices)
    return arrival_cfg


def _apply_constraint_overrides(target_cfg, args):
    if getattr(args, "rreq", None) is not None:
        target_cfg["Rreq"] = float(args.rreq)
        target_cfg["reliability_target"] = float(args.rreq)
    return target_cfg


def _generator_config(config, args):
    gen_cfg = dict(config.get("generator", {}))
    dataset_cfg = dict(config.get("dataset", {}))
    if not gen_cfg:
        gen_cfg = {
            key: value
            for key, value in dataset_cfg.items()
            if key not in ("type", "source", "path", "train_path", "test_path")
        }
    if args.num_workflows is not None:
        gen_cfg["num_workflows"] = int(args.num_workflows)
    _apply_arrival_overrides(gen_cfg, args)
    _apply_constraint_overrides(gen_cfg, args)
    return _apply_pegasus_scale_overrides(gen_cfg, args)


def _build_effective_config(
    config,
    args,
    *,
    seed,
    episodes,
    rollout_length,
    save_every,
    eval_interval,
    eval_episodes,
    workflow_source,
    eval_workflow_source,
    agent_config,
):
    effective = copy.deepcopy(config)
    effective.setdefault("training", {})
    effective["training"].update(
        {
            "seed": int(seed),
            "episodes": int(episodes),
            "rollout_length": int(rollout_length),
            "save_every": int(save_every),
            "eval_interval": int(eval_interval),
            "eval_episodes": int(eval_episodes),
            "update_mode": str(config.get("training", {}).get("update_mode", "episode_end")),
            "rollout_length_mode": "compatibility_only",
        }
    )
    effective["agent"] = copy.deepcopy(agent_config)
    dataset_type = _resolve_dataset_type(config, args)
    effective.setdefault("dataset", {})
    effective["dataset"]["type"] = dataset_type
    if args.num_workflows is not None:
        effective["dataset"]["num_workflows"] = int(args.num_workflows)
    _apply_arrival_overrides(effective["dataset"], args)
    _apply_constraint_overrides(effective["dataset"], args)
    _apply_constraint_overrides(effective.setdefault("simulation", {}), args)
    if dataset_type == "pegasus":
        effective["generator"] = _generator_config(config, args)
    effective.setdefault("runtime", {})
    effective["runtime"].update(
        {
            "workflow_source": workflow_source["description"],
            "eval_workflow_source": eval_workflow_source["description"],
            "max_ready_tasks": agent_config["shared"]["state"]["max_ready_tasks"],
            "max_servers": agent_config["shared"]["state"]["max_servers"],
        }
    )
    return effective


def _generate_pegasus_workflows(config, args, resources, seed):
    gen_cfg = _generator_config(config, args)
    return create_workflows(gen_cfg, source="pegasus", seed=seed, resources=resources)


def _load_alibaba_workflows(config, args, seed, resources):
    dataset_cfg = dict(config.get("dataset", {}))
    if args.num_workflows is not None:
        dataset_cfg["max_workflows"] = int(args.num_workflows)
    _apply_arrival_overrides(dataset_cfg, args)
    _apply_constraint_overrides(dataset_cfg, args)
    return create_workflows(dataset_cfg, source="alibaba_json", seed=seed, resources=resources)


def _build_pool_sampler(config, args=None, resources=None):
    dataset_cfg = dict(config.get("dataset", {}))
    gen_cfg = dict(config.get("generator", {}))
    if args is not None:
        _apply_arrival_overrides(dataset_cfg, args)
        _apply_constraint_overrides(dataset_cfg, args)
    train_path = dataset_cfg.get("train_path", dataset_cfg.get("path", "outputs/alibaba_train.json"))
    arrival_rate = float(dataset_cfg.get("arrival_rate", gen_cfg.get("arrival_rate", 0.01)))
    return WorkflowSampler.from_file(
        train_path,
        arrival_rate=arrival_rate,
        mapping_config=dataset_cfg,
        resources=resources,
    )


def _build_eval_pool_sampler(config, args=None, resources=None):
    dataset_cfg = dict(config.get("dataset", {}))
    gen_cfg = dict(config.get("generator", {}))
    if args is not None:
        _apply_arrival_overrides(dataset_cfg, args)
        _apply_constraint_overrides(dataset_cfg, args)
    test_path = dataset_cfg.get("test_path", dataset_cfg.get("eval_path", "outputs/alibaba_test.json"))
    arrival_rate = float(dataset_cfg.get("arrival_rate", gen_cfg.get("arrival_rate", 0.01)))
    return WorkflowSampler.from_file(
        test_path,
        arrival_rate=arrival_rate,
        mapping_config=dataset_cfg,
        resources=resources,
    )


def _resolve_pool_sample_count(config, args):
    if args.num_workflows is not None:
        return int(args.num_workflows)
    dataset_cfg = dict(config.get("dataset", {}))
    gen_cfg = dict(config.get("generator", {}))
    return int(dataset_cfg.get("num_workflows", gen_cfg.get("num_workflows", 100)))


def _prepare_workflow_source(config, args, resources, seed):
    dataset_type = _resolve_dataset_type(config, args)
    if dataset_type == "pegasus":
        gen_cfg = _generator_config(config, args)
        arrival_process = gen_cfg.get("arrival_process", "multi_device_poisson")
        desc = (
            f"pegasus (num_workflows={gen_cfg.get('num_workflows')}, "
            f"tasks={gen_cfg.get('min_tasks', gen_cfg.get('num_tasks'))}-"
            f"{gen_cfg.get('max_tasks', gen_cfg.get('num_tasks'))}, "
            f"arrival_process={arrival_process}, "
            f"arrival_rate={gen_cfg.get('arrival_rate')}, "
            f"num_mobile_devices={gen_cfg.get('num_mobile_devices', 1)})"
        )
        return {
            "type": dataset_type,
            "description": desc,
            "fixed_workflows": None,
            "sampler": None,
            "sample_count": 0,
        }
    if dataset_type == "alibaba_json":
        fixed_workflows = _load_alibaba_workflows(config, args, seed, resources)
        return {
            "type": dataset_type,
            "description": f"alibaba_json (fixed {len(fixed_workflows)} workflows)",
            "fixed_workflows": fixed_workflows,
            "sampler": None,
            "sample_count": 0,
        }
    if dataset_type == "alibaba_pool":
        sampler = _build_pool_sampler(config, args, resources)
        sample_count = _resolve_pool_sample_count(config, args)
        return {
            "type": dataset_type,
            "description": (
                f"alibaba_pool (sample {sample_count}/episode; "
                f"pool={len(sampler)}; fresh {sampler.arrival_process} arrivals per episode)"
            ),
            "fixed_workflows": None,
            "sampler": sampler,
            "sample_count": sample_count,
        }
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def _prepare_eval_workflow_source(config, args, resources, seed):
    dataset_type = _resolve_dataset_type(config, args)
    if dataset_type == "alibaba_pool":
        sampler = _build_eval_pool_sampler(config, args, resources)
        sample_count = _resolve_pool_sample_count(config, args)
        return {
            "type": dataset_type,
            "description": (
                f"alibaba_pool_eval (sample {sample_count}/episode; "
                f"pool={len(sampler)}; fresh {sampler.arrival_process} arrivals per episode)"
            ),
            "fixed_workflows": None,
            "sampler": sampler,
            "sample_count": sample_count,
        }
    return _prepare_workflow_source(config, args, resources, seed)


def _workflows_for_episode(source, config, args, resources, episode_seed):
    if source["type"] == "pegasus":
        return _generate_pegasus_workflows(config, args, resources, episode_seed)
    if source["type"] == "alibaba_json":
        return source["fixed_workflows"]
    if source["type"] == "alibaba_pool":
        return source["sampler"].sample(source["sample_count"], episode_seed)
    raise ValueError(f"Unknown dataset type: {source['type']}")


def fmt_metric(value, digits=4):
    if value in (None, "n/a"):
        return "n/a"
    try:
        number = float(value)
        if number != 0.0 and abs(number) < 10 ** (-digits):
            return f"{number:.{digits}e}"
        return f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def deadline_status_text(summary):
    success = summary.get("deadline_success_count")
    total = summary.get("deadline_total_count")
    violation = summary.get("deadline_violation_count")
    rate = summary.get("deadline_success_rate")
    if total in (None, 0):
        return "n/a"
    return (
        f"{success}/{total} met "
        f"(viol={violation}, rate={fmt_metric(rate)})"
    )


def should_trace_decision(trace_config, step):
    if not trace_config.get("enabled", False):
        return False
    limit = trace_config.get("limit")
    return limit is None or step < int(limit)


def log_decision_trace(episode, step, state, high_decision, low_decision):
    ready_count = len(state.get("ready_tasks", []))
    task = state["ready_tasks"][0] if state.get("ready_tasks") else {}
    server = high_decision["observation"].servers[high_decision["server_index"]]
    backup_server_id = high_decision.get("backup_server_id")
    print(
        f"[decision] ep={episode} step={step + 1} time={fmt_metric(state.get('time'))} "
        f"ready={ready_count} "
        f"task={high_decision['workflow_id']}:t{high_decision['task_id']} "
        f"workload={fmt_metric(task.get('workload_cycles'))}cycles "
        f"absolute_deadline={fmt_metric(task.get('absolute_deadline'))} "
        f"server={high_decision['server_id']}({server.get('type', 'n/a')}) "
        f"backup_server={backup_server_id if backup_server_id is not None else 'none'} "
        f"server_queue={fmt_metric(server.get('queue_load'))}s "
        f"server_util={fmt_metric(server.get('cpu_utilization'))} "
        f"rho={fmt_metric(low_decision['rho'])} "
        f"logp_H={fmt_metric(high_decision['log_prob'])} "
        f"logp_H_primary={fmt_metric(high_decision.get('primary_log_prob'))} "
        f"logp_H_backup={fmt_metric(high_decision.get('backup_log_prob'))} "
        f"H_primary={fmt_metric(high_decision.get('primary_entropy'))} "
        f"H_backup={fmt_metric(high_decision.get('backup_entropy'))} "
        f"H_joint={fmt_metric(high_decision.get('entropy'))} "
        f"logp_L={fmt_metric(low_decision['log_prob'])}"
    )


def log_effect_trace(episode, step, info, high_reward, low_reward, completed):
    event = info.get("event", {})
    completed_ids = ",".join(str(item.get("workflow_id")) for item in completed) if completed else "-"
    print(
        f"[effect]   ep={episode} step={step + 1} "
        f"workflow={event.get('workflow_id', 'n/a')} task={event.get('task_id', 'n/a')} "
        f"server={event.get('server_id', 'n/a')} rho={fmt_metric(event.get('rho'))} "
        f"start={fmt_metric(event.get('start_time'))}s "
        f"finish={fmt_metric(event.get('finish_time'))}s "
        f"exec={fmt_metric(event.get('exec_time'))}s "
        f"energy={fmt_metric(event.get('energy'))}J "
        f"high_r={fmt_metric(high_reward)} low_r={fmt_metric(low_reward)} "
        f"completed={completed_ids}"
    )


def log_update_metrics(episode, steps, high_metrics, low_metrics):
    print(
        f"[update] episode={episode} steps={steps} "
        f"high(loss={fmt_metric(high_metrics.get('loss'))}, "
        f"actor={fmt_metric(high_metrics.get('actor_loss'))}, "
        f"policy={fmt_metric(high_metrics.get('policy_loss'))}, "
        f"value={fmt_metric(high_metrics.get('value_loss'))}, "
        f"entropy={fmt_metric(high_metrics.get('entropy'))}, "
        f"clip={fmt_metric(high_metrics.get('clip_fraction'))}, "
        f"D={fmt_metric(high_metrics.get('deadline_constraint_loss'))}, "
        f"Rel={fmt_metric(high_metrics.get('reliability_constraint_loss'))}, "
        f"J_D={fmt_metric(high_metrics.get('J_deadline'))}, "
        f"J_Rel={fmt_metric(high_metrics.get('J_reliability'))}, "
        f"penD={fmt_metric(high_metrics.get('deadline_penalty_after_kappa'))}, "
        f"penRel={fmt_metric(high_metrics.get('reliability_penalty_after_kappa'))}, "
        f"same_pb={fmt_metric(high_metrics.get('primary_backup_same_rate'))}, "
        f"logp_pb={fmt_metric(high_metrics.get('joint_log_prob_mean'))}, "
        f"H_pb={fmt_metric(high_metrics.get('joint_entropy_mean'))}, "
        f"boundary={fmt_metric(high_metrics.get('boundary_rate'))}) "
        f"low(loss={fmt_metric(low_metrics.get('loss'))}, "
        f"actor={fmt_metric(low_metrics.get('actor_loss'))}, "
        f"policy={fmt_metric(low_metrics.get('policy_loss'))}, "
        f"value={fmt_metric(low_metrics.get('value_loss'))}, "
        f"entropy={fmt_metric(low_metrics.get('entropy'))}, "
        f"clip={fmt_metric(low_metrics.get('clip_fraction'))}, "
        f"D={fmt_metric(low_metrics.get('deadline_constraint_loss'))}, "
        f"Rel={fmt_metric(low_metrics.get('reliability_constraint_loss'))}, "
        f"J_D={fmt_metric(low_metrics.get('J_deadline'))}, "
        f"J_Rel={fmt_metric(low_metrics.get('J_reliability'))}, "
        f"penD={fmt_metric(low_metrics.get('deadline_penalty_after_kappa'))}, "
        f"penRel={fmt_metric(low_metrics.get('reliability_penalty_after_kappa'))}, "
        f"boundary={fmt_metric(low_metrics.get('boundary_rate'))}, "
        f"u={fmt_metric(low_metrics.get('u_mean'))}+/-{fmt_metric(low_metrics.get('u_std'))}, "
        f"rho={fmt_metric(low_metrics.get('rho_mean'))}+/-{fmt_metric(low_metrics.get('rho_std'))})"
    )


def log_episode_metrics(
    episode,
    reward,
    energy,
    steps,
    high_metrics,
    low_metrics,
    env,
    server_selection_counts=None,
    backup_request_counts=None,
    backup_execution_counts=None,
    replica_energy_stats=None,
):
    workflow_results = env._build_workflow_results()
    summary = compute_summary(workflow_results, env.event_log, env.servers)
    deadline_text = deadline_status_text(summary)
    low_server_text = format_low_server_stats(low_metrics)
    queue_text = format_server_queue_stats(compute_server_queue_stats(env.event_log))
    replica_energy_stats = replica_energy_stats or compute_replica_energy_stats(env.event_log)
    backup_requested_tasks = sum((backup_request_counts or {}).values())
    backup_started_tasks = int(replica_energy_stats.get("backup_task_count", 0))
    print(
        f"[train] ep={episode} steps={steps}\n"
        f"  objective  reward={fmt_metric(reward)}  "
        f"energy={fmt_metric(summary.get('total_energy', energy))}J  "
        f"E_ref={fmt_metric(high_metrics.get('mean_energy_ref'))}J  "
        f"E_actual={fmt_metric(high_metrics.get('mean_energy_actual'))}J  "
        f"E_save={fmt_metric(high_metrics.get('mean_energy_saving'))}J  "
        f"energy_norm={fmt_metric(high_metrics.get('energy_norm'))}\n"
        f"  workflow   finished={summary.get('total_finished_workflows', 'n/a')}  "
        f"deadline={deadline_text}  "
        f"makespan={fmt_metric(summary.get('avg_workflow_makespan'))}s  "
        f"max_makespan={fmt_metric(summary.get('max_workflow_makespan'))}s  "
        f"compute_span={fmt_metric(summary.get('avg_workflow_compute_span'))}s  "
        f"wait={fmt_metric(summary.get('avg_workflow_waiting_time'))}s  "
        f"util={fmt_metric(summary.get('avg_server_utilization'))}\n"
        f"  task_time  total={fmt_metric(summary.get('avg_task_total_time'))}s  "
        f"max_total={fmt_metric(summary.get('max_task_total_time'))}s  "
        f"wait={fmt_metric(summary.get('avg_task_waiting_time'))}s  "
        f"max_wait={fmt_metric(summary.get('max_task_waiting_time'))}s  "
        f"comm={fmt_metric(summary.get('avg_task_communication_time'))}s  "
        f"exec={fmt_metric(summary.get('avg_task_execution_time'))}s\n"
        f"  high       loss={fmt_metric(high_metrics.get('loss'))}  "
        f"actor={fmt_metric(high_metrics.get('actor_loss'))}  "
        f"policy={fmt_metric(high_metrics.get('policy_loss'))}  "
        f"value={fmt_metric(high_metrics.get('value_loss'))}  "
        f"D={fmt_metric(high_metrics.get('deadline_constraint_loss'))}  "
        f"Rel={fmt_metric(high_metrics.get('reliability_constraint_loss'))}  "
        f"penD={fmt_metric(high_metrics.get('deadline_penalty_after_kappa'))}  "
        f"penRel={fmt_metric(high_metrics.get('reliability_penalty_after_kappa'))}\n"
        f"[PPO-H] loss={fmt_metric(high_metrics.get('loss'))} "
        f"actor={fmt_metric(high_metrics.get('actor_loss'))} "
        f"policy={fmt_metric(high_metrics.get('policy_loss'))} "
        f"value={fmt_metric(high_metrics.get('value_loss'))} "
        f"entropy={fmt_metric(high_metrics.get('entropy'))} "
        f"clip={fmt_metric(high_metrics.get('clip_fraction'))} "
        f"completed={fmt_metric(high_metrics.get('completed_workflow_count'), digits=0)}\n"
        f"        D_usage_mean={fmt_metric(high_metrics.get('high_deadline_usage_mean'))} "
        f"D_usage_max={fmt_metric(high_metrics.get('high_deadline_usage_max'))} "
        f"D_active={fmt_metric(high_metrics.get('high_deadline_active_rate'))} "
        f"D_violation={fmt_metric(high_metrics.get('high_deadline_violation_rate'))} "
        f"D_surrogate={fmt_metric(high_metrics.get('high_deadline_surrogate_mean'))} "
        f"D_policy={fmt_metric(high_metrics.get('high_deadline_policy_loss'))} "
        f"D_penalty={fmt_metric(high_metrics.get('high_deadline_penalty_after_kappa'))} "
        f"D_viol_inactive={fmt_metric(high_metrics.get('high_deadline_violation_but_inactive_rate'))}\n"
        f"        R_active={fmt_metric(high_metrics.get('high_reliability_active_rate'))} "
        f"R_violation={fmt_metric(high_metrics.get('high_reliability_violation_rate'))} "
        f"R_surrogate={fmt_metric(high_metrics.get('high_reliability_surrogate_mean'))} "
        f"R_policy={fmt_metric(high_metrics.get('high_reliability_policy_loss'))} "
        f"R_penalty={fmt_metric(high_metrics.get('high_reliability_penalty_after_kappa'))}\n"
        f"  low        loss={fmt_metric(low_metrics.get('loss'))}  "
        f"actor={fmt_metric(low_metrics.get('actor_loss'))}  "
        f"policy={fmt_metric(low_metrics.get('policy_loss'))}  "
        f"value={fmt_metric(low_metrics.get('value_loss'))}  "
        f"D={fmt_metric(low_metrics.get('deadline_constraint_loss'))}  "
        f"Rel={fmt_metric(low_metrics.get('reliability_constraint_loss'))}  "
        f"penD={fmt_metric(low_metrics.get('deadline_penalty_after_kappa'))}  "
        f"penRel={fmt_metric(low_metrics.get('reliability_penalty_after_kappa'))}  "
        f"boundary={fmt_metric(low_metrics.get('boundary_rate'))}  "
        f"u={fmt_metric(low_metrics.get('u_mean'))}+/-{fmt_metric(low_metrics.get('u_std'))}  "
        f"rho={fmt_metric(low_metrics.get('rho_mean'))}+/-{fmt_metric(low_metrics.get('rho_std'))}  "
        f"alpha={fmt_metric(low_metrics.get('alpha_mean'))}  "
        f"beta={fmt_metric(low_metrics.get('beta_mean'))}  "
        f"normE={fmt_metric(low_metrics.get('energy_norm'))}  "
        f"server_groups={fmt_metric(low_metrics.get('server_group_count'), digits=0)} "
        f"return_mode={low_metrics.get('low_return_mode', 'n/a')}\n"
        f"  constraints completed={fmt_metric(high_metrics.get('completed_workflow_count'), digits=0)}  "
        f"D_rate={fmt_metric(high_metrics.get('deadline_violation_rate'))}  "
        f"Rel_rate={fmt_metric(high_metrics.get('reliability_violation_rate'))}  "
        f"D_cost={fmt_metric(high_metrics.get('mean_deadline_cost_per_completed_workflow'))}  "
        f"Rel_cost={fmt_metric(high_metrics.get('mean_reliability_cost_per_completed_workflow'))}\n"
        f"  low/server {low_server_text}\n"
        f"  primary    {format_server_selection_counts(server_selection_counts)}\n"
        f"  backup_req {format_server_selection_counts(backup_request_counts)}\n"
        f"  backup_exe {format_server_selection_counts(backup_execution_counts)}\n"
        f"  replica    primary_energy={fmt_metric(replica_energy_stats.get('primary_energy'))}J  "
        f"backup_energy={fmt_metric(replica_energy_stats.get('backup_energy'))}J  "
        f"backup_requested_tasks={backup_requested_tasks}  "
        f"backup_started_tasks={backup_started_tasks}\n"
        f"  queue      {queue_text}"
    )


def format_workflow_deadline_rows(workflow_results):
    if not workflow_results:
        return "  workflows  n/a"
    rows = []
    for workflow in sorted_workflow_results(workflow_results):
        finish_time = workflow.get("finish_time")
        absolute_deadline = workflow.get("absolute_deadline")
        deadline_satisfied = workflow.get("deadline_satisfied")
        if deadline_satisfied is None:
            status = "pending"
        else:
            status = "met" if bool(deadline_satisfied) else "miss"
        rows.append(
            "  workflow_deadline "
            f"wid={workflow.get('workflow_id')} "
            f"arrival={fmt_metric(workflow.get('arrival_time'))}s "
            f"finish={fmt_metric(finish_time)}s "
            f"deadline={fmt_metric(absolute_deadline)}s "
            f"response={fmt_metric(workflow.get('response_time'))}s "
            f"relative_deadline={fmt_metric(workflow.get('relative_deadline'))}s "
            f"reference_makespan={fmt_metric(workflow.get('reference_makespan'))}s "
            f"multiplier={fmt_metric(workflow.get('deadline_multiplier'))} "
            f"tardiness={fmt_metric(workflow.get('deadline_violation'))}s "
            f"status={status}"
        )
    return "\n".join(rows)


def workflow_deadline_records(workflow_results):
    records = []
    for workflow in sorted_workflow_results(workflow_results):
        finish_time = workflow.get("finish_time")
        absolute_deadline = workflow.get("absolute_deadline")
        deadline_satisfied = workflow.get("deadline_satisfied")
        records.append(
            {
                "workflow_id": workflow.get("workflow_id"),
                "arrival_time": workflow.get("arrival_time"),
                "finish_time": finish_time,
                "compute_finish_time": workflow.get("compute_finish_time"),
                "absolute_deadline": absolute_deadline,
                "relative_deadline": workflow.get("relative_deadline"),
                "reference_makespan": workflow.get("reference_makespan"),
                "deadline_multiplier": workflow.get("deadline_multiplier"),
                "deadline_type": workflow.get("deadline_type"),
                "response_time": workflow.get("response_time"),
                "makespan": workflow.get("makespan"),
                "compute_span": workflow.get("compute_span"),
                "waiting_time": workflow.get("waiting_time"),
                "deadline_violation": workflow.get("deadline_violation"),
                "deadline_satisfied": deadline_satisfied,
                "status": (
                    "pending"
                    if deadline_satisfied is None
                    else ("met" if bool(deadline_satisfied) else "miss")
                ),
                "total_energy": workflow.get("total_energy"),
                "reliability": workflow.get("reliability"),
                "reliability_req": workflow.get("reliability_req"),
                "reliability_met": workflow.get("reliability_met"),
            }
        )
    return records


def sorted_workflow_results(workflow_results):
    return sorted(workflow_results, key=lambda item: workflow_sort_key(item.get("workflow_id")))


def workflow_sort_key(workflow_id):
    try:
        return (0, int(workflow_id))
    except (TypeError, ValueError):
        return (1, str(workflow_id))


def format_server_selection_counts(counts):
    if not counts:
        return "n/a"
    rows = sorted(counts.items(), key=lambda item: int(item[0]))
    return "; ".join(f"s{server_id}={count}" for server_id, count in rows)


def format_low_server_stats(low_metrics, limit=5):
    counts = low_metrics.get("server_transition_counts") or {}
    mean_rho = low_metrics.get("server_mean_rho") or {}
    mean_energy = low_metrics.get("server_mean_energy") or {}
    if not counts:
        return "n/a"
    rows = sorted(counts.items(), key=lambda item: int(item[0]))[:limit]
    parts = []
    for server_id, count in rows:
        parts.append(
            f"s{server_id}:n={count},rho={fmt_metric(mean_rho.get(server_id))},"
            f"E={fmt_metric(mean_energy.get(server_id))}"
        )
    hidden = len(counts) - len(rows)
    suffix = f" ... +{hidden} servers" if hidden > 0 else ""
    return "; ".join(parts) + suffix


def compute_server_queue_stats(event_log):
    stats = {}
    for event in event_log:
        if event.get("queue_type", "execution") != "execution":
            continue
        server_id = event.get("server_id")
        if server_id is None:
            continue
        key = str(int(server_id))
        row = stats.setdefault(
            key,
            {
                "tasks": 0,
                "queued_tasks": 0,
                "wait_sum": 0.0,
                "wait_max": 0.0,
            },
        )
        wait = float(event.get("server_queue_waiting_time", 0.0) or 0.0)
        row["tasks"] += 1
        row["wait_sum"] += wait
        row["wait_max"] = max(row["wait_max"], wait)
        if wait > 1e-12:
            row["queued_tasks"] += 1

    result = {}
    for server_id, row in sorted(stats.items(), key=lambda item: int(item[0])):
        tasks = max(row["tasks"], 1)
        result[server_id] = {
            "tasks": row["tasks"],
            "queued_tasks": row["queued_tasks"],
            "queue_rate": row["queued_tasks"] / tasks,
            "mean_wait": row["wait_sum"] / tasks,
            "max_wait": row["wait_max"],
        }
    return result


def format_server_queue_stats(queue_stats, limit=5):
    if not queue_stats:
        return "n/a"
    rows = sorted(
        queue_stats.items(),
        key=lambda item: (item[1]["queue_rate"], item[1]["mean_wait"], item[1]["tasks"]),
        reverse=True,
    )[:limit]
    parts = []
    for server_id, row in rows:
        parts.append(
            f"s{server_id}:q={row['queued_tasks']}/{row['tasks']},"
            f"rate={fmt_metric(row['queue_rate'])},"
            f"wait={fmt_metric(row['mean_wait'])}s,"
            f"max={fmt_metric(row['max_wait'])}s"
        )
    hidden = len(queue_stats) - len(rows)
    suffix = f" ... +{hidden} servers" if hidden > 0 else ""
    return "; ".join(parts) + suffix


def summarize_server_queue_stats(queue_stats):
    if not queue_stats:
        return {
            "server_queue_total_tasks": 0,
            "server_queue_queued_tasks": 0,
            "server_queue_rate": 0.0,
            "server_queue_mean_wait": 0.0,
            "server_queue_max_wait": 0.0,
        }
    total_tasks = sum(row["tasks"] for row in queue_stats.values())
    queued_tasks = sum(row["queued_tasks"] for row in queue_stats.values())
    wait_sum = sum(row["mean_wait"] * row["tasks"] for row in queue_stats.values())
    max_wait = max(row["max_wait"] for row in queue_stats.values())
    return {
        "server_queue_total_tasks": total_tasks,
        "server_queue_queued_tasks": queued_tasks,
        "server_queue_rate": queued_tasks / max(total_tasks, 1),
        "server_queue_mean_wait": wait_sum / max(total_tasks, 1),
        "server_queue_max_wait": max_wait,
    }


def compute_replica_energy_stats(event_log):
    primary_events = [
        event
        for event in event_log
        if (
            event.get("replica_role", event.get("queue_type")) == "primary"
            or event.get("queue_type") == "execution"
        )
    ]
    backup_events = [
        event
        for event in event_log
        if event.get("replica_role") == "backup" or event.get("queue_type") == "backup"
    ]
    primary_energy = sum(float(event.get("energy", 0.0) or 0.0) for event in primary_events)
    backup_energy = sum(float(event.get("energy", 0.0) or 0.0) for event in backup_events)
    return {
        "primary_task_count": len(primary_events),
        "backup_task_count": len(backup_events),
        "primary_energy": primary_energy,
        "backup_energy": backup_energy,
        "replica_total_energy": primary_energy + backup_energy,
    }


def completed_workflows_from_step(prev_results, next_results):
    prev_completed = {
        item["workflow_id"]
        for item in prev_results
        if item.get("finish_time") is not None
    }
    completed = []
    for item in next_results:
        if item.get("finish_time") is not None and item["workflow_id"] not in prev_completed:
            completed.append(item)
    return completed


def update_if_ready(agent, high_buffer, low_buffer, rollout_length, force=False):
    metrics = {}
    if high_buffer.size == 0 and low_buffer.size == 0:
        return metrics
    if not force and high_buffer.size < rollout_length and low_buffer.size < rollout_length:
        return metrics
    if high_buffer.size > 0:
        high_metrics = agent.high_agent.update(high_buffer)
        metrics.update({f"high_{key}": value for key, value in high_metrics.items()})
        metrics["_high_raw"] = high_metrics
        high_buffer.clear()
    if low_buffer.size > 0:
        low_metrics = agent.low_dvfs_agent.update(low_buffer)
        metrics.update({f"low_{key}": value for key, value in low_metrics.items()})
        metrics["_low_raw"] = low_metrics
        low_buffer.clear()
    return metrics


def select_hierarchical_action_with_low_safety(env, agent, state, deterministic=False):
    high_decision = agent.high_agent.select_assignment(state, deterministic=deterministic)
    safety_metrics = env.compute_low_safety_metrics(
        high_decision["workflow_id"],
        high_decision["task_id"],
        high_decision["primary_server_id"],
    )
    low_decision = agent.server_low_agent_pool.select_frequency_for_server(
        high_observation=high_decision["observation"],
        task_index=high_decision["task_index"],
        server_index=high_decision["primary_server_index"],
        deterministic=deterministic,
        safety_metrics=safety_metrics,
    )
    env_action = {
        "workflow_id": high_decision["workflow_id"],
        "task_id": high_decision["task_id"],
        "server_id": high_decision["primary_server_id"],
        "rho": low_decision["rho"],
        "backup": True,
        "backup_server_id": high_decision["backup_server_id"],
    }
    return high_decision, low_decision, env_action, safety_metrics


def workflow_constraint_metadata(env, workflow_id, previous_partials):
    workflow_id = str(workflow_id)
    workflow_state = env.workflow_by_id[workflow_id]
    current_partial_makespan = compute_partial_workflow_makespan(
        workflow_state.get("task_finish_times", {}).values(),
        workflow_state["arrival_time"],
    )
    current_partial_log_risk = compute_partial_workflow_log_risk(
        workflow_state.get("reliability_product", 1.0),
        eps=P3O_COST_EPS,
    )
    previous = previous_partials.setdefault(
        workflow_id,
        {"makespan": 0.0, "log_risk": 0.0},
    )
    deadline_cost = compute_deadline_cost_increment(
        previous["makespan"],
        current_partial_makespan,
        eps=P3O_COST_EPS,
    )
    reliability_cost = compute_reliability_cost_increment(
        previous["log_risk"],
        current_partial_log_risk,
        eps=P3O_COST_EPS,
    )
    previous["makespan"] = current_partial_makespan
    previous["log_risk"] = current_partial_log_risk
    reliability_req = env._workflow_reliability_requirement(workflow_state)
    relative_deadline = float(env.workflow_relative_deadline(workflow_state))
    deadline_scale = max(relative_deadline, P3O_COST_EPS)
    normalized_deadline_cost = deadline_cost / deadline_scale
    normalized_partial_makespan = current_partial_makespan / deadline_scale
    return {
        "workflow_id": workflow_id,
        "deadline_cost": normalized_deadline_cost,
        "reliability_cost": reliability_cost,
        "deadline_limit": 1.0,
        "reliability_limit": compute_allowed_log_risk(reliability_req, eps=P3O_COST_EPS),
        "current_partial_makespan": normalized_partial_makespan,
        "current_partial_log_risk": current_partial_log_risk,
        "raw_deadline_increment": deadline_cost,
        "raw_partial_makespan": current_partial_makespan,
        "raw_relative_deadline": relative_deadline,
        "normalized_deadline_cost": normalized_deadline_cost,
        "normalized_partial_makespan": normalized_partial_makespan,
        "workflow_done": bool(workflow_state.get("completed", False)),
    }


def finalize_completed_workflows(env, completed, high_buffer, low_buffer, previous_partials, eps=P3O_COST_EPS):
    for item in completed:
        workflow_id = str(item["workflow_id"])
        response_time = item.get("response_time", item.get("makespan"))
        if response_time is not None:
            previous = previous_partials.setdefault(workflow_id, {"makespan": 0.0, "log_risk": 0.0})
            relative_deadline = float(item.get("relative_deadline") or env.workflow_relative_deadline(env.workflow_by_id[workflow_id]))
            deadline_scale = max(relative_deadline, P3O_COST_EPS)
            terminal_increment = float(response_time) - float(previous.get("makespan", 0.0))
            if terminal_increment < -float(eps):
                raise RuntimeError(
                    f"Final response time is below partial makespan for workflow={workflow_id}: "
                    f"response_time={response_time}, previous={previous.get('makespan')}"
                )
            if terminal_increment < 0.0:
                terminal_increment = 0.0
            previous["makespan"] = float(response_time)
            normalized_terminal_increment = terminal_increment / deadline_scale
            normalized_final_response = float(response_time) / deadline_scale
            for buffer in (high_buffer, low_buffer):
                buffer.add_to_last_workflow_transition(
                    workflow_id,
                    deadline_cost_delta=normalized_terminal_increment,
                    partial_makespan=normalized_final_response,
                    info_updates={
                        "terminal_deadline_increment": normalized_terminal_increment,
                        "terminal_raw_deadline_increment": terminal_increment,
                        "final_response_time": float(response_time),
                        "raw_partial_makespan": float(response_time),
                        "raw_relative_deadline": relative_deadline,
                        "normalized_partial_makespan": normalized_final_response,
                        "normalized_deadline_cost_terminal": normalized_terminal_increment,
                    },
                )
                indices = [idx for idx, wid in enumerate(buffer.workflow_ids) if wid == workflow_id]
                total_normalized = sum(float(buffer.deadline_costs[idx]) for idx in indices)
                if abs(total_normalized - normalized_final_response) > 1e-5:
                    raise RuntimeError(
                        "Normalized deadline cost telescope mismatch: "
                        f"workflow={workflow_id}, sum={total_normalized}, "
                        f"final_response/deadline={normalized_final_response}"
                    )
        high_buffer.mark_workflow_done(workflow_id)
        low_buffer.mark_workflow_done(workflow_id)


def apply_low_workflow_terminal_penalties(env, low_buffer, workflow_results, coef):
    by_workflow = {}
    for index, workflow_id in enumerate(low_buffer.workflow_ids):
        if workflow_id is not None:
            by_workflow.setdefault(str(workflow_id), []).append(index)
    results = {str(item["workflow_id"]): item for item in workflow_results}
    for workflow_id, indices in by_workflow.items():
        result = results.get(workflow_id)
        if not result or result.get("response_time") is None:
            continue
        relative_deadline = float(result.get("relative_deadline") or 0.0)
        if relative_deadline <= 0.0:
            continue
        violation_ratio = max(0.0, float(result["response_time"]) / relative_deadline - 1.0)
        slowdowns = [float(low_buffer.infos[index].get("low_slowdown", 0.0)) for index in indices]
        from src.ppo.hierarchical_reward import compute_workflow_deadline_penalty_distribution

        penalties = compute_workflow_deadline_penalty_distribution(
            slowdowns,
            violation_ratio,
            coef=coef,
        )
        for index, penalty in zip(indices, penalties):
            low_buffer.rewards[index] = float(low_buffer.rewards[index]) - float(penalty)
            low_buffer.infos[index]["low_workflow_deadline_penalty"] = float(penalty)
            low_buffer.infos[index]["low_total_reward"] = float(low_buffer.rewards[index])


def log_cpu_device_diagnostic(agent):
    if getattr(agent.high_agent.device, "type", None) != "cpu":
        return
    if torch is None:
        print(f"[device] agent resolved to CPU; python={sys.executable}; torch=not installed")
        return
    print(
        "[device] agent resolved to CPU; "
        f"python={sys.executable}; "
        f"torch={torch.__version__}; "
        f"cuda_available={torch.cuda.is_available()}; "
        f"cuda_version={torch.version.cuda}; "
        f"device_count={torch.cuda.device_count()}"
    )


def run_training_episode(
    env,
    agent,
    rollout_length,
    reward_config,
    episode=0,
    trace_config=None,
):
    trace_config = trace_config or {}
    state = env.reset()
    high_buffer = PPORolloutBuffer()
    low_buffer = PPORolloutBuffer()
    energy_normalizer = EnergyRewardNormalizer(
        beta=float(reward_config.get("energy_norm_beta", 0.99))
    )
    total_reward = 0.0
    total_energy = 0.0
    total_energy_ref = 0.0
    total_low_reward = 0.0
    steps = 0
    train_metrics = {}
    completed_workflow_ids = set()
    previous_partials = {}
    last_results = env._build_workflow_results()
    server_selection_counts = {
        str(int(server["id"])): 0
        for server in getattr(env, "servers", [])
    }
    backup_request_counts = {
        str(int(server["id"])): 0
        for server in getattr(env, "servers", [])
    }
    backup_execution_counts = {
        str(int(server["id"])): 0
        for server in getattr(env, "servers", [])
    }
    max_steps = sum(env._workflow_task_count(item["workflow"]) for item in env.workflow_states) + 1000

    while not env.done and steps < max_steps:
        if not state["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                current_results = env._build_workflow_results()
                completed = completed_workflows_from_step(last_results, current_results)
                finalize_completed_workflows(env, completed, high_buffer, low_buffer, previous_partials)
                completed_workflow_ids.update(str(item["workflow_id"]) for item in completed)
                last_results = current_results
                env._update_done()
                break
            current_results = env._build_workflow_results()
            completed = completed_workflows_from_step(last_results, current_results)
            finalize_completed_workflows(env, completed, high_buffer, low_buffer, previous_partials)
            completed_workflow_ids.update(str(item["workflow_id"]) for item in completed)
            last_results = current_results
            state = env.get_state()
            continue

        high_decision, low_decision, env_action, safety_metrics = select_hierarchical_action_with_low_safety(
            env,
            agent,
            state,
            deterministic=False,
        )
        energy_ref = env.estimate_primary_energy_ref_rho1(env_action)
        selected_server_id = str(int(env_action["server_id"]))
        server_selection_counts.setdefault(selected_server_id, 0)
        server_selection_counts[selected_server_id] += 1
        backup_server_id = env_action.get("backup_server_id")
        if backup_server_id is not None:
            backup_server_key = str(int(backup_server_id))
            backup_request_counts.setdefault(backup_server_key, 0)
            backup_request_counts[backup_server_key] += 1
        trace_this_step = should_trace_decision(trace_config, steps)
        if trace_this_step:
            log_decision_trace(episode, steps, state, high_decision, low_decision)
        next_state, env_reward, done, info = env.step(env_action)
        constraint_meta = workflow_constraint_metadata(
            env,
            high_decision["workflow_id"],
            previous_partials,
        )
        event = info.get("event", {})
        executed_backup_server = event.get("backup_server")
        if executed_backup_server is not None:
            backup_execution_key = str(int(executed_backup_server))
            backup_execution_counts.setdefault(backup_execution_key, 0)
            backup_execution_counts[backup_execution_key] += 1
        next_results = env._build_workflow_results()

        energy = extract_task_energy(info, env_reward)
        energy_norm = energy_normalizer.current(energy)
        event = info.get("event", {})
        primary_energy_actual = float(event.get("primary_energy", energy))
        low_energy_reward = compute_low_energy_saving_reward(energy_ref, primary_energy_actual)
        local_deadline_penalty = compute_low_local_deadline_penalty(
            event.get("finish_time", event.get("primary_aft", safety_metrics["estimated_ast"])),
            safety_metrics["task_latest_finish_time"],
            safety_metrics["workflow_relative_deadline"],
            coef=float(reward_config.get("low_local_deadline_penalty_coef", 10.0)),
        )
        infeasible_penalty = (
            float(reward_config.get("low_infeasible_penalty", 1.0))
            if low_decision.get("forced_max_frequency")
            else 0.0
        )
        high_reward = compute_high_level_time_reward(constraint_meta["raw_deadline_increment"])
        high_energy_reward = -float(energy_ref) / max(float(energy_norm), P3O_COST_EPS)
        low_reward = low_energy_reward - local_deadline_penalty - infeasible_penalty
        completed = completed_workflows_from_step(last_results, next_results)
        if trace_this_step:
            log_effect_trace(episode, steps, info, high_reward, low_reward, completed)

        next_high_obs = agent.high_agent.build_high_state(next_state)
        low_next_state = agent.server_low_agent_pool.build_low_state(
            next_high_obs,
            high_decision["task_index"],
            high_decision["server_index"],
            safety_metrics=safety_metrics,
        )

        high_buffer.add(
            state=high_decision["state"],
            action=[
                high_decision["primary_server_index"],
                high_decision["backup_server_index"],
            ],
            log_prob=high_decision["log_prob"],
            value=high_decision["value"],
            reward=high_reward,
            done=done,
            next_state=next_high_obs.vector,
            deadline_cost=constraint_meta["deadline_cost"],
            reliability_cost=constraint_meta["reliability_cost"],
            deadline_limit=constraint_meta["deadline_limit"],
            reliability_limit=constraint_meta["reliability_limit"],
            current_partial_makespan=constraint_meta["current_partial_makespan"],
            current_partial_log_risk=constraint_meta["current_partial_log_risk"],
            deadline_value=high_decision.get("deadline_value", 0.0),
            reliability_value=high_decision.get("reliability_value", 0.0),
            raw_task_energy=energy,
            workflow_id=high_decision["workflow_id"],
            workflow_done=constraint_meta["workflow_done"],
            masks={
                "server_mask": high_decision["server_mask"],
            },
            info={
                "workflow_id": high_decision["workflow_id"],
                "task_id": high_decision["task_id"],
                "server_id": high_decision["primary_server_id"],
                "primary_server_id": high_decision["primary_server_id"],
                "backup_server_id": high_decision["backup_server_id"],
                "primary_log_prob": high_decision.get("primary_log_prob"),
                "backup_log_prob": high_decision.get("backup_log_prob"),
                "joint_log_prob": high_decision["log_prob"],
                "primary_entropy": high_decision.get("primary_entropy"),
                "backup_entropy": high_decision.get("backup_entropy"),
                "entropy": high_decision.get("entropy"),
                "energy_ref": energy_ref,
                "energy_actual": energy,
                "energy_saving": energy_ref - energy,
                "energy_norm": energy_norm,
                "high_energy_reward": high_energy_reward,
                "high_time_reward": high_reward,
                "energy": energy,
                "raw_task_energy": energy,
                **constraint_meta,
            },
        )

        low_buffer.add(
            state=low_decision["state"],
            action=low_decision["action_u"],
            log_prob=low_decision["log_prob"],
            value=low_decision["value"],
            reward=low_reward,
            done=done,
            next_state=low_next_state,
            deadline_cost=constraint_meta["deadline_cost"],
            reliability_cost=constraint_meta["reliability_cost"],
            deadline_limit=constraint_meta["deadline_limit"],
            reliability_limit=constraint_meta["reliability_limit"],
            current_partial_makespan=constraint_meta["current_partial_makespan"],
            current_partial_log_risk=constraint_meta["current_partial_log_risk"],
            deadline_value=low_decision.get("deadline_value", 0.0),
            reliability_value=low_decision.get("reliability_value", 0.0),
            raw_task_energy=energy,
            workflow_id=high_decision["workflow_id"],
            workflow_done=constraint_meta["workflow_done"],
            actor_update_mask=low_decision.get("actor_update_mask", True),
            info={
                "workflow_id": high_decision["workflow_id"],
                "task_id": high_decision["task_id"],
                "server_index": high_decision["server_index"],
                "server_id": high_decision["server_id"],
                "action_u": low_decision["action_u"],
                "rho": low_decision["rho"],
                "rho_boundary": low_decision.get("rho_boundary", False),
                "rho_alpha": low_decision.get("rho_alpha"),
                "rho_beta": low_decision.get("rho_beta"),
                "energy_ref": energy_ref,
                "energy_actual": energy,
                "energy_saving": energy_ref - energy,
                "energy_norm": energy_norm,
                "high_energy_reward": high_energy_reward,
                "high_time_reward": high_reward,
                "energy": energy,
                "raw_task_energy": energy,
                "primary_energy_actual": primary_energy_actual,
                "energy_ref_rho1": energy_ref,
                "low_energy_reward": low_energy_reward,
                "low_local_deadline_penalty": local_deadline_penalty,
                "low_workflow_deadline_penalty": 0.0,
                "low_infeasible_penalty": infeasible_penalty,
                "low_total_reward": low_reward,
                "rho_safe": low_decision.get("rho_safe"),
                "rho_margin": float(low_decision.get("rho", 0.0)) - float(low_decision.get("rho_safe", 0.0)),
                "below_safe": float(low_decision.get("rho", 0.0)) + 1e-8 < float(low_decision.get("rho_safe", 0.0)),
                "low_action_feasible": low_decision.get("low_action_feasible"),
                "forced_max_frequency": low_decision.get("forced_max_frequency"),
                "actor_update_mask": low_decision.get("actor_update_mask", True),
                "low_slowdown": max(
                    float(event.get("primary_exec_time", event.get("exec_time", 0.0)) or 0.0)
                    - float(safety_metrics.get("execution_time_rho_1", 0.0)),
                    0.0,
                ),
                **{f"low_safe_{key}": value for key, value in safety_metrics.items()},
                **constraint_meta,
            },
        )

        finalize_completed_workflows(env, completed, high_buffer, low_buffer, previous_partials)
        completed_workflow_ids.update(str(item["workflow_id"]) for item in completed)
        last_results = next_results
        energy_normalizer.update(energy)

        total_reward += high_reward
        total_low_reward += low_reward
        total_energy += energy
        total_energy_ref += energy_ref
        state = next_state
        steps += 1

    final_results = env._build_workflow_results()
    final_completed = completed_workflows_from_step(last_results, final_results)
    finalize_completed_workflows(env, final_completed, high_buffer, low_buffer, previous_partials)
    completed_workflow_ids.update(
        str(item["workflow_id"])
        for item in final_completed
    )
    last_results = final_results
    apply_low_workflow_terminal_penalties(
        env,
        low_buffer,
        final_results,
        coef=float(reward_config.get("low_workflow_deadline_penalty_coef", 5.0)),
    )
    env_completed_count = len([item for item in final_results if item.get("finish_time") is not None])
    high_completed_count = len(high_buffer.completed_workflow_ids)
    low_completed_count = len(low_buffer.completed_workflow_ids)
    if env.done and not (
        env_completed_count == high_completed_count == low_completed_count
    ):
        raise RuntimeError(
            "Completed workflow bookkeeping mismatch: "
            f"env_completed={env_completed_count}, "
            f"high_completed={high_completed_count}, "
            f"low_completed={low_completed_count}"
        )

    final_metrics = update_if_ready(
        agent,
        high_buffer,
        low_buffer,
        rollout_length=rollout_length,
        force=True,
    )
    final_metrics.pop("_high_raw", None)
    final_metrics.pop("_low_raw", None)
    train_metrics.update(final_metrics)
    completed_count = len(completed_workflow_ids)
    train_metrics.update(
        {
            "mean_high_reward": total_reward / max(steps, 1),
            "mean_low_reward": total_low_reward / max(steps, 1),
            "total_high_reward": total_reward,
            "total_low_reward": total_low_reward,
            "mean_energy_ref": total_energy_ref / max(steps, 1),
            "mean_energy_actual": total_energy / max(steps, 1),
            "mean_energy_saving": (total_energy_ref - total_energy) / max(steps, 1),
            "completed_workflow_count": completed_count,
            "env_completed_workflow_count": env_completed_count,
            "high_completed_workflow_count": high_completed_count,
            "low_completed_workflow_count": low_completed_count,
            "energy_norm": float(energy_normalizer.value or 0.0),
        }
    )

    workflow_results = env._build_workflow_results()
    summary = compute_summary(workflow_results, env.event_log, env.servers)
    replica_energy_stats = compute_replica_energy_stats(env.event_log)
    return {
        "steps": steps,
        "done": env.done,
        "total_reward": total_reward,
        "total_energy": total_energy,
        "workflow_results": workflow_results,
        "event_log": list(env.event_log),
        "summary": summary,
        "train_metrics": train_metrics,
        "server_selection_counts": server_selection_counts,
        "backup_request_counts": backup_request_counts,
        "backup_execution_counts": backup_execution_counts,
        "replica_energy_stats": replica_energy_stats,
    }


def run_evaluation_episode(env, agent, reward_config, deterministic=True):
    state = env.reset()
    total_high_reward = 0.0
    total_low_reward = 0.0
    total_low_energy_reward = 0.0
    total_low_local_deadline_penalty = 0.0
    total_low_infeasible_penalty = 0.0
    total_energy_ref = 0.0
    total_energy = 0.0
    steps = 0
    max_steps = sum(env._workflow_task_count(item["workflow"]) for item in env.workflow_states) + 1000
    previous_partials = {}

    while not env.done and steps < max_steps:
        if not state["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            state = env.get_state()
            continue

        high_decision, low_decision, env_action, safety_metrics = select_hierarchical_action_with_low_safety(
            env,
            agent,
            state,
            deterministic=deterministic,
        )
        energy_ref = env.estimate_primary_energy_ref_rho1(env_action)
        next_state, env_reward, _, info = env.step(env_action)
        constraint_meta = workflow_constraint_metadata(
            env,
            high_decision["workflow_id"],
            previous_partials,
        )

        energy = extract_task_energy(info, env_reward)
        event = info.get("event", {})
        primary_energy_actual = float(event.get("primary_energy", energy))
        high_reward = compute_high_level_time_reward(constraint_meta["raw_deadline_increment"])
        low_energy_reward = compute_low_energy_saving_reward(energy_ref, primary_energy_actual)
        local_deadline_penalty = compute_low_local_deadline_penalty(
            event.get("finish_time", event.get("primary_aft", safety_metrics["estimated_ast"])),
            safety_metrics["task_latest_finish_time"],
            safety_metrics["workflow_relative_deadline"],
            coef=float(reward_config.get("low_local_deadline_penalty_coef", 10.0)),
        )
        infeasible_penalty = (
            float(reward_config.get("low_infeasible_penalty", 1.0))
            if low_decision.get("forced_max_frequency")
            else 0.0
        )
        low_reward = low_energy_reward - local_deadline_penalty - infeasible_penalty
        total_high_reward += high_reward
        total_low_reward += low_reward
        total_low_energy_reward += low_energy_reward
        total_low_local_deadline_penalty += local_deadline_penalty
        total_low_infeasible_penalty += infeasible_penalty
        total_energy_ref += energy_ref
        total_energy += energy
        state = next_state
        steps += 1

    workflow_results = env._build_workflow_results()
    summary = compute_summary(workflow_results, env.event_log, env.servers)
    return {
        "steps": steps,
        "done": env.done,
        "total_reward": total_high_reward + total_low_reward,
        "total_high_reward": total_high_reward,
        "total_low_reward": total_low_reward,
        "total_low_energy_reward": total_low_energy_reward,
        "total_low_local_deadline_penalty": total_low_local_deadline_penalty,
        "total_low_workflow_deadline_penalty": 0.0,
        "total_low_infeasible_penalty": total_low_infeasible_penalty,
        "total_primary_energy_ref_rho1": total_energy_ref,
        "total_energy": total_energy,
        "workflow_results": workflow_results,
        "event_log": list(env.event_log),
        "summary": summary,
    }


def run_evaluation(
    agent,
    eval_source,
    config,
    args,
    resources,
    sim_config,
    reward_config,
    base_seed,
    eval_episodes,
):
    rows = []
    for eval_index in range(eval_episodes):
        eval_seed = base_seed + eval_index
        workflows = _workflows_for_episode(eval_source, config, args, resources, eval_seed)
        env = build_env(workflows, resources, sim_config, seed=eval_seed)
        result = run_evaluation_episode(env, agent, reward_config)
        summary = result["summary"]
        rows.append(
            {
                "eval_index": eval_index + 1,
                "steps": result["steps"],
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
                "max_makespan": summary.get("max_workflow_makespan"),
                "compute_span": summary.get("avg_workflow_compute_span"),
                "max_compute_span": summary.get("max_workflow_compute_span"),
                "response_time": summary.get("avg_workflow_response_time"),
                "max_response_time": summary.get("max_workflow_response_time"),
                "waiting_time": summary.get("avg_workflow_waiting_time"),
                "max_waiting_time": summary.get("max_workflow_waiting_time"),
                "avg_task_total_time": summary.get("avg_task_total_time"),
                "max_task_total_time": summary.get("max_task_total_time"),
                "avg_task_waiting_time": summary.get("avg_task_waiting_time"),
                "max_task_waiting_time": summary.get("max_task_waiting_time"),
                "avg_task_communication_time": summary.get("avg_task_communication_time"),
                "max_task_communication_time": summary.get("max_task_communication_time"),
                "avg_task_execution_time": summary.get("avg_task_execution_time"),
                "deadline_violation": deadline_violation(summary),
                "deadline_success_count": summary.get("deadline_success_count"),
                "deadline_total_count": summary.get("deadline_total_count"),
                "deadline_success_rate": summary.get("deadline_success_rate"),
                "avg_workflow_reliability": summary.get("avg_workflow_reliability"),
                "reliability_violation": reliability_violation(summary),
                "reliability_success_count": summary.get("reliability_success_count"),
                "reliability_total_count": summary.get("reliability_total_count"),
                "reliability_success_rate": summary.get("reliability_success_rate"),
                "total_finished_workflows": summary.get("total_finished_workflows"),
                "avg_server_utilization": summary.get("avg_server_utilization"),
                "backup_replica_count": summary.get("backup_replica_count"),
                "backup_energy": summary.get("backup_energy"),
            }
        )
    return _average_eval_rows(rows)


def _average_eval_rows(rows):
    if not rows:
        return {}
    result = {"eval_episodes": len(rows)}
    keys = [
        "steps",
        "total_reward",
        "total_energy",
        "makespan",
        "max_makespan",
        "compute_span",
        "max_compute_span",
        "response_time",
        "max_response_time",
        "waiting_time",
        "max_waiting_time",
        "avg_task_total_time",
        "max_task_total_time",
        "avg_task_waiting_time",
        "max_task_waiting_time",
        "avg_task_communication_time",
        "max_task_communication_time",
        "avg_task_execution_time",
        "deadline_violation",
        "deadline_success_count",
        "deadline_total_count",
        "deadline_success_rate",
        "avg_workflow_reliability",
        "reliability_violation",
        "reliability_success_count",
        "reliability_total_count",
        "reliability_success_rate",
        "total_finished_workflows",
        "avg_server_utilization",
        "backup_replica_count",
        "backup_energy",
    ]
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) not in (None, "n/a")
        ]
        result[key] = sum(values) / len(values) if values else None
    return result


def log_eval_metrics(episode, metrics):
    deadline_text = (
        "n/a"
        if metrics.get("deadline_total_count") in (None, 0)
        else (
            f"{fmt_metric(metrics.get('deadline_success_count'))}/"
            f"{fmt_metric(metrics.get('deadline_total_count'))} met "
            f"(viol={fmt_metric(metrics.get('deadline_violation'))}, "
            f"rate={fmt_metric(metrics.get('deadline_success_rate'))})"
        )
    )
    print(
        f"[eval] ep={episode} eval_eps={metrics.get('eval_episodes', 'n/a')} "
        f"reward={fmt_metric(metrics.get('total_reward'))} "
        f"energy={fmt_metric(metrics.get('total_energy'))}J "
        f"finished={fmt_metric(metrics.get('total_finished_workflows'))} "
        f"deadline={deadline_text} "
        f"makespan={fmt_metric(metrics.get('makespan'))}s "
        f"wait={fmt_metric(metrics.get('waiting_time'))}s "
        f"util={fmt_metric(metrics.get('avg_server_utilization'))}"
    )


def deadline_violation(summary):
    if "deadline_violation_count" in summary:
        return summary["deadline_violation_count"]
    if "deadline_success_rate" in summary and summary["deadline_success_rate"] is not None:
        return 1.0 - float(summary["deadline_success_rate"])
    return "n/a"


def reliability_violation(summary):
    if "reliability_violation_count" in summary:
        return summary["reliability_violation_count"]
    if "reliability_success_rate" in summary and summary["reliability_success_rate"] is not None:
        return 1.0 - float(summary["reliability_success_rate"])
    return "n/a"


def _strip_metric_prefix(metrics, prefix):
    result = {}
    marker = f"{prefix}_"
    for key, value in metrics.items():
        if key.startswith(marker):
            result[key[len(marker) :]] = value
    return result


def train_hierarchical_ppo(env, config):
    agent = HierarchicalPPOAgent(
        config=config.get("agent", {}),
        seed=int(config.get("seed", 42)),
    )
    rollout_length = int(config.get("rollout_length", 128))
    num_episodes = int(config.get("num_episodes", config.get("episodes", 1000)))
    trace_config = {
        "enabled": bool(config.get("trace_decisions", False)),
        "limit": config.get("trace_decision_limit"),
    }
    latest_result = None

    for episode in range(1, num_episodes + 1):
        latest_result = run_training_episode(
            env,
            agent,
            rollout_length=rollout_length,
            reward_config=config,
            episode=episode,
            trace_config=trace_config,
        )
        high_metrics = _strip_metric_prefix(latest_result["train_metrics"], "high")
        low_metrics = _strip_metric_prefix(latest_result["train_metrics"], "low")
        log_episode_metrics(
            episode=episode,
            reward=latest_result["total_reward"],
            energy=latest_result["total_energy"],
            steps=latest_result["steps"],
            high_metrics=high_metrics,
            low_metrics=low_metrics,
            env=env,
            server_selection_counts=latest_result["server_selection_counts"],
            backup_request_counts=latest_result["backup_request_counts"],
            backup_execution_counts=latest_result["backup_execution_counts"],
            replica_energy_stats=latest_result["replica_energy_stats"],
        )

    return agent


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    training_cfg = dict(config.get("training", {}))
    seed = args.seed if args.seed is not None else int(training_cfg.get("seed", 42))
    episodes = (
        args.train_episodes
        or args.episodes
        or int(training_cfg.get("train_episodes", training_cfg.get("episodes", 100)))
    )
    rollout_length = args.rollout_length or int(training_cfg.get("rollout_length", 128))
    save_every = args.save_every if args.save_every is not None else int(training_cfg.get("save_every", 10))
    eval_interval = (
        args.eval_interval
        if args.eval_interval is not None
        else int(training_cfg.get("eval_interval", 50))
    )
    eval_episodes = (
        args.eval_episodes
        if args.eval_episodes is not None
        else int(training_cfg.get("eval_episodes", 5))
    )
    trace_config = {
        "enabled": bool(args.trace_decisions or training_cfg.get("trace_decisions", False)),
        "limit": (
            args.trace_decision_limit
            if args.trace_decision_limit is not None
            else training_cfg.get("trace_decision_limit")
        ),
    }
    checkpoint_root = Path(training_cfg.get("checkpoint_dir", "results/hierarchical_ppo"))
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root
    run_name = training_cfg.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = checkpoint_root / str(run_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sim_config = dict(config.get("simulation", {}))
    if "low_deadline_safety_margin_ratio" in training_cfg:
        sim_config["low_deadline_safety_margin_ratio"] = float(
            training_cfg["low_deadline_safety_margin_ratio"]
        )
    _apply_constraint_overrides(sim_config, args)
    if getattr(args, "rreq", None) is not None:
        sim_config["default_reliability"] = float(args.rreq)
    configured_use_p3o = bool(training_cfg.get("use_p3o", training_cfg.get("use_p3o_constraints", True)))
    use_p3o = configured_use_p3o if args.use_p3o is None else bool(args.use_p3o)
    p3o_config = {
        "use_p3o": use_p3o,
        "use_p3o_constraints": use_p3o,
        "reward_gamma": float(training_cfg.get("reward_gamma", 0.99)),
        "reward_gae_lambda": float(training_cfg.get("reward_gae_lambda", 0.95)),
        "cost_gamma": float(training_cfg.get("cost_gamma", 1.0)),
        "cost_gae_lambda": float(training_cfg.get("cost_gae_lambda", 1.0)),
        "kappa_deadline": (
            args.kappa_deadline
            if args.kappa_deadline is not None
            else float(training_cfg.get("kappa_deadline", 5.0))
        ),
        "kappa_reliability": (
            args.kappa_reliability
            if args.kappa_reliability is not None
            else float(training_cfg.get("kappa_reliability", 5.0))
        ),
        "cost_value_coef": (
            args.cost_value_coef
            if args.cost_value_coef is not None
            else float(training_cfg.get("cost_value_coef", 1.0))
        ),
        "deadline_gate_margin_ratio": float(training_cfg.get("deadline_gate_margin_ratio", 0.05)),
        "deadline_severity_coef": float(training_cfg.get("deadline_severity_coef", 2.0)),
        "deadline_violation_clip": float(training_cfg.get("deadline_violation_clip", 1.0)),
        "reliability_gate_margin_ratio": float(training_cfg.get("reliability_gate_margin_ratio", 0.05)),
        "reliability_severity_coef": float(training_cfg.get("reliability_severity_coef", 2.0)),
        "reliability_violation_clip_ratio": float(training_cfg.get("reliability_violation_clip_ratio", 1.0)),
        "server_balanced_minibatch": False,
    }
    legacy_deadline_limit = (
        args.deadline_cost_limit
        if args.deadline_cost_limit is not None
        else training_cfg.get("deadline_cost_limit")
    )
    legacy_reliability_limit = (
        args.reliability_cost_limit
        if args.reliability_cost_limit is not None
        else training_cfg.get("reliability_cost_limit")
    )
    if legacy_deadline_limit is not None:
        p3o_config["deadline_cost_limit"] = float(legacy_deadline_limit)
    if legacy_reliability_limit is not None:
        p3o_config["reliability_cost_limit"] = float(legacy_reliability_limit)
    reward_config = {
        **training_cfg,
        "use_p3o_constraints": use_p3o,
    }

    resources = load_resources(
        str(PROJECT_ROOT / config["resources"]["path"])
        if not Path(config["resources"]["path"]).is_absolute()
        else config["resources"]["path"],
        num_edge=config["resources"]["num_edge"],
        num_cloud=config["resources"]["num_cloud"],
        seed=seed,
    )

    workflow_source = _prepare_workflow_source(config, args, resources, seed)
    eval_workflow_source = _prepare_eval_workflow_source(config, args, resources, seed)
    dataset_type = workflow_source["type"]

    agent_config = copy.deepcopy(config.get("agent", config.get("scheduler", {}).get("params", {}).get("agent", {})))
    if "shared" not in agent_config:
        state_cfg = agent_config.get("state", config.get("scheduler", {}).get("params", {}).get("state", {}))
        shared = dict(agent_config)
        shared["state"] = dict(state_cfg)
        agent_config = {"shared": shared}
    agent_config.setdefault("shared", {}).setdefault("device", "auto")
    agent_config["shared"].setdefault("state", {})
    agent_config["shared"]["state"]["max_ready_tasks"] = infer_ready_task_capacity(
        config,
        args,
        workflow_source,
    )
    agent_config["shared"]["state"]["max_servers"] = infer_server_count(resources)
    agent_config["shared"]["state"]["time_horizon"] = float(
        sim_config.get(
            "time_horizon",
            agent_config["shared"]["state"].get("time_horizon", 3600.0),
        )
    )
    if getattr(args, "debug_high_state", False):
        agent_config["shared"]["state"]["debug_high_state"] = True
    agent_config.setdefault("low", {}).setdefault(
        "rho_min",
        float(sim_config.get("rho_min", 0.1)),
    )
    agent_config.setdefault("low", {})["reward_gamma"] = 0.0
    agent_config.setdefault("low", {})["gamma"] = 0.0
    agent_config.setdefault("low", {})["use_p3o"] = False
    agent_config.setdefault("low", {})["server_balanced_minibatch"] = True
    agent_config.setdefault("high", {}).update(p3o_config)

    effective_config = _build_effective_config(
        config,
        args,
        seed=seed,
        episodes=episodes,
        rollout_length=rollout_length,
        save_every=save_every,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        workflow_source=workflow_source,
        eval_workflow_source=eval_workflow_source,
        agent_config=agent_config,
    )
    effective_config_path = checkpoint_dir / "effective_config.json"
    with effective_config_path.open("w", encoding="utf-8") as file:
        json.dump(effective_config, file, ensure_ascii=False, indent=2)

    agent = HierarchicalPPOAgent(agent_config, seed=seed)
    print(describe_runtime_devices(agent.high_agent.device))
    log_cpu_device_diagnostic(agent)
    print(
        "Hierarchical PPO train: "
        f"episodes={episodes}, update_mode=episode_end, "
        f"rollout_length={rollout_length} (compatibility_only), "
        f"dataset={workflow_source['description']}"
    )
    print(
        "Hierarchical PPO state: "
        f"max_ready_tasks={agent_config['shared']['state']['max_ready_tasks']}, "
        f"max_servers={agent_config['shared']['state']['max_servers']}"
    )
    print(
        "Hierarchical PPO eval: "
        f"interval={eval_interval}, eval_episodes={eval_episodes}, "
        f"dataset={eval_workflow_source['description']}"
    )

    history = []
    eval_history = []
    episode_detail_dir = checkpoint_dir / "episode_details"
    episode_detail_dir.mkdir(parents=True, exist_ok=True)
    for episode in range(1, episodes + 1):
        episode_seed = seed + episode
        workflows = _workflows_for_episode(
            workflow_source,
            config,
            args,
            resources,
            episode_seed,
        )

        env = build_env(workflows, resources, sim_config, seed=episode_seed)
        result = run_training_episode(
            env,
            agent,
            rollout_length=rollout_length,
            reward_config=reward_config,
            episode=episode,
            trace_config=trace_config,
        )
        metrics = result["train_metrics"]
        summary = result["summary"]
        replica_energy_stats = result["replica_energy_stats"]
        server_queue_stats = compute_server_queue_stats(result["event_log"])
        server_queue_summary = summarize_server_queue_stats(server_queue_stats)
        workflow_deadlines = workflow_deadline_records(result["workflow_results"])
        episode_detail_path = episode_detail_dir / f"episode_{episode:06d}.json"
        episode_detail = {
            "episode": episode,
            "seed": episode_seed,
            "steps": result["steps"],
            "done": result["done"],
            "summary": summary,
            "train_metrics": metrics,
            "workflow_deadlines": workflow_deadlines,
            "workflow_results": result["workflow_results"],
            "event_log": result["event_log"],
            "server_selection_counts": result["server_selection_counts"],
            "backup_request_counts": result["backup_request_counts"],
            "backup_execution_counts": result["backup_execution_counts"],
            "server_queue_stats": server_queue_stats,
            "replica_energy_stats": replica_energy_stats,
        }
        with episode_detail_path.open("w", encoding="utf-8") as file:
            json.dump(episode_detail, file, ensure_ascii=False, indent=2)
        row = {
            "episode": episode,
            "seed": episode_seed,
            "total_reward": result["total_reward"],
            "total_energy": summary.get("total_energy", result["total_energy"]),
            "makespan": summary.get("avg_workflow_makespan"),
            "max_makespan": summary.get("max_workflow_makespan"),
            "compute_span": summary.get("avg_workflow_compute_span"),
            "max_compute_span": summary.get("max_workflow_compute_span"),
            "response_time": summary.get("avg_workflow_response_time"),
            "max_response_time": summary.get("max_workflow_response_time"),
            "waiting_time": summary.get("avg_workflow_waiting_time"),
            "max_waiting_time": summary.get("max_workflow_waiting_time"),
            "avg_task_total_time": summary.get("avg_task_total_time"),
            "max_task_total_time": summary.get("max_task_total_time"),
            "avg_task_waiting_time": summary.get("avg_task_waiting_time"),
            "max_task_waiting_time": summary.get("max_task_waiting_time"),
            "avg_task_communication_time": summary.get("avg_task_communication_time"),
            "max_task_communication_time": summary.get("max_task_communication_time"),
            "avg_task_execution_time": summary.get("avg_task_execution_time"),
            "total_finished_workflows": summary.get("total_finished_workflows"),
            "deadline_success_count": summary.get("deadline_success_count"),
            "deadline_total_count": summary.get("deadline_total_count"),
            "deadline_violation": deadline_violation(summary),
            "deadline_success_rate": summary.get("deadline_success_rate"),
            "avg_workflow_reliability": summary.get("avg_workflow_reliability"),
            "reliability_success_count": summary.get("reliability_success_count"),
            "reliability_total_count": summary.get("reliability_total_count"),
            "reliability_violation": reliability_violation(summary),
            "reliability_success_rate": summary.get("reliability_success_rate"),
            "avg_server_utilization": summary.get("avg_server_utilization"),
            "backup_replica_count": summary.get("backup_replica_count"),
            "backup_energy": summary.get("backup_energy"),
            "primary_energy": replica_energy_stats["primary_energy"],
            "primary_task_count": replica_energy_stats["primary_task_count"],
            "backup_task_count": replica_energy_stats["backup_task_count"],
            "backup_requested_task_count": sum(result["backup_request_counts"].values()),
            "backup_started_task_count": replica_energy_stats["backup_task_count"],
            "replica_total_energy": replica_energy_stats["replica_total_energy"],
            **server_queue_summary,
            "server_queue_stats": server_queue_stats,
            "server_selection_counts": result["server_selection_counts"],
            "backup_request_counts": result["backup_request_counts"],
            "backup_execution_counts": result["backup_execution_counts"],
            "workflow_deadlines": workflow_deadlines,
            "episode_detail_path": str(episode_detail_path.relative_to(checkpoint_dir)),
            **metrics,
        }
        history.append(row)
        log_episode_metrics(
            episode=episode,
            reward=result["total_reward"],
            energy=row["total_energy"],
            steps=result["steps"],
            high_metrics=_strip_metric_prefix(metrics, "high"),
            low_metrics=_strip_metric_prefix(metrics, "low"),
            env=env,
            server_selection_counts=result["server_selection_counts"],
            backup_request_counts=result["backup_request_counts"],
            backup_execution_counts=result["backup_execution_counts"],
            replica_energy_stats=replica_energy_stats,
        )

        if eval_interval > 0 and episode % eval_interval == 0:
            eval_metrics = run_evaluation(
                agent=agent,
                eval_source=eval_workflow_source,
                config=config,
                args=args,
                resources=resources,
                sim_config=sim_config,
                reward_config=reward_config,
                base_seed=seed + 100000 + episode * max(eval_episodes, 1),
                eval_episodes=eval_episodes,
            )
            eval_row = {"episode": episode, **eval_metrics}
            eval_history.append(eval_row)
            log_eval_metrics(episode, eval_metrics)

        if save_every > 0 and episode % save_every == 0:
            checkpoint_path = checkpoint_dir / f"hierarchical_ppo_ep{episode}.pt"
            agent.save(str(checkpoint_path), extra={"episode": episode, "config": effective_config})
            history_path = checkpoint_dir / "train_history.json"
            with history_path.open("w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2)
            eval_history_path = checkpoint_dir / "eval_history.json"
            with eval_history_path.open("w", encoding="utf-8") as file:
                json.dump(eval_history, file, ensure_ascii=False, indent=2)
            print(f"checkpoint saved: {checkpoint_path}")

    final_checkpoint = checkpoint_dir / "hierarchical_ppo_final.pt"
    agent.save(str(final_checkpoint), extra={"episode": episodes, "config": effective_config})
    history_path = checkpoint_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    eval_history_path = checkpoint_dir / "eval_history.json"
    with eval_history_path.open("w", encoding="utf-8") as file:
        json.dump(eval_history, file, ensure_ascii=False, indent=2)
    print(f"final checkpoint saved: {final_checkpoint}")
    print(f"effective config saved: {effective_config_path}")
    print(f"history saved: {history_path}")
    print(f"eval history saved: {eval_history_path}")


if __name__ == "__main__":
    main()
