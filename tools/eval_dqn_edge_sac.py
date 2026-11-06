from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.D3QN.hierarchical_d3qn_agent import HierarchicalD3QNAgent
from src.baselines.dqn_edge import DQNEdgeConfig
from src.simulator.metrics import compute_summary
from tools.hierarchical_common import (
    build_env,
    build_resources,
    build_workflows,
    load_json,
)
from tools.train_dqn_edge import build_dqn_edge_runtime
from tools.train_hierarchical_d3qn import configure_d3qn_agent_state


DEFAULT_DQN_CHECKPOINT = (
    "results/dqn_edge/20260824_091640/best_checkpoint.pt"
)
DEFAULT_LOW_CHECKPOINT = (
    "results/hierarchical_d3qn_sac_server_ctde/20260824_094111/"
    "checkpoint_ep000540_low.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one trained DQN-Edge upper policy with rho=1 and SAC."
    )
    parser.add_argument("--dqn-config", default="configs/dqn_edge_small_20edge_6cloud.json")
    parser.add_argument("--low-config", default="configs/v2_small_20edge_6cloud_dvfs_v3_future_esp.json")
    parser.add_argument("--dqn-checkpoint", default=DEFAULT_DQN_CHECKPOINT)
    parser.add_argument("--low-checkpoint", default=DEFAULT_LOW_CHECKPOINT)
    parser.add_argument("--seed-start", type=int, default=242000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def _resolve(path):
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def _load_dqn_runtime(config, resources, checkpoint, device):
    payload = torch.load(
        str(checkpoint), map_location=device, weights_only=False
    )
    values = dict(payload.get("config", config["agent"]["dqn_edge"]))
    values["device"] = device
    agent_config = DQNEdgeConfig.from_mapping(values)
    runtime = build_dqn_edge_runtime(
        config,
        resources,
        agent_config,
        variant=payload.get("variant", config["agent"].get("variant")),
        state_config=payload.get("state_config"),
        objective_config=payload.get("objective_config"),
    )
    runtime["agent"].load_checkpoint(payload)
    return runtime, payload


def _load_low_bundle(config, resources, checkpoint, seed):
    agent_config = configure_d3qn_agent_state(config, resources)
    bundle = HierarchicalD3QNAgent(agent_config, seed=seed)
    if bundle.low_agent is None or bundle.low_algorithm != "sac":
        raise RuntimeError("The lower checkpoint requires a configured SAC agent.")
    payload = torch.load(
        str(checkpoint), map_location=bundle.low_agent.device, weights_only=False
    )
    low_state = payload.get("low_sac_state_dict")
    if low_state is None:
        low_state = payload.get("low_model_state_dict")
    if low_state is None:
        raise RuntimeError(f"Checkpoint has no SAC state: {checkpoint}")
    bundle.low_agent.load_state_dict(low_state)
    bundle.low_agent.actor.eval()
    bundle.low_agent.critic_1.eval()
    bundle.low_agent.critic_2.eval()
    return bundle, payload


def _selected_task_state(env, state, task):
    selected_key = (str(task["workflow_id"]), int(task["task_id"]))
    first = state["ready_tasks"][0]
    first_key = (str(first["workflow_id"]), int(first["task_id"]))
    if selected_key == first_key:
        return state
    result = dict(state)
    result["ready_tasks"] = [
        task,
        *[
            row
            for row in state["ready_tasks"]
            if (str(row["workflow_id"]), int(row["task_id"])) != selected_key
        ],
    ]
    result["decision_context"] = env.build_hierarchical_decision_context(
        task["workflow_id"], task["task_id"]
    )
    return result


def run_episode(env, scheduler, low_bundle, *, use_sac):
    env.reset()
    scheduler.reward_function.reset_episode()
    scheduler.state_builder.reset_episode()
    rhos = []
    steps = 0
    max_steps = (
        sum(env._workflow_task_count(row["workflow"]) for row in env.workflow_states)
        + 1000
    )
    while not env.done and steps < max_steps:
        state = env.get_state()
        if not state["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            continue
        (
            _vector,
            _context,
            _action_id,
            _weights,
            task,
            _workflow_state,
            action,
        ) = scheduler.select_action(env, training=False)
        decision_state = _selected_task_state(env, state, task)
        if use_sac:
            observation = low_bundle.high_agent.build_high_state(decision_state)
            server_id = int(action["server_id"])
            server_indices = [
                index
                for index, server in enumerate(observation.servers)
                if int(server["id"]) == server_id
            ]
            if len(server_indices) != 1:
                raise RuntimeError(f"Cannot map selected server {server_id}.")
            safety_metrics = decision_state["decision_context"][
                "server_safety_metrics"
            ][str(server_id)]
            low_decision = low_bundle.low_agent.select_frequency(
                observation,
                task_index=0,
                server_index=server_indices[0],
                deterministic=True,
                safety_metrics=safety_metrics,
            )
            rho = float(low_decision["rho"])
        else:
            rho = 1.0
        action = {**action, "rho": rho}
        next_state, _reward, done, _info = env.step(action)
        rhos.append(rho)
        steps += 1
        while not done and not next_state["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            next_state, done = env.get_state(), env.done
    if steps >= max_steps and not env.done:
        raise RuntimeError("DQN-Edge + SAC episode exceeded maximum steps.")
    summary = compute_summary(
        env._build_workflow_results(), env.event_log, env.servers
    )
    return {
        **summary,
        "rho_mean": float(np.mean(rhos)) if rhos else 1.0,
        "rho_std": float(np.std(rhos)) if rhos else 0.0,
        "steps": steps,
    }


def _aggregate(rows):
    keys = (
        "total_energy",
        "deadline_success_rate",
        "reliability_success_rate",
        "avg_workflow_response_time",
        "rho_mean",
    )
    return {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "std": float(np.std([float(row[key]) for row in rows])),
        }
        for key in keys
    }


def evaluate(args):
    dqn_config = load_json(args.dqn_config)
    low_config = load_json(args.low_config)
    resource_seed = int(dqn_config["training"].get("seed", 42))
    resources = build_resources(dqn_config, resource_seed)
    dqn_checkpoint = _resolve(args.dqn_checkpoint)
    low_checkpoint = _resolve(args.low_checkpoint)
    rho1_runtime, dqn_payload = _load_dqn_runtime(
        dqn_config, resources, dqn_checkpoint, args.device
    )
    sac_runtime, _ = _load_dqn_runtime(
        dqn_config, resources, dqn_checkpoint, args.device
    )
    low_bundle, low_payload = _load_low_bundle(
        low_config, resources, low_checkpoint, resource_seed
    )
    count = min(args.num_seeds, 2) if args.smoke_test else args.num_seeds
    rho1_rows = []
    sac_rows = []
    for seed in range(args.seed_start, args.seed_start + count):
        workflows = build_workflows(
            dqn_config, resources, seed, smoke_test=args.smoke_test
        )
        simulation = {
            **dqn_config.get("simulation", {}),
            "enable_backup_duplication": False,
        }
        rho1_env = build_env(workflows, resources, simulation, seed)
        sac_env = build_env(workflows, resources, simulation, seed)
        rho1 = run_episode(
            rho1_env, rho1_runtime["scheduler"], low_bundle, use_sac=False
        )
        sac = run_episode(
            sac_env, sac_runtime["scheduler"], low_bundle, use_sac=True
        )
        rho1["seed"] = seed
        sac["seed"] = seed
        rho1_rows.append(rho1)
        sac_rows.append(sac)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "rho1_energy": rho1["total_energy"],
                    "sac_energy": sac["total_energy"],
                    "rho1_deadline": rho1["deadline_success_rate"],
                    "sac_deadline": sac["deadline_success_rate"],
                    "sac_rho": sac["rho_mean"],
                },
                ensure_ascii=False,
            )
        )
    rho1_aggregate = _aggregate(rho1_rows)
    sac_aggregate = _aggregate(sac_rows)
    rho1_energy = rho1_aggregate["total_energy"]["mean"]
    sac_energy = sac_aggregate["total_energy"]["mean"]
    result = {
        "algorithm": "dqn_edge_upper_with_ctde_sac_lower",
        "dqn_checkpoint": str(dqn_checkpoint.resolve()),
        "dqn_checkpoint_episode": int(dqn_payload.get("episode", 0)),
        "low_checkpoint": str(low_checkpoint.resolve()),
        "low_checkpoint_episode": int(low_payload.get("episode", 0)),
        "seed_start": args.seed_start,
        "num_seeds": count,
        "rho1": {"aggregate": rho1_aggregate, "rows": rho1_rows},
        "sac": {"aggregate": sac_aggregate, "rows": sac_rows},
        "paired": {
            "energy_delta_j": float(sac_energy - rho1_energy),
            "energy_saving_ratio": float(
                (rho1_energy - sac_energy) / max(rho1_energy, 1e-8)
            ),
            "deadline_delta": float(
                sac_aggregate["deadline_success_rate"]["mean"]
                - rho1_aggregate["deadline_success_rate"]["mean"]
            ),
            "response_time_delta_s": float(
                sac_aggregate["avg_workflow_response_time"]["mean"]
                - rho1_aggregate["avg_workflow_response_time"]["mean"]
            ),
        },
    }
    output = (
        _resolve(args.output)
        if args.output
        else dqn_checkpoint.parent / "eval_with_ctde_sac_ep540_20seed.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["paired"], ensure_ascii=False))
    print(f"output={output}")
    return result


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
