"""Break each task's wall-clock into ready-wait / contention / comm / exec.

Answers one question: how much of a task's life is spent waiting because the
*chosen* server was already busy, as opposed to waiting for data, moving data,
or actually computing.  The contention share is the quantity the beta_q term
in the Deadline head is meant to push down.

The four components are disjoint and reconstruct the task's span from the
moment its inputs are ready to the moment it finishes:

    data_ready --[ready_wait]--> base_ready --[contention]--> ast
               --[exec]--> aft                    (comm is inside ready_wait's
                                                   upstream, reported apart)

* ``ready_wait``  = base_ready_time - data_ready_time.  The task sat in the
  global ready queue; it would have waited wherever it was sent, so no action
  can be blamed for it.
* ``contention``  = ast - base_ready_time.  The server the action picked was
  busy.  This one *is* the action's fault, and is what beta_q charges.
* ``comm``        = data_ready_time - dependency_ready_time.  Input transfer.
* ``exec``        = the execution itself, which rho scales.

Also reports how concentrated the placement is, because contention and
crowding are the same phenomenon seen from two ends.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from tools.train_coupled_hierarchical_d3qn_sac import select_agent_class


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=642000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--fixed-rho-one",
        action="store_true",
        help="Evaluate the upper level alone; contention is rho-independent, "
        "so this isolates placement from DVFS stretching.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def _f(event, key, default=0.0):
    value = event.get(key)
    return default if value is None else float(value)


def episode_breakdown(event_log):
    """Aggregate the primary replicas of one episode."""
    ready_wait = contention = comm = exec_time = 0.0
    contended_tasks = 0
    tasks = 0
    per_server = Counter()
    contention_per_server = Counter()
    for event in event_log:
        if str(event.get("replica_role", "primary")) != "primary":
            continue
        # ``event_log`` rows use unsuffixed keys; the reward's step event uses
        # the ``*_primary`` suffixed copies of the same numbers.
        ast = _f(event, "start_time")
        base_ready = _f(event, "base_ready_time")
        data_ready = _f(event, "data_ready_time")
        dependency_ready = _f(event, "dependency_ready_time")
        task_contention = max(0.0, ast - base_ready)
        tasks += 1
        ready_wait += max(0.0, base_ready - data_ready)
        contention += task_contention
        comm += max(0.0, data_ready - dependency_ready)
        exec_time += _f(event, "exec_time")
        if task_contention > 1e-9:
            contended_tasks += 1
        server = event.get("server_id")
        per_server[server] += 1
        contention_per_server[server] += task_contention
    total = ready_wait + contention + comm + exec_time
    return {
        "tasks": tasks,
        "ready_wait": ready_wait,
        "contention": contention,
        "comm": comm,
        "exec": exec_time,
        "total": total,
        "contended_tasks": contended_tasks,
        "per_server": per_server,
        "contention_per_server": contention_per_server,
    }


def _gini(counts):
    values = np.sort(np.asarray(list(counts), dtype=np.float64))
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    index = np.arange(1, values.size + 1)
    return float(
        (2.0 * (index * values).sum()) / (values.size * values.sum())
        - (values.size + 1.0) / values.size
    )


def main():
    args = parse_args()
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    agent_class = select_agent_class(config)
    with redirect_stdout(StringIO()):
        resources = build_resources(
            config, int(config.get("training", {}).get("seed", 42))
        )
    objective = HierarchicalObjective(config.get("objective", {}))
    use_low = not args.fixed_rho_one

    agent_config = configure_agent_state(config, resources, smoke_test=False)
    agent_config["use_low_agent"] = bool(use_low)
    agent = agent_class(agent_config, seed=args.seed_start)
    agent.load_checkpoint(Path(args.checkpoint).resolve())
    agent.high_agent.epsilon = 0.0

    totals = Counter()
    per_server = Counter()
    contention_per_server = Counter()
    rows = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        with redirect_stdout(StringIO()):
            workflows = build_workflows(config, resources, seed, smoke_test=False)
            env = build_env(
                workflows, resources, dict(config.get("simulation", {})), seed
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
                fixed_low_rho_one=not use_low,
                high_algorithm="d3qn",
            )
        breakdown = episode_breakdown(env.event_log)
        metrics = result["metrics"]
        for key in ("tasks", "ready_wait", "contention", "comm", "exec", "total",
                    "contended_tasks"):
            totals[key] += breakdown[key]
        per_server.update(breakdown["per_server"])
        contention_per_server.update(breakdown["contention_per_server"])
        rows.append(
            {
                "eval_seed": seed,
                "contention_share": breakdown["contention"] / max(breakdown["total"], 1e-12),
                "mean_contention_seconds": breakdown["contention"] / max(breakdown["tasks"], 1),
                "contended_task_fraction": breakdown["contended_tasks"] / max(breakdown["tasks"], 1),
                "total_energy": float(metrics.get("total_energy", 0.0)),
                "deadline_success_rate": float(metrics.get("deadline_success_rate", 0.0)),
                "reliability_success_rate": float(metrics.get("reliability_success_rate", 0.0)),
                "rho_mean": float(metrics.get("rho_mean", 0.0)),
            }
        )

    total = max(float(totals["total"]), 1e-12)
    servers_total = len(resources["servers"]) if isinstance(resources, dict) else len(resources)
    counts = sorted(per_server.values(), reverse=True)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seeds": f"{args.seed_start}-{args.seed_start + args.num_seeds - 1}",
        "mode": "rho1" if args.fixed_rho_one else "sac",
        "tasks": int(totals["tasks"]),
        "share_ready_wait": float(totals["ready_wait"]) / total,
        "share_contention": float(totals["contention"]) / total,
        "share_comm": float(totals["comm"]) / total,
        "share_exec": float(totals["exec"]) / total,
        "mean_contention_seconds": float(totals["contention"]) / max(int(totals["tasks"]), 1),
        "contended_task_fraction": float(totals["contended_tasks"]) / max(int(totals["tasks"]), 1),
        "servers_used": len(per_server),
        "servers_total": servers_total,
        "placement_gini": _gini(per_server.values()),
        "top1_placement_share": (counts[0] / sum(counts)) if counts else 0.0,
        "top5_placement_share": (sum(counts[:5]) / sum(counts)) if counts else 0.0,
        "mean_total_energy": float(np.mean([r["total_energy"] for r in rows])),
        "mean_deadline_success_rate": float(np.mean([r["deadline_success_rate"] for r in rows])),
        "mean_reliability_success_rate": float(np.mean([r["reliability_success_rate"] for r in rows])),
        "mean_rho": float(np.mean([r["rho_mean"] for r in rows])),
    }

    print(f"[{summary['mode']}] {summary['seeds']}  tasks={summary['tasks']}")
    print(
        "  time split:  ready_wait={share_ready_wait:.4f}  "
        "contention={share_contention:.4f}  comm={share_comm:.4f}  "
        "exec={share_exec:.4f}".format(**summary)
    )
    print(
        "  contention:  mean={mean_contention_seconds:.3f}s/task  "
        "tasks_that_waited={contended_task_fraction:.4f}".format(**summary)
    )
    print(
        "  placement:   servers_used={servers_used}/{servers_total}  "
        "gini={placement_gini:.4f}  top1={top1_placement_share:.4f}  "
        "top5={top5_placement_share:.4f}".format(**summary)
    )
    print(
        "  outcome:     E={mean_total_energy:.1f}J  "
        "deadline={mean_deadline_success_rate:.4f}  "
        "rel={mean_reliability_success_rate:.4f}  "
        "rho={mean_rho:.4f}".format(**summary)
    )

    if args.output:
        payload = {
            "summary": summary,
            "rows": rows,
            "placement_counts": {str(k): int(v) for k, v in per_server.most_common()},
            "contention_seconds_by_server": {
                str(k): float(v) for k, v in contention_per_server.most_common()
            },
        }
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
