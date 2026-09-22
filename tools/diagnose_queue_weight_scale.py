"""Pick beta_q from measurement instead of guessing, and check it travels.

The Deadline head's target is currently ``Sum_t Delta-C_t / D_w`` plus a
terminal miss credit distributed proportionally to the same Delta-C.  Adding
``beta_q * q_t / (N_w * D_w)`` closes the blind spot where a task that is not
its workflow's longest pole contributes exactly zero.  But *how much* to add
is a scale question, and this run answers it without training anything:

1. Measure both terms on a trained rollout at ``beta_q = 0`` -- the diagnostic
   ``high_queue_delay_normalized`` is populated regardless of the weight, so
   the queue mass is observable before it is ever charged.

2. Solve for the ``beta_q`` that puts the queue term at a target share of the
   dense cost.  Too small and the term is unlearnable noise; too large and the
   head stops tracking the makespan it is actually constrained on.

3. Run the same solve on every scale given and compare.  Both terms are built
   to be O(1) per workflow -- Delta-C telescopes to ``C_w / D_w ~ 1`` and the
   queue term is divided by ``N_w`` -- so a portable beta_q should come out
   similar on small and middle.  If it does not, the difference is the
   phenomenon (middle queues more), not the normalization, and the two scales
   need their share targets matched rather than their raw weights copied.

4. Report the knock-on effect on lambda.  Growing the Deadline target grows
   ``Q_D``, and the selector ranks on ``Q_E - lambda_D * Q_D``, so adding the
   queue term implicitly amplifies lambda by the same factor.  Holding the
   energy/Deadline balance fixed means dividing lambda by that factor at the
   same time -- otherwise the change is confounded with a lambda increase,
   which the sweep already showed to be harmful.
"""
from __future__ import annotations

import argparse
import json
import sys
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


EPS = 1e-9
DEFAULT_SHARES = "0.10,0.20,0.30"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="LABEL:CONFIG:CHECKPOINT",
        help="Repeatable. Colon-separated, e.g. "
        "medium:configs/v2_medium....json:results/.../checkpoint_ep000740.pt",
    )
    parser.add_argument("--seed-start", type=int, default=642000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--shares", default=DEFAULT_SHARES)
    parser.add_argument(
        "--fixed-rho-one",
        action="store_true",
        help="rho does not change the contention wait, but it does change "
        "Delta-C; keep the upper level alone for a clean placement read.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def parse_case(text):
    label, _, rest = text.partition(":")
    config, _, checkpoint = rest.partition(":")
    if not (label and config and checkpoint):
        raise ValueError(f"bad --case {text!r}; want LABEL:CONFIG:CHECKPOINT")
    return label, config, checkpoint


def workflow_terms(buffer):
    """Per workflow: the Delta-C mass, the queue mass at beta_q=1, and the miss.

    Both are summed over the workflow's own transitions, which is the level at
    which the miss credit is distributed, so shares computed here are the
    shares the training target will actually see.
    """
    rows = []
    for workflow_id, indices in buffer.workflow_groups():
        deadlines = np.array(
            [max(float(buffer.deadline_limits[i]), EPS) for i in indices],
            dtype=np.float64,
        )
        increments = np.array(
            [max(0.0, float(buffer.deadline_costs[i])) for i in indices],
            dtype=np.float64,
        ) / deadlines
        queue = np.array(
            [
                max(0.0, float(buffer.infos[i].get("high_queue_delay_normalized", 0.0)))
                for i in indices
            ],
            dtype=np.float64,
        )
        final = indices[-1]
        completion = float(buffer.partial_makespans[final])
        limit = max(float(buffer.deadline_limits[final]), EPS)
        rows.append(
            {
                "workflow_id": str(workflow_id),
                "tasks": len(indices),
                "increment_mass": float(increments.sum()),
                "queue_mass_at_beta1": float(queue.sum()),
                "zero_increment_tasks": int((increments <= EPS).sum()),
                "queue_mass_on_zero_increment": float(
                    queue[increments <= EPS].sum()
                ),
                "miss": float(completion > limit + EPS),
            }
        )
    return rows


def run_case(label, config_path, checkpoint_path, args, shares):
    torch.set_num_threads(1)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
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
    agent.load_checkpoint(Path(checkpoint_path).resolve())
    agent.high_agent.epsilon = 0.0
    trained_lambda = float(agent.high_agent.deadline_lambda)
    miss_penalty = float(getattr(agent.high_agent, "deadline_miss_penalty", 0.0))
    configured_beta_q = float(getattr(agent.high_agent, "deadline_queue_weight", 0.0))

    rows = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        high_buffer = HighRolloutBuffer()
        with redirect_stdout(StringIO()):
            workflows = build_workflows(config, resources, seed, smoke_test=False)
            env = build_env(
                workflows, resources, dict(config.get("simulation", {})), seed
            )
            collect_d3qn_episode(
                env,
                agent,
                objective,
                high_buffer,
                LowRolloutBuffer(),
                training=False,
                high_deterministic=True,
                low_deterministic=True,
                fixed_low_rho_one=not use_low,
                high_algorithm="d3qn",
            )
        rows.extend(workflow_terms(high_buffer))

    increment = np.array([row["increment_mass"] for row in rows])
    queue = np.array([row["queue_mass_at_beta1"] for row in rows])
    tasks = np.array([row["tasks"] for row in rows], dtype=np.float64)
    misses = np.array([row["miss"] for row in rows])
    zero_inc = np.array([row["zero_increment_tasks"] for row in rows], dtype=np.float64)
    queue_on_zero = np.array([row["queue_mass_on_zero_increment"] for row in rows])

    # Per-workflow shares, then averaged: the target is applied per workflow,
    # so a workflow-weighted mean is the honest summary.  A pooled ratio would
    # let a few huge workflows set the weight for all the rest.
    beta_for_share = {}
    for share in shares:
        # share = beta*Q / (I + beta*Q)  ->  beta = share/(1-share) * I/Q
        ratio = float(increment.sum() / max(queue.sum(), EPS))
        beta_for_share[share] = share / max(1.0 - share, EPS) * ratio

    summary = {
        "label": label,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "workflows": int(len(rows)),
        "mean_tasks_per_workflow": float(tasks.mean()),
        "mean_increment_mass": float(increment.mean()),
        "mean_queue_mass_at_beta1": float(queue.mean()),
        "queue_over_increment": float(queue.sum() / max(increment.sum(), EPS)),
        "mean_zero_increment_task_fraction": float((zero_inc / tasks).mean()),
        "queue_mass_hidden_on_zero_increment": float(
            queue_on_zero.sum() / max(queue.sum(), EPS)
        ),
        "miss_rate": float(misses.mean()),
        "miss_penalty": miss_penalty,
        "mean_miss_credit_mass": float(miss_penalty * misses.mean()),
        "trained_lambda_deadline": trained_lambda,
        "configured_beta_q": configured_beta_q,
        "beta_for_share": {f"{k:.2f}": v for k, v in beta_for_share.items()},
    }
    return summary, rows


def main():
    args = parse_args()
    shares = [float(v) for v in args.shares.split(",") if v.strip()]
    cases = [parse_case(text) for text in args.case]
    mode = "rho1" if args.fixed_rho_one else "sac"
    seeds = f"{args.seed_start}-{args.seed_start + args.num_seeds - 1}"

    summaries = []
    for label, config_path, checkpoint_path in cases:
        summary, _ = run_case(label, config_path, checkpoint_path, args, shares)
        summaries.append(summary)

    print("=" * 78)
    print(f"DEADLINE-HEAD TARGET MASS AT beta_q = 0   [{mode}]  seeds {seeds}")
    print("=" * 78)
    print(f"{'scale':<10}{'N_w':>7}{'Sum dC/D':>11}{'Sum q/(N D)':>13}"
          f"{'q/dC':>8}{'zero-dC':>10}{'q hidden':>10}{'miss':>7}")
    for row in summaries:
        print(f"{row['label']:<10}{row['mean_tasks_per_workflow']:>7.1f}"
              f"{row['mean_increment_mass']:>11.4f}"
              f"{row['mean_queue_mass_at_beta1']:>13.4f}"
              f"{row['queue_over_increment']:>8.3f}"
              f"{row['mean_zero_increment_task_fraction']:>10.1%}"
              f"{row['queue_mass_hidden_on_zero_increment']:>10.1%}"
              f"{row['miss_rate']:>7.3f}")
    print()
    print("  Sum dC/D    telescopes to C_w/D_w, so ~1 means the workflow used")
    print("              its whole Deadline; this is the existing dense signal.")
    print("  zero-dC     share of tasks with Delta-C_t = 0 -- invisible to the")
    print("              head today, and invisible again in the miss credit.")
    print("  q hidden    share of the queue mass sitting on exactly those tasks;")
    print("              this is what beta_q makes visible and nothing else can.")

    print()
    print("=" * 78)
    print("beta_q REQUIRED FOR A GIVEN QUEUE SHARE OF THE DENSE COST")
    print("=" * 78)
    header = f"{'share':>8}" + "".join(f"{row['label']:>14}" for row in summaries)
    print(header)
    for share in shares:
        key = f"{share:.2f}"
        line = f"{share:>7.0%}"
        for row in summaries:
            line += f"{row['beta_for_share'][key]:>14.3f}"
        print(line)
    if len(summaries) > 1:
        spread = [
            max(row["beta_for_share"][f"{share:.2f}"] for row in summaries)
            / max(
                min(row["beta_for_share"][f"{share:.2f}"] for row in summaries),
                EPS,
            )
            for share in shares
        ]
        print(f"\n  cross-scale spread of the required beta_q: "
              f"x{max(spread):.2f}")
        if max(spread) <= 1.25:
            print("  -> one beta_q carries both scales; set it once.")
        else:
            print("  -> the same beta_q would mean different things per scale.")
            print("     Match the SHARE on each scale, not the raw weight.")

    print()
    print("=" * 78)
    print("KNOCK-ON EFFECT ON lambda_D  (the selector ranks Q_E - lambda_D*Q_D)")
    print("=" * 78)
    print(f"{'scale':>10}{'lambda now':>12}{'share':>8}{'Q_D grows':>12}"
          f"{'lambda should be':>19}")
    for row in summaries:
        for share in shares:
            grow = 1.0 / max(1.0 - share, EPS)
            print(f"{row['label']:>10}{row['trained_lambda_deadline']:>12.3f}"
                  f"{share:>8.0%}{grow:>12.3f}"
                  f"{row['trained_lambda_deadline'] / grow:>19.3f}")
    print()
    print("  Adding the queue term inflates the Deadline target by 1/(1-share).")
    print("  Leaving lambda alone therefore also raises the effective penalty,")
    print("  and the lambda sweep showed that direction costs DSR and energy.")
    print("  Dividing lambda by the same factor isolates the change.")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {"mode": mode, "seeds": seeds, "shares": shares,
                 "cases": summaries},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
