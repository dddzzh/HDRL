"""Numerical diagnostic trainer for the standalone reliability-aware high D3QN.

This intentionally does not import or modify the production hierarchical
trainer.  It creates controlled transitions so Q-head and Lagrangian behavior
can be checked before wiring the module into real workflow collection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE.is_dir() and str(LOCAL_SITE) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.D3QN.reliability_aware_high import (
    ReliabilityAwareConstrainedD3QN,
    ReliabilityLagrangian,
    reliability_cost,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/reliability_aware_high_diagnostic.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    agent = ReliabilityAwareConstrainedD3QN(
        state_dim=args.state_dim, action_dim=args.action_dim,
        hidden_dim=128, hidden_layers=2, batch_size=args.batch_size,
        replay_size=max(5000, args.batch_size * 10), device="auto", seed=args.seed,
    )
    lagrangian = ReliabilityLagrangian(lambda_init=1.0, dual_lr=0.001, lambda_max=100.0)
    history = []
    for step in range(args.steps):
        state = rng.normal(0.0, 0.5, args.state_dim).astype(np.float32)
        next_state = rng.normal(0.0, 0.5, args.state_dim).astype(np.float32)
        action = int(rng.integers(0, args.action_dim))
        # Controlled risk mixture: most transitions are feasible, some violate
        # R_required=0.9 so the reliability head and multiplier receive signal.
        actual_r = float(rng.uniform(0.84, 0.995))
        cost = reliability_cost(actual_r, 0.9)
        energy_reward = float(rng.normal(0.0, 1.0))
        deadline_cost = float(rng.choice([0.0, 0.05, 0.2], p=[0.75, 0.2, 0.05]))
        done = bool(step % 25 == 24)
        mask = np.ones(args.action_dim, dtype=bool)
        agent.remember(state, action, energy_reward, deadline_cost, cost,
                       next_state, done, mask, mask)
        lambda_row = lagrangian.update(cost)
        metrics = agent.learn(1.0, lagrangian.lambda_reliability)
        if metrics and (step % 50 == 0 or step == args.steps - 1):
            energy, deadline, reliability = agent.action_q_values(state)
            combined, _, _, _ = agent.combined_action_q_values(
                state, 1.0, lagrangian.lambda_reliability)
            action_selected = int(np.argmax(combined))
            row = {
                "step": step,
                "actual_reliability": actual_r,
                "reliability_cost": cost,
                **lambda_row,
                "selected_action": action_selected,
                "energy_q_mean": float(np.mean(energy)),
                "deadline_q_mean": float(np.mean(deadline)),
                "reliability_q_mean": float(np.mean(reliability)),
                "combined_q_mean": float(np.mean(combined)),
                **metrics,
            }
            history.append(row)
            print(
                f"step={step} cost={cost:.5f} lambda_R={lagrangian.lambda_reliability:.5f} "
                f"loss_R={metrics.get('reliability_loss', float('nan')):.5f} "
                f"qR={row['reliability_q_mean']:.5f}", flush=True,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "trainer": "standalone_reliability_aware_high_diagnostic",
        "steps": args.steps,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "history": history,
        "final_lambda_reliability": lagrangian.lambda_reliability,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
