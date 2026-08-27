from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CoupledResult:
    success_rate: float
    energy_saving: float
    high_slack_use: float
    low_slack_use: float
    high_lambda: float
    low_lambda: float


def _finite_difference_miss_gradient(delay, coefficient, thresholds, step=0.01):
    current = float(np.mean(delay > thresholds))
    perturbed = float(np.mean((delay + coefficient * step) > thresholds))
    return (perturbed - current) / step


def simulate_coupled_control(
    *,
    seed,
    shared,
    episodes=800,
    workflows_per_episode=100,
    budget=0.0,
    shared_lambda_init=1.0,
):
    """Compare two prices with one shared price in a coupled toy CMDP.

    ``high_slack_use`` and ``low_slack_use`` are normalized energy-saving
    decisions.  Both reduce energy, while their weighted sum consumes the one
    workflow Deadline slack resource.  The two agents therefore optimize a
    genuinely coupled constraint rather than two independent constraints.
    """
    rng = np.random.default_rng(seed)
    high_action = 0.45
    low_action = 0.45
    high_lambda = 1.0
    low_lambda = 10.0
    shared_lambda = float(shared_lambda_init)
    actor_lr = 0.012
    high_dual_lr = 0.001
    low_dual_lr = 0.01
    shared_dual_lr = 0.01
    high_delay_coefficient = 0.55
    low_delay_coefficient = 0.45
    high_energy_gradient = 0.42
    low_energy_gradient = 0.58
    warmup = 50
    tail_success = []
    tail_energy = []

    for episode in range(episodes):
        # Every workflow is feasible at zero slowdown.  Tightness varies, but
        # there is no irreducible miss, so a zero-miss policy exists.
        thresholds = np.clip(
            rng.normal(loc=0.62, scale=0.075, size=workflows_per_episode),
            0.42,
            0.85,
        )
        delay = (
            high_delay_coefficient * high_action
            + low_delay_coefficient * low_action
        )
        miss_rate = float(np.mean(delay > thresholds))
        high_miss_gradient = _finite_difference_miss_gradient(
            delay, high_delay_coefficient, thresholds
        )
        low_miss_gradient = _finite_difference_miss_gradient(
            delay, low_delay_coefficient, thresholds
        )

        if shared:
            high_price = low_price = shared_lambda
        else:
            high_price = high_lambda
            low_price = low_lambda
        high_action = float(
            np.clip(
                high_action
                + actor_lr
                * (high_energy_gradient - high_price * high_miss_gradient),
                0.0,
                1.0,
            )
        )
        low_action = float(
            np.clip(
                low_action
                + actor_lr
                * (low_energy_gradient - low_price * low_miss_gradient),
                0.0,
                1.0,
            )
        )

        if episode >= warmup:
            if shared:
                shared_lambda = max(
                    0.0,
                    shared_lambda + shared_dual_lr * (miss_rate - budget),
                )
            else:
                high_lambda = max(
                    0.0,
                    high_lambda + high_dual_lr * (miss_rate - budget),
                )
                low_lambda = max(
                    0.0,
                    low_lambda + low_dual_lr * (miss_rate - budget),
                )

        if episode >= episodes - 200:
            energy_saving = (
                high_energy_gradient * high_action
                + low_energy_gradient * low_action
            )
            tail_success.append(1.0 - miss_rate)
            tail_energy.append(energy_saving)

    if shared:
        high_lambda = low_lambda = shared_lambda
    return CoupledResult(
        success_rate=float(np.mean(tail_success)),
        energy_saving=float(np.mean(tail_energy)),
        high_slack_use=high_action,
        low_slack_use=low_action,
        high_lambda=high_lambda,
        low_lambda=low_lambda,
    )


def summarize_history(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    paired = [
        row
        for row in rows
        if "lambda_deadline" in row and "low_deadline_lambda" in row
    ]
    if not paired:
        return None
    gaps = np.asarray(
        [
            abs(float(row["low_deadline_lambda"]) - float(row["lambda_deadline"]))
            for row in paired
        ],
        dtype=np.float64,
    )
    last = paired[-1]
    return {
        "episodes": len(paired),
        "mean_absolute_lambda_gap": float(np.mean(gaps)),
        "max_absolute_lambda_gap": float(np.max(gaps)),
        "final_high_lambda": float(last["lambda_deadline"]),
        "final_low_lambda": float(last["low_deadline_lambda"]),
        "final_absolute_lambda_gap": float(gaps[-1]),
        "final_train_deadline_success_rate": float(
            last.get("deadline_success_rate", 0.0)
        ),
    }


def aggregate_simulation(seeds, shared, shared_lambda_init=1.0):
    values = [
        simulate_coupled_control(
            seed=seed,
            shared=shared,
            shared_lambda_init=shared_lambda_init,
        )
        for seed in seeds
    ]
    fields = CoupledResult.__dataclass_fields__
    return {
        name: {
            "mean": float(np.mean([getattr(row, name) for row in values])),
            "std": float(np.std([getattr(row, name) for row in values])),
        }
        for name in fields
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = range(args.seeds)
    result = {
        "scope": (
            "Diagnostic only: the coupled simulation isolates multiplier "
            "coordination; it is not an Edge-Cloud training result."
        ),
        "deadline_miss_budget": 0.0,
        "history": summarize_history(args.history) if args.history else None,
        "coupled_simulation": {
            "separate_current_asymmetric_lambdas": aggregate_simulation(
                seeds, shared=False
            ),
            "shared_lambda_init_1": aggregate_simulation(
                seeds, shared=True, shared_lambda_init=1.0
            ),
            "shared_lambda_init_10": aggregate_simulation(
                seeds, shared=True, shared_lambda_init=10.0
            ),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
