from __future__ import annotations

import math


P3O_COST_EPS = 1e-8


class EnergyRewardNormalizer:
    def __init__(self, beta=0.99, eps=P3O_COST_EPS):
        self.beta = float(beta)
        self.eps = float(eps)
        self.value = None

    def current(self, energy_actual):
        if self.value is None or self.value <= self.eps:
            return max(float(energy_actual), self.eps)
        return max(float(self.value), self.eps)

    def update(self, energy_actual):
        energy_actual = max(float(energy_actual), self.eps)
        if self.value is None or self.value <= self.eps:
            self.value = energy_actual
        else:
            beta = min(0.9999, max(0.0, self.beta))
            self.value = beta * self.value + (1.0 - beta) * energy_actual
        return float(self.value)


def extract_task_energy(info, env_reward):
    if "energy" in info:
        return float(info["energy"])
    if "task_energy" in info:
        return float(info["task_energy"])
    if "actual_energy" in info:
        return float(info["actual_energy"])
    event = info.get("event", {})
    if "energy" in event:
        return float(event["energy"])
    if env_reward is not None:
        return float(-env_reward)
    raise ValueError(
        "Environment step info must provide task energy via 'energy', "
        "'task_energy', 'actual_energy', or info['event']['energy']."
    )


def compute_energy_decomposed_rewards(energy_ref, energy_actual, energy_norm):
    energy_ref = float(energy_ref)
    energy_actual = float(energy_actual)
    energy_norm = max(float(energy_norm), P3O_COST_EPS)
    high_reward = -energy_ref / energy_norm
    low_reward = (energy_ref - energy_actual) / energy_norm
    expected_total = -energy_actual / energy_norm
    if abs((high_reward + low_reward) - expected_total) > 1e-6:
        raise AssertionError("Hierarchical energy reward decomposition is inconsistent.")
    return float(high_reward), float(low_reward)


def compute_high_level_time_reward(partial_makespan_increment):
    """High-level objective: minimize workflow partial makespan increments."""
    return -float(partial_makespan_increment)


def compute_time_high_energy_low_rewards(
    partial_makespan_increment,
    energy_ref,
    energy_actual,
    energy_norm,
):
    """High-level time objective with the original low-level energy reward."""
    _, low_reward = compute_energy_decomposed_rewards(
        energy_ref,
        energy_actual,
        energy_norm,
    )
    return (
        float(compute_high_level_time_reward(partial_makespan_increment)),
        float(low_reward),
    )


def compute_low_energy_saving_reward(energy_ref_rho1, primary_energy_actual, eps=P3O_COST_EPS):
    energy_ref_rho1 = max(float(energy_ref_rho1), float(eps))
    primary_energy_actual = float(primary_energy_actual)
    saving_ratio = (energy_ref_rho1 - primary_energy_actual) / energy_ref_rho1
    return float(max(-1.0, min(1.0, saving_ratio)))


def compute_low_local_deadline_penalty(
    actual_aft,
    task_latest_finish_time,
    workflow_relative_deadline,
    coef=10.0,
    eps=P3O_COST_EPS,
):
    violation_ratio = max(0.0, float(actual_aft) - float(task_latest_finish_time)) / max(
        float(workflow_relative_deadline),
        float(eps),
    )
    return float(float(coef) * violation_ratio)


def compute_workflow_deadline_penalty_distribution(
    slowdowns,
    workflow_deadline_violation_ratio,
    coef=5.0,
    eps=P3O_COST_EPS,
):
    slowdowns = [max(0.0, float(value)) for value in slowdowns]
    if not slowdowns:
        return []
    total = sum(slowdowns)
    if total > float(eps):
        weights = [value / total for value in slowdowns]
    else:
        weights = [1.0 / len(slowdowns)] * len(slowdowns)
    total_penalty = float(coef) * max(0.0, float(workflow_deadline_violation_ratio))
    return [float(total_penalty * weight) for weight in weights]


def compute_partial_workflow_makespan(
    scheduled_task_finish_times,
    workflow_arrival_time,
):
    finish_times = [float(value) for value in scheduled_task_finish_times if value is not None]
    if not finish_times:
        return 0.0
    return float(max(finish_times) - float(workflow_arrival_time))


def compute_deadline_cost_increment(
    previous_partial_makespan,
    current_partial_makespan,
    eps=P3O_COST_EPS,
):
    increment = float(current_partial_makespan) - float(previous_partial_makespan)
    if increment < -float(eps):
        raise RuntimeError(
            "Partial workflow makespan must be non-decreasing: "
            f"previous={previous_partial_makespan}, current={current_partial_makespan}"
        )
    if increment < 0.0:
        return 0.0
    return float(increment)


def compute_partial_workflow_log_risk(current_partial_reliability, eps=P3O_COST_EPS):
    reliability = min(1.0, max(float(eps), float(current_partial_reliability)))
    return float(-math.log(reliability))


def compute_reliability_cost_increment(
    previous_partial_log_risk,
    current_partial_log_risk,
    eps=P3O_COST_EPS,
):
    increment = float(current_partial_log_risk) - float(previous_partial_log_risk)
    if increment < -float(eps):
        raise RuntimeError(
            "Partial workflow log-risk must be non-decreasing: "
            f"previous={previous_partial_log_risk}, current={current_partial_log_risk}"
        )
    if increment < 0.0:
        return 0.0
    return float(increment)


def compute_allowed_log_risk(reliability_req, eps=P3O_COST_EPS):
    reliability_req = min(1.0, max(float(eps), float(reliability_req)))
    return float(-math.log(reliability_req))
