from __future__ import annotations

import math


EPS = 1e-8


def build_low_sac_reward_components(env, env_action, info, reference_metrics):
    """Return action-attributable reward components for low SAC replay."""
    event = dict((info or {}).get("event", {}))
    workflow_id = str(env_action["workflow_id"])
    task_id = int(env_action["task_id"])
    deadline = max(float(reference_metrics["workflow_relative_deadline"]), EPS)

    energy_ref = max(float(reference_metrics["total_energy"]), EPS)
    energy_actual = float(event.get("energy", 0.0))
    energy_saving = energy_ref - energy_actual
    energy_saving_ratio = energy_saving / energy_ref

    reference_aft = float(reference_metrics["final_aft"])
    actual_aft = float(event.get("final_aft", reference_aft))
    time_increase_seconds = max(0.0, actual_aft - reference_aft)
    time_increase = time_increase_seconds / deadline

    latest_finish = float(reference_metrics["task_latest_finish_time"])
    deadline_ref = max(0.0, reference_aft - latest_finish) / deadline
    deadline_actual = max(0.0, actual_aft - latest_finish) / deadline
    deadline_delta = max(0.0, deadline_actual - deadline_ref)
    deadline_crossed = float(deadline_actual > EPS and deadline_ref <= EPS)

    workflow_state = env.workflow_by_id[workflow_id]
    reliability_req = float(env._workflow_reliability_requirement(workflow_state))
    reliability_ref = float(reference_metrics["final_reliability"])
    reliability_actual = float(event.get("final_reliability", reliability_ref))
    reliability_ref_violation = _projected_reliability_violation(
        env, workflow_state, task_id, reliability_ref, reliability_req
    )
    reliability_actual_violation = _projected_reliability_violation(
        env, workflow_state, task_id, reliability_actual, reliability_req
    )
    reliability_delta = max(0.0, reliability_actual_violation - reliability_ref_violation)
    reliability_crossed = float(
        reliability_actual_violation > EPS and reliability_ref_violation <= EPS
    )

    return {
        "energy_saving": float(energy_saving),
        "energy_saved": float(energy_saving),
        "energy_saving_ratio": float(energy_saving_ratio),
        "energy_reference": float(energy_ref),
        "energy_actual": float(energy_actual),
        "time_increase": float(time_increase),
        "time_increase_seconds": float(time_increase_seconds),
        "deadline_violation": float(deadline_delta),
        "reliability_violation": float(reliability_delta),
        "deadline_crossed": deadline_crossed,
        "reliability_crossed": reliability_crossed,
    }


def _projected_reliability_violation(
    env, workflow_state, task_id, candidate_reliability, reliability_req
):
    if reliability_req <= 0.0:
        return 0.0
    log_projected = float(
        env._estimated_workflow_log_reliability(
            workflow_state, task_id, candidate_reliability
        )
    )
    log_required = math.log(max(reliability_req, EPS))
    risk_budget = max(-log_required, EPS)
    return max(0.0, log_required - log_projected) / risk_budget
