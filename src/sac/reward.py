from __future__ import annotations

import math

from src.shared.hierarchical_state_reward import (
    compute_dvfs_reward_components,
    low_reliability_counterfactual_cost,
    normalized_sub_deadline_violation,
)


EPS = 1e-8


def build_low_sac_reward_components(
    env,
    env_action,
    info,
    reference_metrics,
    *,
    slack_floor_ratio=0.05,
    reward_profile="ep740_lagrangian_v1",
):
    """Return slack-aware reward components for low SAC replay."""
    event = dict((info or {}).get("event", {}))
    workflow_id = str(env_action["workflow_id"])
    task_id = int(env_action["task_id"])
    deadline = max(float(reference_metrics["workflow_relative_deadline"]), EPS)

    reference_aft = float(reference_metrics["final_aft"])
    actual_aft = float(event.get("final_aft", reference_aft))
    partial_makespan_before = max(
        0.0, float(reference_metrics.get("workflow_partial_makespan_before", 0.0))
    )
    workflow_arrival_time = float(
        reference_metrics.get(
            "workflow_arrival_time",
            env.workflow_by_id[workflow_id].get("arrival_time", 0.0),
        )
    )
    dvfs_components = compute_dvfs_reward_components(
        deadline=deadline,
        partial_makespan_before=partial_makespan_before,
        workflow_arrival_time=workflow_arrival_time,
        reference_aft=reference_aft,
        actual_aft=actual_aft,
        reference_energy=float(reference_metrics["total_energy"]),
        actual_energy=float(event.get("energy", 0.0)),
        slack_floor_ratio=slack_floor_ratio,
    )
    raw_energy_saving_joules = float(dvfs_components["energy_saving"])
    reward_profile = str(reward_profile).strip().lower()
    if reward_profile == "sub_deadline_lagrangian_v1":
        sub_deadline = reference_metrics.get("task_sub_deadline_relative")
        if sub_deadline is None:
            raise KeyError(
                "sub_deadline_lagrangian_v1 requires task_sub_deadline_relative"
            )
        workflow_task_count = max(
            float(reference_metrics.get("workflow_task_count", 1.0)), 1.0
        )
        reference_sub_cost = normalized_sub_deadline_violation(
            finish_time=reference_aft,
            workflow_arrival_time=workflow_arrival_time,
            sub_deadline_relative=sub_deadline,
            workflow_task_count=workflow_task_count,
        )
        actual_sub_cost = normalized_sub_deadline_violation(
            finish_time=actual_aft,
            workflow_arrival_time=workflow_arrival_time,
            sub_deadline_relative=sub_deadline,
            workflow_task_count=workflow_task_count,
        )
        deadline_counterfactual_cost = max(
            0.0, actual_sub_cost - reference_sub_cost
        )
        deadline_counterfactual_crossed = float(
            actual_sub_cost > EPS and reference_sub_cost <= EPS
        )
        # Energy and local Deadline cost now share the same dimensionless,
        # per-workflow normalization.  No replay z-score is used by this
        # profile.
        energy_ratio_per_task = (
            float(dvfs_components["energy_saving_ratio"])
            / workflow_task_count
        )
        dvfs_components.update(
            {
                "energy_saving": energy_ratio_per_task,
                "energy_saving_normalized": energy_ratio_per_task,
                "slack_time_cost": 0.0,
                "sub_deadline_relative": float(sub_deadline),
                "reference_sub_deadline_cost": reference_sub_cost,
                "actual_sub_deadline_cost": actual_sub_cost,
                "sub_deadline_constraint_cost": deadline_counterfactual_cost,
                "sub_deadline_crossed_by_dvfs": deadline_counterfactual_crossed,
            }
        )
    else:
        deadline_counterfactual_cost = float(
            dvfs_components["deadline_constraint_cost"]
        )
        deadline_counterfactual_crossed = float(
            dvfs_components["deadline_constraint_crossed"]
        )
    # Attribute only the extra Deadline damage caused by DVFS.  A workflow that
    # is already late at rho=1 is an upper-placement failure and must not give
    # the lower policy the full system-miss penalty.
    workflow_state = env.workflow_by_id[workflow_id]
    reliability_ref = float(reference_metrics["final_reliability"])
    reliability_actual = float(event.get("final_reliability", reliability_ref))
    reliability = low_reliability_counterfactual_cost(
        env,
        workflow_state,
        task_id,
        reliability_ref,
        reliability_actual,
    )

    # Raw task reliabilities are exposed so a reward variant can charge the
    # *risk budget actually consumed* instead of the projected violation.  The
    # projection assumes every not-yet-scheduled task achieves
    # ``remaining_task_reliability_estimate`` (0.999), so it stays at zero while
    # early DVFS decisions are already spending the workflow's budget.
    required_reliability = float(
        env._workflow_reliability_requirement(env.workflow_by_id[workflow_id])
    )
    risk_budget = max(-math.log(max(required_reliability, EPS)), EPS)

    return {
        **dvfs_components,
        "energy_saved": raw_energy_saving_joules,
        "energy_saving_joules": raw_energy_saving_joules,
        "task_reliability_reference": float(reliability_ref),
        "task_reliability_actual": float(reliability_actual),
        "workflow_reliability_risk_budget": float(risk_budget),
        "reliability_budget_share_spent": float(
            max(
                0.0,
                math.log(max(float(reliability_ref), EPS))
                - math.log(max(float(reliability_actual), EPS)),
            )
            / risk_budget
        ),
        # Filled with the workflow-level cooperative energy credit before the
        # episode transitions are committed to SAC replay.
        "team_energy_reward": 0.0,
        "deadline_constraint_cost": deadline_counterfactual_cost,
        "deadline_constraint_crossed": deadline_counterfactual_crossed,
        "deadline_counterfactual_cost": deadline_counterfactual_cost,
        "deadline_counterfactual_crossed": deadline_counterfactual_crossed,
        "deadline_reward_profile": reward_profile,
        "reliability_violation": reliability["reliability_violation_delta"],
        "reliability_crossed": reliability["reliability_crossed"],
    }
