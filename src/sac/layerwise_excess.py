"""Conditional cost of the lower DVFS action on an already selected server."""
import math


def added_excess(actual, full_frequency, limit):
    """Extra violation over the same decision at full frequency."""
    return max(0., max(0., actual - limit) - max(0., full_frequency - limit))


def dvfs_excess(event, safety, remaining_hazard_budget, hazard_budget):
    """Use the existing deadline forecast and a reserved workflow risk budget."""
    duration = float(event["exec_time"])
    reference_duration = float(safety["execution_time_rho_1"])
    hazard = float(event["failure_rate"]) * duration
    reference_hazard = -math.log(float(safety["task_reliability_rho_1"]))
    if math.isclose(float(event["rho"]), 1., abs_tol=1e-12):
        # Avoid a spurious increment from reconstructing the same hazard via log.
        assert math.isclose(hazard, reference_hazard, abs_tol=1e-12)
        assert math.isclose(duration, reference_duration, abs_tol=1e-10)
        return 0., 0.
    return (
        added_excess(duration, reference_duration, safety["safe_execution_budget"])
        / safety["workflow_relative_deadline"],
        added_excess(hazard, reference_hazard, remaining_hazard_budget) / hazard_budget,
    )
