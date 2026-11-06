"""Shared-environment HEFT reference for an opt-in training experiment.

Generate legacy workflows first, then replace only their deadline reference.
This preserves all legacy RNG draws, including each 2x/3x multiplier.
"""
from __future__ import annotations

import copy
import math

from src.schedulers.heft import HEFTScheduler
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv


def assign_shared_heft_deadlines(workflows, resources, sim_config, seed):
    """Return new workflows and audit rows; never mutate inputs or train a policy.

    The existing online HEFT scheduler handles released, dependency-ready tasks
    across workflows, selecting upward rank then queue-aware EFT at rho=1.
    Deadlines are per-workflow response times times the original multiplier,
    not a common environment makespan. Reference scheduling ignores deadlines.
    """
    if not workflows or not resources.get("servers"):
        raise ValueError("Shared HEFT requires workflows and servers.")
    ids = [str(w.workflow_id) for w in workflows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate workflow IDs.")
    if sim_config.get("enable_backup_duplication", False):
        raise ValueError("This reference requires the original no-backup protocol.")
    for workflow in workflows:
        values = (workflow.arrival_time, workflow.reference_makespan,
                  workflow.relative_deadline, workflow.absolute_deadline)
        if any(v is None or not math.isfinite(float(v)) for v in values):
            raise ValueError("Generate complete finite legacy deadlines first.")
        if workflow.reference_makespan <= 0 or workflow.deadline_multiplier not in (2, 3):
            raise ValueError("Expected positive legacy reference and 2x/3x multiplier.")

    # Both env and scheduler own local RNGs. No learned model or global RNG is used.
    env = EdgeCloudWorkflowEnv(workflows, resources, copy.deepcopy(sim_config), seed=seed)
    result = env.run(HEFTScheduler(params={"rho": 1.0}, seed=seed))
    rows = result["workflow_results"]
    by_id = {str(row["workflow_id"]): row for row in rows}
    if not env.done or len(rows) != len(ids) or set(by_id) != set(ids):
        raise RuntimeError("Shared HEFT reference did not complete every workflow.")

    updated = copy.deepcopy(workflows)
    audit = []
    for workflow in updated:
        response = float(by_id[str(workflow.workflow_id)]["response_time"])
        if not math.isfinite(response) or response <= 0:
            raise RuntimeError("Invalid shared HEFT response time.")
        relative = float(workflow.deadline_multiplier) * response
        absolute = float(workflow.arrival_time) + relative
        audit.append({
            "workflow_id": str(workflow.workflow_id),
            "arrival_time": float(workflow.arrival_time),
            "deadline_multiplier": float(workflow.deadline_multiplier),
            "isolated_reference": float(workflow.reference_makespan),
            "isolated_relative_deadline": float(workflow.relative_deadline),
            "shared_reference": response,
            "relative_deadline": relative,
            "absolute_deadline": absolute,
        })
        workflow.reference_makespan = response
        workflow.relative_deadline = relative
        workflow.absolute_deadline = absolute
        workflow.deadline_reference = "shared_environment_online_heft_rho1"
    return updated, audit
