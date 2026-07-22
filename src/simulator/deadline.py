from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

from src.schedulers.heft import reference_heft_makespan
from src.simulator.workflow import Workflow


WorkflowLike = Union[Workflow, Mapping[str, Any]]

DEADLINE_PROBABILITIES = {
    "tight": 0.8,
    "medium": 0.5,
    "loose": 0.2,
}


def validate_deadline_type(deadline_type: str) -> str:
    deadline_type = str(deadline_type).lower()
    if deadline_type not in DEADLINE_PROBABILITIES:
        raise ValueError(
            f"invalid deadline_type={deadline_type!r}; expected one of "
            f"{sorted(DEADLINE_PROBABILITIES)}"
        )
    return deadline_type


def sample_deadline_multiplier(deadline_type: str, rng) -> float:
    prob_2x = DEADLINE_PROBABILITIES[validate_deadline_type(deadline_type)]
    return 2.0 if float(rng.random()) < prob_2x else 3.0


def cai_2025_deadline_fields(
    workflow: WorkflowLike,
    arrival_time: float,
    servers: Sequence[Mapping[str, Any]],
    *,
    deadline_type: str,
    rng,
    link_bandwidth_mbps: Optional[Mapping[str, float]] = None,
    rho: float = 1.0,
    cache: Optional[MutableMapping[str, dict[str, Any]]] = None,
) -> dict[str, float | str]:
    deadline_type = validate_deadline_type(deadline_type)
    reference_makespan = reference_heft_makespan(
        workflow,
        servers,
        link_bandwidth_mbps,
        rho=rho,
        cache=cache,
    )
    deadline_multiplier = sample_deadline_multiplier(deadline_type, rng)
    relative_deadline = deadline_multiplier * reference_makespan
    absolute_deadline = float(arrival_time) + relative_deadline
    if reference_makespan <= 0:
        raise ValueError(f"reference_makespan must be positive, got {reference_makespan}")
    if absolute_deadline < float(arrival_time):
        raise ValueError("absolute_deadline must be >= arrival_time")
    if relative_deadline <= 0:
        raise ValueError("relative_deadline must be positive")
    if deadline_multiplier not in {2.0, 3.0}:
        raise ValueError(f"invalid deadline_multiplier={deadline_multiplier}")
    return {
        "deadline_type": deadline_type,
        "reference_makespan": float(reference_makespan),
        "deadline_multiplier": float(deadline_multiplier),
        "relative_deadline": float(relative_deadline),
        "absolute_deadline": float(absolute_deadline),
    }
