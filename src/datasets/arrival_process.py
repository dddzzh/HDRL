from __future__ import annotations

import random
from typing import Any, Mapping


DEFAULT_POISSON_ARRIVAL_RATE = 0.01
CAI_2025_ARRIVAL_PROCESS = "cai_2025_exponential"
POISSON_ARRIVAL_DEFINITION = (
    "Cai et al. (2025): one system-wide workflow stream; the intervals "
    "between consecutive workflows follow Exp(lambda), where lambda is the "
    "expected number of workflow arrivals per second."
)


def resolve_arrival_rate(config: Mapping[str, Any] | None, default=DEFAULT_POISSON_ARRIVAL_RATE):
    config = config or {}
    if "mean_interarrival" in config and config["mean_interarrival"] is not None:
        raise ValueError("mean_interarrival is no longer supported; use arrival_rate/lambdaRate only.")
    value = _first_present(
        config,
        ("arrival_rate", "lambdaRate", "lambda_rate", "lambda_arrival"),
    )
    if value is None:
        value = default
    value = float(value)
    if value <= 0.0:
        raise ValueError("arrival_rate/lambdaRate must be positive")
    return value


def create_cai_2025_time_table(num_workflows, lambda_rate, rng=None):
    """Generate one exponential workflow-arrival stream as in Cai et al. (2025)."""
    if int(num_workflows) < 0:
        raise ValueError("num_workflows must be non-negative")
    lambda_rate = float(lambda_rate)
    if lambda_rate <= 0.0:
        raise ValueError("arrival_rate/lambdaRate must be positive")
    rng = rng or random

    arrival_times = []
    current_time = 0.0
    for index in range(int(num_workflows)):
        if index > 0:
            current_time += rng.expovariate(lambda_rate)
        arrival_times.append(float(current_time))
    return arrival_times


def sample_configured_arrivals(count, config, start_time=0.0, rng=None):
    config = config or {}
    arrival_process = str(config.get("arrival_process", CAI_2025_ARRIVAL_PROCESS)).lower()
    if arrival_process != CAI_2025_ARRIVAL_PROCESS:
        raise ValueError(
            f"Unsupported arrival_process={arrival_process!r}; expected "
            f"{CAI_2025_ARRIVAL_PROCESS!r}."
        )
    lambda_rate = resolve_arrival_rate(config)
    py_rng = _as_python_rng(rng)
    start_time = float(start_time)
    relative_times = create_cai_2025_time_table(count, lambda_rate, rng=py_rng)
    arrival_times = [start_time + value for value in relative_times]
    # The scheduler no longer models one arrival stream per IoT device.  This
    # placeholder is retained only for the serialized Workflow schema.
    source_device_ids = [0 for _ in arrival_times]
    inter_arrival_times = []
    previous = start_time
    for index, arrival_time in enumerate(arrival_times):
        inter_arrival_times.append(0.0 if index == 0 else max(0.0, arrival_time - previous))
        previous = arrival_time
    return arrival_times, inter_arrival_times, source_device_ids, CAI_2025_ARRIVAL_PROCESS


def sample_poisson_arrivals(count, arrival_rate, start_time=0.0, rng=None):
    """Return Cai-style single-stream arrivals and inter-arrival intervals."""
    arrivals, intervals, _, _ = sample_configured_arrivals(
        count,
        {
            "arrival_process": CAI_2025_ARRIVAL_PROCESS,
            "arrival_rate": arrival_rate,
        },
        start_time,
        rng,
    )
    return arrivals, intervals


def debug_check_poisson_arrivals(rng=None):
    arrival_times, _, source_device_ids, _ = sample_configured_arrivals(
        1000,
        {
            "arrival_process": CAI_2025_ARRIVAL_PROCESS,
            "arrival_rate": DEFAULT_POISSON_ARRIVAL_RATE,
        },
        0.0,
        rng,
    )
    assert len(arrival_times) == 1000
    assert len(source_device_ids) == 1000
    assert all(arrival_times[index] >= arrival_times[index - 1] for index in range(1, len(arrival_times)))
    return arrival_times


def _as_python_rng(rng):
    if rng is None:
        return random
    if isinstance(rng, random.Random):
        return rng
    if isinstance(rng, int):
        return random.Random(int(rng))
    if hasattr(rng, "integers"):
        return random.Random(int(rng.integers(0, 2**32 - 1)))
    return rng


def _first_present(config, keys):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None
