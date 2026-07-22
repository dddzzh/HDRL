from __future__ import annotations

import random
from typing import Any, Mapping


DEFAULT_POISSON_ARRIVAL_RATE = 0.01
POISSON_ARRIVAL_DEFINITION = (
    "Reference createTimeTable arrivals: each mobile device starts at 0, "
    "then advances by random.expovariate(lambdaRate); all device streams are "
    "merged and the earliest numWorkflows arrivals are kept."
)
MULTI_DEVICE_POISSON_ARRIVAL_DEFINITION = POISSON_ARRIVAL_DEFINITION


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


def create_time_table(num_mobile_devices, num_workflows, lambda_rate, rng=None):
    if int(num_mobile_devices) <= 0:
        raise ValueError("num_mobile_devices must be positive")
    if int(num_workflows) < 0:
        raise ValueError("num_workflows must be non-negative")
    lambda_rate = float(lambda_rate)
    if lambda_rate <= 0.0:
        raise ValueError("lambdaRate must be positive")
    rng = rng or random

    time_table = []
    for _ in range(int(num_mobile_devices)):
        workflow_list = [0.0]
        for _ in range(1, int(num_workflows)):
            interval = rng.expovariate(lambda_rate)
            workflow_list.append(workflow_list[-1] + interval)
        time_table.append(workflow_list)

    all_times = [
        (float(time_value), int(device_id))
        for device_id, sublist in enumerate(time_table)
        for time_value in sublist
    ]
    return sorted(all_times, key=lambda item: item[0])[: int(num_workflows)]


def sample_configured_arrivals(count, config, start_time=0.0, rng=None):
    config = config or {}
    if str(config.get("arrival_process", "multi_device_poisson")).lower() not in {
        "multi_device_poisson",
        "device_poisson",
        "senior_poisson",
    }:
        raise ValueError("Only reference multi-device Poisson arrivals are supported.")
    lambda_rate = resolve_arrival_rate(config)
    num_mobile_devices = int(
        config.get("num_mobile_devices", config.get("num_devices", config.get("mobile_devices", 1)))
    )
    py_rng = _as_python_rng(rng)
    rows = create_time_table(num_mobile_devices, count, lambda_rate, rng=py_rng)
    start_time = float(start_time)
    arrival_times = [start_time + float(time_value) for time_value, _ in rows]
    source_device_ids = [int(device_id) for _, device_id in rows]
    inter_arrival_times = []
    previous = start_time
    for index, arrival_time in enumerate(arrival_times):
        inter_arrival_times.append(0.0 if index == 0 else max(0.0, arrival_time - previous))
        previous = arrival_time
    return arrival_times, inter_arrival_times, source_device_ids, "multi_device_poisson"


def sample_multi_device_poisson_arrivals(count, arrival_rate, start_time=0.0, rng=None, num_mobile_devices=1):
    config = {
        "arrival_rate": arrival_rate,
        "num_mobile_devices": num_mobile_devices,
        "arrival_process": "multi_device_poisson",
    }
    arrival_times, inter_arrival_times, source_device_ids, _ = sample_configured_arrivals(
        count,
        config,
        start_time,
        rng,
    )
    return arrival_times, inter_arrival_times, source_device_ids


def sample_poisson_arrivals(*args, **kwargs):
    raise ValueError("Single-stream poisson arrivals were removed; use create_time_table/sample_configured_arrivals.")


def resolve_mean_interarrival(*args, **kwargs):
    raise ValueError("mean_interarrival was removed; use resolve_arrival_rate.")


def poisson_arrival_rate(value):
    return resolve_arrival_rate({"arrival_rate": value})


def debug_check_poisson_arrivals(rng=None):
    arrival_times, _, source_device_ids, _ = sample_configured_arrivals(
        1000,
        {"arrival_rate": DEFAULT_POISSON_ARRIVAL_RATE, "num_mobile_devices": 3},
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
