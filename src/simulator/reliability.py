from __future__ import annotations

import math
from typing import Mapping, Optional


def clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def server_frequency_hz(server: Mapping[str, float], rho: float) -> float:
    return float(rho) * float(server["max_freq_hz"])


def server_min_frequency_hz(server: Mapping[str, float]) -> float:
    if "min_freq_hz" in server:
        return float(server["min_freq_hz"])
    return float(server.get("min_rho", 0.1)) * float(server["max_freq_hz"])


def reliability_exponent(server: Mapping[str, float], default: float = 1.0) -> float:
    for key in ("failure_rate_exponent", "reliability_exponent", "failure_exponent"):
        if key in server and server[key] is not None:
            return float(server[key])
    return float(default)


def frequency_dependent_failure_rate(
    server: Mapping[str, float],
    rho: float,
    exponent_default: float = 1.0,
) -> float:
    """Failure rate after DVFS.

    Model:
        lambda_k(f) = lambda_k,max * 10 ** (d_k * (f_k,max - f) / (f_k,max - f_k,min))

    ``failure_rate`` in the resource config is treated as lambda_k,max, the
    failure rate at max frequency. ``f`` is the actual DVFS frequency.
    """

    lambda_at_max = float(server.get("failure_rate", server.get("failure_rate_max", 0.0)))
    if lambda_at_max <= 0:
        return 0.0

    f_max = float(server["max_freq_hz"])
    f_min = server_min_frequency_hz(server)
    f_actual = min(f_max, max(f_min, server_frequency_hz(server, rho)))
    denominator = max(f_max - f_min, 1e-12)
    d_k = reliability_exponent(server, exponent_default)
    scale = 10.0 ** (d_k * (f_max - f_actual) / denominator)
    return float(lambda_at_max * scale)


def single_replica_reliability(failure_rate: float, execution_time: float) -> float:
    """R(t) = exp(-lambda * t)."""

    lambda_t = max(0.0, float(failure_rate)) * max(0.0, float(execution_time))
    return clamp_probability(math.exp(-lambda_t))


def task_reliability(
    primary_reliability: float,
    backup_reliability: Optional[float] = None,
    backup_enabled: bool = False,
) -> float:
    """Task reliability with optional primary-backup execution.

    z_i = 0: R_i = R_i^p
    z_i = 1: R_i = 1 - (1 - R_i^p)(1 - R_i^b)
    """

    primary = clamp_probability(primary_reliability)
    if not backup_enabled:
        return primary

    backup = clamp_probability(backup_reliability if backup_reliability is not None else 0.0)
    return clamp_probability(1.0 - (1.0 - primary) * (1.0 - backup))


def task_reliability_from_server(
    server: Mapping[str, float],
    rho: float,
    execution_time: float,
    exponent_default: float = 1.0,
) -> tuple[float, float]:
    failure_rate = frequency_dependent_failure_rate(server, rho, exponent_default)
    return single_replica_reliability(failure_rate, execution_time), failure_rate


def workflow_reliability(task_reliabilities) -> float:
    reliability = 1.0
    for value in task_reliabilities:
        reliability *= clamp_probability(value)
    return clamp_probability(reliability)
