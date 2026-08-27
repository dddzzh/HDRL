from __future__ import annotations

import math
from enum import Enum


class FaultToleranceTechnique(Enum):
    PB = 0
    TMR = 1
    NMR = 2


def voltage_dependent_fault_rate(voltage, voltage_max, lambda_0=1e-6, d=2.0):
    return float(lambda_0) * 10.0 ** ((float(voltage_max) - float(voltage)) / float(d))


def basic_reliability(fault_rate, execution_time_ms):
    return math.exp(-float(fault_rate) * float(execution_time_ms))


def pb_reliability(fault_rate, execution_time_ms, c=0.98):
    x = float(fault_rate) * float(execution_time_ms)
    return (1.0 + float(c) * x) * math.exp(-x)


def tmr_reliability(fault_rate, execution_time_ms):
    x = float(fault_rate) * float(execution_time_ms)
    return 3.0 * math.exp(-2.0 * x) - 2.0 * math.exp(-3.0 * x)


def nmr_reliability(fault_rate, execution_time_ms):
    x = float(fault_rate) * float(execution_time_ms)
    return 6.0 * math.exp(-5.0 * x) - 15.0 * math.exp(-4.0 * x) + 10.0 * math.exp(-3.0 * x)


def get_additional_replica_count(technique):
    return {FaultToleranceTechnique.PB: 1, FaultToleranceTechnique.TMR: 2, FaultToleranceTechnique.NMR: 4}[FaultToleranceTechnique(technique)]


def select_backup_server_by_min_utilization(servers, excluded_server_ids=None):
    excluded = {int(value) for value in (excluded_server_ids or ())}
    candidates = [server for server in servers if int(server.get("id", server.get("server_id"))) not in excluded]
    if not candidates:
        raise ValueError("No backup server remains after exclusions.")
    return min(candidates, key=lambda server: float(server.get("cpu_utilization", server.get("utilization", 0.0))))
