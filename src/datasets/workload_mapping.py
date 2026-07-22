"""把 Alibaba trace 的相对计算强度映射成仿真器用的 workload（cycles）。

设计原则（见需求第六~九节）：

* Alibaba 的 ``duration * (plan_cpu/100) * max(1, instance_num)`` 只是
  **相对计算强度权重 raw_weight**，量纲是 core-seconds，不是 cycles，
  绝不能直接当 workload_cycles 用。
* 真实 workload 由实验配置给定的 [workload_gc_min, workload_gc_max] 范围控制，
  按 raw_weight 的 log 归一化（p5/p95 截断）映射进该范围，再加轻微抖动。
* 边通信数据量 data_size_mb 不来自 trace，由配置范围内随机生成。
* 所有随机都走传入的 rng，保证给定 seed 可复现。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional

import numpy as np

LEGACY_WORKLOAD_FIELDS = ("task_length_mi", "task_length", "workload", "workload_mi")


class WorkloadMappingConfig:
    def __init__(
        self,
        workload_gc_min: float = 0.5,
        workload_gc_max: float = 20.0,
        jitter_min: float = 0.8,
        jitter_max: float = 1.2,
        data_size_mb_min: float = 1.0,
        data_size_mb_max: float = 30.0,
    ):
        self.workload_gc_min = workload_gc_min
        self.workload_gc_max = workload_gc_max
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self.data_size_mb_min = data_size_mb_min
        self.data_size_mb_max = data_size_mb_max

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]]) -> "WorkloadMappingConfig":
        config = dict(config or {})
        return cls(
            workload_gc_min=float(config.get("workload_gc_min", 0.5)),
            workload_gc_max=float(config.get("workload_gc_max", 20.0)),
            jitter_min=float(config.get("workload_jitter_min", 0.8)),
            jitter_max=float(config.get("workload_jitter_max", 1.2)),
            data_size_mb_min=float(config.get("data_size_mb_min", 1.0)),
            data_size_mb_max=float(config.get("data_size_mb_max", 30.0)),
        )


def compute_raw_weight(task: Mapping[str, Any]) -> float:
    """Trace 派生的相对计算强度（NOT cycles）。

    优先用预计算的 ``raw_weight``，否则用 duration/plan_cpu/instance_num 重算；
    再不行用 plan_cpu*instance_num；都缺失则返回 0（触发 uniform fallback）。
    """
    if task.get("raw_weight") is not None:
        return max(0.0, float(task["raw_weight"]))

    duration = task.get("duration")
    plan_cpu = task.get("plan_cpu")
    instance_num = task.get("instance_num")
    inst = max(1, int(instance_num)) if instance_num is not None else 1

    if duration is not None and plan_cpu is not None:
        return max(0.0, float(duration) * (float(plan_cpu) / 100.0) * inst)
    if plan_cpu is not None:
        return max(0.0, float(plan_cpu) * inst)
    return 0.0


class WorkloadMapper:
    """根据一个工作流母池的 raw_weight 统计量，把任务映射到 cycle workload。"""

    def __init__(
        self,
        pool: Iterable[Mapping[str, Any]],
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.cfg = WorkloadMappingConfig.from_mapping(config)

        z_values = []
        for workflow in pool:
            for task in workflow.get("tasks", []):
                weight = compute_raw_weight(task)
                if weight > 0:
                    z_values.append(math.log1p(weight))

        if z_values:
            arr = np.asarray(z_values, dtype=float)
            self.z_p5 = float(np.percentile(arr, 5))
            self.z_p95 = float(np.percentile(arr, 95))
        else:
            self.z_p5 = 0.0
            self.z_p95 = 0.0
        # no usable spread -> fall back to uniform sampling within the gc range
        self.has_spread = self.z_p95 > self.z_p5

    def assign_workload_cycles(self, task: Mapping[str, Any], rng) -> float:
        cfg = self.cfg
        weight = compute_raw_weight(task)
        if not self.has_spread or weight <= 0:
            workload_gc = rng.uniform(cfg.workload_gc_min, cfg.workload_gc_max)
        else:
            z = math.log1p(weight)
            z_norm = (z - self.z_p5) / (self.z_p95 - self.z_p5)
            z_norm = min(1.0, max(0.0, z_norm))
            workload_gc = cfg.workload_gc_min + z_norm * (
                cfg.workload_gc_max - cfg.workload_gc_min
            )

        jitter = rng.uniform(cfg.jitter_min, cfg.jitter_max)
        workload_gc = workload_gc * jitter
        workload_gc = min(cfg.workload_gc_max, max(cfg.workload_gc_min, workload_gc))
        return workload_gc * 1e9

    def assign_edge_data_mb(self, rng) -> float:
        return rng.uniform(self.cfg.data_size_mb_min, self.cfg.data_size_mb_max)

    def apply(self, workflow: dict[str, Any], rng) -> dict[str, Any]:
        """就地为 workflow 的每个 task 赋 workload_cycles、每条边赋 data_size_mb。"""
        for task in workflow.get("tasks", []):
            task["workload_cycles"] = self.assign_workload_cycles(task, rng)
            # legacy workload fields are not trusted as cycles; drop them.
            for field in LEGACY_WORKLOAD_FIELDS:
                task.pop(field, None)
        for edge in workflow.get("edges", []):
            edge["data_size_mb"] = self.assign_edge_data_mb(rng)
        return workflow
