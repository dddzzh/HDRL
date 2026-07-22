"""统一的工作流生成入口。

通过一个函数在「随机合成」和「数据集加载（如阿里 trace）」之间切换，
方便训练 / 评估 / 实验脚本复用。

用法::

    from src.datasets.workflow_factory import create_workflows

    # 随机生成（每个 episode 换 seed 即可得到不同工作流）
    wfs = create_workflows(
        {"source": "pegasus", "num_workflows": 100, "arrival_rate": 0.01},
        seed=episode_seed,
        resources=resources,
    )

    # 阿里数据集（固定数据集）
    wfs = create_workflows(
        {"source": "alibaba_json", "path": "outputs/alibaba_train.json"},
        seed=42,
    )
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from src.datasets.alibaba_dataset import AlibabaWorkflowDataset
from src.datasets.arrival_process import (
    DEFAULT_POISSON_ARRIVAL_RATE,
    resolve_arrival_rate,
    sample_configured_arrivals,
)
from src.datasets.pegasus_workflow_generator import PegasusWorkflowGenerator
from src.datasets.workload_mapping import WorkloadMapper
from src.simulator.deadline import cai_2025_deadline_fields, validate_deadline_type
from src.simulator.workflow import build_parents_children
from src.simulator.workflow import Workflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PEGASUS = "pegasus"
SOURCE_ALIBABA = "alibaba_json"
VALID_SOURCES = (SOURCE_PEGASUS, SOURCE_ALIBABA)

WorkflowList = list


def create_workflows(
    config: Optional[Mapping[str, Any]] = None,
    *,
    source: Optional[str] = None,
    seed: int = 42,
    resources: Optional[Mapping[str, Any]] = None,
) -> WorkflowList:
    """根据配置生成一批工作流。

    Args:
        config: 工作流配置。可含 ``source``（或 ``type``）以及对应来源的参数。
        source: 显式指定来源，覆盖 ``config`` 里的设置。
        seed: 随机种子（随机生成用它造工作流；阿里数据集用它做可选 shuffle）。
        resources: 资源字典（含 servers / network）。提供后随机生成会用真实
            服务器与链路计算 deadline，更准确。

    Returns:
        工作流列表（随机来源是 ``Workflow`` 对象；阿里来源是规范化 dict，
        二者都能直接喂给 ``EdgeCloudWorkflowEnv``）。
    """
    config = dict(config or {})
    resolved_source = _resolve_source(config, source)

    if resolved_source == SOURCE_PEGASUS:
        return _create_pegasus(config, seed=seed, resources=resources)
    if resolved_source == SOURCE_ALIBABA:
        return _create_alibaba(config, seed=seed, resources=resources)
    raise ValueError(
        f"Unknown workflow source: {resolved_source!r} (expected one of {VALID_SOURCES})"
    )


def _resolve_source(config: Mapping[str, Any], source: Optional[str]) -> str:
    resolved = source or config.get("source") or config.get("type") or SOURCE_PEGASUS
    return str(resolved)


def _create_pegasus(
    config: Mapping[str, Any],
    *,
    seed: int,
    resources: Optional[Mapping[str, Any]],
) -> WorkflowList:
    generator_cfg = {
        key: value
        for key, value in config.items()
        if key not in ("source", "type")
    }
    generator = PegasusWorkflowGenerator(generator_cfg, seed=seed)
    if resources is not None:
        return generator.generate(
            servers=resources["servers"],
            link_bandwidth_mbps=resources["network"]["link_bandwidth_mbps"],
        )
    return generator.generate()


TASK_DEADLINE_FIELDS = ("deadline", "sub_deadline", "local_deadline", "relative_deadline_s", "deadline_ms")


def _create_alibaba(
    config: Mapping[str, Any],
    *,
    seed: int,
    resources: Optional[Mapping[str, Any]],
) -> WorkflowList:
    dataset_cfg = {
        key: value
        for key, value in config.items()
        if key not in ("source", "type")
    }
    dataset_cfg.setdefault("path", "outputs/alibaba_train.json")
    dataset_cfg["path"] = _resolve_path(dataset_cfg["path"])

    workflows = AlibabaWorkflowDataset(dataset_cfg, seed=seed).load()

    # map trace-derived relative weights -> cycle workloads + synth edge data sizes
    mapper = WorkloadMapper(workflows, config)
    rng = random.Random(seed)
    reference_makespan_cache = {}
    for workflow in workflows:
        mapper.apply(workflow, rng)
        apply_workflow_constraints(
            workflow,
            dataset_cfg,
            resources=resources,
            rng=rng,
            reference_makespan_cache=reference_makespan_cache,
        )
    return workflows


def resolve_deadline_type(config: Optional[Mapping[str, Any]], default: str = "medium") -> str:
    config = dict(config or {})
    return validate_deadline_type(config.get("deadline_type", default))


def resolve_reliability_requirement(config: Optional[Mapping[str, Any]], default: float = 0.9) -> float:
    config = dict(config or {})
    for key in ("Rreq", "rreq", "reliability_req", "reliability_target", "default_reliability"):
        if key in config and config[key] is not None:
            return float(config[key])
    return float(default)


def apply_workflow_constraints(
    workflow,
    config: Optional[Mapping[str, Any]],
    *,
    resources: Optional[Mapping[str, Any]],
    rng: Optional[random.Random] = None,
    reference_makespan_cache: Optional[dict[str, float]] = None,
):
    """Apply workflow-level deadline/reliability constraints and remove task deadlines."""
    config = dict(config or {})
    _clear_task_deadlines(workflow)
    rreq = resolve_reliability_requirement(config)
    if isinstance(workflow, Workflow):
        workflow.reliability = rreq
    else:
        workflow["reliability"] = rreq

    if resources is None:
        return workflow

    if _has_complete_deadline_fields(workflow):
        return workflow

    deadline_type = resolve_deadline_type(config)
    rng = rng or random.Random(int(config.get("seed", 42)))
    servers = resources["servers"]
    links = resources.get("network", {}).get("link_bandwidth_mbps")
    rho = float(config.get("deadline_heft_rho", config.get("heft_rho", 1.0)))
    if not isinstance(workflow, Workflow):
        build_parents_children(workflow, default_edge_data_mb=config.get("data_size_mb_min", 1.0))
    arrival_time = workflow.arrival_time if isinstance(workflow, Workflow) else workflow["arrival_time"]
    deadline_info = cai_2025_deadline_fields(
        workflow,
        float(arrival_time),
        servers,
        deadline_type=deadline_type,
        rng=rng,
        link_bandwidth_mbps=links,
        rho=rho,
        cache=reference_makespan_cache,
    )
    _set_deadline_fields(workflow, deadline_info)
    _log_deadline_fields(workflow, deadline_info, config)
    return workflow


def _has_complete_deadline_fields(workflow) -> bool:
    if isinstance(workflow, Workflow):
        return (
            workflow.reference_makespan is not None
            and workflow.deadline_multiplier is not None
            and workflow.relative_deadline is not None
            and workflow.absolute_deadline is not None
        )
    return all(
        workflow.get(field) is not None
        for field in ("reference_makespan", "deadline_multiplier", "relative_deadline", "absolute_deadline")
    )


def _set_deadline_fields(workflow, deadline_info: Mapping[str, Any]) -> None:
    reference_makespan = float(deadline_info["reference_makespan"])
    deadline_multiplier = float(deadline_info["deadline_multiplier"])
    relative_deadline = float(deadline_info["relative_deadline"])
    absolute_deadline = float(deadline_info["absolute_deadline"])
    if isinstance(workflow, Workflow):
        workflow.reference_makespan = reference_makespan
        workflow.deadline_multiplier = deadline_multiplier
        workflow.relative_deadline = relative_deadline
        workflow.absolute_deadline = absolute_deadline
        workflow.deadline_type = str(deadline_info["deadline_type"])
        return
    workflow["reference_makespan"] = reference_makespan
    workflow["deadline_multiplier"] = deadline_multiplier
    workflow["relative_deadline"] = relative_deadline
    workflow["absolute_deadline"] = absolute_deadline
    workflow["deadline_type"] = str(deadline_info["deadline_type"])


def _log_deadline_fields(workflow, deadline_info: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if not bool(config.get("log_deadline_generation", config.get("trace_deadline_generation", False))):
        return
    workflow_id = workflow.workflow_id if isinstance(workflow, Workflow) else workflow.get("workflow_id")
    workflow_type = (
        getattr(workflow, "workflow_type", None)
        if isinstance(workflow, Workflow)
        else workflow.get("workflow_type", workflow.get("type", "unknown"))
    )
    arrival_time = workflow.arrival_time if isinstance(workflow, Workflow) else workflow.get("arrival_time")
    print(
        "DEADLINE "
        f"workflow_id={workflow_id} "
        f"workflow_type={workflow_type or 'unknown'} "
        f"arrival_time={float(arrival_time):.6f} "
        f"reference_makespan={float(deadline_info['reference_makespan']):.6f} "
        f"deadline_type={deadline_info['deadline_type']} "
        f"deadline_multiplier={float(deadline_info['deadline_multiplier']):.1f} "
        f"relative_deadline={float(deadline_info['relative_deadline']):.6f} "
        f"absolute_deadline={float(deadline_info['absolute_deadline']):.6f}"
    )


def _clear_task_deadlines(workflow) -> None:
    if isinstance(workflow, Workflow):
        return
    for task in workflow.get("tasks", []):
        for field in TASK_DEADLINE_FIELDS:
            task.pop(field, None)


def _resolve_path(path: Union[str, Path]) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


class WorkflowSampler:
    """从一个固定母池中按 episode 随机抽取工作流，并现场生成泊松到达。

    适合「每个 episode 从训练/测试池随机抽 N 个工作流」的实验设置：
    母池只加载一次，每次 ``sample`` 用给定 seed 抽样并赋新的泊松到达时间，
    保证每集不同且可复现。

    用法::

        sampler = WorkflowSampler.from_file("outputs/alibaba_train.json")
        wfs = sampler.sample(num_workflows=100, seed=episode_seed)
    """

    def __init__(
        self,
        pool: list,
        *,
        arrival_rate: float = DEFAULT_POISSON_ARRIVAL_RATE,
        start_time: float = 0.0,
        mapping_config: Optional[Mapping[str, Any]] = None,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not pool:
            raise ValueError("workflow pool is empty")
        self.pool = pool
        arrival_config = dict(mapping_config or {})
        arrival_config.setdefault("arrival_rate", arrival_rate)
        arrival_config["arrival_process"] = "multi_device_poisson"
        self.arrival_process = "multi_device_poisson"
        self.arrival_rate = resolve_arrival_rate(arrival_config)
        self.start_time = float(start_time)
        self.arrival_config = arrival_config
        self.resources = resources
        # workload (cycles) + edge data size mapping is computed from the whole pool
        self.mapper = WorkloadMapper(pool, mapping_config)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        arrival_rate: float = DEFAULT_POISSON_ARRIVAL_RATE,
        start_time: float = 0.0,
        mapping_config: Optional[Mapping[str, Any]] = None,
        resources: Optional[Mapping[str, Any]] = None,
    ) -> "WorkflowSampler":
        resolved = _resolve_path(path)
        with open(resolved, "r", encoding="utf-8") as file:
            data = json.load(file)
        pool = data["workflows"] if isinstance(data, dict) else data
        return cls(
            pool,
            arrival_rate=arrival_rate,
            start_time=start_time,
            mapping_config=mapping_config,
            resources=resources,
        )

    def __len__(self) -> int:
        return len(self.pool)

    def sample(self, num_workflows: int, seed: int) -> WorkflowList:
        if num_workflows <= 0:
            raise ValueError("num_workflows must be positive")
        rng = random.Random(seed)
        count = min(num_workflows, len(self.pool))
        chosen = rng.sample(self.pool, count)

        sampled: WorkflowList = []
        reference_makespan_cache = {}
        arrival_times, inter_arrival_times, source_device_ids, arrival_process = sample_configured_arrivals(
            count,
            self.arrival_config,
            self.start_time,
            random.Random(seed),
        )
        for index, template in enumerate(chosen):
            workflow = copy.deepcopy(template)
            workflow.pop("arrival_time", None)
            workflow["arrival_index"] = index
            workflow["arrival_time"] = float(arrival_times[index])
            workflow["inter_arrival_time"] = float(inter_arrival_times[index])
            workflow["arrival_process"] = arrival_process
            workflow["arrival_rate"] = float(self.arrival_rate)
            workflow["source_device_id"] = source_device_ids[index]
            workflow["workflow_id"] = f"{workflow.get('workflow_id', 'wf')}_ep{index}"
            # assign cycle workloads + synth edge data sizes (seed-reproducible)
            self.mapper.apply(workflow, rng)
            apply_workflow_constraints(
                workflow,
                self.arrival_config,
                resources=self.resources,
                rng=rng,
                reference_makespan_cache=reference_makespan_cache,
            )
            sampled.append(workflow)
        return sampled
