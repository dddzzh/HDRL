"""从全量阿里工作流中采样母池并切分训练/测试集。

流程：
1. 流式读取 `outputs/alibaba_workflows_all.json`（3GB+，用 ijson 避免占满内存）。
2. 蓄水池采样（reservoir sampling）随机抽取 ``--pool-size`` 个工作流（内存恒定）。
3. 精简字段（tasks 留 id/plan_cpu/plan_mem/instance_num/duration/raw_weight，edges 留 src/dst）。
4. 固定 seed 打乱后按 ``--test-ratio`` 切成训练 / 测试两份，写出精简 JSON。

到达时间不在此写入：训练时由采样器按 episode 现场生成泊松到达。

用法::

    python tools/split_alibaba_dataset.py --pool-size 50000 --test-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator

import ijson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "alibaba_workflows_all.json"
DEFAULT_TRAIN_OUT = PROJECT_ROOT / "outputs" / "alibaba_train.json"
DEFAULT_TEST_OUT = PROJECT_ROOT / "outputs" / "alibaba_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a master pool from the full Alibaba workflow set and split train/test."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--test-out", default=str(DEFAULT_TEST_OUT))
    parser.add_argument("--pool-size", type=int, default=50000, help="Master pool size (<=0 keeps all)")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _raw_weight(duration: float, plan_cpu: float, instance_num: int) -> float:
    # trace-derived RELATIVE compute-intensity weight (core-seconds), NOT cycles.
    return float(duration) * (float(plan_cpu) / 100.0) * max(1, int(instance_num))


def _strip_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Keep only structural + resource-request fields needed by the mapping layer.

    Intentionally drops the old (wrong) `workload` / `task_length_mi` fields; the
    sampler derives `workload_cycles` from `raw_weight` at experiment time.
    """
    tasks = []
    for task in workflow.get("tasks", []):
        plan_cpu = float(task.get("plan_cpu", 0.0) or 0.0)
        plan_mem = float(task.get("plan_mem", 0.0) or 0.0)
        instance_num = int(task.get("instance_num", 1) or 1)
        duration = float(task.get("duration", 0.0) or 0.0)
        tasks.append(
            {
                "id": int(task["id"]),
                "plan_cpu": plan_cpu,
                "plan_mem": plan_mem,
                "instance_num": instance_num,
                "duration": duration,
                "raw_weight": _raw_weight(duration, plan_cpu, instance_num),
            }
        )
    edges = [{"src": int(edge["src"]), "dst": int(edge["dst"])} for edge in workflow.get("edges", [])]
    return {
        "workflow_id": str(workflow.get("workflow_id")),
        "n_tasks": len(tasks),
        "n_edges": len(edges),
        "tasks": tasks,
        "edges": edges,
    }


def _stream_workflows(input_path: Path) -> Iterator[dict[str, Any]]:
    with input_path.open("rb") as file:
        yield from ijson.items(file, "workflows.item")


def reservoir_sample(input_path: Path, pool_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    total = 0
    keep_all = pool_size <= 0

    for workflow in _stream_workflows(input_path):
        total += 1
        stripped = _strip_workflow(workflow)
        if keep_all or len(reservoir) < pool_size:
            reservoir.append(stripped)
        else:
            index = rng.randint(0, total - 1)
            if index < pool_size:
                reservoir[index] = stripped
        if total % 100000 == 0:
            print(f"  scanned {total} workflows, reservoir={len(reservoir)}")

    print(f"Total workflows scanned: {total}")
    return reservoir


def write_split(path: Path, workflows: list[dict[str, Any]], split: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Alibaba Cluster Trace v2018 (sampled master pool)",
        "split": split,
        "seed": seed,
        "num_workflows": len(workflows),
        "network": {"edge_data_mb": 1.0},
        "workflows": workflows,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Sampling master pool (size={args.pool_size}) from {input_path} ...")
    pool = reservoir_sample(input_path, args.pool_size, args.seed)

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_test = int(len(pool) * args.test_ratio)
    test_workflows = pool[:n_test]
    train_workflows = pool[n_test:]

    write_split(Path(args.train_out), train_workflows, "train", args.seed)
    write_split(Path(args.test_out), test_workflows, "test", args.seed)

    print(f"Master pool size: {len(pool)}")
    print(f"Train workflows: {len(train_workflows)} -> {args.train_out}")
    print(f"Test workflows:  {len(test_workflows)} -> {args.test_out}")


if __name__ == "__main__":
    main()
