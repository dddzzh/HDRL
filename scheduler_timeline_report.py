from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppo.dual_agent import HighLevelPPOAgent
from src.schedulers.registry import create_scheduler
from src.simulator.metrics import compute_summary
from src.simulator.resource import load_resources
from staged_p3o import HighOnlyTrainingAgent
from tools import train_hierarchical_ppo as train


def parse_args():
    parser = argparse.ArgumentParser(description="Compare schedulers and export queue timeline/Gantt reports.")
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--algorithms", nargs="+", default=["high_only", "fcfs", "heft"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["pegasus", "alibaba_json", "alibaba_pool"], default=None)
    parser.add_argument("--num-workflows", type=int, default=None)
    parser.add_argument("--task-count", type=int, default=None)
    parser.add_argument("--max-ready-tasks", type=int, default=None)
    parser.add_argument(
        "--arrival-process",
        choices=["poisson", "multi_device_poisson", "device_poisson", "senior_poisson"],
        default=None,
    )
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--num-mobile-devices", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-rho", type=float, default=None)
    parser.add_argument(
        "--mean-interarrival",
        type=float,
        default=None,
        help="Override Poisson mean interarrival time; larger means lower arrival intensity.",
    )
    parser.add_argument("--output-dir", default="results/timeline_reports")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(args, checkpoint):
    if args.config:
        path = resolve_path(args.config)
    else:
        path = checkpoint.parent / "effective_config.json"
        if not path.exists():
            path = PROJECT_ROOT / "configs" / "hierarchical_ppo_default.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file), path


def make_train_args(args):
    return SimpleNamespace(
        dataset=args.dataset,
        num_workflows=args.num_workflows,
        task_count=args.task_count,
        max_ready_tasks=args.max_ready_tasks,
        arrival_process=args.arrival_process,
        arrival_rate=args.arrival_rate,
        mean_interarrival=args.mean_interarrival,
        num_mobile_devices=args.num_mobile_devices,
        device=args.device,
    )


def build_resources(config, seed):
    return load_resources(
        str(PROJECT_ROOT / config["resources"]["path"])
        if not Path(config["resources"]["path"]).is_absolute()
        else config["resources"]["path"],
        num_edge=config["resources"]["num_edge"],
        num_cloud=config["resources"]["num_cloud"],
        seed=seed,
    )


def build_high_only_agent(config, args, resources, source, checkpoint):
    agent_config = copy.deepcopy(config.get("agent", {}))
    agent_config.setdefault("shared", {}).setdefault("device", args.device)
    agent_config["shared"]["device"] = args.device
    agent_config["shared"].setdefault("state", {})
    train_args = make_train_args(args)
    agent_config["shared"]["state"]["max_ready_tasks"] = train.infer_ready_task_capacity(
        config,
        train_args,
        source,
    )
    agent_config["shared"]["state"]["max_servers"] = train.infer_server_count(resources)
    agent_config["shared"]["state"]["time_horizon"] = float(
        config.get("simulation", {}).get(
            "time_horizon",
            agent_config["shared"]["state"].get("time_horizon", 3600.0),
        )
    )
    high_cfg = {**agent_config["shared"], **dict(agent_config.get("high", {}))}
    high_cfg["checkpoint"] = str(checkpoint)
    fixed_rho = args.fixed_rho
    if fixed_rho is None:
        fixed_rho = float(config.get("simulation", {}).get("default_rho", 1.0))
    return HighOnlyTrainingAgent(HighLevelPPOAgent(high_cfg, seed=args.seed), fixed_rho)


def run_high_only(env, agent):
    result = train.run_evaluation_episode(env, agent, reward_config={}, deterministic=True)
    return result["summary"], result["event_log"]


def run_baseline(env, name):
    scheduler = create_scheduler(name, params={"rho": 1.0}, seed=0)
    result = env.run(scheduler)
    return compute_summary(result["workflow_results"], result["event_log"], env.servers), result["event_log"]


def queue_snapshot(event_log, servers):
    server_ids = sorted(int(server["id"]) for server in servers)
    rows = []
    for server_id in server_ids:
        changes = []
        event_by_key = {}
        for index, event in enumerate(event_log):
            if int(event.get("server_id", -1)) != server_id:
                continue
            if event.get("queue_type", "execution") != "execution":
                continue
            key = f"{index}:{event.get('workflow_id')}:t{event.get('task_id')}"
            event_by_key[key] = event
            ready_time = float(event.get("base_ready_time", event.get("start_time", 0.0)))
            start_time = float(event.get("start_time", ready_time))
            finish_time = float(event.get("finish_time", start_time))
            changes.append((ready_time, 0, key))
            changes.append((start_time, 1, key))
            changes.append((finish_time, 2, key))
        queued = set()
        running = set()
        for time, kind, key in sorted(changes, key=lambda item: (item[0], item[1])):
            if kind == 0:
                queued.add(key)
            elif kind == 1:
                queued.discard(key)
                running.add(key)
            else:
                running.discard(key)
            if not queued and not running:
                continue
            queued_items = sorted(queued)[:20]
            running_items = sorted(running)[:5]
            rows.append(
                {
                    "time": time,
                    "server_id": server_id,
                    "queued_count": len(queued),
                    "running_count": len(running),
                    "queued_tasks": ";".join(
                        f"{event_by_key[item].get('workflow_id')}:t{event_by_key[item].get('task_id')}"
                        for item in queued_items
                    ),
                    "running_tasks": ";".join(
                        f"{event_by_key[item].get('workflow_id')}:t{event_by_key[item].get('task_id')}"
                        for item in running_items
                    ),
                }
            )
    rows.sort(key=lambda row: (row["time"], row["server_id"]))
    return rows


def write_queue_csv(path, rows):
    fields = ["time", "server_id", "queued_count", "running_count", "queued_tasks", "running_tasks"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_gantt_svg(path, event_log, servers, title):
    primary = [
        event
        for event in event_log
        if event.get("replica_role", "primary") == "primary"
        and event.get("queue_type", "execution") == "execution"
    ]
    server_ids = sorted({int(server["id"]) for server in servers})
    used_server_ids = sorted({int(event.get("server_id", -1)) for event in primary})
    if used_server_ids:
        server_ids = [server_id for server_id in server_ids if server_id in set(used_server_ids)]
    server_index = {server_id: index for index, server_id in enumerate(server_ids)}
    max_finish = max((float(event.get("finish_time", 0.0)) for event in primary), default=1.0)
    width = 1800
    row_h = 24
    left = 120
    top = 50
    height = top + max(1, len(server_ids)) * row_h + 78
    plot_w = width - left - 40
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{left}" y="28" font-size="12" fill="#555">server execution timeline, time 0 - {max_finish:.3f}s</text>',
    ]
    for tick in range(6):
        x = left + plot_w * tick / 5
        t = max_finish * tick / 5
        lines.append(f'<line x1="{x:.2f}" y1="{top - 8}" x2="{x:.2f}" y2="{height - 42}" stroke="#eee"/>')
        lines.append(f'<text x="{x - 16:.2f}" y="{height - 22}" font-size="10" fill="#666">{t:.1f}s</text>')
    for server_id, idx in server_index.items():
        y = top + idx * row_h
        lines.append(f'<text x="18" y="{y + 16}" font-size="11">server {server_id}</text>')
        lines.append(f'<line x1="{left}" y1="{y + row_h - 2}" x2="{width - 30}" y2="{y + row_h - 2}" stroke="#eee"/>')
    for event in primary:
        server_id = int(event.get("server_id", -1))
        if server_id not in server_index:
            continue
        workflow_id = str(event.get("workflow_id"))
        task_id = event.get("task_id")
        y = top + server_index[server_id] * row_h + 3
        start = float(event.get("start_time", 0.0))
        finish = float(event.get("finish_time", start))
        ready = float(event.get("base_ready_time", start))
        deadline = event.get("deadline")
        wait = max(0.0, start - ready)
        missed = deadline is not None and finish > float(deadline)
        x = left + start / max_finish * plot_w
        w = max(1.0, (finish - start) / max_finish * plot_w)
        color = colors[hash(workflow_id) % len(colors)]
        label = f"w{workflow_id}:t{task_id}"
        deadline_text = "n/a" if deadline is None else f"{float(deadline):.3f}"
        stroke = '#d62728' if missed else '#333'
        stroke_w = '1.2' if missed else '0.2'
        lines.append(
            f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="{row_h - 7}" '
            f'fill="{color}" opacity="0.82" stroke="{stroke}" stroke-width="{stroke_w}">'
            f'<title>{html.escape(label)} server={server_id} ready={ready:.3f} '
            f'start={start:.3f} finish={finish:.3f} wait={wait:.3f} deadline={deadline_text}</title></rect>'
        )
        if w >= 34:
            lines.append(
                f'<text x="{x + 3:.2f}" y="{y + row_h - 12}" font-size="9" fill="white">'
                f'{html.escape(str(task_id))}</text>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_queue_peaks(name, rows):
    peaks = sorted(rows, key=lambda row: (row["queued_count"], row["running_count"]), reverse=True)[:10]
    print(f"[{name}] top queue snapshots")
    for row in peaks:
        print(
            f"  t={row['time']:.4f} server={row['server_id']} "
            f"queued={row['queued_count']} running={row['running_count']} "
            f"queued_tasks={row['queued_tasks'][:120]}"
        )


def main():
    args = parse_args()
    checkpoint = resolve_path(args.high_checkpoint)
    config, config_path = load_config(args, checkpoint)
    if args.mean_interarrival is not None:
        config.setdefault("dataset", {})["mean_interarrival"] = float(args.mean_interarrival)
        config.setdefault("generator", {})["mean_interarrival"] = float(args.mean_interarrival)
        config.setdefault("dataset", {}).pop("arrival_rate", None)
        config.setdefault("generator", {}).pop("arrival_rate", None)
    if args.arrival_rate is not None:
        config.setdefault("dataset", {})["arrival_rate"] = float(args.arrival_rate)
        config.setdefault("generator", {})["arrival_rate"] = float(args.arrival_rate)
        config.setdefault("dataset", {}).pop("mean_interarrival", None)
        config.setdefault("generator", {}).pop("mean_interarrival", None)
    if args.arrival_process is not None:
        config.setdefault("dataset", {})["arrival_process"] = args.arrival_process
        config.setdefault("generator", {})["arrival_process"] = args.arrival_process
    if args.num_mobile_devices is not None:
        config.setdefault("dataset", {})["num_mobile_devices"] = int(args.num_mobile_devices)
        config.setdefault("generator", {})["num_mobile_devices"] = int(args.num_mobile_devices)
    resources = build_resources(config, args.seed)
    train_args = make_train_args(args)
    source = train._prepare_eval_workflow_source(config, train_args, resources, args.seed)
    workflows = train._workflows_for_episode(source, config, train_args, resources, args.seed)
    output_dir = resolve_path(args.output_dir) / checkpoint.parent.name / checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    high_agent = None
    results = {}
    for name in args.algorithms:
        env = train.build_env(copy.deepcopy(workflows), resources, copy.deepcopy(config.get("simulation", {})), seed=args.seed)
        if name == "high_only":
            if high_agent is None:
                high_agent = build_high_only_agent(config, args, resources, source, checkpoint)
            summary, event_log = run_high_only(env, high_agent)
        else:
            summary, event_log = run_baseline(env, name)
        queue_rows = queue_snapshot(event_log, env.servers)
        algo_dir = output_dir / name
        algo_dir.mkdir(parents=True, exist_ok=True)
        write_queue_csv(algo_dir / "server_queue_timeline.csv", queue_rows)
        with (algo_dir / "event_log.json").open("w", encoding="utf-8") as file:
            json.dump(event_log, file, ensure_ascii=False, indent=2)
        write_gantt_svg(algo_dir / "server_gantt.svg", event_log, env.servers, f"{name} server/task Gantt")
        write_gantt_svg(algo_dir / "workflow_gantt.svg", event_log, env.servers, f"{name} server/task Gantt")
        results[name] = {
            "summary": summary,
            "queue_timeline": str(algo_dir / "server_queue_timeline.csv"),
            "event_log": str(algo_dir / "event_log.json"),
            "gantt": str(algo_dir / "server_gantt.svg"),
        }
        print(
            f"[{name}] energy={train.fmt_metric(summary.get('total_energy'))}J "
            f"deadline_rate={train.fmt_metric(summary.get('deadline_success_rate'))} "
            f"finished={train.fmt_metric(summary.get('total_finished_workflows'))}"
        )
        print_queue_peaks(name, queue_rows)
    payload = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "dataset": source["description"],
        "seed": args.seed,
        "results": results,
    }
    with (output_dir / "comparison_report.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"report saved: {output_dir / 'comparison_report.json'}")


if __name__ == "__main__":
    main()
