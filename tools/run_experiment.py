import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run dynamic edge-cloud workflow experiments.")
    parser.add_argument("--config", default="configs/experiment_default.json")
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--print-details", action="store_true")
    parser.add_argument("--max-workflows", type=int, default=20)
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--trace-scheduling", action="store_true")
    parser.add_argument("--trace-limit", type=int, default=100)
    parser.add_argument("--num-edge", type=int, default=None, help="Number of edge servers")
    parser.add_argument("--num-cloud", type=int, default=None, help="Number of cloud servers")
    return parser.parse_args()


def _fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_workflow_arrivals(workflow_results, max_rows):
    rows = sorted(workflow_results, key=lambda row: (row["arrival_time"], row["workflow_id"]))
    total = len(rows)
    shown = rows[:max_rows]
    print(f"\nWorkflow arrivals (showing {len(shown)}/{total}):")
    print("  index | workflow_id | arrival_time | finish_time | response_time | waiting_time")
    for index, row in enumerate(shown):
        print(
            "  "
            f"{index:>5} | "
            f"{row['workflow_id']} | "
            f"{_fmt(row['arrival_time'])} | "
            f"{_fmt(row['finish_time'])} | "
            f"{_fmt(row['response_time'])} | "
            f"{_fmt(row['waiting_time'])}"
        )


def _print_schedule_events(event_log, workflow_results, max_rows):
    arrivals = {
        str(row["workflow_id"]): row["arrival_time"]
        for row in workflow_results
    }
    total = len(event_log)
    shown = event_log[:max_rows]
    print(f"\nSchedule events (showing {len(shown)}/{total}):")
    print("  index | time | workflow_id | task_id | arrival_time | server_id | rho | start | finish | exec | energy")
    for index, event in enumerate(shown):
        workflow_id = str(event["workflow_id"])
        print(
            "  "
            f"{index:>5} | "
            f"{_fmt(event['start_time'])} | "
            f"{workflow_id} | "
            f"{event['task_id']} | "
            f"{_fmt(arrivals.get(workflow_id))} | "
            f"{event['server_id']} | "
            f"{_fmt(event['rho'])} | "
            f"{_fmt(event['start_time'])} | "
            f"{_fmt(event['finish_time'])} | "
            f"{_fmt(event['exec_time'])} | "
            f"{_fmt(event['energy'])}"
        )


def main():
    args = parse_args()
    runner = ExperimentRunner(
        args.config,
        project_root=PROJECT_ROOT,
        overrides={
            "scheduler": args.scheduler,
            "dataset": args.dataset,
            "seeds": args.seeds,
            "num_edge": args.num_edge,
            "num_cloud": args.num_cloud,
        },
    )
    if args.trace_scheduling:
        runner.config["simulation"]["trace_scheduling"] = True
        runner.config["simulation"]["trace_limit"] = args.trace_limit

    result = runner.run_all()
    config = result["config"]
    scheduler_name = config["scheduler"]["name"]

    for seed, output_dir, seed_result in result["seed_outputs"]:
        summary = seed_result["summary"]
        print(f"Experiment: {config['experiment_name']}")
        print(f"Scheduler: {scheduler_name}")
        print(f"Seed: {seed}")
        print(f"Finished workflows: {summary['total_finished_workflows']}")
        print(f"Total energy: {_fmt(summary['total_energy'])}")
        print(f"Average response time: {_fmt(summary['avg_workflow_response_time'])}")
        print(f"Average makespan: {_fmt(summary['avg_workflow_makespan'])}")
        print(f"Deadline success rate: {_fmt(summary['deadline_success_rate'])}")
        print(f"Results saved to: {output_dir}")
        if args.print_details:
            _print_workflow_arrivals(seed_result["workflow_results"], args.max_workflows)
            _print_schedule_events(
                seed_result["event_log"],
                seed_result["workflow_results"],
                args.max_events,
            )
    print(f"Aggregate summary: {result['aggregate_path']}")


if __name__ == "__main__":
    main()
