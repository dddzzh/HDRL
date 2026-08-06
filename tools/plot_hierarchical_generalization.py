from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODE_LABELS = {
    "d3qn_sac": "D3QN + SAC",
    "heft_same_sac": "HEFT + same SAC",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot hierarchical generalization evaluation.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_summary(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in (
            "scale_value", "num_workflows", "num_edge", "num_cloud", "num_servers",
            "n", "mean", "std", "ci95_half_width", "min", "max",
        ):
            row[key] = float(row[key])
    return rows


def series(rows, dimension, metric, mode):
    selected = [
        row for row in rows
        if row["dimension"] == dimension and row["metric"] == metric and row["mode"] == mode
    ]
    selected.sort(key=lambda row: row["scale_value"])
    return (
        [row["scale_value"] for row in selected],
        [row["mean"] for row in selected],
        [row["ci95_half_width"] for row in selected],
    )


def plot_panel(axis, rows, dimension, metric, title, ylabel, *, percent=False):
    colors = ("tab:blue", "tab:green")
    markers = ("o", "s")
    modes = sorted({row["mode"] for row in rows})
    for index, mode in enumerate(modes):
        x, mean, error = series(rows, dimension, metric, mode)
        if not x:
            continue
        if percent:
            mean = [value * 100.0 for value in mean]
            error = [value * 100.0 for value in error]
        axis.errorbar(
            x,
            mean,
            yerr=error,
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            linewidth=1.8,
            capsize=3,
            label=MODE_LABELS.get(mode, mode),
        )
    axis.set_title(title)
    axis.set_xlabel("Workflow count" if dimension == "workflow_count" else "VM count")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.legend(frameon=False, fontsize=9)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    rows = read_summary(input_dir / "generalization_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plot_panel(
        axes[0, 0], rows, "workflow_count", "energy_per_workflow",
        "Workload scaling: energy", "Energy per workflow (J)",
    )
    plot_panel(
        axes[0, 1], rows, "workflow_count", "average_response_time",
        "Workload scaling: response", "Average response (s)",
    )
    plot_panel(
        axes[0, 2], rows, "workflow_count", "deadline_success_rate",
        "Workload scaling: deadlines", "Deadline success (%)", percent=True,
    )
    plot_panel(
        axes[1, 0], rows, "vm_count", "energy_per_workflow",
        "Resource scaling: energy", "Energy per workflow (J)",
    )
    plot_panel(
        axes[1, 1], rows, "vm_count", "average_response_time",
        "Resource scaling: response", "Average response (s)",
    )
    plot_panel(
        axes[1, 2], rows, "vm_count", "deadline_success_rate",
        "Resource scaling: deadlines", "Deadline success (%)", percent=True,
    )
    fig.suptitle("D3QN + SAC generalization (mean and 95% CI)")
    output = Path(args.output).resolve() if args.output else input_dir / "generalization_overview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"plot={output}")


if __name__ == "__main__":
    main()
