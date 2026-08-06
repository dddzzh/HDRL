from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "total_energy": "Energy (J)",
    "total_low_sac_energy_reference": "Reference energy (J)",
    "low_sac_total_energy_saving_ratio": "SAC saving ratio",
    "deadline_success_rate": "Deadline success rate",
    "reliability_success_rate": "Reliability success rate",
    "average_response_time": "Average response time (s)",
    "global_makespan": "Global makespan (s)",
    "mean_task_time_regret": "Mean task time regret",
    "rho_mean": "Mean rho",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot hierarchical D3QN+SAC convergence and seed evaluation.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--best-eval", default=None, help="JSONL containing one row per held-out seed.")
    parser.add_argument("--final-eval", default=None, help="JSONL containing one row per held-out seed.")
    parser.add_argument("--heft-eval", default=None, help="JSONL for HEFT using the same low-level checkpoint.")
    parser.add_argument("--seed-start", type=int, default=342000)
    parser.add_argument("--best-episode", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=5)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def numeric(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def smooth(values, window):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def style_axis(axis, best_episode):
    axis.grid(True, alpha=0.22, linewidth=0.7)
    if best_episode is not None:
        axis.axvline(best_episode, color="tab:red", linestyle="--", linewidth=1.1, alpha=0.8)


def plot_series(axis, episodes, rows, keys, labels, window, *, percent=False):
    colors = ("tab:blue", "tab:orange", "tab:green")
    for index, (key, label) in enumerate(zip(keys, labels)):
        values = numeric(rows, key)
        if percent:
            values = values * 100.0
        color = colors[index]
        axis.plot(episodes, values, color=color, alpha=0.18, linewidth=0.9)
        axis.plot(episodes, smooth(values, window), color=color, linewidth=2.0, label=label)
    if len(keys) > 1:
        axis.legend(frameon=False, fontsize=9)


def plot_convergence(run_dir, output_dir, best_episode, window):
    rows = load_jsonl(run_dir / "eval_history.jsonl")
    episodes = numeric(rows, "episode")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)

    plot_series(
        axes[0, 0], episodes, rows,
        ("total_energy", "total_low_sac_energy_reference"),
        ("Actual", "rho=1 reference"), window,
    )
    axes[0, 0].set_title("Evaluation energy")
    axes[0, 0].set_ylabel("J")

    plot_series(
        axes[0, 1], episodes, rows,
        ("deadline_success_rate", "reliability_success_rate"),
        ("Deadline", "Reliability"), window, percent=True,
    )
    axes[0, 1].set_title("Constraint success")
    axes[0, 1].set_ylabel("Success rate (%)")
    axes[0, 1].set_ylim(-2, 102)

    plot_series(
        axes[1, 0], episodes, rows,
        ("average_response_time", "global_makespan"),
        ("Average response", "Global makespan"), window,
    )
    axes[1, 0].set_title("Timing")
    axes[1, 0].set_ylabel("Seconds")

    plot_series(
        axes[1, 1], episodes, rows,
        ("mean_task_time_regret",), ("Task regret",), window,
    )
    axes[1, 1].set_title("Upper-level time regret")

    plot_series(
        axes[2, 0], episodes, rows,
        ("low_sac_total_energy_saving_ratio",), ("Saving ratio",), window, percent=True,
    )
    axes[2, 0].set_title("Lower-level SAC energy saving")
    axes[2, 0].set_ylabel("Saving ratio (%)")

    plot_series(
        axes[2, 1], episodes, rows,
        ("mean_high_makespan_reward", "mean_high_low_reward_bonus"),
        ("Time component", "Energy component"), window,
    )
    axes[2, 1].set_title("Upper-level reward components")

    for axis in axes.flat:
        style_axis(axis, best_episode)
        axis.set_xlabel("Training episode")
    if best_episode is not None:
        fig.suptitle(f"D3QN + SAC convergence (red dashed line: selected best checkpoint at ep {best_episode})")
    else:
        fig.suptitle("D3QN + SAC convergence")
    path = output_dir / "convergence_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    # Student-t critical value for 95% CI and df=19; the expected evaluation size is 20.
    t_critical = 2.093 if n == 20 else 1.96
    ci95 = float(t_critical * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_half_width": ci95,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def write_seed_analysis(best_rows, final_rows, output_dir, seed_start, heft_rows=None):
    if len(best_rows) != len(final_rows):
        raise ValueError("best and final evaluations must contain the same number of seeds.")
    names = ["best_ep580", "final_ep1500"]
    labels = ["D3QN best ep580", "D3QN final ep1500"]
    groups = [best_rows, final_rows]
    if heft_rows is not None:
        if len(best_rows) != len(heft_rows):
            raise ValueError("HEFT evaluation must contain the same number of seeds.")
        names.append("heft_same_sac")
        labels.append("HEFT + same SAC")
        groups.append(heft_rows)
    summary = {
        name: {key: describe(numeric(rows, key)) for key in METRICS}
        for name, rows in zip(names, groups)
    }
    summary["paired_final_minus_best"] = {
        key: describe(numeric(final_rows, key) - numeric(best_rows, key))
        for key in METRICS
    }
    if heft_rows is not None:
        summary["paired_heft_minus_best"] = {
            key: describe(numeric(heft_rows, key) - numeric(best_rows, key))
            for key in METRICS
        }
    summary["seed_start"] = int(seed_start)
    summary["seed_end"] = int(seed_start + len(best_rows) - 1)

    with (output_dir / "seed20_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    fields = ["checkpoint", "eval_seed", *METRICS.keys()]
    with (output_dir / "seed20_per_seed.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for checkpoint, rows in zip(names, groups):
            for offset, row in enumerate(rows):
                writer.writerow({
                    "checkpoint": checkpoint,
                    "eval_seed": seed_start + offset,
                    **{key: row[key] for key in METRICS},
                })

    panels = (
        ("total_energy", "Energy (J)", 1.0),
        ("average_response_time", "Average response (s)", 1.0),
        ("global_makespan", "Global makespan (s)", 1.0),
        ("deadline_success_rate", "Deadline success (%)", 100.0),
        ("low_sac_total_energy_saving_ratio", "SAC saving (%)", 100.0),
        ("mean_task_time_regret", "Task time regret", 1.0),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    positions = np.arange(1.0, len(groups) + 1.0)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for axis, (key, title, scale) in zip(axes.flat, panels):
        data = [numeric(rows, key) * scale for rows in groups]
        axis.boxplot(data, positions=positions, widths=0.5, showfliers=False)
        for group_index, values in enumerate(data):
            jitter = np.linspace(-0.09, 0.09, len(values))
            axis.scatter(
                np.full(len(values), positions[group_index]) + jitter,
                values,
                s=18,
                alpha=0.55,
                color=colors[group_index],
            )
            axis.scatter(
                positions[group_index], np.mean(values), marker="D", s=45,
                color=colors[group_index], zorder=4,
            )
        axis.set_xticks(positions, labels, rotation=8)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.22, linewidth=0.7)
    fig.suptitle(f"Held-out evaluation across seeds {seed_start}-{seed_start + len(best_rows) - 1} (diamonds: means)")
    path = output_dir / "seed20_checkpoint_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, summary


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_path = plot_convergence(
        run_dir, output_dir, args.best_episode, max(1, int(args.smooth_window))
    )
    print(f"convergence={convergence_path}")

    if args.best_eval and args.final_eval:
        best_rows = load_jsonl(args.best_eval)
        final_rows = load_jsonl(args.final_eval)
        heft_rows = load_jsonl(args.heft_eval) if args.heft_eval else None
        comparison_path, summary = write_seed_analysis(
            best_rows, final_rows, output_dir, args.seed_start, heft_rows
        )
        print(f"seed_comparison={comparison_path}")
        print(json.dumps({
            "best_energy": summary["best_ep580"]["total_energy"],
            "final_energy": summary["final_ep1500"]["total_energy"],
            "paired_energy_delta": summary["paired_final_minus_best"]["total_energy"],
        }, indent=2))


if __name__ == "__main__":
    main()
