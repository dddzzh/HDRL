"""Plot the existing 20-seed DRLEC comparison results.

Outputs:
  energy_comparison.png
  deadline_success_comparison.png
  energy_deadline_with_std.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 9, "savefig.dpi": 600})


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def load_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def summary_report(path: str):
    data = load_json(path)
    return data["summary"]


def baseline_summary(path: str, mode: str):
    data = load_json(path)
    for item in data:
        if item.get("mode") == mode:
            return item["metrics"]
    raise KeyError(f"Mode {mode!r} not found in {path}")


def metric(metrics, name):
    if name not in metrics and name == "average_workflow_response_time":
        name = "average_response_time"
    return float(metrics[name]["mean"])


def metric_std(metrics, name):
    if name not in metrics and name == "average_workflow_response_time":
        name = "average_response_time"
    return float(metrics[name]["std"])


def collect_data():
    baselines = "results/comparisons/episode_makespan_20seed/comparison_summary.json"
    dream = "results/comparisons/episode_makespan_20seed/comparison_summary.json"
    rows = [
        ("U1.3\nD3QN", summary_report(
            "results/hierarchical_dual_head_d3qn_dense_miss_credit/20260824_200638/eval_episode_makespan_20seed.json"
        )),
        ("U1.3 + SAC\nseparate", summary_report(
            "results/hierarchical_dual_head_d3qn_sac_freeze_high/20260824_210601/eval_episode_makespan_sac_20seed.json"
        )),
        ("U1.3 + SAC\njoint", summary_report(
            "results/hierarchical_dual_head_d3qn_sac_joint/20260825_113856/eval_episode_makespan_sac_20seed.json"
        )),
        ("DQN-EDGE", baseline_summary(baselines, "dqn_edge")),
        ("OOAC", baseline_summary(
            "results/comparisons/episode_makespan_ooac_20seed/comparison_summary.json",
            "oo_actor_critic"
        )),
        ("Dream-DQN", baseline_summary(dream, "dream_dqn")),
    ]
    result = {"label": [], "deadline": [], "deadline_std": [], "energy": [], "energy_std": []}
    for label, metrics in rows:
        result["label"].append(label)
        result["deadline"].append(metric(metrics, "deadline_success_rate") * 100.0)
        result["deadline_std"].append(metric_std(metrics, "deadline_success_rate") * 100.0)
        result["energy"].append(metric(metrics, "total_energy") / 1000.0)
        result["energy_std"].append(metric_std(metrics, "total_energy") / 1000.0)
    return result


def style(ax):
    ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#bdbdbd", alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    data = collect_data()
    x = np.arange(len(data["label"]))
    colors = ["#1f77b4", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    hatches = ["//", "\\\\", "xx", "..", "++", "oo"]

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)
    bars = ax.bar(x, data["energy"], yerr=data["energy_std"], capsize=4, color=colors, edgecolor="black", linewidth=0.7, hatch=hatches, error_kw={"elinewidth": 0.9, "ecolor": "black"})
    ax.set_ylabel("Total energy (kJ)")
    ax.set_title("20-seed total energy comparison")
    ax.set_xticks(x, data["label"])
    style(ax)
    for bar, value in zip(bars, data["energy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "energy_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)
    bars = ax.bar(x, data["deadline"], yerr=data["deadline_std"], capsize=4, color=colors, edgecolor="black", linewidth=0.7, hatch=hatches, error_kw={"elinewidth": 0.9, "ecolor": "black"})
    ax.set_ylabel("Deadline success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("20-seed Deadline success-rate comparison")
    ax.set_xticks(x, data["label"])
    style(ax)
    for bar, value in zip(bars, data["deadline"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "deadline_success_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
    axes[0].errorbar(x, data["energy"], yerr=data["energy_std"], fmt="o", capsize=5, lw=1.5, color="#2F5597", ecolor="#333333")
    axes[0].set_title("Energy (mean ± std)")
    axes[0].set_ylabel("Total energy (kJ)")
    axes[1].errorbar(x, data["deadline"], yerr=data["deadline_std"], fmt="o", capsize=5, lw=1.5, color="#ED7D31", ecolor="#333333")
    axes[1].set_title("Deadline success rate (mean ± std)")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_ylim(0, 105)
    for ax in axes:
        ax.set_xticks(x, data["label"])
        ax.tick_params(axis="x", labelsize=8)
        style(ax)
    fig.suptitle("20-seed mean ± standard deviation")
    fig.tight_layout()
    fig.savefig(OUT / "energy_deadline_with_std.png", bbox_inches="tight")
    plt.close(fig)

    (OUT / "plot_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"plots written to {OUT}")


if __name__ == "__main__":
    main()
