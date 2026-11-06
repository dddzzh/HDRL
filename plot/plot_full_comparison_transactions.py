"""Generate the publication comparison figure suite from the final 20-seed runs."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/comparison_small_medium_20260921.csv"
OUT = ROOT / "results/comparison_figures/transactions_20260924"

DEADLINES = ("loose", "medium", "tight")
SCALES = ("small", "medium")
METHODS = ("D3QN+SAC (ours)", "DQN-Edge", "DReaM-DQN", "OOAC")
SHORT = {
    "D3QN+SAC (ours)": "Ours",
    "DQN-Edge": "DQN-Edge",
    "DReaM-DQN": "DReaM",
    "OOAC": "OOAC",
}
COLORS = {
    "D3QN+SAC (ours)": "#0B4F6C",
    "DQN-Edge": "#E69F00",
    "DReaM-DQN": "#009E73",
    "OOAC": "#8E6C8A",
}
MARKERS = {"D3QN+SAC (ours)": "*", "DQN-Edge": "s", "DReaM-DQN": "D", "OOAC": "o"}
HATCHES = {"D3QN+SAC (ours)": "", "DQN-Edge": "//", "DReaM-DQN": "\\\\", "OOAC": "xx"}

RAW_FILES = {
    ("small", "D3QN+SAC (ours)"): ROOT / "results/comparison_small_regimes_242000_20/20260916_future_esp_ep1140/comparison_rows.csv",
    ("small", "DQN-Edge"): ROOT / "results/comparison_small_regimes_242000_20/dqn_edge_20260917_092338/comparison_rows.csv",
    ("small", "DReaM-DQN"): ROOT / "results/comparison_small_regimes_242000_20/dream_dqn_20260918_090425/comparison_rows.csv",
    ("small", "OOAC"): ROOT / "results/comparison_small_regimes_242000_20/oo_actor_critic_20260917_092358/comparison_rows.csv",
    ("medium", "D3QN+SAC (ours)"): ROOT / "results/comparison_medium_regimes_242000_20/20260917_ours_dqn_oo_final/comparison_rows.csv",
    ("medium", "DQN-Edge"): ROOT / "results/comparison_medium_regimes_242000_20/20260917_ours_dqn_oo_final/comparison_rows.csv",
    ("medium", "DReaM-DQN"): ROOT / "results/comparison_medium_regimes_242000_20/dream_dqn_20260917_100220/comparison_rows.csv",
    ("medium", "OOAC"): ROOT / "results/comparison_medium_regimes_242000_20/20260917_ours_dqn_oo_final/comparison_rows.csv",
}
RAW_MODES = {
    "D3QN+SAC (ours)": "d3qn_sac",
    "DQN-Edge": "dqn_edge",
    "DReaM-DQN": "dream_dqn",
    "OOAC": "oo_actor_critic",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary() -> dict[tuple[str, str, str], dict[str, str]]:
    with SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["scale"], row["deadline_type"], row["method"]): row for row in rows}


def load_raw() -> dict[tuple[str, str, str], list[dict[str, str]]]:
    result: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for (scale, method), path in RAW_FILES.items():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for deadline in DEADLINES:
            selected = [
                row
                for row in rows
                if row["scale"] == scale
                and row["deadline_type"] == deadline
                and row["mode"] == RAW_MODES[method]
            ]
            if len(selected) != 20:
                raise ValueError(f"Expected 20 rows for {(scale, deadline, method)}, got {len(selected)}")
            result[(scale, deadline, method)] = selected
    return result


def validate(summary: dict, raw: dict) -> None:
    for key, rows in raw.items():
        energies = [float(row["total_energy"]) / 1000.0 for row in rows]
        expected = summary[key]
        if abs(statistics.mean(energies) - float(expected["energy_mean_kJ"])) > 0.02:
            raise ValueError(f"Energy mean mismatch for {key}")
        if abs(statistics.stdev(energies) - float(expected["energy_std_kJ"])) > 0.02:
            raise ValueError(f"Energy std mismatch for {key}")
        if {int(row["eval_seed"]) for row in rows} != set(range(242000, 242020)):
            raise ValueError(f"Unexpected seed set for {key}")


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.45, color="#B8B8B8", alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_energy_distributions(summary: dict, raw: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.25))
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    rng = np.random.default_rng(20260924)

    for ax, scale in zip(axes, SCALES):
        for method_index, method in enumerate(METHODS):
            for deadline_index, deadline in enumerate(DEADLINES):
                values = [float(row["total_energy"]) / 1000.0 for row in raw[(scale, deadline, method)]]
                pos = deadline_index + offsets[method_index]
                box = ax.boxplot(
                    [values],
                    positions=[pos],
                    widths=0.16,
                    patch_artist=True,
                    showfliers=False,
                    showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 2.6},
                    medianprops={"color": "black", "linewidth": 0.85},
                    whiskerprops={"color": "#444444", "linewidth": 0.65},
                    capprops={"color": "#444444", "linewidth": 0.65},
                    boxprops={"edgecolor": "black", "linewidth": 0.6},
                )
                box["boxes"][0].set_facecolor(COLORS[method])
                box["boxes"][0].set_alpha(0.88 if method == METHODS[0] else 0.68)
                jitter = rng.normal(0.0, 0.018, len(values))
                ax.scatter(
                    np.full(len(values), pos) + jitter,
                    values,
                    s=4.5,
                    color=COLORS[method],
                    edgecolors="none",
                    alpha=0.42,
                    zorder=2,
                )
        ax.set_xticks(range(3), [name.capitalize() for name in DEADLINES])
        ax.set_xlabel("Deadline type")
        ax.set_ylabel("Total energy (kJ)")
        ax.set_title(f"({chr(97 + SCALES.index(scale))}) {scale.capitalize()} scale")
        style_axes(ax)

    handles = [Patch(facecolor=COLORS[m], edgecolor="black", label=SHORT[m], alpha=0.82) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.5, w_pad=1.2)
    save(fig, "fig01_energy_distributions")


def plot_esp_heatmap(summary: dict) -> None:
    columns = [(scale, deadline) for scale in SCALES for deadline in DEADLINES]
    matrix = np.array(
        [[100.0 * float(summary[(scale, deadline, method)]["esp_at_090"]) for scale, deadline in columns] for method in METHODS]
    )
    cmap = LinearSegmentedColormap.from_list("drlec_esp", ["#D6604D", "#F6E8C3", "#92C5DE", "#0B4F6C"])
    fig, ax = plt.subplots(figsize=(7.16, 2.55))
    image = ax.imshow(matrix, cmap=cmap, vmin=40, vmax=100, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.0f}%", ha="center", va="center", color="white" if value >= 95 else "#202020", fontweight="bold" if value == 100 else "normal")
            if value == 100:
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#0B4F6C", linewidth=1.4))

    ax.set_xticks(range(6), ["Loose", "Medium", "Tight"] * 2)
    ax.set_yticks(range(4), [SHORT[m] for m in METHODS])
    ax.tick_params(length=0)
    ax.set_title("ESP@90 across scales and deadline types")
    ax.text(1, -0.82, "Small scale", ha="center", fontweight="bold")
    ax.text(4, -0.82, "Medium scale", ha="center", fontweight="bold")
    ax.axvline(2.5, color="white", linewidth=2.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("ESP@90 (%)")
    fig.text(0.01, 0.01, "Outlined cells satisfy the 100% ESP target; reliability is 100% in every cell.", fontsize=6.8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), pad=0.6)
    save(fig, "fig02_esp90_heatmap")


def plot_tradeoff(summary: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.85), sharey=True)
    for row_index, scale in enumerate(SCALES):
        for col_index, deadline in enumerate(DEADLINES):
            ax = axes[row_index, col_index]
            ax.axhspan(40, 100, color="#F8E5E1", alpha=0.55, zorder=0)
            ax.axhline(100, color="#B2182B", linestyle="--", linewidth=0.8, zorder=1)
            for method in METHODS:
                item = summary[(scale, deadline, method)]
                energy = float(item["energy_mean_kJ"])
                energy_std = float(item["energy_std_kJ"])
                esp = 100.0 * float(item["esp_at_090"])
                ax.errorbar(
                    energy,
                    esp,
                    xerr=energy_std,
                    fmt=MARKERS[method],
                    ms=7 if method == METHODS[0] else 4.3,
                    color=COLORS[method],
                    markeredgecolor="black",
                    markeredgewidth=0.45,
                    elinewidth=0.65,
                    capsize=2,
                    zorder=3,
                )
            ax.set_title(f"({chr(97 + row_index * 3 + col_index)}) {scale.capitalize()}-{deadline.capitalize()}")
            ax.set_xlabel("Energy (kJ)")
            if col_index == 0:
                ax.set_ylabel("ESP@90 (%)")
            ax.set_ylim(40, 103)
            style_axes(ax)

    handles = [
        plt.Line2D([], [], marker=MARKERS[m], color=COLORS[m], linestyle="None", markeredgecolor="black", markeredgewidth=0.45, markersize=7 if m == METHODS[0] else 5, label=SHORT[m])
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.997))
    fig.text(0.01, 0.01, "Preferred region: lower energy and ESP@90 = 100%.", fontsize=6.8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.92), pad=0.55, w_pad=0.8, h_pad=1.0)
    save(fig, "fig03_energy_esp_tradeoff")


def plot_savings(summary: dict) -> None:
    labels: list[str] = []
    savings: list[float] = []
    valid: list[bool] = []
    for scale in SCALES:
        for deadline in DEADLINES:
            labels.append(f"{scale[0].upper()}-{deadline[0].upper()}")
            ours = summary[(scale, deadline, METHODS[0])]
            baselines = [summary[(scale, deadline, method)] for method in METHODS[1:]]
            feasible_baselines = [row for row in baselines if row["feasible"].lower() == "true"]
            is_valid = ours["feasible"].lower() == "true" and bool(feasible_baselines)
            valid.append(is_valid)
            if is_valid:
                best = min(float(row["energy_mean_kJ"]) for row in feasible_baselines)
                savings.append(100.0 * (best - float(ours["energy_mean_kJ"])) / best)
            else:
                savings.append(0.0)

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(labels))
    colors = ["#0B4F6C" if ok else "#D9D9D9" for ok in valid]
    bars = ax.bar(x, savings, color=colors, edgecolor="black", linewidth=0.6, width=0.68)
    for bar, value, ok in zip(bars, savings, valid):
        if ok:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}%", ha="center", fontsize=7)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.8, "N/A", ha="center", color="#B2182B", fontsize=7, fontweight="bold")
    ax.axvline(2.5, color="#777777", linewidth=0.8)
    ax.text(1, -4.3, "Small", ha="center", fontweight="bold")
    ax.text(4, -4.3, "Medium", ha="center", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Energy saving (%)")
    ax.set_title("Savings over the best feasible baseline")
    ax.set_ylim(0, max(savings) + 5)
    style_axes(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.6)
    save(fig, "fig04_saving_vs_best_feasible")


def main() -> None:
    configure_style()
    summary = load_summary()
    raw = load_raw()
    validate(summary, raw)
    plot_energy_distributions(summary, raw)
    plot_esp_heatmap(summary)
    plot_tradeoff(summary)
    plot_savings(summary)
    print(f"validated_groups={len(raw)}")
    print(f"written={OUT}")


if __name__ == "__main__":
    main()
