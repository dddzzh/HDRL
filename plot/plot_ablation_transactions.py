"""Draw the small-scenario ablation figure in an IEEE Transactions style."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/ablation_results/small_main_ablation/ablation_summary.csv"
OUT = ROOT / "results/ablation_results/small_main_ablation/figures"

DEADLINES = ("loose", "medium", "tight")
LABELS = {
    "Full D3QN+SAC": "Full",
    "Full checkpoint with rho=1": r"Full, $\rho=1$",
    "Independently trained full high with rho=1": r"High only, $\rho=1$",
    "HEFT+SAC": "HEFT+SAC",
    "HEFT with rho=1": r"HEFT, $\rho=1$",
    "High without deadline head with rho=1": "w/o deadline head",
}
ORDER = tuple(LABELS)
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#7F7F7F")
HATCHES = ("", "//", "\\\\", "xx", "++", "..")


def load_rows() -> list[dict[str, str]]:
    with DATA.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["variant"] in LABELS]
    present = {row["variant"] for row in rows}
    required = {"Full D3QN+SAC", "Full checkpoint with rho=1"}
    if not required <= present:
        raise ValueError(f"Missing required variants: {sorted(required - present)}")
    return rows


def main() -> None:
    rows = load_rows()
    lookup = {(r["variant"], r["deadline_type"]): r for r in rows}
    methods = [method for method in ORDER if all((method, d) in lookup for d in DEADLINES)]

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

    x = np.arange(len(DEADLINES), dtype=float)
    width = 0.78 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35))

    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * width
        energy = [float(lookup[(method, d)]["energy_mean_kj"]) for d in DEADLINES]
        energy_std = [float(lookup[(method, d)]["energy_std_kj"]) for d in DEADLINES]
        esp = [100.0 * float(lookup[(method, d)]["esp_at_0_90"]) for d in DEADLINES]
        feasible = [lookup[(method, d)]["feasible"].strip().lower() == "true" for d in DEADLINES]

        bars = axes[0].bar(
            x + offset,
            energy,
            width,
            yerr=energy_std,
            capsize=2,
            color=COLORS[index],
            edgecolor="black",
            linewidth=0.55,
            hatch=HATCHES[index],
            error_kw={"elinewidth": 0.65, "capthick": 0.65},
            label=LABELS[method],
            zorder=3,
        )
        axes[1].bar(
            x + offset,
            esp,
            width,
            color=COLORS[index],
            edgecolor="black",
            linewidth=0.55,
            hatch=HATCHES[index],
            zorder=3,
        )
        for deadline_index, (bar, ok) in enumerate(zip(bars, feasible)):
            if not ok:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    "×",
                    color="#B2182B",
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    zorder=5,
                )
                axes[1].text(
                    x[deadline_index] + offset,
                    max(2.0, esp[deadline_index] + 2.0),
                    "×",
                    color="#B2182B",
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    zorder=5,
                )

    for ax in axes:
        ax.set_xticks(x, [name.capitalize() for name in DEADLINES])
        ax.grid(axis="y", linestyle="--", linewidth=0.45, color="#B8B8B8", alpha=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("(a) Energy consumption")
    axes[0].set_ylabel("Total energy (kJ)")
    axes[0].set_ylim(bottom=0)
    axes[0].text(
        0.01,
        0.97,
        "×  constraint target not satisfied",
        color="#B2182B",
        fontsize=6.8,
        transform=axes[0].transAxes,
        va="top",
    )

    axes[1].set_title("(b) Deadline robustness")
    axes[1].set_ylabel("ESP@90 (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_yticks(np.arange(0, 101, 20))
    axes[1].axhline(100, color="#B2182B", linestyle="--", linewidth=0.9, zorder=2)
    axes[1].text(
        0.99,
        0.965,
        "Target: ESP@90 = 100%",
        color="#B2182B",
        fontsize=6.5,
        ha="right",
        transform=axes[1].transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.5, w_pad=1.2)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"small_ablation_transactions.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"methods={methods}")
    print(f"written={OUT}")


if __name__ == "__main__":
    main()
