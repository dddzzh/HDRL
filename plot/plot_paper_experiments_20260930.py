"""Generate publication-ready experimental figures from the formal test runs.

The formal comparison data are split across several result directories because
the four policies were evaluated separately.  This script validates and merges
those paired 20-seed runs, then exports vector PDF/SVG and 300-dpi PNG figures.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

INPUTS = [
    ROOT / "results/comparison_small_regimes_242000_20/20260916_future_esp_ep1140/comparison_rows.csv",
    ROOT / "results/comparison_small_regimes_242000_20/dqn_edge_20260917_092338/comparison_rows.csv",
    ROOT / "results/comparison_small_regimes_242000_20/dream_dqn_20260918_090425/comparison_rows.csv",
    ROOT / "results/comparison_small_regimes_242000_20/oo_actor_critic_20260917_092358/comparison_rows.csv",
    ROOT / "results/comparison_medium_regimes_242000_20/20260917_ours_dqn_oo_final/comparison_rows.csv",
    ROOT / "results/comparison_medium_regimes_242000_20/dream_dqn_20260917_100220/comparison_rows.csv",
]

ABLATION = ROOT / "results/ablation_results/small_main_ablation/ablation_summary.csv"

METHODS = ["d3qn_sac", "oo_actor_critic", "dream_dqn", "dqn_edge"]
LABEL = {
    "d3qn_sac": "Ours",
    "oo_actor_critic": "OOAC",
    "dream_dqn": "DReaM-DQN",
    "dqn_edge": "DQN-Edge",
}
COLOR = {
    "d3qn_sac": "#0072B2",
    "oo_actor_critic": "#009E73",
    "dream_dqn": "#CC79A7",
    "dqn_edge": "#E69F00",
}
MARKER = {
    "d3qn_sac": "o",
    "oo_actor_critic": "s",
    "dream_dqn": "D",
    "dqn_edge": "^",
}
SCALES = ["small", "medium"]
SCALE_LABEL = {"small": "Small (20 edge, 6 cloud)", "medium": "Medium (28 edge, 9 cloud)"}
DEADLINES = ["tight", "medium", "loose"]
DEADLINE_LABEL = {"tight": "Tight", "medium": "Medium", "loose": "Loose"}
T_CRIT_95_DF19 = 2.093


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "lines.markersize": 5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def read_formal_rows() -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str, int]] = set()
    for path in INPUTS:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                mode = raw["mode"]
                if mode not in METHODS:
                    continue
                key = (raw["scale"], raw["deadline_type"], mode, int(raw["eval_seed"]))
                if key in seen:
                    # The medium combined run intentionally supersedes an earlier
                    # ours-only directory, which is therefore not listed in INPUTS.
                    raise ValueError(f"Duplicate formal result row: {key}")
                seen.add(key)
                rows.append(
                    {
                        "scale": raw["scale"],
                        "deadline": raw["deadline_type"],
                        "mode": mode,
                        "seed": int(raw["eval_seed"]),
                        "energy_kj": float(raw["total_energy"]) / 1000.0,
                        "deadline_success": float(raw["deadline_success_rate"]),
                        "reliability_success": float(raw["reliability_success_rate"]),
                    }
                )

    expected_seeds = set(range(242000, 242020))
    for scale in SCALES:
        for deadline in DEADLINES:
            for mode in METHODS:
                subset = [r for r in rows if r["scale"] == scale and r["deadline"] == deadline and r["mode"] == mode]
                seeds = {r["seed"] for r in subset}
                if len(subset) != 20 or seeds != expected_seeds:
                    raise ValueError(
                        f"Incomplete paired test set for {scale}/{deadline}/{mode}: "
                        f"n={len(subset)}, seeds={sorted(seeds)}"
                    )
    return rows


def group_rows(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scale"], row["deadline"], row["mode"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda r: r["seed"])
    return grouped


def metrics(values: list[dict]) -> dict[str, float]:
    energy = np.asarray([r["energy_kj"] for r in values], dtype=float)
    deadline = np.asarray([r["deadline_success"] for r in values], dtype=float)
    reliability = np.asarray([r["reliability_success"] for r in values], dtype=float)
    n = len(values)
    std = float(energy.std(ddof=1))
    return {
        "n": n,
        "energy_mean": float(energy.mean()),
        "energy_std": std,
        "energy_ci95": T_CRIT_95_DF19 * std / np.sqrt(n),
        "esp90": float((deadline >= 0.90 - 1e-12).mean()),
        "reliability_rate": float(reliability.mean()),
        "reliability_esp100": float((reliability >= 1.0 - 1e-12).mean()),
        "joint_esp90": float(((deadline >= 0.90 - 1e-12) & (reliability >= 1.0 - 1e-12)).mean()),
    }


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLOR[m],
            marker=MARKER[m],
            markerfacecolor=COLOR[m],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=LABEL[m],
        )
        for m in METHODS
    ]


def plot_main_matrix(grouped: dict) -> None:
    x = np.arange(len(DEADLINES), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.75), sharex="col")

    for row_idx, scale in enumerate(SCALES):
        ax_energy, ax_esp = axes[row_idx]
        for mode in METHODS:
            summaries = [metrics(grouped[(scale, d, mode)]) for d in DEADLINES]
            means = np.asarray([s["energy_mean"] for s in summaries])
            ci = np.asarray([s["energy_ci95"] for s in summaries])
            esp = np.asarray([100.0 * s["esp90"] for s in summaries])

            ax_energy.errorbar(
                x,
                means,
                yerr=ci,
                color=COLOR[mode],
                marker=MARKER[mode],
                capsize=2.2,
                label=LABEL[mode],
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
            ax_esp.plot(x, esp, color=COLOR[mode], linewidth=1.15)
            for idx, value in enumerate(esp):
                feasible = value >= 90.0 - 1e-9
                ax_esp.scatter(
                    x[idx],
                    value,
                    s=28,
                    marker=MARKER[mode],
                    facecolor=COLOR[mode] if feasible else "white",
                    edgecolor=COLOR[mode],
                    linewidth=1.0,
                    zorder=4,
                )

        ax_energy.set_ylabel("Total energy (kJ)")
        ax_energy.set_title(f"({chr(97 + row_idx * 2)}) {SCALE_LABEL[scale]}: energy")
        ax_energy.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax_energy.set_axisbelow(True)

        ax_esp.axhspan(0, 90, color="#F2F2F2", zorder=0)
        ax_esp.axhline(90, color="#555555", linestyle="--", linewidth=0.9)
        ax_esp.set_ylim(-4, 104)
        ax_esp.set_ylabel(r"Deadline ESP@90 (\%)")
        ax_esp.set_title(f"({chr(98 + row_idx * 2)}) {SCALE_LABEL[scale]}: feasibility")
        ax_esp.text(
            0.02,
            0.05,
            "Reliability pass rate = 100% for all methods",
            transform=ax_esp.transAxes,
            fontsize=6.8,
            color="#444444",
        )
        ax_esp.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax_esp.set_axisbelow(True)

    for ax in axes[-1]:
        ax.set_xticks(x, [DEADLINE_LABEL[d] for d in DEADLINES])
        ax.set_xlabel("Deadline regime")

    fig.legend(handles=legend_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.subplots_adjust(top=0.88, wspace=0.28, hspace=0.34)
    export(fig, "fig_main_energy_deadline_matrix")


def plot_energy_distributions(grouped: dict) -> None:
    rng = np.random.default_rng(20260930)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.85), sharey="row")
    short_labels = ["Ours", "OOAC", "DReaM-\nDQN", "DQN-\nEdge"]

    def draw_scenario(
        ax: plt.Axes,
        scale: str,
        deadline: str,
        local_rng: np.random.Generator,
        set_local_ylim: bool = True,
    ) -> None:
        all_values = []
        for method_idx, mode in enumerate(METHODS):
            values = np.asarray(
                [r["energy_kj"] for r in grouped[(scale, deadline, mode)]], dtype=float
            )
            all_values.extend(values.tolist())
            summary = metrics(grouped[(scale, deadline, mode)])
            feasible = summary["joint_esp90"] >= 0.90 - 1e-12
            box = ax.boxplot(
                [values],
                positions=[method_idx],
                widths=0.56,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#222222", "linewidth": 1.0},
                boxprops={"linewidth": 0.9, "edgecolor": COLOR[mode]},
                whiskerprops={"linewidth": 0.8, "color": COLOR[mode]},
                capprops={"linewidth": 0.8, "color": COLOR[mode]},
            )
            patch = box["boxes"][0]
            patch.set_facecolor(COLOR[mode] if feasible else "white")
            patch.set_alpha(0.52 if feasible else 1.0)
            if not feasible:
                patch.set_hatch("////")
            jitter = local_rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(
                np.full(len(values), method_idx) + jitter,
                values,
                s=7,
                color=COLOR[mode],
                alpha=0.44,
                linewidths=0,
                zorder=3,
            )

        if set_local_ylim:
            span = max(all_values) - min(all_values)
            ax.set_ylim(min(all_values) - 0.10 * span, max(all_values) + 0.12 * span)
        ax.set_xticks(range(len(METHODS)), short_labels)
        ax.set_xlim(-0.62, len(METHODS) - 0.38)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)

    for row_idx, scale in enumerate(SCALES):
        for col_idx, deadline in enumerate(DEADLINES):
            ax = axes[row_idx, col_idx]
            draw_scenario(ax, scale, deadline, rng, set_local_ylim=False)
            panel = chr(97 + row_idx * len(DEADLINES) + col_idx)
            ax.set_title(f"({panel}) {DEADLINE_LABEL[deadline]}")
            if col_idx == 0:
                ax.set_ylabel(f"{SCALE_LABEL[scale]}\nTotal energy (kJ)")

            # Export every scenario as a self-contained figure using the same style.
            single_fig, single_ax = plt.subplots(figsize=(3.45, 2.75))
            single_rng = np.random.default_rng(20260930 + row_idx * 10 + col_idx)
            draw_scenario(single_ax, scale, deadline, single_rng)
            single_ax.set_ylabel("Total energy (kJ)")
            single_ax.set_title(f"{SCALE_LABEL[scale]} | {DEADLINE_LABEL[deadline]} deadline")
            single_ax.text(
                0.98,
                0.02,
                "Hatched: ESP@90 < 0.90",
                transform=single_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.3,
                color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
            )
            export(single_fig, f"fig_energy_distribution_{scale}_{deadline}")

        row_values = [
            r["energy_kj"]
            for deadline in DEADLINES
            for mode in METHODS
            for r in grouped[(scale, deadline, mode)]
        ]
        row_span = max(row_values) - min(row_values)
        axes[row_idx, 0].set_ylim(
            min(row_values) - 0.08 * row_span,
            max(row_values) + 0.10 * row_span,
        )

    fig.text(
        0.995,
        0.005,
        "Hatched box: ESP@90 < 0.90",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#555555",
    )
    fig.subplots_adjust(top=0.94, bottom=0.13, left=0.105, right=0.985, wspace=0.18, hspace=0.38)
    export(fig, "fig_energy_distributions")


def plot_energy_feasibility_tradeoff(grouped: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.75), sharex=True)

    for row_idx, scale in enumerate(SCALES):
        for col_idx, deadline in enumerate(DEADLINES):
            ax = axes[row_idx, col_idx]
            ax.axvspan(0, 0.90, color="#F2F2F2", zorder=0)
            ax.axvline(0.90, color="#555555", linestyle="--", linewidth=0.85)
            energy_values = []
            for mode in METHODS:
                summary = metrics(grouped[(scale, deadline, mode)])
                x = summary["esp90"]
                y = summary["energy_mean"]
                energy_values.append(y)
                feasible = summary["joint_esp90"] >= 0.90 - 1e-12
                ax.scatter(
                    x,
                    y,
                    s=45 if mode == "d3qn_sac" else 34,
                    marker=MARKER[mode],
                    facecolor=COLOR[mode] if feasible else "white",
                    edgecolor=COLOR[mode],
                    linewidth=1.2,
                    zorder=4,
                )
            pad = max(8.0, (max(energy_values) - min(energy_values)) * 0.16)
            ax.set_ylim(min(energy_values) - pad, max(energy_values) + pad)
            ax.set_xlim(-0.03, 1.03)
            ax.grid(color="#E0E0E0", linewidth=0.5)
            ax.set_axisbelow(True)
            panel = chr(97 + row_idx * len(DEADLINES) + col_idx)
            ax.set_title(f"({panel}) {DEADLINE_LABEL[deadline]} deadline")
            if col_idx == 0:
                ax.set_ylabel(f"{SCALE_LABEL[scale]}\nMean energy (kJ)")
            if row_idx == len(SCALES) - 1:
                ax.set_xlabel("Deadline ESP@90")
            if row_idx == 0 and col_idx == 0:
                ax.text(
                    0.03,
                    0.95,
                    "Gray: infeasible region",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=6.6,
                    color="#555555",
                )

    fig.legend(handles=legend_handles(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.subplots_adjust(top=0.88, wspace=0.34, hspace=0.32)
    export(fig, "fig_energy_feasibility_tradeoff")


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 10000) -> tuple[float, float, float]:
    median = float(np.median(values))
    samples = rng.choice(values, size=(reps, len(values)), replace=True)
    boot = np.median(samples, axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return median, float(low), float(high)


def plot_paired_savings(grouped: dict) -> None:
    rng = np.random.default_rng(20260930)
    baselines = ["oo_actor_critic", "dream_dqn", "dqn_edge"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.05), sharex=True)

    for ax, scale, panel in zip(axes, SCALES, ["(a)", "(b)"]):
        rows = []
        for deadline in DEADLINES:
            ours_by_seed = {r["seed"]: r["energy_kj"] for r in grouped[(scale, deadline, "d3qn_sac")]}
            for baseline in baselines:
                base_by_seed = {r["seed"]: r["energy_kj"] for r in grouped[(scale, deadline, baseline)]}
                savings = np.asarray(
                    [100.0 * (base_by_seed[s] - ours_by_seed[s]) / base_by_seed[s] for s in sorted(ours_by_seed)],
                    dtype=float,
                )
                rows.append((deadline, baseline, *bootstrap_median_ci(savings, rng)))

        y = np.arange(len(rows))[::-1]
        for y_pos, (deadline, baseline, median, low, high) in zip(y, rows):
            ax.errorbar(
                median,
                y_pos,
                xerr=[[median - low], [high - median]],
                fmt=MARKER[baseline],
                color=COLOR[baseline],
                markerfacecolor=COLOR[baseline],
                markeredgecolor="white",
                markeredgewidth=0.55,
                capsize=2,
                linewidth=1.0,
            )
            ax.text(high + 0.8, y_pos, f"{median:.1f}%", va="center", fontsize=6.6, color=COLOR[baseline])

        ax.axvline(0, color="#444444", linewidth=0.8)
        ax.set_yticks(
            y,
            [f"{DEADLINE_LABEL[d]} | {LABEL[b]}" for d, b, *_ in rows],
        )
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_xlabel("Paired energy saving of Ours (\%)")
        ax.set_title(f"{panel} {SCALE_LABEL[scale]}")

    fig.subplots_adjust(wspace=0.60)
    export(fig, "fig_paired_energy_savings")


def read_ablation() -> list[dict]:
    with ABLATION.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_ablation_tradeoff() -> None:
    rows = read_ablation()
    variants = [
        "Full D3QN+SAC",
        "Full checkpoint with rho=1",
        "Independently trained full high with rho=1",
        "HEFT+SAC",
        "HEFT with rho=1",
        "High without deadline head with rho=1",
    ]
    variant_label = {
        "Full D3QN+SAC": "Full",
        "Full checkpoint with rho=1": r"Full, $\rho=1$",
        "Independently trained full high with rho=1": r"Upper only, $\rho=1$",
        "HEFT+SAC": "HEFT+SAC",
        "HEFT with rho=1": r"HEFT, $\rho=1$",
        "High without deadline head with rho=1": r"No deadline head, $\rho=1$",
    }
    colors = ["#0072B2", "#56B4E9", "#9467BD", "#009E73", "#7F7F7F", "#D55E00"]
    markers = ["o", "s", "D", "^", "v", "X"]

    by_key = {(r["deadline_type"], r["variant"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.65), sharey=True)
    for ax, deadline, panel in zip(axes, DEADLINES, ["(a)", "(b)", "(c)"]):
        ax.axhspan(0, 0.90, color="#F2F2F2", zorder=0)
        ax.axhline(0.90, color="#555555", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.75)
        for variant, color, marker in zip(variants, colors, markers):
            r = by_key[(deadline, variant)]
            if variant == "Full D3QN+SAC":
                delta = 0.0
            elif r["delta_energy_vs_full_pct"]:
                delta = float(r["delta_energy_vs_full_pct"])
            else:
                full_energy = float(by_key[(deadline, "Full D3QN+SAC")]["energy_mean_kj"])
                delta = 100.0 * (float(r["energy_mean_kj"]) - full_energy) / full_energy
            esp = float(r["esp_at_0_90"])
            feasible = r["feasible"].strip().lower() == "true"
            ax.scatter(
                delta,
                esp,
                s=43,
                marker=marker,
                facecolor=color if feasible else "white",
                edgecolor=color,
                linewidth=1.1,
                zorder=4,
            )
        ax.set_title(f"{panel} {DEADLINE_LABEL[deadline]}")
        ax.set_xlabel("Energy change vs. full (\%)")
        ax.set_xlim(-90, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(color="#E0E0E0", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Deadline ESP@90")
    handles = [
        Line2D([0], [0], linestyle="none", marker=m, color=c, markerfacecolor=c, label=variant_label[v])
        for v, c, m in zip(variants, colors, markers)
    ]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=6.8)
    fig.subplots_adjust(top=0.73, wspace=0.26)
    export(fig, "fig_ablation_energy_feasibility")


def write_data_artifacts(rows: list[dict], grouped: dict) -> None:
    selected = OUT / "formal_comparison_selected_columns.csv"
    with selected.open("w", encoding="utf-8", newline="") as handle:
        fields = ["scale", "deadline", "mode", "seed", "energy_kj", "deadline_success", "reliability_success"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["scale"], r["deadline"], r["mode"], r["seed"])))

    summary_path = OUT / "formal_comparison_figure_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "scale",
            "deadline",
            "mode",
            "n",
            "energy_mean_kj",
            "energy_std_kj",
            "energy_ci95_kj",
            "deadline_esp_at_0_90",
            "reliability_success_rate",
            "reliability_esp_at_1_00",
            "joint_esp_at_0_90",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scale in SCALES:
            for deadline in DEADLINES:
                for mode in METHODS:
                    s = metrics(grouped[(scale, deadline, mode)])
                    writer.writerow(
                        {
                            "scale": scale,
                            "deadline": deadline,
                            "mode": mode,
                            "n": s["n"],
                            "energy_mean_kj": f"{s['energy_mean']:.6f}",
                            "energy_std_kj": f"{s['energy_std']:.6f}",
                            "energy_ci95_kj": f"{s['energy_ci95']:.6f}",
                            "deadline_esp_at_0_90": f"{s['esp90']:.6f}",
                            "reliability_success_rate": f"{s['reliability_rate']:.6f}",
                            "reliability_esp_at_1_00": f"{s['reliability_esp100']:.6f}",
                            "joint_esp_at_0_90": f"{s['joint_esp90']:.6f}",
                        }
                    )

    manifest = OUT / "README.md"
    source_lines = "\n".join(f"- `{p.relative_to(ROOT).as_posix()}`" for p in INPUTS)
    manifest.write_text(
        "# Experimental figure assets\n\n"
        "These figures use the formal configurations stated in the manuscript: "
        "20 edge + 6 cloud servers for the small setting and 28 edge + 9 cloud servers "
        "for the medium setting. Every method is evaluated on the same 20 workflow seeds "
        "(`242000`--`242019`) for tight, medium, and loose deadlines.\n\n"
        "## Source files\n\n"
        f"{source_lines}\n\n"
        "The ablation figure uses `results/ablation_results/small_main_ablation/ablation_summary.csv`.\n\n"
        "## Metric definitions\n\n"
        "- **Total energy:** sum of computation energy over the 20 workflows in one test seed; figures report kJ.\n"
        "- **Deadline ESP@90:** fraction of test seeds whose workflow deadline success rate is at least 0.90.\n"
        "- **Reliability pass:** a seed passes when its reliability success rate is 1.00.\n"
        "- **Joint feasibility:** both the Deadline ESP criterion and the reliability criterion hold.\n"
        "- **95% CI:** Student-t interval for mean energy with 19 degrees of freedom.\n"
        "- **Paired saving CI:** percentile bootstrap CI of the median seed-wise energy-saving ratio (10,000 resamples).\n\n"
        "## Figure files\n\n"
        "- `fig_energy_distributions`: 2-by-3 scenario matrix of 20-seed energy boxplots with every raw test point.\n"
        "- `fig_energy_distribution_<scale>_<deadline>`: six self-contained scenario figures in the same style.\n"
        "- Filled boxes meet joint feasibility; hatched boxes have Deadline ESP@90 below 0.90.\n\n"
        "Every figure is exported as PDF, SVG, and 300-dpi PNG.\n",
        encoding="utf-8",
    )


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_formal_rows()
    grouped = group_rows(rows)
    write_data_artifacts(rows, grouped)
    plot_energy_distributions(grouped)
    print(f"Wrote figures and data artifacts to {OUT}")


if __name__ == "__main__":
    main()
