"""Style previews; --all exports the selected A/C styles for four scenarios."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_experiments_20260930 import (
    ROOT, configure_style, group_rows, metrics,
)

OUT = ROOT / "docs/figures/main_style_options"
METHODS = ("d3qn_sac", "dqn_edge", "dream_dqn", "oo_actor_critic")
LABELS = ("D3QN+SAC", "DQN-Edge", "DReaM-DQN", "OOAC")
COLORS = ("#176B86", "#D89535", "#739C66", "#9A7399")
MARKERS = ("o", "s", "^", "D")
HATCHES = ("", "//", "..", "\\\\")
DEADLINES = ("loose", "medium", "tight")
SCALES = ("small", "medium")
TITLES = ("S26: Small, 20 edge + 6 cloud", "M37: Medium, 28 edge + 9 cloud")
ENERGY_LIMITS = {}


def load_data(all_scenarios=False):
    source = ROOT / "docs/figures/formal_comparison_selected_columns.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            {**row, "seed": int(row["seed"]),
             "energy_kj": float(row["energy_kj"]),
             "deadline_success": float(row["deadline_success"]),
             "reliability_success": float(row["reliability_success"])}
            for row in csv.DictReader(stream)
        ]
    grouped = group_rows(rows)
    with (ROOT / "docs/figures/formal_comparison_figure_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        summary = {
            (r["scale"], r["deadline"], r["mode"]): r
            for r in csv.DictReader(stream)
        }
    assert len(grouped) == 24
    for key, values in grouped.items():
        assert len(values) == 20
        assert {r["seed"] for r in values} == set(range(242000, 242020))
        actual, expected = metrics(values), summary[key]
        for a, b in (
            ("energy_mean", "energy_mean_kj"),
            ("energy_std", "energy_std_kj"),
            ("esp90", "deadline_esp_at_0_90"),
        ):
            assert abs(actual[a] - float(expected[b])) < 2e-6, (key, a)
    if not all_scenarios:
        return grouped
    grouped = {
        ("S26" if scale == "small" else "M37", d, mode): values
        for (scale, d, mode), values in grouped.items()
    }
    for scenario, directory, scale in (
        ("S37", "S37_final", "small"),
        ("M53", "M53_complete_4alg_20260927", "medium"),
    ):
        base = ROOT / "results/scenario_matrix_evaluation" / directory
        with (base / "comparison_rows.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            extra = []
            for row in csv.DictReader(stream):
                if row["mode"] not in METHODS:
                    continue
                assert row["scale"] == scale
                assert int(row["num_workflows"]) == 20
                extra.append({
                    "scale": scenario, "deadline": row["deadline_type"],
                    "mode": row["mode"], "seed": int(row["eval_seed"]),
                    "energy_kj": float(row["total_energy"]) / 1000,
                    "deadline_success": float(row["deadline_success_rate"]),
                    "reliability_success": float(row["reliability_success_rate"]),
                })
        additional = group_rows(extra)
        with (base / "comparison_summary.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            reference = {
                (r["deadline_type"], r["mode"], r["metric"]): r
                for r in csv.DictReader(stream)
            }
        assert len(additional) == 12, scenario
        for (_, d, mode), values in additional.items():
            assert len(values) == 20
            assert {r["seed"] for r in values} == set(range(242000, 242020))
            actual = metrics(values)
            energy = reference[d, mode, "total_energy"]
            assert abs(actual["energy_mean"] * 1000 - float(energy["mean"])) < .002
            assert abs(actual["energy_std"] * 1000 - float(energy["std"])) < .002
            assert abs(actual["esp90"] - float(
                reference[d, mode, "deadline_esp@0.90"]["mean"]
            )) < 1e-12
            assert actual["reliability_esp100"] == 1
        grouped.update(additional)
    assert len(grouped) == 48
    for scenarios in (("S26", "S37"), ("M37", "M53")):
        largest = max(
            max(max(r["energy_kj"] for r in values),
                metrics(values)["energy_mean"] + metrics(values)["energy_std"])
            for (s, _, _), values in grouped.items() if s in scenarios
        ) * 1000
        limit = np.ceil(largest / 100000) * 100000
        ENERGY_LIMITS.update({s: limit for s in scenarios})
    return grouped


def canvas():
    full = len(SCALES) == 4
    fig, axes = plt.subplots(2, len(SCALES),
                             figsize=(14.32 if full else 7.16, 4.5))
    fig.subplots_adjust(left=.045 if full else .09, right=.985, bottom=.16,
                        top=.85, hspace=.55, wspace=.42 if full else .29)
    for col, (scale, title) in enumerate(zip(SCALES, TITLES)):
        axes[0, col].set_title(f"({chr(97 + col)}) {title}", pad=8)
        axes[1, col].set_title(
            f"({chr(97 + len(SCALES) + col)}) Deadline success probability",
            pad=8)
        for row in range(2):
            ax = axes[row, col]
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#DDE1E4", linewidth=.45)
            ax.set_axisbelow(True)
            ax.set_xticks(range(3), ["Loose", "Medium", "Tight"])
            ax.set_xlim(-.52, 2.52)
        axes[0, col].set_ylabel("Total energy (J)")
        axes[0, col].set_ylim(
            0, ENERGY_LIMITS.get(scale, 410000 if col == 0 else 750000))
        axes[0, col].ticklabel_format(axis="y", style="sci",
                                    scilimits=(0, 0), useMathText=True)
        axes[1, col].set_ylabel("ESP@90 (%)")
        axes[1, col].set_ylim(0, 111)
        axes[1, col].set_yticks([0, 25, 50, 75, 100])
        axes[1, col].set_xlabel("Deadline regime")
    return fig, axes


def finish(fig, stem, note):
    handles = [Patch(facecolor=c, edgecolor="#333333", label=label)
               for c, label in zip(COLORS, LABELS)]
    if stem == "option_B_marked_lines":
        handles = [
            Line2D([0], [0], color=c, marker=m, linewidth=1.25, label=label)
            for c, m, label in zip(COLORS, MARKERS, LABELS)
        ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(.54, .99))
    fig.text(.5, .025, note, ha="center", fontsize=7, color="#444444")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=320)
    plt.close(fig)


def grouped_bars(data):
    fig, axes = canvas()
    for col, scale in enumerate(SCALES):
        for j, mode in enumerate(METHODS):
            stats = [metrics(data[scale, d, mode]) for d in DEADLINES]
            x = np.arange(3) + (j - 1.5) * .19
            common = dict(width=.17, color=COLORS[j], edgecolor="#333333",
                          linewidth=.55, hatch=HATCHES[j])
            axes[0, col].bar(
                x, [s["energy_mean"] * 1000 for s in stats],
                yerr=[s["energy_std"] * 1000 for s in stats], **common,
                error_kw=dict(lw=.75, capsize=2, capthick=.75),
            )
            axes[1, col].bar(x, [s["esp90"] * 100 for s in stats], **common)
    finish(fig, "option_A_grouped_bars",
           "Energy: mean +/- sample SD; ESP@90: 20 fixed test seeds.")


def marked_lines(data):
    fig, axes = canvas()
    for col, scale in enumerate(SCALES):
        for j, mode in enumerate(METHODS):
            stats = [metrics(data[scale, d, mode]) for d in DEADLINES]
            # Categorical offsets keep equal-ESP markers individually visible.
            x = np.arange(3) + (j - 1.5) * .035
            common = dict(color=COLORS[j], marker=MARKERS[j],
                          markersize=4.6, linewidth=1.25,
                          markeredgecolor="white", markeredgewidth=.45)
            axes[0, col].errorbar(
                x, [s["energy_mean"] * 1000 for s in stats],
                yerr=[s["energy_std"] * 1000 for s in stats],
                capsize=2, elinewidth=.7, **common,
            )
            axes[1, col].plot(x, [s["esp90"] * 100 for s in stats], **common)
    finish(fig, "option_B_marked_lines",
           "Energy: mean +/- sample SD; marker offsets separate equal ESP values.")


def distributions(data):
    fig, axes = canvas()
    for col, scale in enumerate(SCALES):
        for j, mode in enumerate(METHODS):
            samples = [[r["energy_kj"] * 1000 for r in data[scale, d, mode]]
                       for d in DEADLINES]
            x = np.arange(3) + (j - 1.5) * .19
            boxes = axes[0, col].boxplot(
                samples, positions=x, widths=.145, patch_artist=True,
                manage_ticks=False, whis=1.5, showfliers=True,
                medianprops=dict(color="#222222", linewidth=.9),
                boxprops=dict(edgecolor="#333333", linewidth=.6),
                whiskerprops=dict(color="#444444", linewidth=.6),
                capprops=dict(color="#444444", linewidth=.6),
                flierprops=dict(marker=".", markersize=2.8,
                                markerfacecolor=COLORS[j],
                                markeredgecolor=COLORS[j]),
            )
            for box in boxes["boxes"]:
                box.set_facecolor(COLORS[j])
                box.set_alpha(.85)
        ax = axes[1, col]
        esp = np.array([
            [metrics(data[scale, d, mode])["esp90"] * 100 for d in DEADLINES]
            for mode in METHODS
        ])
        ax.grid(False)
        ax.imshow(esp, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(3.5, -.5)
        ax.set_yticks(range(4), LABELS)
        ax.set_ylabel("")
        ax.tick_params(axis="both", length=0)
        for row in range(4):
            for d in range(3):
                value = esp[row, d]
                ax.text(d, row, f"{value:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if value >= 65 else "#222222")
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1)
        ax.tick_params(which="minor", length=0)
    finish(fig, "option_C_boxplots_matrix",
           "Energy: median, IQR, 1.5-IQR whiskers and outliers; cells: ESP@90.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Export A/C for S26, S37, M37 and M53.")
    args = parser.parse_args()
    if args.all:
        OUT = ROOT / "docs/figures/main_comparison_all_scenarios"
        SCALES = ("S26", "S37", "M37", "M53")
        TITLES = ("S26: Small, 20 edge + 6 cloud",
                  "S37: Small, 28 edge + 9 cloud",
                  "M37: Medium, 28 edge + 9 cloud",
                  "M53: Medium, 40 edge + 13 cloud")
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data(args.all)
    grouped_bars(data)
    if not args.all:
        marked_lines(data)
    distributions(data)
    if args.all:
        fields = ("scenario", "deadline", "method", "n", "energy_mean_j",
                  "energy_std_j", "esp90_percent", "reliability_percent")
        with (OUT / "plotted_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for scenario in SCALES:
                for d in DEADLINES:
                    for mode in METHODS:
                        m = metrics(data[scenario, d, mode])
                        writer.writerow({
                            "scenario": scenario, "deadline": d, "method": mode,
                            "n": m["n"], "energy_mean_j": m["energy_mean"] * 1000,
                            "energy_std_j": m["energy_std"] * 1000,
                            "esp90_percent": m["esp90"] * 100,
                            "reliability_percent": m["reliability_esp100"] * 100,
                        })
    print(f"Validated {sum(map(len, data.values()))} seed rows and "
          f"{len(data)} summary cells; saved {'A/C' if args.all else 'A/B/C'}.")
