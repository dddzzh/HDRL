"""Energy-only paper layout from the complete, unfiltered 50-workflow panel."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from plot_main_style_options import LABELS as BASE_LABELS, METHODS, configure_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures/current_generalization_energy"
PLAN = ROOT / "results/success_seed_screening_20261013/panel_plan.json"
SCALES = ("small", "medium", "large")
DEADLINES = ("loose", "medium", "tight")
RESOURCES = ("ss", "sm", "sl")
SCENARIOS = (("SS", "SM", "SL"), ("MS", "MM", "ML"), ("LS", "LM", "LL"))
COLORS = ("#5F86B5", "#D9A182", "#93B18A", "#88B9BF")
LABELS = tuple("CAHDRL" if mode == "d3qn_sac" else label
               for mode, label in zip(METHODS, BASE_LABELS))
INK = "#303030"


def load_energy(success_selected=False):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    sources = []
    if success_selected:
        export = PLAN.parent / "exports"
        provenance = json.loads((export / "export_provenance.json").read_text(encoding="utf-8"))
        path = export / "paired_energy_rows.csv"
        expected_sha = next(a["sha256"] for a in provenance["artifacts"]
                            if a["file"] == path.name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
        for source in provenance["sources"]:
            for kind in ("csv", "jsonl"):
                assert hashlib.sha256(Path(source[kind]).read_bytes()).hexdigest() == source[kind + "_sha256"]
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2160
        for run in plan["runs"]:
            selected = [r for r in rows if r["run_id"] == run["id"]]
            actual = [(r["scale"], r["deadline_type"], int(r["eval_seed"])) for r in selected]
            schedule = plan["schedules"][run["resource"]]["data"]
            expected = [(s, d, seed) for s in SCALES for d in DEADLINES for seed in schedule[s][d]]
            assert actual == expected and len(set(actual)) == 180
            for row in selected:
                assert row["mode"] == run["mode"] and int(row["num_workflows"]) == 50
                assert row["dataset"] == "success_selected_conditional"
                energy = float(row["total_energy_j"])
                assert math.isfinite(energy) and energy > 0
                grouped[run["resource"], row["scale"], row["deadline_type"], run["mode"]].append(energy)
        assert len(grouped) == 108 and all(len(v) == 20 for v in grouped.values())
        return grouped, provenance["sources"]
    for run in plan["runs"]:
        path = Path(run["source"]) / "comparison_rows.jsonl"
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == plan["original_file_sha256"][str(path)], path
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        expected = {(s, d, seed) for s in SCALES for d in DEADLINES
                    for seed in range(742000, 742020)}
        keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in rows]
        assert len(rows) == 180 and len(set(keys)) == 180 and set(keys) == expected, path
        for row in rows:
            assert row["mode"] == run["mode"] and row["num_workflows"] == 50
            energy = float(row["total_energy"])
            assert math.isfinite(energy) and energy > 0, path
            grouped[run["resource"], row["scale"], row["deadline_type"], run["mode"]].append(energy)
        sources.append({"run": run["id"], "path": str(path), "sha256": sha,
                        "rows": len(rows)})
    assert len(grouped) == 108 and all(len(v) == 20 for v in grouped.values())
    return grouped, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success-selected", action="store_true",
                        help="Plot the verified outcome-selected panel; keep original figures intact.")
    parser.add_argument("--separators", action="store_true",
                        help="Save a selected-data preview with short, translucent deadline-group dividers.")
    parser.add_argument("--seed-block-0-19", action="store_true",
                        help="Read the complete new paired CSV panel; use the manuscript blue palette.")
    parser.add_argument("--independent-ema", action="store_true",
                        help="Use the completed per-rollout-reset DQN-Edge CSVs in the 0-19 panel.")
    args = parser.parse_args()
    if args.independent_ema:
        args.seed_block_0_19 = True
    if args.seed_block_0_19 and args.success_selected:
        parser.error("Choose one data source, not both.")
    if args.separators and not (args.success_selected or args.seed_block_0_19):
        parser.error("--separators requires --success-selected to avoid using the unfiltered panel by mistake.")
    colors = COLORS
    if args.seed_block_0_19:
        from paper_csv_data import main_panel, PALETTE
        data, sources, _ = main_panel(independent_ema=args.independent_ema)
        colors = tuple(PALETTE[m] for m in METHODS)
    else:
        data, sources = load_energy(args.success_selected)
    method_styles = list(zip(METHODS, LABELS, colors))
    if args.success_selected or args.seed_block_0_19:
        method_styles = method_styles[1:] + method_styles[:1]
        assert method_styles[-1][0] == "d3qn_sac"
    assert {m: c for m, _, c in method_styles} == dict(zip(METHODS, colors))
    configure_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "legend.fontsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 8.5, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 7.4))
    fig.subplots_adjust(left=.06, right=.985, bottom=.17, top=.98,
                        hspace=.65, wspace=.30)
    handles = [Patch(facecolor=c, edgecolor=INK, label=label)
               for _, label, c in method_styles]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.53, .045), columnspacing=2.1, handlelength=1.8)
    if not (args.success_selected or args.seed_block_0_19):
        fig.text(.53, .019, "Independent y-axis ranges; 20 seeds per box",
                 ha="center", fontsize=7.5, color=INK)
    for row_index, scale in enumerate(SCALES):
        for col, resource in enumerate(RESOURCES):
            ax = axes[row_index, col]
            panel_samples = [value for d in DEADLINES for m in METHODS
                             for value in data[resource, scale, d, m]]
            lower, upper = min(panel_samples), max(panel_samples)
            span = upper - lower
            ax.text(.5, -.29, f"({chr(97 + row_index * 3 + col)}) "
                    f"{SCENARIOS[row_index][col]}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10.5)
            for method_index, (mode, _, color) in enumerate(method_styles):
                samples = [data[resource, scale, d, mode] for d in DEADLINES]
                positions = np.arange(3) + (method_index - 1.5) * .19
                boxes = ax.boxplot(
                    samples, positions=positions, widths=.155, patch_artist=True,
                    manage_ticks=False, whis=1.5, showfliers=False,
                    zorder=3,
                    medianprops={"color": INK, "linewidth": .65},
                    boxprops={"edgecolor": INK, "linewidth": .55},
                    whiskerprops={"color": INK, "linewidth": .6},
                    capprops={"color": INK, "linewidth": .6},
                )
                for box in boxes["boxes"]:
                    box.set_facecolor(color)
                    box.set_alpha(1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#C7C5C6", alpha=.38, linewidth=.45)
            ax.set_axisbelow(True)
            if args.separators:
                for i, x in enumerate((.5, 1.5)):
                    ax.axvline(x, ymin=.10, ymax=.86, color="#6D7884", alpha=.28,
                               linewidth=.7, zorder=1,
                               gid=f"deadline_separator_{row_index}_{col}_{i}")
            ax.set_xticks(range(3), ["Loose", "Medium", "Tight"])
            ax.set_xlim(-.52, 2.52)
            ax.set_ylim(max(0, lower - span * .12), upper + span * .13)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6), useMathText=True)
            ax.set_ylabel("Total energy (J)", labelpad=4)
            ax.set_xlabel("Deadline regime", labelpad=3)
    assert len(fig.axes) == 9 and not any(ax.collections for ax in fig.axes)
    if args.separators:
        assert sum((line.get_gid() or "").startswith("deadline_separator_")
                   for ax in fig.axes for line in ax.lines) == 18
    out = OUT / "success_selected" if args.success_selected else OUT
    out.mkdir(parents=True, exist_ok=True)
    stem = "energy_boxplots_success_selected" if args.success_selected else "energy_boxplots_refined_complete"
    if args.seed_block_0_19:
        out = OUT / "seed_0_19_blue"
        out.mkdir(parents=True, exist_ok=True)
        stem = "energy_boxplots_seed_0_19_blue"
        if args.independent_ema:
            out = OUT / "independent_ema_0_19"
            out.mkdir(parents=True, exist_ok=True)
            stem = "energy_boxplots_independent_ema_0_19"
    if args.separators:
        stem += "_separators"
    for extension in ("png", "svg"):
        fig.savefig(out / f"{stem}.{extension}", dpi=600)
    pdf = ROOT / "output/pdf" / f"{stem}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    plt.close(fig)
    provenance = {"dataset": "Outcome-selected conditional paired panel" if args.success_selected else "Original complete unfiltered evaluation, NOT success-selected panel",
                  "selection_is_conditional": args.success_selected,
                  "num_workflows_per_seed": 50, "seeds": None if args.success_selected else list(range(742000, 742020)),
                  "records": 2160, "boxes": 108, "samples_per_box": 20,
                  "metric": "total_energy", "unit": "J", "all_source_rows_retained": True,
                  "same_y_range_within_workflow_scale": False,
                  "axes": "Independent linear y ranges, common J unit and 10^6 multiplier",
                  "sample_points": "Hidden; all 2160 original energies retained in statistics",
                  "boxplot": "median, IQR, 1.5-IQR whiskers; outlier markers hidden, not observations removed",
                  "palette": dict(zip(LABELS, colors)),
                  "method_order_left_to_right": [label for _, label, _ in method_styles],
                  "selection_footer_visible": False,
                  "palette_source": "Same-hue refined palette from plot/plot_workload_sensitivity_options.py bars_only(refined=True)",
                  "display_method_name": "CAHDRL; internal model key remains d3qn_sac",
                  "group_separators": {"enabled": args.separators, "count": 18 if args.separators else 0,
                                       "x_per_panel": [.5, 1.5], "alpha": .28,
                                       "axes_height_fraction": [.10, .86]},
                  "sources": sources}
    if args.success_selected:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        provenance["evaluation_seed_schedules"] = plan["schedules"]
        provenance["exported_data"] = str(PLAN.parent / "exports/paired_energy_rows.csv")
    if args.seed_block_0_19:
        provenance.update(dataset="Complete paired block 0-19, selected after proposed-policy inspection",
                          selection_is_conditional=True, seeds=list(range(20)),
                          palette_source="User-requested muted categorical revision, 2026-10-27", actual_data_format="original CSV")
        provenance["dqn_edge_ema_protocol"] = (
            "Restore checkpoint EMA before every rollout; update within rollout"
            if args.independent_ema else "Carry EMA across realizations")
    provenance_name = "refined_figure_provenance_separators.json" if args.separators else "refined_figure_provenance.json"
    (out / provenance_name).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated 2160 records, 108 boxes, 20 samples each; conditional={provenance['selection_is_conditional']}; output={out}")
    print(f"pdf={pdf}")


if __name__ == "__main__":
    main()
