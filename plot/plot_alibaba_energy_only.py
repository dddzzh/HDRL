"""Redraw only the energy panel from the unfiltered Alibaba evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.plot_main_style_options import configure_style

SOURCE = ROOT / "results/alibaba_sl111_100wf_20261015"
OUT = ROOT / "docs/figures/alibaba_energy_only"
METHODS = ("dqn_edge", "dream_dqn", "oo_actor_critic", "d3qn_sac")
LABELS = ("DQN-Edge", "DReaM-DQN", "OOAC", "CAHDRL")
COLORS = ("#F5AA83", "#BBDDB2", "#BAE0E3", "#A8C0E4")
DEADLINES = ("loose", "medium", "tight")


def load_data(csv_source=False, independent_ema=False):
    from plot.paper_csv_data import CURRENT, current_dataset
    if (CURRENT/'manifest.json').exists():
        groups,sources={},[]
        for mode in METHODS:
            records,files=current_dataset('alibaba',mode)
            assert [(r['deadline_type'],r['eval_seed']) for r in records] == [
                (d,s) for d in DEADLINES for s in range(742000,742020)]
            for deadline in DEADLINES:
                sample=[r for r in records if r['deadline_type']==deadline]
                assert len(sample)==20 and all(r['num_workflows']==100 for r in sample)
                groups[mode,deadline]=np.array([r['total_energy'] for r in sample],dtype=float)
            sources.extend(files)
        plan_path=CURRENT/'provenance/results/alibaba_sl111_100wf_20261015/evaluation_plan.json'
        return groups,sources,hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_path = SOURCE / "evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert plan["selection_is_conditional"] is False
    assert plan["num_servers"] == 111 and plan["num_workflows"] == 100
    assert plan["seeds"] == list(range(742000, 742020))
    groups, sources = {}, []
    for mode in METHODS:
        if independent_ema and mode == "dqn_edge":
            from plot.paper_csv_data import independent_extra_rows
            rows, source = independent_extra_rows("alibaba")
            path = Path(source["path"])
            completion = json.loads((path.parent / "completion.json").read_text(encoding="utf-8"))
            assert completion["records"] == 60 and not completion["failures"]
        else:
            path = SOURCE / mode / "comparison_rows.jsonl"
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if csv_source:
                from plot.paper_csv_data import read_csv
                path = path.with_suffix(".csv")
                rows = read_csv(path)
            completion = json.loads((path.parent / "completion.json").read_text(encoding="utf-8"))
            provenance = json.loads((path.parent / "run_provenance.json").read_text(encoding="utf-8"))
            assert completion["records"] == 60 and not completion["failures"]
            assert provenance["plan_sha256"] == plan_sha and provenance["unselected"] is True
        keys = [(r["deadline_type"], r["eval_seed"]) for r in rows]
        assert keys == [(d, s) for d in DEADLINES for s in plan["seeds"]]
        assert len(set(keys)) == 60
        for row in rows:
            assert row["mode"] == mode and row["num_workflows"] == 100
            sample = plan["samples"][row["deadline_type"]][str(row["eval_seed"])]
            assert all(row[k] == sample[k] for k in ("physical_sha256", "deadlines_sha256", "tasks", "edges"))
            assert row["deadline_success_rate"] >= .9 - 1e-9
            assert row["reliability_success_rate"] >= 1 - 1e-9
        for deadline in DEADLINES:
            energy = np.array([r["total_energy"] for r in rows if r["deadline_type"] == deadline], dtype=float)
            assert len(energy) == 20 and np.isfinite(energy).all() and (energy > 0).all()
            groups[mode, deadline] = energy
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=60))
    assert len(groups) == 12
    return groups, sources, plan_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separators", action="store_true", help="Save a separate preview with subtle group dividers.")
    parser.add_argument("--paper-blue", action="store_true", help="Read original CSV and use the manuscript blue palette.")
    parser.add_argument("--independent-ema", action="store_true", help="Use completed per-rollout EMA-reset DQN-Edge CSV.")
    args = parser.parse_args()
    if args.independent_ema:
        args.paper_blue = True
    data, sources, plan_sha = load_data(csv_source=args.paper_blue, independent_ema=args.independent_ema)
    colors = COLORS
    if args.paper_blue:
        from plot.paper_csv_data import PALETTE
        colors = tuple(PALETTE[m] for m in METHODS)
    configure_style()
    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    fig.subplots_adjust(left=.13, right=.98, top=.95, bottom=.25)
    ink = "#444444"
    for i, (mode, color) in enumerate(zip(METHODS, colors)):
        ax.boxplot([data[mode, d] for d in DEADLINES],
            positions=np.arange(3) + (i - 1.5) * .19,
            widths=.155, manage_ticks=False, patch_artist=True, whis=1.5,
            showfliers=False, zorder=3,
            boxprops=dict(facecolor=color, edgecolor=ink, linewidth=.8),
            medianprops=dict(color=ink, linewidth=1),
            whiskerprops=dict(color=ink, linewidth=.8),
            capprops=dict(color=ink, linewidth=.8))
    ax.set_xticks(range(3), ["Loose", "Medium", "Tight"])
    ax.set(xlabel="Deadline regime", ylabel="Total energy (J)", xlim=(-.52, 2.52))
    values = np.concatenate(list(data.values()))
    ax.set_ylim(min(80000, values.min()*.9), max(320000, values.max()*1.05))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(5, 5), useMathText=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#C7C5C6", alpha=.4, linewidth=.55)
    ax.set_axisbelow(True)
    if args.separators:
        for i, x in enumerate((.5, 1.5)):
            ax.axvline(x, ymin=.10, ymax=.86, color="#6D7884", alpha=.28,
                       linewidth=.7, zorder=1, gid=f"deadline_separator_{i}")
        assert sum((line.get_gid() or "").startswith("deadline_separator_") for line in ax.lines) == 2
    handles = [Patch(facecolor=c, edgecolor=ink, linewidth=.8, label=l) for c, l in zip(colors, LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, .035),
               ncol=4, frameon=False, columnspacing=1.4, handlelength=1.5)
    assert len(fig.axes) == 1 and len(ax.patches) == 12
    assert not fig.texts and not ax.collections
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "alibaba_energy_boxplot" + ("_separators" if args.separators else "")
    if args.paper_blue:
        stem += "_blue"
    if args.independent_ema:
        stem += "_independent_ema"
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600)
    pdf = ROOT / "output/pdf" / f"{stem}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    plt.close(fig)
    provenance = dict(dataset="Alibaba-derived, 111 servers, 100 workflows",
        selection_is_conditional=False, records=240, boxes=12, samples_per_box=20,
        unit="J", methods_left_to_right=list(LABELS), colors=dict(zip(LABELS, colors)),
        retained_observations="all 240 original energies, no success filtering",
        boxplot="median, IQR, 1.5-IQR whiskers; outlier markers hidden, observations retained",
        removed=["ESP panel", "reliability attainment panel", "bottom explanatory footnote", "subfigure label"],
        original_three_panel_pdf_modified_by_this_script=False, manuscript_modified_by_this_script=False,
        evaluation_plan_sha256=plan_sha, sources=sources, pdf=str(pdf),
        group_separators=dict(enabled=args.separators, x=[.5, 1.5],
                              alpha=.28, axes_height_fraction=[.10, .86]))
    provenance_name = "provenance_separators.json" if args.separators else "provenance.json"
    if args.paper_blue:
        provenance_name = "provenance_blue.json"
    if args.independent_ema:
        provenance_name = "provenance_independent_ema.json"
        provenance["dqn_edge_ema_protocol"] = "Restore checkpoint EMA before each realization; update within realization"
    (OUT / provenance_name).write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Validated 240 unfiltered records, 12 boxes, 20 observations each.")
    print(pdf)


if __name__ == "__main__":
    main()
