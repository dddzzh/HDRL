"""Audit existing results and export manuscript tables/figures; never run an agent."""
import csv
import json
from pathlib import Path
import statistics
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_main_style_options import load_data, METHODS, LABELS, COLORS, DEADLINES
from plot_paper_experiments_20260930 import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/manuscript_revised_20261006/experiment_assets"
SCENARIOS = ("S26", "S37", "M37", "M53")
ABLATION = ROOT / "results/ablation_results/small_main_ablation/ablation_summary.csv"
VARIANTS = (
    "Full D3QN+SAC", "Full checkpoint with rho=1",
    "Independently trained full high with rho=1", "HEFT+SAC",
    "HEFT with rho=1", "High without deadline head with rho=1",
)
ABL_LABELS = ("Full", r"Full, $\rho=1$", r"High only, $\rho=1$",
              "HEFT+SAC", r"HEFT, $\rho=1$", "No deadline head")
RUNS = {
    "S26": ("v2_small_20edge_6cloud_dvfs_v3_future_esp/20260916_100106", 1140),
    "S37": ("v2_small_28edge_9cloud_dvfs_v3_future_esp/20260924_160451", None),
    "M37": ("v2_medium_28edge_9cloud_dvfs_v3_future_esp/20260916_102004", 1060),
    "M53": ("v2_medium_40edge_13cloud_dvfs_v3_future_esp/20260925_170500", 1180),
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def export(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / (name + ".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main_figure(data):
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.75),
                             gridspec_kw={"height_ratios": [1.5, 1]})
    fig.subplots_adjust(left=.075, right=.992, bottom=.105, top=.84,
                        wspace=.30, hspace=.50)
    for col, scenario in enumerate(SCENARIOS):
        ax = axes[0, col]
        for j, method in enumerate(METHODS):
            boxes = ax.boxplot(
                [[r["energy_kj"] * 1000 for r in data[scenario, d, method]]
                 for d in DEADLINES], positions=np.arange(3) + (j - 1.5) * .18,
                widths=.145, patch_artist=True, manage_ticks=False,
                boxprops={"linewidth": .6}, whiskerprops={"linewidth": .6},
                capprops={"linewidth": .6}, medianprops={"color": "#333333", "linewidth": .8},
                flierprops={"marker": "o", "markersize": 2, "markeredgewidth": 0,
                            "markerfacecolor": COLORS[j]},
            )
            for box in boxes["boxes"]:
                box.set_facecolor(COLORS[j])
                box.set_alpha(.88)
        ax.set_title(f"({chr(97+col)}) {scenario}")
        ax.set_xlim(-.5, 2.5)
        ax.set_xticks(range(3), ["L", "M", "T"])
        ax.set_ylim(0, 600000 if col < 2 else 800000)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.grid(axis="y", color="#DDE1E4", linewidth=.45)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("Total energy (J)")
        ax.spines[["top", "right"]].set_visible(False)
        heat = axes[1, col]
        values = np.array([[metrics(data[scenario, d, m])["esp90"] * 100
                            for d in DEADLINES] for m in METHODS])
        heat.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(4):
            for j in range(3):
                heat.text(j, i, f"{values[i,j]:.0f}%", ha="center", va="center",
                          color="white" if values[i,j] >= 60 else "#222222", fontsize=6)
        heat.set_xticks(range(3), ["L", "M", "T"])
        heat.set_yticks(range(4), LABELS if col == 0 else [""] * 4, fontsize=6)
        heat.set_xticks(np.arange(-.5, 3, 1), minor=True)
        heat.set_yticks(np.arange(-.5, 4, 1), minor=True)
        heat.grid(which="minor", color="white", linewidth=.6)
        heat.tick_params(which="both", length=0)
        heat.set_xlabel("Deadline regime")
        heat.set_title(f"({chr(101+col)}) ESP@90")
        for spine in heat.spines.values():
            spine.set_visible(False)
    fig.legend([Patch(facecolor=c) for c in COLORS], LABELS, ncol=4,
               frameon=False, loc="upper center", fontsize=7,
               bbox_to_anchor=(.53, .99))
    export(fig, "main_comparison")


def ablation_figure(rows):
    lookup = {(r["variant"], r["deadline_type"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    palette = (*COLORS[:2], "#739C66", "#9A7399", "#56B4E9", "#7F7F7F")
    hatches = ("", "//", "\\\\", "xx", "++", "..")
    for i, variant in enumerate(VARIANTS):
        x = np.arange(3) + (i - 2.5) * .12
        rs = [lookup[variant, d] for d in DEADLINES]
        axes[0].bar(x, [float(r["energy_mean_kj"]) * 1000 for r in rs], width=.11,
                    yerr=[float(r["energy_std_kj"]) * 1000 for r in rs],
                    color=palette[i], hatch=hatches[i], edgecolor="#333333", linewidth=.5,
                    error_kw={"lw": .6, "capsize": 1.5}, label=ABL_LABELS[i])
        axes[1].bar(x, [float(r["esp_at_0_90"]) * 100 for r in rs], width=.11,
                    color=palette[i], hatch=hatches[i], edgecolor="#333333", linewidth=.5)
    for ax in axes:
        ax.set_xticks(range(3), ["Loose", "Medium", "Tight"])
        ax.grid(axis="y", color="#DDE1E4", linewidth=.45)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Total energy (J)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[0].set_title("(a) Energy: mean and sample SD")
    axes[1].set_ylabel("ESP@90 (%)")
    axes[1].set_ylim(0, 108)
    axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[1].axhline(90, color="#555555", lw=.7, ls="--")
    axes[1].set_title("(b) Deadline service attainment")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.02), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .83))
    export(fig, "ablation")


def learning_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    audit = {}
    for color, (scenario, (run, selected)) in zip(COLORS, RUNS.items()):
        path = ROOT / "results" / run / "eval_history.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_ep = {}
        for r in records:
            by_ep.setdefault(int(r["episode"]), []).append(r)
        eps, energies, esp = [], [], []
        for ep, rs in sorted(by_ep.items()):
            samples = []
            for r in rs:
                samples.extend(r.get("validation_blocks", [r]))
            # Prefer explicit block rows; never count a pooled row a second time.
            unique = {int(r.get("eval_seed", 542000)): r for r in samples}
            assert unique, (scenario, ep)
            rates = []
            for r in unique.values():
                rates.append(float(r.get("deadline_esp_0p90", statistics.mean(
                    float(x) >= .90 - 1e-12 for x in r["deadline_success_rate_samples"]))))
            eps.append(ep)
            energies.append(statistics.mean(float(r["total_energy"]) for r in unique.values()))
            esp.append(min(rates) * 100)
        axes[0].plot(eps, energies, color=color, lw=.9, label=scenario)
        axes[1].plot(eps, esp, color=color, lw=.9, label=scenario)
        audit[scenario] = {"history": str(path.relative_to(ROOT)), "last_logged_episode": max(eps),
                           "points": len(eps), "selected_checkpoint_episode": selected,
                           "last_validation_energy_j": energies[-1], "last_min_block_esp90": esp[-1] / 100}
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.grid(color="#DDE1E4", lw=.45)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("(a) Validation energy")
    axes[0].set_ylabel("Total energy (J)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[1].set_title("(b) Minimum validation-block ESP")
    axes[1].set_ylabel("ESP@90 (%)")
    axes[1].set_ylim(-3, 105)
    axes[1].set_yticks([0, 25, 50, 75, 100])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .87))
    export(fig, "learning_dynamics")
    return audit


def tables(data, ablation):
    main_rows, savings = [], []
    for scenario in SCENARIOS:
        for j, method in enumerate(METHODS):
            cells = []
            for d in DEADLINES:
                s = metrics(data[scenario, d, method])
                cell = f'{s["energy_mean"]*1000:,.2f} $\\pm$ {s["energy_std"]*1000:,.2f}'
                if method == METHODS[0]:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            main_rows.append(f'{scenario if j == 0 else ""} & {LABELS[j]} & ' + " & ".join(cells) + r" \\")
        if scenario != SCENARIOS[-1]:
            main_rows.append(r"\midrule")
        for d in DEADLINES:
            ours = metrics(data[scenario, d, METHODS[0]])
            feasible = [(m, metrics(data[scenario, d, m])) for m in METHODS[1:]
                        if metrics(data[scenario, d, m])["esp90"] >= .9 - 1e-12]
            best, stat = min(feasible, key=lambda ms: ms[1]["energy_mean"])
            assert ours["esp90"] >= .9 - 1e-12
            savings.append({"scenario": scenario, "deadline": d, "baseline": best,
                            "saving_percent": (1 - ours["energy_mean"] / stat["energy_mean"]) * 100})
    (OUT / "main_energy_rows.tex").write_text("\n".join(main_rows) + "\n", encoding="utf-8")
    rows = []
    lookup = {(r["variant"], r["deadline_type"]): r for r in ablation}
    for variant, label in zip(VARIANTS, ABL_LABELS):
        cells = []
        for d in DEADLINES:
            r = lookup[variant, d]
            e = f'{float(r["energy_mean_kj"])*1000:,.2f} $\\pm$ {float(r["energy_std_kj"])*1000:,.2f}'
            if variant == VARIANTS[0]:
                e = r"\textbf{" + e + "}"
            if float(r["esp_at_0_90"]) < .9 - 1e-12:
                e = r"\textcolor{gray}{" + e + "}"
            cells += [e, f'{float(r["esp_at_0_90"])*100:.0f}\\%']
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    (OUT / "ablation_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return savings


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 7, "axes.titlesize": 8, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    data = load_data(True)  # Existing loader checks every original, unfiltered 20-seed block.
    ablation = read_csv(ABLATION)
    assert len(ablation) == 18
    assert {(r["variant"], r["deadline_type"]) for r in ablation} == {(v, d) for v in VARIANTS for d in DEADLINES}
    assert all(float(r["reliability_esp_at_1_00"]) == 1 for r in ablation)
    main_figure(data)
    ablation_figure(ablation)
    learning = learning_figure()
    savings = tables(data, ablation)
    audit = {"primary_test_seeds": [242000, 242019], "main_groups": len(data),
             "main_seed_rows": sum(len(x) for x in data.values()), "ablation_test_seeds": [742000, 742019],
             "paper_feasibility": {"esp90_min": .90, "reliability_attainment": 1.0},
             "energy_savings_vs_lowest_energy_feasible_baseline": savings, "learning": learning}
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    verify_manuscript()
    print(json.dumps(audit, indent=2))


def verify_manuscript():
    """Static consistency only: not a substitute for a TeX compilation."""
    folder = OUT.parent
    sources = [folder / "main.tex", folder / "experiments.tex",
               OUT / "main_energy_rows.tex", OUT / "ablation_rows.tex"]
    contents = [p.read_text(encoding="utf-8") for p in sources]
    text = "\n".join(contents)
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    assert len(labels) == len(set(labels)), "Duplicate LaTeX labels"
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text))
    assert refs <= set(labels), f"Undefined references: {refs - set(labels)}"
    bib = (folder / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
    cites = {key.strip() for match in re.findall(r"\\cite\{([^}]+)\}", text)
             for key in match.split(",")}
    assert cites <= keys, f"Missing bibliography entries: {cites - keys}"
    for path, content in zip(sources, contents):
        clean = re.sub(r"(?<!\\)%[^\n]*", "", content)
        stack = []
        for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", clean):
            if kind == "begin":
                stack.append(name)
            else:
                assert stack and stack.pop() == name, (path.name, name)
        assert not stack, (path.name, stack)
        assert len(re.findall(r"(?<!\\)\{", clean)) == len(re.findall(r"(?<!\\)\}", clean)), path.name
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        assert (folder / name).is_file(), f"Missing figure: {name}"
    for name, count, columns in (("main_energy_rows.tex", 16, 5), ("ablation_rows.tex", 6, 7)):
        rows = [line for line in (OUT / name).read_text(encoding="utf-8").splitlines() if " & " in line]
        assert len(rows) == count and all(line.count(" & ") == columns - 1 for line in rows), name
    print("PASS: labels, citations, environments, braces, figures and table rows")


if __name__ == "__main__":
    main()
