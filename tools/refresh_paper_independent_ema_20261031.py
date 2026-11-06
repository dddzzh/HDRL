"""Audit original CSVs, compute paired tests, and emit a narrow manuscript patch.

No training, evaluation, checkpoint selection, or raw-data replacement occurs.
"""
import argparse
import csv
import difflib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plot"))
from plot.paper_csv_data import main_panel, read_csv, sha, SCALES, DEADLINES, MODES, RESOURCES
from plot.analyze_main_wilcoxon import wilcoxon_exact, correct, latex_p

OUT = ROOT / "docs/figures/paper_refresh_independent_ema_20261031"
NEW = ROOT / "results/dqn_edge_independent_ema_0_19_20261029"
TEX = ROOT / "paper/manuscript/main2.tex"
ORDER = ("dqn_edge", "dream_dqn", "oo_actor_critic", "d3qn_sac")
LABELS = dict(zip(ORDER, ("DQN-Edge", "DReaM-DQN", "OOAC", "CAHDRL")))
SCENARIOS = ("SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL")


def key_for(scenario, deadline, mode):
    return ({"S": "ss", "M": "sm", "L": "sl"}[scenario[1]],
            {"S": "small", "M": "medium", "L": "large"}[scenario[0]], deadline, mode)


def verify_new():
    plan = json.loads((NEW / "frozen_plan.json").read_text(encoding="utf-8"))
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for model in plan["models"]:
        folder = NEW / model["resource"]
        completion = json.loads((folder / "completion.json").read_text())
        manifest = json.loads((folder / "experiment_manifest.json").read_text())
        log = (NEW / f"{model['resource']}.log").read_text(encoding="utf-8-sig")
        assert "Traceback" not in log and log.rstrip().endswith(
            f"[independent-ema-complete] {model['resource']} records=180")
        assert not (folder / "failure.json").exists()
        assert completion["records"] == completion["ema_reset_count"] == 180
        assert completion["source_hashes_verified"]
        assert manifest["plan_sha256"] == sha(NEW / "frozen_plan.json")
        assert manifest["model_initialization_seed"] == 742000
        assert manifest["workflow_seeds"] == list(range(20))
        assert manifest["training"] is False
        data = read_csv(folder / "comparison_rows.csv")
        failures = []
        for row in data:
            assert row["ema_reset_before_rollout"] is True
            assert row["ema_initial_sha256"] == model["ema_initial_sha256"]
            assert row["network_sha256"] == model["network_sha256"]
            if row["deadline_success_rate"] < .9-1e-9 or row["reliability_success_rate"] < 1-1e-9:
                failures.append(row)
        assert len(failures) == len(completion["failures"])
    return plan


def write_csv(name, records):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0, center-half), min(1, center+half)


def audit():
    verify_new()
    grouped, sources, cells = main_panel(independent_ema=True)
    old_grouped, _, old_cells = main_panel()
    assert all(grouped[k] == old_grouped[k] for k in grouped if k[-1] != "dqn_edge")
    OUT.mkdir(parents=True, exist_ok=True)
    records, tests, pairs, changes = [], [], [], []
    for key, cell in cells.items():
        resource, scale, deadline, mode = key
        row = dict(resource=resource, scale=scale, deadline_type=deadline, mode=mode, **cell)
        for metric in ("esp_at_90", "reliability_attainment"):
            successes = round(cell[metric] * 20)
            row[metric + "_wilson95_low"], row[metric + "_wilson95_high"] = wilson(successes, 20)
        records.append(row)
        if mode == "dqn_edge":
            previous = old_cells[key]["energy_mean_J"]
            current = cell["energy_mean_J"]
            changes.append(dict(resource=resource, scale=scale, deadline_type=deadline,
                                old_energy_mean_J=previous, new_energy_mean_J=current,
                                change_percent=100*(current/previous-1)))
    source_rows = []
    for source in sources:
        path = Path(source["path"])
        resource = next(r for r in RESOURCES if r in str(path))[:2]
        for row in read_csv(path):
            source_rows.append(dict(resource=resource, scale=row["scale"],
                deadline_type=row["deadline_type"], mode=row["mode"], eval_seed=row["eval_seed"],
                num_workflows=row["num_workflows"], total_energy_J=row["total_energy"],
                deadline_success_rate=row["deadline_success_rate"],
                reliability_success_rate=row["reliability_success_rate"],
                source_csv=str(path.relative_to(ROOT))))
    assert len(source_rows) == 2160
    lookup = {(r["resource"], r["scale"], r["deadline_type"], r["mode"], r["eval_seed"]): r
              for r in source_rows}
    assert len(lookup) == 2160
    for scenario in SCENARIOS:
        for deadline in DEADLINES:
            ours_key = key_for(scenario, deadline, "d3qn_sac")
            for mode in ORDER[:-1]:
                base_key = key_for(scenario, deadline, mode)
                differences = []
                for seed in range(20):
                    ours = lookup[ours_key + (seed,)]["total_energy_J"]
                    baseline = lookup[base_key + (seed,)]["total_energy_J"]
                    difference = baseline - ours
                    differences.append(difference)
                    pairs.append(dict(scenario=scenario, deadline_type=deadline,
                        baseline=LABELS[mode], eval_seed=seed, cahdrl_energy_J=ours,
                        baseline_energy_J=baseline, difference_J=difference))
                statistic, p = wilcoxon_exact(differences)
                tests.append(dict(scenario=scenario, deadline_type=deadline, baseline=LABELS[mode],
                    n=20, statistic=statistic, p_raw=p,
                    positive_pairs=sum(d > 0 for d in differences),
                    zero_pairs=sum(d == 0 for d in differences),
                    median_energy_reduction_J=statistics.median(differences)))
    assert len(tests) == 81 and len(pairs) == 1620
    correct(tests)
    for label in LABELS.values():
        if label == "CAHDRL":
            continue
        # The compact manuscript table is valid only if all condition-level tests agree.
        selected = [t for t in tests if t["baseline"] == label]
        assert len({(t["p_raw"], t["p_bonferroni"], t["p_fdr_bh"]) for t in selected}) == 1
    write_csv("main_panel_original_csv_rows.csv", source_rows)
    write_csv("main_cells_J.csv", records)
    write_csv("dqn_edge_old_vs_new.csv", changes)
    write_csv("wilcoxon_energy_pairs_J.csv", pairs)
    write_csv("wilcoxon_by_condition.csv", tests)
    protected = {}
    for source in sources:
        path = Path(source["path"])
        for actual in (path, path.with_suffix(".jsonl")):
            protected[str(actual.relative_to(ROOT))] = sha(actual)
    for directory in ("alibaba_sl111_100wf_20261015", "ablation_sm74_new_eval_20261015",
                      "workflow_count_sm74_20261017", "arrival_rate_sm74_20261017"):
        for path in (ROOT / "results" / directory).rglob("comparison_rows.*"):
            if path.suffix in (".csv", ".jsonl"):
                protected[str(path.relative_to(ROOT))] = sha(path)
    for resource in RESOURCES:
        for suffix in ("csv", "jsonl"):
            path = ROOT / f"results/baselines_seed_block_0_19_20261026/{resource}_dqn_edge/comparison_rows.{suffix}"
            protected[str(path.relative_to(ROOT))] = sha(path)
    metadata = dict(records=2160, cells=108, changed_cells=27, unit="J", seeds=list(range(20)),
        dqn_edge_ema_protocol="Restore saved checkpoint EMA before every rollout; update within rollout",
        selection_is_conditional=True, source_sha256=protected, sources=sources,
        tests=tests, correction_family=81,
        unchanged_experiments=["Alibaba", "ablations", "workflow count", "arrival rate"],
        exact_test="Two-sided signed-rank sign permutation; zeros discarded, average tied ranks",
        inference_scope="Fixed checkpoints and reported workload block; not independent training repeats")
    (OUT / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = ["# Main experiment refresh: independent DQN-Edge EMA", "",
        "All energies were read from original CSVs and cross-checked against JSONL.",
        "Only DQN-Edge main-experiment data changed; the other 1,620 main records and separate experiments remain intact.",
        "The 540 new records restore the checkpoint EMA before each rollout, preserving within-rollout updates.", "",
        "## Paired energy tests", "",
        "81 exact two-sided Wilcoxon signed-rank tests: 27 conditions x 3 baselines, n=20 each.",
        "Differences are baseline energy minus CAHDRL energy in J, paired by resource, scale, deadline and actual seed.",
        "Bonferroni and BH adjustments cover the entire family of 81, alpha=0.05.", "",
        "| Baseline | Raw p | Bonferroni | BH-FDR | Significant conditions |",
        "|---|---:|---:|---:|---:|"]
    for mode in ORDER[:-1]:
        selected = [t for t in tests if t["baseline"] == LABELS[mode]]
        t = selected[0]
        significant = sum(t["p_bonferroni"] < .05 for t in selected)
        lines.append(f"| {LABELS[mode]} | {t['p_raw']:.9g} | {t['p_bonferroni']:.9g} | {t['p_fdr_bh']:.9g} | {significant}/27 |")
    low, high = wilson(20, 20)
    lines += ["", "Identical p-values follow from the same direction of all 20 paired differences, not identical measurements.",
        "Signed-rank inference assumes independent paired differences and sign symmetry under the null.",
        "The seed block was adopted after inspecting the proposed policy; independent EMA does not remove that selection limitation.",
        "Inference is conditional on these workloads/checkpoints and does not establish population feasibility or training robustness.",
        "", "## Wilson intervals (a separate quantity, not a Wilcoxon test)", "",
        f"Cell-level ESP/Rel. 95% Wilson bounds are in main_cells_J.csv. For 20 successes in 20 batches: [{100*low:.2f}%, {100*high:.2f}%].",
        "These are descriptive binomial intervals under an independent-batch assumption, not a correction for outcome-based seed selection.",
        "", "Raw CSV/JSONL files are preserved. Source hashes and exact source paths are recorded in audit.json.",
        "Old manuscript, figures, plot sources and old DQN-Edge CSVs: paper/manuscript/backups/independent_ema_20261031/.",
        "", "Reproduce audit: python tools/refresh_paper_independent_ema_20261031.py --audit",
        "", "Method reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audited 2160 records; new DQN-Edge 540; unchanged other records 1620; tested {len(tests)} contrasts.")
    return cells, tests


def manuscript_patch(original, cells, tests, all_experiments=False):
    updated = original
    first = updated.index("\\label{tab:constraint_results}")
    last = updated.index("\\bottomrule", first)
    block = updated[first:last]
    deadline = None
    changed = 0
    for line in block.splitlines():
        match = re.match(r"^(Loose|Medium|Tight)? & (SS|SM|SL|MS|MM|ML|LS|LM|LL) & ", line)
        if not match:
            continue
        if match[1]:
            deadline = match[1].lower()
        scenario = match[2]
        parts = line.split(" & ")
        cell = cells[key_for(scenario, deadline, "dqn_edge")]
        parts[2:5] = [f"{cell['energy_mean_J']:.2f}", f"{100*cell['esp_at_90']:g}\\%",
                      f"{100*cell['reliability_attainment']:g}\\%"]
        updated = updated.replace(line, " & ".join(parts), 1)
        changed += 1
    assert changed == 27
    old = "DQN-Edge carries its smoothed attention state between test\nrealizations and is evaluated in the recorded order."
    if old not in updated:
        old = "For DQN-Edge, the\nsmoothed attention state is carried across test realizations, which are evaluated in the recorded order."
    new = ("For the main experiment, DQN-Edge restores the same checkpoint-loaded\n"
           "attention EMA before each realization and updates it within that\n"
           "realization. Network parameters remain fixed. The separate Alibaba\n"
           "and sensitivity experiments retain their recorded evaluation protocols.")
    if all_experiments:
        new = ("For DQN-Edge, every realization in the main, Alibaba, and sensitivity experiments "
               "starts from the same checkpoint-loaded attention EMA and updates it within that "
               "realization only. Network parameters remain fixed; attention state is not carried "
               "between realizations.")
    assert updated.count(old) == 1
    updated = updated.replace(old, new, 1)
    start = updated.index("Exact two-sided Wilcoxon tests", updated.index("\\label{subsec:statistical_significance}"))
    end = updated.index("\n\n", start)
    updated = updated[:start] + (
        "Exact two-sided Wilcoxon signed-rank tests~\\cite{wilcoxon1945} compare 20 paired energy differences in J per condition. "
        "Bonferroni and Benjamini--Hochberg corrections~\\cite{benjamini1995} cover all 81 comparisons "
        "(27 conditions, three baselines), including DQN-Edge with realization-wise EMA reset. "
        "CAHDRL consumes less energy in every tested pair; all comparisons remain significant at $0.05$ after correction "
        "(Table~\\ref{tab:main_wilcoxon}). Identical $p$-values reflect uniformly signed differences, not identical energies. "
        "Inference concerns the reported workload blocks and fixed checkpoints, not training-seed variability or population feasibility."
    ) + updated[end:]
    updated = updated.replace("energy: 27 tests per baseline; --- denotes exclusion.",
                              "energy: 27 tests per baseline; corrections cover all 81 tests.", 1)
    start = updated.index("\\midrule", updated.index("\\label{tab:main_wilcoxon}")) + len("\\midrule")
    end = updated.index("\\bottomrule", start)
    lines = []
    for mode in ORDER[:-1]:
        test = next(t for t in tests if t["baseline"] == LABELS[mode])
        lines.append(LABELS[mode] + " & " + " & ".join(latex_p(test[k]) for k in
                     ("p_raw", "p_bonferroni", "p_fdr_bh")) + r" \\")
    updated = updated[:start] + "\n" + "\n".join(lines) + "\n" + updated[end:]
    reductions = []
    for scenario in SCENARIOS:
        for deadline in DEADLINES:
            ours = cells[key_for(scenario, deadline, "d3qn_sac")]["energy_mean_J"]
            best = min(cells[key_for(scenario, deadline, mode)]["energy_mean_J"] for mode in ORDER[:-1])
            reductions.append(100*(1-ours/best))
    span = f"{min(reductions):.2f}--{max(reductions):.2f}"
    updated = updated.replace("10.31--57.26", span)
    # Author manuscript changes through apply_patch, not direct file writes.
    diff = list(difflib.unified_diff(original.splitlines(True), updated.splitlines(True), n=3))
    print("*** Begin Patch\n*** Update File: " + str(TEX))
    print(re.sub(r"^@@ .*@@$", "@@", "".join(diff[2:]), flags=re.MULTILINE), end="")
    print("*** End Patch")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--tex-patch", action="store_true")
    args = parser.parse_args()
    if args.tex_patch:
        _, _, cells = main_panel(independent_ema=True)
        metadata = json.loads((OUT / "audit.json").read_text())
        manuscript_patch(TEX.read_text(encoding="utf-8"), cells, metadata["tests"])
    else:
        audit()
