"""Exact paired energy tests, blocking repeated conditions by test seed."""

import csv
import math
from collections import Counter
from statistics import mean

from plot_main_style_options import ROOT, METHODS, LABELS, DEADLINES, load_data

OUT = ROOT / "docs/figures/main_comparison_all_scenarios"


def wilcoxon_exact(differences):
    """Two-sided signed-rank sign permutation; discard zeros, average ties."""
    values = sorted((abs(d), d > 0) for d in differences if d != 0)
    assert all(math.isfinite(d) for d in differences)
    if not values:
        return 0.0, 1.0
    ranks, positive, start = [], 0, 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end][0] == values[start][0]:
            end += 1
        rank = start + 1 + end  # twice the average rank: integer even with ties
        ranks.extend([rank] * (end - start))
        positive += rank * sum(sign for _, sign in values[start:end])
        start = end
    statistic = min(positive, sum(ranks) - positive)
    counts = Counter({0: 1})
    for rank in ranks:
        previous = list(counts.items())
        for total, count in previous:
            counts[total + rank] += count
    p = min(1.0, 2 * sum(c for total, c in counts.items()
                        if total <= statistic) / 2 ** len(ranks))
    return statistic / 2, p


def correct(rows):
    n, running = len(rows), 1.0
    ordered = sorted(rows, key=lambda r: r["p_raw"])
    for i in range(n - 1, -1, -1):
        row = ordered[i]
        running = min(running, row["p_raw"] * n / (i + 1))
        row["p_fdr_bh"] = running
        row["p_bonferroni"] = min(1.0, n * row["p_raw"])


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_p(p):
    mantissa, exponent = f"{p:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def main():
    # Check against the exact example in the official SciPy documentation.
    assert wilcoxon_exact([6, 8, 14, 16, 23, 24, 28, 29, 41, -48,
                          49, 56, 60, -67, 75]) == (24.0, 0.041259765625)
    assert wilcoxon_exact([1, -1]) == (1.5, 1.0)
    assert wilcoxon_exact([0, 0]) == (0.0, 1.0)
    assert wilcoxon_exact(list(range(1, 21))) == (0.0, 2 / 2**20)
    data = load_data(True)
    lookup = {key: {r["seed"]: r["energy_kj"] for r in rows}
              for key, rows in data.items()}
    conditions = [(s, d) for s in ("S26", "S37", "M37", "M53")
                  for d in DEADLINES]
    seeds = list(range(242000, 242020))
    overall, details, pairs = [], [], []
    for mode, label in zip(METHODS[1:], LABELS[1:]):
        contrasts = {seed: [] for seed in seeds}
        for scenario, deadline in conditions:
            differences = []
            for seed in seeds:
                ours = lookup[scenario, deadline, METHODS[0]][seed]
                baseline = lookup[scenario, deadline, mode][seed]
                assert ours > 0 and baseline > 0
                difference = (baseline - ours) / baseline
                contrasts[seed].append(difference)
                differences.append(difference)
                pairs.append(dict(scenario=scenario, deadline=deadline,
                                  baseline=label, seed=seed, ours_j=ours * 1000,
                                  baseline_j=baseline * 1000,
                                  relative_energy_reduction=difference))
            w, p = wilcoxon_exact(differences)
            details.append(dict(scenario=scenario, deadline=deadline,
                                baseline=label, n=20, statistic=w, p_raw=p,
                                positive_pairs=sum(d > 0 for d in differences)))
        differences = [mean(contrasts[seed]) for seed in seeds]
        w, p = wilcoxon_exact(differences)
        overall.append(dict(baseline=label, n=20, conditions=12, statistic=w,
                            p_raw=p, positive_pairs=sum(d > 0 for d in differences),
                            mean_relative_reduction_percent=100 * mean(differences)))
    correct(overall)
    correct(details)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("wilcoxon_energy_overall.csv", overall)
    write_csv("wilcoxon_energy_by_condition.csv", details)
    write_csv("wilcoxon_energy_pairs.csv", pairs)
    table = r"""\begin{table}[t]
\centering
\caption{Two-sided paired Wilcoxon signed-rank tests of total energy: our method versus each baseline.}
\label{tab:wilcoxon_energy}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{@{}lrrr@{}}
\toprule
Baseline & Raw $p$ & Bonferroni & BH--FDR \\
\midrule
"""
    for row in overall:
        table += row["baseline"] + " & " + " & ".join(
            latex_p(row[k]) for k in ("p_raw", "p_bonferroni", "p_fdr_bh")
        ) + r" \\" + "\n"
    table += r"""\bottomrule
\end{tabular}
\vspace{2pt}
\begin{minipage}{\columnwidth}
\footnotesize
For each of the 20 fixed test seeds, the relative energy reduction
$(E_{\mathrm{baseline}}-E_{\mathrm{ours}})/E_{\mathrm{baseline}}$
is averaged over four resource--workload scenarios and three deadline regimes.
The 20 seed-level contrasts, rather than the 240 repeated condition-level
measurements, are tested against zero. Both corrections cover the three
baseline comparisons; the significance level is $0.05$.
\end{minipage}
\end{table}
"""
    (OUT / "wilcoxon_energy_table.tex").write_text(table, encoding="utf-8")
    report = "# Paired energy significance tests\n\n"
    report += "Primary data: original fixed test seeds 242000--242019; no seed replacement.\n\n"
    report += "Each seed is one block. Its relative energy reduction is averaged equally over S26/S37/M37/M53 and loose/medium/tight before a two-sided exact signed-rank test. Zeros are discarded; tied absolute contrasts receive average ranks. P-values enumerate the sign-permutation distribution via integer rank-sum counting.\n\n"
    report += "| Baseline | n | W | Raw p | Bonferroni p | BH-FDR p | Positive seed contrasts | Mean reduction |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in overall:
        report += f"| {r['baseline']} | 20 | {r['statistic']:g} | {r['p_raw']:.8g} | {r['p_bonferroni']:.8g} | {r['p_fdr_bh']:.8g} | {r['positive_pairs']}/20 | {r['mean_relative_reduction_percent']:.2f}% |\n"
    if all(r["p_bonferroni"] < .05 and r["p_fdr_bh"] < .05 for r in overall):
        report += "\nAll three overall comparisons survive both corrections.\n"
    if all(r["positive_pairs"] == 20 for r in overall):
        report += "Identical p-values result from all 20 seed-level contrasts having the same sign, not identical energy measurements.\n\n"
    report += "The per-condition CSV contains 36 supplementary tests, corrected as a separate family of 36. These tests do not establish deadline feasibility, reliability superiority, training-seed reproducibility, or generalization beyond the tested workloads. Signed-rank inference assumes independent seed blocks and a symmetric distribution of contrasts under the null.\n\n"
    report += "Implementation check: reproduces W=24 and exact p=0.041259765625 from the [official SciPy Wilcoxon example](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).\n\n"
    report += "Reproduce: `D:\\Miniconda\\envs\\py310\\python.exe plot/analyze_main_wilcoxon.py`\n"
    (OUT / "WILCOXON_REPORT.md").write_text(report, encoding="utf-8")
    for row in overall:
        print(row)


if __name__ == "__main__":
    main()
