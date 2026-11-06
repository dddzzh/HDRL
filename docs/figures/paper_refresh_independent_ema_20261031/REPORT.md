# Main experiment refresh: independent DQN-Edge EMA

All energies were read from original CSVs and cross-checked against JSONL.
Only DQN-Edge main-experiment data changed; the other 1,620 main records and separate experiments remain intact.
The 540 new records restore the checkpoint EMA before each rollout, preserving within-rollout updates.

## Paired energy tests

81 exact two-sided Wilcoxon signed-rank tests: 27 conditions x 3 baselines, n=20 each.
Differences are baseline energy minus CAHDRL energy in J, paired by resource, scale, deadline and actual seed.
Bonferroni and BH adjustments cover the entire family of 81, alpha=0.05.

| Baseline | Raw p | Bonferroni | BH-FDR | Significant conditions |
|---|---:|---:|---:|---:|
| DQN-Edge | 1.90734863e-06 | 0.000154495239 | 1.90734863e-06 | 27/27 |
| DReaM-DQN | 1.90734863e-06 | 0.000154495239 | 1.90734863e-06 | 27/27 |
| OOAC | 1.90734863e-06 | 0.000154495239 | 1.90734863e-06 | 27/27 |

Identical p-values follow from the same direction of all 20 paired differences, not identical measurements.
Signed-rank inference assumes independent paired differences and sign symmetry under the null.
The seed block was adopted after inspecting the proposed policy; independent EMA does not remove that selection limitation.
Inference is conditional on these workloads/checkpoints and does not establish population feasibility or training robustness.

## Wilson intervals (a separate quantity, not a Wilcoxon test)

Cell-level ESP/Rel. 95% Wilson bounds are in main_cells_J.csv. For 20 successes in 20 batches: [83.89%, 100.00%].
These are descriptive binomial intervals under an independent-batch assumption, not a correction for outcome-based seed selection.

Raw CSV/JSONL files are preserved. Source hashes and exact source paths are recorded in audit.json.
Old manuscript, figures, plot sources and old DQN-Edge CSVs: paper/manuscript/backups/independent_ema_20261031/.

Reproduce audit: python tools/refresh_paper_independent_ema_20261031.py --audit

Method reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
