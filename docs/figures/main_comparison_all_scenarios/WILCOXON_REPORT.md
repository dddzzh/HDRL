# Paired energy significance tests

Primary data: original fixed test seeds 242000--242019; no seed replacement.

Each seed is one block. Its relative energy reduction is averaged equally over S26/S37/M37/M53 and loose/medium/tight before a two-sided exact signed-rank test. Zeros are discarded; tied absolute contrasts receive average ranks. P-values enumerate the sign-permutation distribution via integer rank-sum counting.

| Baseline | n | W | Raw p | Bonferroni p | BH-FDR p | Positive seed contrasts | Mean reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| DQN-Edge | 20 | 0 | 1.9073486e-06 | 5.7220459e-06 | 1.9073486e-06 | 20/20 | 50.04% |
| DReaM-DQN | 20 | 0 | 1.9073486e-06 | 5.7220459e-06 | 1.9073486e-06 | 20/20 | 21.74% |
| OOAC | 20 | 0 | 1.9073486e-06 | 5.7220459e-06 | 1.9073486e-06 | 20/20 | 29.38% |

All three overall comparisons survive both corrections.
Identical p-values result from all 20 seed-level contrasts having the same sign, not identical energy measurements.

The per-condition CSV contains 36 supplementary tests, corrected as a separate family of 36. These tests do not establish deadline feasibility, reliability superiority, training-seed reproducibility, or generalization beyond the tested workloads. Signed-rank inference assumes independent seed blocks and a symmetric distribution of contrasts under the null.

Implementation check: reproduces W=24 and exact p=0.041259765625 from the [official SciPy Wilcoxon example](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

Reproduce: `D:\Miniconda\envs\py310\python.exe plot/analyze_main_wilcoxon.py`
