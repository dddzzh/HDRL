# Complete main comparison figures

The selected A and C styles cover S26, S37, M37 and M53, each with Loose,
Medium and Tight deadlines and four methods. Both figures use the same
960 test rows (48 groups of 20 seeds).

Layout: four scenarios horizontally aligned, ordered S26, S37, M37, M53.
The upper row shows energy and the lower row shows ESP@90.

## Sources

| Scenarios | Seed-level CSV | Summary for independent cross-check |
|---|---|---|
| S26 / M37 | docs/figures/formal_comparison_selected_columns.csv | docs/figures/formal_comparison_figure_summary.csv |
| S37 | results/scenario_matrix_evaluation/S37_final/comparison_rows.csv | comparison_summary.csv in the same folder |
| M53 | results/scenario_matrix_evaluation/M53_complete_4alg_20260927/comparison_rows.csv | comparison_summary.csv in the same folder |

All groups use the original fixed seeds 242000-242019. One selected checkpoint
per method and scenario is reused across the three deadline regimes.

## Interpretation

- Energy is plotted in J. Source kJ values are multiplied by 1000.
- A: bar heights are means; error bars are sample standard deviations.
- C: boxes show the median and IQR, with 1.5-IQR whiskers and visible outliers.
- ESP@90 is the fraction of test seeds with at least 90% timely workflows.
- Reliability attainment is 100% for every group.
- Small scenarios share one energy-axis range; Medium scenarios share another.
- Four scenarios correspond to separately trained resource/workflow settings.
- The adjacent plotted_summary.csv contains all displayed aggregate values.

## Outputs

- option_A_grouped_bars: complete mean-energy and ESP bar chart.
- option_C_boxplots_matrix: complete energy distributions and ESP matrices.
- PNG for preview; PDF and SVG for vector publication output.

To regenerate from the CSV files:

    & 'D:\Miniconda\envs\py310\python.exe' plot/plot_main_style_options.py --all
