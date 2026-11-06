# Final Experimental Figure Design

## Selected style

The final result figures use only energy-distribution boxplots. Each boxplot is based on the same 20 paired workflow seeds (`242000`--`242019`) and overlays every raw test point. The six scenarios are:

- Small (20 edge, 6 cloud): tight, medium, and loose deadlines.
- Medium (28 edge, 9 cloud): tight, medium, and loose deadlines.

Each panel compares Ours, OOAC, DReaM-DQN, and DQN-Edge. Filled boxes satisfy the joint constraint criterion. A hatched box indicates that Deadline ESP@90 is below 0.90. Reliability success is 1.00 for every method and scenario in this formal comparison.

## Files

- `figures/fig_energy_distributions.pdf`: combined 2-by-3 paper figure.
- `figures/fig_energy_distribution_small_tight.pdf`
- `figures/fig_energy_distribution_small_medium.pdf`
- `figures/fig_energy_distribution_small_loose.pdf`
- `figures/fig_energy_distribution_medium_tight.pdf`
- `figures/fig_energy_distribution_medium_medium.pdf`
- `figures/fig_energy_distribution_medium_loose.pdf`

Every figure is also available as SVG and 300-dpi PNG.

## Manuscript-ready caption

> **Distributions of total computation energy under different resource scales and deadline regimes.** Each box shows the interquartile range over 20 paired test seeds, the central line denotes the median, whiskers extend to 1.5 times the interquartile range, and translucent points show all individual seeds. Filled boxes satisfy the joint Deadline and reliability criterion, whereas hatched boxes have Deadline ESP@90 below 0.90. All methods achieve a reliability success rate of 1.00 in every scenario.

## Data provenance

The source paths and metric definitions are recorded in `figures/README.md`. The selected raw columns and aggregated values are stored in `figures/formal_comparison_selected_columns.csv` and `figures/formal_comparison_figure_summary.csv`.

The plotting script is `plot/plot_paper_experiments_20260930.py`.
