# Experimental figure assets

These figures use the formal configurations stated in the manuscript: 20 edge + 6 cloud servers for the small setting and 28 edge + 9 cloud servers for the medium setting. Every method is evaluated on the same 20 workflow seeds (`242000`--`242019`) for tight, medium, and loose deadlines.

## Source files

- `results/comparison_small_regimes_242000_20/20260916_future_esp_ep1140/comparison_rows.csv`
- `results/comparison_small_regimes_242000_20/dqn_edge_20260917_092338/comparison_rows.csv`
- `results/comparison_small_regimes_242000_20/dream_dqn_20260918_090425/comparison_rows.csv`
- `results/comparison_small_regimes_242000_20/oo_actor_critic_20260917_092358/comparison_rows.csv`
- `results/comparison_medium_regimes_242000_20/20260917_ours_dqn_oo_final/comparison_rows.csv`
- `results/comparison_medium_regimes_242000_20/dream_dqn_20260917_100220/comparison_rows.csv`

The ablation figure uses `results/ablation_results/small_main_ablation/ablation_summary.csv`.

## Metric definitions

- **Total energy:** sum of computation energy over the 20 workflows in one test seed; figures report kJ.
- **Deadline ESP@90:** fraction of test seeds whose workflow deadline success rate is at least 0.90.
- **Reliability pass:** a seed passes when its reliability success rate is 1.00.
- **Joint feasibility:** both the Deadline ESP criterion and the reliability criterion hold.
- **95% CI:** Student-t interval for mean energy with 19 degrees of freedom.
- **Paired saving CI:** percentile bootstrap CI of the median seed-wise energy-saving ratio (10,000 resamples).

## Figure files

- `fig_energy_distributions`: 2-by-3 scenario matrix of 20-seed energy boxplots with every raw test point.
- `fig_energy_distribution_<scale>_<deadline>`: six self-contained scenario figures in the same style.
- Filled boxes meet joint feasibility; hatched boxes have Deadline ESP@90 below 0.90.

Every figure is exported as PDF, SVG, and 300-dpi PNG.
