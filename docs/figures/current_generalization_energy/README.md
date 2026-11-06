# Energy-only generalization boxplots: original complete data

This layout preview contains the complete **unfiltered** evaluation, not the
unfinished success-selected experiment. All failed-seed energies remain included.
The original evaluation data and running evaluation processes are untouched.

- Columns: 37 (28 edge + 9 cloud), 74 (56 + 18), 111 (84 + 27) servers.
- Rows: Small, Medium, Large workflows. Each rollout contains 50 workflows.
- Each panel: Loose / Medium / Tight deadlines and four fixed algorithms.
- Each box: 20 original seeds 742000-742019; 108 boxes and 2160 observations.
- Total energy is in J. The common multiplier is 10^6, not 10^5 from the older
  reference figure. Panels within a workload row share the same y-axis range.
- Boxes show median and interquartile range; whiskers extend to 1.5 IQR;
  outliers are shown, never removed. No success-rate plots are included.
- Palette and serif typography follow the supplied Option C reference.
- The revised wide layout uses abbreviated captions (a) SS through (i) LL
  below the panels, with the four-method legend at the very bottom.
- `figure_provenance.json` records all 12 raw source paths and SHA256 values.
- PNG is 600 dpi; SVG and PDF are vector figures. The PDF is in
  ../../../output/pdf/energy_boxplots_original_complete.pdf.

Energy alone does not establish deadline/reliability feasibility. This is a
layout trial; the paper should separately disclose the protocol and attainment
metrics. Success-selected data must not be presented as unfiltered evaluation.

Regenerate with the existing Python environment:

    python plot/plot_current_generalization_energy.py

## Refined layout (2026-10-14)

The current generator exports `energy_boxplots_refined_complete` (PNG/SVG and
PDF in output/pdf); the previous original-complete images remain untouched.
It retains exactly the same 2160 unfiltered energies and palette. Taller panels,
reduced blank space, and thinner median lines improve legibility. At the user's
request, the latest version is a pure boxplot: no sample dots or outlier markers.
All 20 observations per box remain in the quartile and whisker calculations;
none are removed or modified. Each panel has its own labelled linear y-axis range,
with common units J and multiplier 10^6. Compare numerical ticks, not vertical
positions across different panels. SS through LL and the bottom legend remain.
The provenance for this variant is `refined_figure_provenance.json`.

The latest pure-boxplot colors now match the method-framework source palette
exactly (opaque fills): D3QN+SAC #A8C0E4, DQN-Edge #F5AA83, DReaM-DQN #BBDDB2,
OOAC #BAE0E3. Neutral strokes use #303030 and subtle grids use #C7C5C6.

## Updated outcome-selected version

At the user's explicit request after the paired panel completed, the new result
is exported separately in `success_selected/energy_boxplots_success_selected`
(PNG/SVG), with its PDF under output/pdf. Use `--success-selected` to regenerate.
Its sources are the new 2160-row paired CSV export, not the original unfiltered
evaluation. The footer explicitly labels outcome-selected seeds. Layout, palette,
bottom legend and hidden point markers remain unchanged. Original figures above
are not overwritten. Full raw CSV, plot columns and 108-group summary are in
results/success_seed_screening_20261013/exports/.
