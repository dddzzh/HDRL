# Main experiment: three style options

These previews use the same S26 and M37 main-experiment data. Choose a style
before extending the layout to S37 and M53.

| Option | Presentation | Reference inspiration |
|---|---|---|
| A | Grouped energy bars with sample-SD error bars; grouped ESP bars | DReaM Fig. 9; HiRL Fig. 3 |
| B | Energy means with sample-SD error bars; marked ESP lines | Fault-Tolerant Aware Task Offloading, Figs. 9-10 and 13-14 |
| C | Grouped energy boxplots; annotated ESP matrix | Multi-Agent Hierarchical workflow paper Fig. 4; fault-aware paper Fig. 11; HiRL Fig. 4 |

## Data and units

- Inputs: ../formal_comparison_selected_columns.csv and
  ../formal_comparison_figure_summary.csv.
- S26 = Small workflows, 20 edge + 6 cloud servers.
- M37 = Medium workflows, 28 edge + 9 cloud servers.
- All three deadline regimes use the same selected model per method/scenario.
- Each cell contains 20 fixed test seeds, 242000-242019.
- Energy values in the source CSV are kJ; plotted values are J (multiply by 1000).
- The energy axis uses scientific notation; x10^5 applies to every tick.
- ESP@90 is the proportion of test seeds whose within-seed deadline success
  rate reaches 90%. It is not mean workflow deadline success.
- All reliability attainment values are 100%, so reliability is not repeated
  in the style previews.
- A/B error bars show sample standard deviation across test seeds, not a
  confidence interval across independent training runs.
- C boxes show the median and interquartile range. Whiskers use 1.5 IQR;
  observations beyond them remain visible as outliers.
- B uses small horizontal categorical offsets so coincident ESP markers remain
  identifiable; the values themselves are unchanged.

The generator checks all 480 seed-level rows and all 24 aggregate cells before
exporting PNG, PDF and SVG:

    & 'D:\Miniconda\envs\py310\python.exe' plot/plot_main_style_options.py

Figures borrow presentation conventions, not artwork, from the four reference
papers in paper/. The four-method palette and panel organization are shared
across these options.
