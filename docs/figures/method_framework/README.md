# D3QN + SAC method framework

## Files

- Publication vector PDF: [d3qn_sac_framework.pdf](../../../output/pdf/d3qn_sac_framework.pdf).
- Editable vector SVG: [d3qn_sac_framework.svg](d3qn_sac_framework.svg).
- High-resolution PNG: [d3qn_sac_framework.png](d3qn_sac_framework.png).
- Actual PDF rendering: [PDF preview](d3qn_sac_framework_pdf_preview.png).
- Reproducible drawing: `tools/draw_method_framework.py`.

The figure is 7.4 x 5.55 inches. Use the full double-column text width. Times New Roman text, STIX mathematics, embedded PDF fonts and no raster images in the PDF. The detailed three-column layout uses exactly the reference palette: #E0EAF6, #A8C0E4, #BBDDB2, #C7C5C6, #F5AA83 and #BAE0E3, with neutral strokes and white backgrounds. Blue identifies upper learning, green lower learning, cyan execution/target critics, and orange constraint adaptation/losses. Solid and dashed lines remain distinguishable in grayscale. The previous text-heavy PDF, SVG and PNG are retained in previous_text_layout/, and the simpler icon version is retained in previous_icon_layout/.

## Paper insertion

Copy the PDF to the manuscript project, retaining its filename.

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{d3qn_sac_framework.pdf}
    \caption{Overview of the proposed hierarchical scheduling framework.
    The upper D3QN selects a server through three independent dueling
    estimators, while the lower SAC determines its operating frequency.
    Upper-level learning uses separate target estimators and Double-Q
    targets; lower-level learning uses a stochastic actor and twin
    soft-Q critics. Workflow feedback and fixed-schedule full-frequency
    counterfactuals provide credit assignment and constraint adaptation.
    Solid arrows denote online data and actions; dashed arrows denote
    replay feedback and learning updates.}
    \label{fig:framework}
\end{figure*}
```

## Current code alignment

Version anchor: `tools/train_coupled_hierarchical_d3qn_sac.py` with the current `configs/v3_sm_74_legacy_train.json` and `configs/v3_sl_111_legacy_train.json`. Their active class is `ConfiguredLagrangianReliabilityAgent`. Resource counts and workflow-generation protocols are experimental settings, not separate algorithm modules.

| Figure element | Implementation evidence |
|---|---|
| Server selection followed by frequency control | `collect_d3qn_episode` in `tools/train_hierarchical_d3qn.py` |
| Three independent dueling estimators; only deadline Q clipped nonnegative | `src/D3QN/reliability_aware_high.py`, `src/D3QN/dual_head_constrained.py` |
| Critical-path budget and global urgency in observations | `src/D3QN/state.py`, `src/ppo/hierarchical_state.py`, `src/simulator/edge_cloud_env.py` |
| SAC actor and twin ordinary soft-Q critics | `src/sac/hierarchical_low_sac.py` |
| Energy-efficient execution floor | `simulation.dvfs_action_floor=energy_efficient`; environment/resource action processing |
| Dense + workflow miss + service-failure credit | `dense_miss_plus_episode_failure_deadline_costs` |
| Fixed-schedule counterfactual and queue externalities | `_workflow_counterfactual_low_deadline_credit` in `tools/train_hierarchical_d3qn.py` |
| Distinct deadline and reliability multiplier updates | High/low deadline controllers, reliability-aware high agent and low reliability controller |

The diagram does not claim a shared upper backbone, lower constraint-specific multi-critics, a hard deadline shield, workflow-wise replay chains, backup replication, or an extra coordination network. The execution floor is an energy-efficiency boundary, not a deadline/reliability guarantee. Actor proposals and executed frequencies are distinguished. Reliability multipliers do not share the deadline PI rule.

The lower actor and both critics learn from the same composed SAC reward, with energy/time shaping and attributed constraint costs. Replay, target-network updates and dual adaptation are summarized at module level; not every optimizer edge is drawn. Hyperparameters, logs and checkpoint screening belong outside the method overview.

## Verification

The drawing checks active config flags, learner choices, credit mode and the frequency floor before exporting. It also checks the current code for the dueling aggregation, independent reliability target, Huber loss, actor and target twin critics. Exact palette membership, text bounds and pairwise text overlap are asserted. The PDF is one vector-only page, checked with pypdf and visually reviewed after Poppler rendering. Icons summarize modules rather than specifying network widths or layer counts; the three upper networks are independent, and the lower learner has one actor and two ordinary soft-Q critics. The displayed Bellman target is schematic: arguments and minibatch averaging are omitted, and d_t denotes transition termination. The target-network copies use soft updates; the actor has no target copy. No training/evaluation program, model, config or experiment result was changed.
