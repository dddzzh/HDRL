# Edge-cloud system overview for the current D3QN+SAC method

This system overview adapts the author's `edge_cloud_network_v4.pdf`; the source PDF is not overwritten. Its embedded cloud, edge-server, radio-tower and device SVG paths are reused as vectors and recolored with the current six-color framework palette. The figure is separate from the detailed learning/network framework.

- Workflow inputs: dynamic DAGs, arrival time, workflow deadline and reliability requirement.
- Shared task dispatch: smallest-workload-first ready queue, not an additional learned ordering agent.
- Upper D3QN chooses the execution server; lower SAC chooses continuous frequency. The old QHH, UASA and MG-DVFS modules and three-action output are removed.
- Solid inter-tier arrows denote dependency-data transfer. The scheduler's execution/control and dashed feedback path are separate.
- Device/radio icons depict application context. Device uplink/downlink timing and energy are outside the current formal scheduling boundary; the modeled communication term concerns inter-server DAG dependencies. The figure does not claim wireless-channel optimization, backup replication, or hard deadline/reliability guarantees.
- The illustrated server count is schematic, not the 37/74/111 experimental inventory. Symbols follow the current system-model manuscript: edge/cloud server sets M^e/M^c, DAG task/edge sets N_k/E_k, arrival a_k, deadline D_k, reliability requirement Theta_k and actions (m_i,rho_i).

PDF: `output/pdf/edge_cloud_system_d3qn_sac.pdf`. Editable SVG and high-resolution PNG are in this folder. Rebuild: `python tools/draw_edge_cloud_system.py`. Source hash and palette are in provenance.json. No model, training, evaluator, existing result or manuscript source was modified.

Suggested caption: **Edge-cloud workflow scheduling system.** Dynamic DAG workflows are dispatched to heterogeneous edge and cloud servers. A shared ready-task queue supplies tasks to the hierarchical controller, whose upper D3QN selects an execution server and lower SAC determines its operating frequency. Workflow execution provides energy, deadline and reliability feedback. Solid inter-server links indicate dependency-data transfer; the dashed controller link indicates execution feedback. End devices illustrate workflow sources rather than separately modeled wireless scheduling actions.

The diagram uses reusable module icons, not literal neural-layer counts. Place this figure in the system-model section; retain the separate detailed method framework in the algorithm section. The PDF is a single vector-only page and has been visually reviewed after Poppler rendering. Text overlap/bounds, palette, current configuration flags, removal of old method labels, and preservation of the source PDF hash are checked.
