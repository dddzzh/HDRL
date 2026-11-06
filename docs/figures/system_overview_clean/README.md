# Redesigned system overview

Independent vector composition for the current paper, not a recolored copy of the earlier network diagram. All device, graph, server, cloud and controller geometry is drawn afresh. The original reference PDF and the rejected layout remain unchanged.

Data path: devices as workflow-source context -> online DAGs with deadline/reliability requirements -> common smallest-workload ready dispatcher -> upper D3QN server decision -> lower SAC continuous frequency decision -> heterogeneous edge/cloud execution. Solid paths represent the online task/action or dependency-data flow, while the dashed return summarizes state and execution feedback. The three colored Q symbols denote independent estimators, not a shared upper backbone. The lower controller icon is not an extra network or safety shield.

The device glyph does not claim modeled wireless scheduling, upload/download energy or mobility. DAG dependencies generate modeled inter-server data transfers. Server icons are schematic; they do not specify the experimental resource counts. The figure complements, rather than duplicates, the detailed method/network diagram.

The six-color palette is the same as the current method framework. All PNG/SVG/PDF outputs use the same source geometry, with embedded vector fonts in the PDF. A single-page, zero-raster-image PDF, current implementation flags, palette membership and text bounds/overlap are checked. The rendered PDF has been visually inspected.

- PDF: `output/pdf/system_overview.pdf`
- PNG/SVG: this directory, `system_overview.png` and `system_overview.svg`
- Rebuild: `python tools/draw_system_overview_clean.py`

Suggested caption: **System overview of dynamic edge-cloud workflow scheduling.** Arriving DAG workflows share heterogeneous edge and cloud servers. A common ready-task dispatcher feeds the hierarchical controller, where the upper D3QN selects an execution server and the lower SAC determines its DVFS ratio. The execution environment returns resource-state and workflow feedback. Inter-server communication carries data associated with workflow dependencies. Device icons illustrate application sources rather than an optimized wireless-access layer.
