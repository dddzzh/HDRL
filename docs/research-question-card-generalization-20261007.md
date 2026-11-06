# Research Question Card: Small-only workload-scale transfer

## Current decision, clarified 2026-10-07

The user reports that the senior trained nine independent models: Small
workflows x three resource scales x three deadline regimes. Each frozen model
tests Small/Medium/Large workflows while keeping its resources and deadline
regime fixed, giving 27 conditions. This is user-reported clarification, not
independent verification of the paper's checkpoint protocol. Larger templates
must stay outside training. A variable-server-count shared network is NOT needed
for this protocol. The proposal below the historical heading is superseded.

Question: can Small-only load diversity, followed if needed by normalized
remaining-work state features, improve zero-shot workload-size transfer?

Evidence: existing `src/D3QN/state.py` already encodes queues, EFT, critical-path
safety, separate busy/wait ratios and global critical-path pressure. Global
ready-task counts use the full queue. The global pressure in
`src/simulator/edge_cloud_env.py` aggregates critical paths, not total unfinished
cycle demand. Missing load/width information is a hypothesis, not a proven cause.

The reports in `results/generalization_diagnostics/L37_failed_seeds_20261007/`
and `L37_failed_seeds_eft_rho1_20261007/` show that full frequency alone does not
rescue most selected failures, while queue-aware EFT rescues some. These tests
use an M37 model, NOT a Small-trained model, and selected failures are NOT an
unbiased ESP estimate or an infeasibility proof.

Minimal pilot: use Small/37 servers/Medium deadline. First vary arrival intensity
episode-wise with Small templates only, retaining 20 workflows, reward/dual/SAC
definitions and task ordering. Choose the range on Small-only feasibility and
validation. If insufficient, add current-workflow remaining-cycle demand and
global released unfinished-cycle demand, normalized by remaining deadlines and
aggregate full-frequency capacity. Do not use future unreleased arrivals. New
state dimensions require retraining; preserve current manuscript checkpoints.

Selection: prespecified Small nominal/heavy validation strata; freeze checkpoint
before Medium/Large tests. Compare original, load diversity only, and diversity
plus state under matched interaction/update budgets, then replicate a promising
pilot before all nine models. Report nominal Small performance as well as transfer
ESP@90, reliability attainment and energy. No guarantee of 100% ESP.

No training code or configuration has been changed.

## Historical proposal — superseded, not the current experiment

Date: 2026-10-07. This is an analysis note, not an implemented method or a change to the current manuscript/results.

## Question

Can one constraint-aware D3QN+SAC policy, trained on nine workload/resource combinations under Medium deadlines, retain deadline ESP and reliability attainment across 27 combinations of workload pool, resources and deadline distribution?

Type: applied / confirmatory.

## Hypotheses

- Server-wise parameter sharing and masked set aggregation remove the current parameter-shape dependence on server count and enforce consistency under candidate-server permutation. They do not by themselves guarantee successful scale transfer.
- Explicitly balanced scenario sampling and sufficient large-template coverage reduce training-distribution gaps.
- Per-configuration constraint monitoring is needed because a low pooled failure rate can hide an unsuccessful configuration.

## Why it matters

The current four configurations are trained separately. Their deadline-distribution robustness is not evidence that one checkpoint works across arbitrary workflow or resource scales. A scale-independent interface would permit a stronger, clearly testable deployment claim.

## Evidence records

### ER-20261007-reference-grid-01

Source: `paper/Multi-Agent Hierarchical Reinforcement Learning for Dynamic Energy-Aware Cloud Workflow Scheduling with Deadline Constraints.pdf`, pp. 8–9, Tables II–IV and Section V-A.

Source type: full paper.

Supports: nine workload/resource configurations, three deadline distributions, 27 evaluation conditions and 30 evaluation runs. Its workload pools are cumulative S, S+M, S+M+L. Its reported arrival rate is 0.03 and realization size is 50 workflows.

Limitation: these pages do not explicitly establish a single unified checkpoint across all scales or specify that nine Medium-only training runs produce the 27 results. Do not attribute either training protocol to the paper without additional confirmation.

Claim strength: supported for the scenario grid; unknown for checkpoint-sharing protocol.

### ER-20261007-code-shape-02

Source: `src/D3QN/state.py`, `src/D3QN/D3QN.py`, `tools/hierarchical_common.py`, and `src/D3QN/hierarchical_d3qn_agent.py`.

Source type: project code.

Supports: state dimension is 12+13N and action dimension is N; N is set from the actual resource pool. The encoder concatenates server slots and the dense advantage output has a separate output per slot. A trained network therefore cannot directly load unchanged for a different N, and there is no enforced server-permutation equivariance.

Limitation: this explains interface incompatibility, not every deadline failure on the same resource pool.

Claim strength: supported.

### ER-20261007-load-coverage-03

Source: `src/datasets/pegasus_workflow_generator.py`, job counts parsed from the 15 XML templates, and resources generated from the v3 resource table with seed 42.

Source type: project code / read-only numerical check.

Supports: cumulative pool means are 27.6, 39.3 and 59.333 tasks. With common task workload and arrival rate, Large demand is 50.98% above Medium demand. Full-frequency aggregate capacities for 26/37/53 servers are 68.15/96.87/138.81 Gcycles/s. Uniform nine-scenario sampling implies task-size-group workflow fractions of approximately 61.1%/27.8%/11.1% for small/middle/large templates.

Limitation: mean offered load does not establish finite-realization deadline feasibility. It omits communication, precedence, heterogeneity, placement hotspots and bursts.

Claim strength: supported for arithmetic; speculative for its contribution to measured transfer failure.

## Proposed experimental definition

- Workflow pools: existing cumulative small / middle / large.
- Resources: existing 20+6 / 28+9 / 40+13 v3 profiles.
- Nine base configurations: SS, SM, SL, MS, MM, ML, LS, LM, LL.
- Train only Medium deadlines; test all three distributions with frozen policies and dual multipliers.
- Option A: nine separately trained checkpoints, one per base configuration. This yields 27 conditions but only deadline-distribution generalization within each configuration.
- Option B, next-version research target: one set-based shared policy trained on a balanced mixture of the nine configurations. This demonstrates one-policy coverage and deadline-distribution transfer; it still does not demonstrate transfer to workload/resource ranges completely absent from training.
- In either protocol, the nine Medium conditions are in-distribution tests on new workload realizations; only the other eighteen conditions change deadline distributions. Do not call all 27 unseen distributions.

## Missing evidence

1. Matched large-pool full-frequency scheduling diagnostics and failed-seed traces; a failed HEFT reference is not an infeasibility proof.
2. Controlled comparisons separating architecture change from training-distribution change.
3. Repeated independent training runs and uniformly defined validation selection.
4. For stronger zero-shot claims: a workload/resource range withheld from training and a prespecified holdout test.

## Support criteria

One unchanged checkpoint and preprocessing pipeline operate correctly across all nine resource/workload combinations; per-configuration ESP@90 meets a prespecified target and reliability attainment is reported separately. Energy is ranked only within the feasible set. Test seeds and scenario definitions are fixed before examining results.

## Falsification criteria

If the new representation resolves dimension/permutation tests but does not improve held-out workload ESP under a matched training budget, do not claim representation alone fixes generalization. If pooled ESP improves while an individual configuration remains poor, do not claim uniform feasibility. If large/small-resource conditions fail even strong full-frequency diagnostics, examine workload and physical feasibility before increasing penalty strength.

## Minimal next action

Keep the current paper version frozen. First implement/test a candidate-wise shared encoder and masked scoring interface in a separate version, retaining three upper estimators, SAC and existing credit. Before all nine configurations, pilot Small+Medium+Large workflow sampling on the existing 37-server pool to separate workload coverage from resource-count transfer. Any implementation requires a subsequent user instruction.

Decision: explore; no code or training configuration modified in this analysis.
