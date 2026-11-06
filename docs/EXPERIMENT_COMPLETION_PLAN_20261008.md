# Minimal experiment-completion plan (2026-10-08)

## Research Question Card

- Question: Can Small-trained placement/DVFS policies satisfy per-seed deadline service levels on unseen Medium workloads while saving energy under a consistent physical and workload-generation model?
- Type: exploratory; scope refinement follows observed Large failures.
- Hypothesis: within fixed resource counts, transfer to Medium is more reliable than transfer to Large; SAC saves energy without losing the declared service level.
- Why it matters: distinguish useful workload transfer from an unsupported claim of arbitrary-scale generalization.
- Current evidence: frozen SS ep500, 37 servers, 20 seeds, Small–Medium ESP@90=100%, Medium–Tight=100%, Large–Tight=10%, Large–Loose=75%; reliability satisfaction=100% in each condition.
- Missing evidence: other deadline regimes, completed SM/SL models, aligned baselines, and independent training repetitions.
- Support: consistent Medium performance across the resource-specific models and deadline regimes, with lower energy than service-level-feasible competitors under a matched protocol.
- Falsification: failures on the other Medium conditions or loss of energy advantage under matched, independently selected comparisons. Do not omit such conditions.
- Minimal next action: finish current trainers; freeze model selection before evaluating the remaining matrix. Large–Loose is complete and confirms a remaining transfer limitation, not universal deadline feasibility.
- Decision: evaluate existing models before adding algorithm modules or starting extra training.

## Evidence and claim limits

`ER-20261008-ss500-transfer-01`: experiment artifact, [report and raw-data links](../results/v3_ss_ep500_probe_20261008/REPORT.md). Observed evidence from one intermediate checkpoint and one diagnostic seed block. Allowed: “SS ep500 passed Medium–Tight on these 20 seeds.” Not allowed: “all Medium conditions generalize,” “Large is physically infeasible,” or “all final models will pass.”

`ER-20261008-baseline-protocol-02`: project configuration evidence. Current `dqn_edge_small_28edge_9cloud.json`, `dream_dqn_small_28edge_9cloud.json`, and `oo_actor_critic_small_28edge_9cloud.json` specify 20 workflows and arrival rate 0.1; `v3_ss_37.json` specifies 50 workflows, arrival rate 0.03, and `sec_job_fcfs_v1`. This demonstrates a configuration mismatch, not the provenance of every historical result. Verify each run's saved configuration before comparison.

## Prioritized experiment set

| Priority | Evidence | Minimum work | Claim answered |
|---|---|---|---|
| 1 | Main matrix: Small/Medium composition × 37/74/111 servers × Loose/Medium/Tight | Evaluate the three currently training models; each stays at its own server count | Deadline and workload transfer under fixed model/resource dimensions |
| 1 | Baseline comparison on the same matrix | First audit saved generator/resource/metric settings; align training and evaluation where mismatched | Energy advantage at a matched service level |
| 2 | Validation learning curves: ESP@90, energy, feasibility | Reuse training logs; label validation curves as validation | Learning stability and checkpoint selection |
| 2 | Full vs frozen Full with rho=1; queue-aware EFT/HEFT with rho=1 | Evaluation only, same seed block and resource pool | Effect of DVFS at fixed weights; non-learning placement reference |
| 2 | Structural ablations already planned | Reuse only genuinely protocol-matched training; otherwise retrain the affected variant | Necessity of learned placement and deadline handling |
| 3 | Large boundary/stress tests | Preserve the Tight failure and complete Loose; clearly list any untested regimes | Limits, not a claim of universal scalability |
| 3 | Timing and training-seed robustness | Measure inference separately from training/evaluation; add independent training seeds if budget permits | Computational overhead and reproducibility |

The main matrix has **18 conditions**, of which **3 are training-distribution conditions and 15 are transfer conditions**. Each policy is trained on Small composition + Medium deadline at its respective resource count. No 37-server checkpoint is silently loaded into a 74/111-server network.

## Rules to lock before final reporting

1. Report ESP@90, reliability satisfaction, and energy in J (mean ± sample SD). ESP@90 is the fraction of seeds with at least 45/50 workflows on time, including equality. Reliability satisfaction is not physical reliability=1.
2. Select checkpoints using only the declared training-distribution validation blocks and the same feasibility/fallback rule. Do not select them using transfer test outcomes.
3. Use all seeds in each predeclared block; no removal of observed failures. This diagnostic block has already been inspected, so subsequent tuning requires a fresh, predeclared confirmation block.
4. Compare energy as a constrained advantage only where methods meet the same declared service-level requirement. Keep infeasible methods' energy visible but do not call their constraint violations “energy savings.”
5. Old-generation September baselines and ablations cannot be pasted into the V3 main table. Re-evaluation of old policies is transfer evidence, not a replacement for matched baseline training.
6. Frozen-policy rho=1 and lambda=0 interventions are diagnostics, not substitutes for independently retrained structural ablations. EFT/HEFT is a reference, not a proven lower bound.
7. Twenty workload seeds measure test-workload variation, not training-seed variation. If reporting only one training seed, state that limitation. For paired energy tests retain per-seed rows, report effect sizes, and correct multiple comparisons where applicable.

Independent training variability is a documented concern in RL evaluation: [Henderson et al., Deep Reinforcement Learning That Matters](https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553). The recommendation here is to improve distinct evidence categories, not inflate the number of plots or hide failed transfers.
