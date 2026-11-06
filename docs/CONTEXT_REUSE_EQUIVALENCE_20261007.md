# Decision-local context reuse: numerical equivalence

Date: 2026-10-07. Scope: only reuse the current task's mean full-frequency execution time across candidate servers in one decision. No queue/communication/available-time cache persists across decisions. The summation order, frequency values and division are unchanged; standalone safety-metric calls keep the uncached fallback. No network dimensions, reward, Deadline generation, frequency bounds, dual update or training config changed.

## Original-source comparison

Before-edit source snapshot: `tmp/context_reuse_equivalence_20261007/edge_cloud_env_before.py`.
SHA256: `566afdeb166e781713f3a49c880d2b6fa5bda72f8769f0878450388fe0a76885`.

12 paired complete trajectories: 37/74/111 servers × future safety on/off × seeds 242000/242001, 654 paired task decisions, with identical CPU deterministic three-head D3QN+SAC weights. Each trajectory contains two generated workflows; arrivals were deliberately compressed and relative Deadline tightened to exercise queues and Deadline pressure. These are diagnostic cases, not formal evaluation data, and no training was performed.

All environment states, decision-context fields/masks, encoded observations, upper/lower decision outputs, actions, step rewards, event logs and final per-workflow results were compared by exact equality (arrays element-wise; no relaxed tolerance). Zero numerical differences. Additional unit check changes a server frequency between context calls to ensure reuse does not retain stale values.

45 related tests passed, including time, generation, configuration, checkpoint-selection and reference-model tests. This supports an arithmetic-preserving implementation change; it does not claim a different learned policy or ESP improvement, nor exhaustive coverage of all possible inputs.

## Local timing, not full-training speed

100 repeated contexts, minimum of 3 measurements; the user's three training processes remained running. These contexts are from the same post-trajectory states for both implementations. Timing is exploratory and not a controlled end-to-end benchmark.

| Servers | Before (s) | After (s) | Local time reduction |
|---:|---:|---:|---:|
| 37 | 0.3642 | 0.3321 | 8.8% |
| 74 | 1.0879 | 0.9417 | 13.4% |
| 111 | 2.0196 | 1.7533 | 13.2% |

No process was stopped. Existing Python processes retain the already imported pre-edit class; this change applies when a process is next launched. Existing checkpoints and training outputs remain compatible. Do not discard current results merely to activate this optimization.

Recheck the permanent tests:

```powershell
& D:\Miniconda\envs\py310\python.exe -m unittest tests.test_context_reuse tests.test_time_model -q
```

Reproduce the original-source trajectory comparison:

```powershell
& D:\Miniconda\envs\py310\python.exe tmp/context_reuse_equivalence_20261007/verify.py
```

Raw comparisons/timing: `tmp/context_reuse_equivalence_20261007/results.json`.
