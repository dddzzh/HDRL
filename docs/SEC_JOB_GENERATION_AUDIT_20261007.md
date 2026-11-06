# Sec-job generation and training audit

## Scope and provenance

Source archive: `C:/Users/Administrator/Desktop/sec_job - 原版_未修改第三个工作_无对比算法.zip`.
Read-only source extraction: `tmp/sec_job_reference_20261007` (Python only; no bundled virtual environment or checkpoints executed).
15 DAG templates were imported separately into `data/sec_job_reference`; existing `data` templates were not replaced.

The user approved migrating the sampling/deadline protocol while retaining DRLEC's cycles/Hz, DAG-edge communication, energy and reliability models. This is an adaptation, not an exact physical-model reproduction of the source project.

## What the supplied source actually does

| Component | Supplied active scripts | Prior DRLEC | New opt-in profile |
|---|---|---|---|
| Workflows per episode | 50 | 20 | 50 |
| Arrival process | NumPy `default_rng(seed)`, cumulative exponential gaps; first arrival positive | Python RNG; first arrival at zero | Source arrival RNG and convention |
| Arrival rate | 0.03/s, mean interval 33.33 s | 0.10/s, mean interval 10 s | 0.03/s |
| DAG sampling | Independent `RandomState(seed)` | Shared stream with arrivals/payload/deadline sampling | Independent `RandomState(seed)` |
| Small templates | CyberShake30, Epigenomics24, Ligo30, Montage25, Sipht29 | Original DRLEC five templates | Exact imported DAGs |
| Medium mixture | 30-size group 30%, 50-size group 70% | Uniform over first ten templates | Source 3:7 mixture |
| Large mixture | 30/50/100-size groups 20%/30%/50% | Uniform over first fifteen templates | Source 2:3:5 mixture |
| Payload seed | Python RNG with `episode_seed + workflow_index` in active environment | One shared Python stream | Per-workflow seed; DRLEC physical ranges retained |
| Deadline factor | Independent local seed: `(seed*1000003 + index*9176 + 20260316) & 0xffffffff`; 2x/3x mixtures | Shared Python RNG, same mixtures | Source independent local seed and probabilities |
| Deadline reference | Active trainers load `deadline_cache_fcfs_vmfirst_...json`, not HEFT cache | Isolated full-frequency HEFT | Full-frequency shared-batch FCFS/first-idle simulation on actual DRLEC resources |

Source entry points checked:

- `a_set/read_xml_opt_Tsize.py`: actual payload defaults are 60–90 MI, 1–3 input/output files, 512–1024 MiB per file.
- `.../env_cloud_workflow_d3qn_state_reworked_multiagent_Tsize_alpha15_DDLCACHE_FCFS.py`: independent deadline RNG, per-seed cache lookup, actual payload seed, reset of template RNG.
- `.../train_hrl_tri_003_alpha15_smallTask_medRes_seed5_ddlFCFS_alpha075_dalphaMix_Medium.py`: actual arrival/workflow/training/evaluation parameters.
- `c_run/.../run_hrl_routeA_medTask_medRes_...Medium.py` and corresponding `largeTask`: mixture weights and final evaluation seeds.
- `b_FCFS_FIFS/...FSFS.py`: FCFS by ready time, randomized ties, earliest-available idle VM with randomized ties.

### Important source caveats

1. Both HEFT and FCFS caches exist in the archive, but the inspected nine training scripts load the FCFS caches. File names and old docstrings are not reliable enough to determine the active algorithm.
2. The FCFS cache metadata labels its policy `FCFS_task + VM_first_idle`. The named precompute script is absent from this archive. Its exact cache-construction trajectory cannot be independently verified from the supplied files alone. The adaptation explicitly defines and recomputes a shared-load FCFS reference; it does not import cache values from a different resource platform.
3. The separate episode-manifest generator declares 2000–3000 MI and different payload/template seed rules, whereas the active environment calls the parser with its 60–90 MI defaults and seed `episode_seed + index`. The new profile follows the active path, not this inconsistent manifest file; its numeric physical workloads remain DRLEC's own.
4. Cache template mismatches only print a warning in the supplied environment; they do not raise, even in strict mode. The adaptation does not use such index-only caches. Its bounded in-memory cache key includes arrivals, complete task/DAG/payload data, server capabilities, link bandwidths and reference seed.
5. The original computes external input/output transmission as well as parent data and uses a host-utilization power model. Those were not migrated: the user chose to preserve DRLEC's physical model and scheduling boundary.

## Changes in this workspace

- `src/datasets/sec_job_workflow_generator.py`: separate protocol, weighted DAG sampling, independent RNGs, event-driven shared FCFS reference with a bounded cache.
- `src/datasets/workflow_factory.py`: initially opt-in; now the current Pegasus default is `sec_job_fcfs_v1`, and retired explicit profiles are rejected.
- `configs/v3_small_28edge_9cloud_sec_job_fcfs.json`: Small + 28 edge / 9 cloud + Medium deadline; otherwise inherits current future-ESP algorithm/objective/constraints.
- `tests/test_sec_job_generation.py`: analytical shared-queue example, chain example, repeatability, deadline identities and arrival/payload-stream isolation.
- Frozen-checkpoint diagnostic: `results/generalization_diagnostics/S37_sec_job_protocol_20261007`.

The new reference dispatcher is a **deadline-construction baseline only**. It does not replace the learned policy's common task-dispatch rule (`smallest_workload`). No state dimension, network, SAC output range, reward or lambda target was changed.

## Training and evaluation speed

The supplied trainer runs 600 episodes and evaluates seed `[1]` after every episode. It saves the best model solely by this seed's energy; it does not apply the current two-block ESP/reliability gate. Its Host/VM learners update after individual assignments and store zero next states with `done=1`, while the Manager has phase-level transitions. That is a different credit-learning design, not a drop-in performance optimization for D3QN+SAC.

Its final evaluation scripts use 30 deterministic seeds, `2..31`. In this supplied version those seed values overlap the training episode seeds `1..600`. These scripts alone therefore do not establish unseen-seed generalization; do not copy that overlap into the new protocol.

The new config uses evaluations every 50 episodes, two blocks of five seeds each, with periodic checkpoints every 50 episodes. For 1200 episodes this gives 24*10=240 screening rollouts, versus 60*40=2400 previously: **90% fewer screening rollouts** (not a measured 10x end-to-end speedup). Each new rollout has 50 workflows rather than 20, so wall-time savings are smaller.

Five-seed blocks are only preliminary screening. Re-rank periodic candidate checkpoints with the original two blocks of 20 seeds, then evaluate the selected checkpoint on a separate untouched test block. A 5/5 screening pass is not a formal acceptance result. No production trainer or replay-update rule was changed.

Training command (not launched):

```powershell
& D:\Miniconda\envs\py310\python.exe tools\train_coupled_hierarchical_d3qn_sac.py --config configs\v3_small_28edge_9cloud_sec_job_fcfs.json
```

## How to interpret the diagnostic

The frozen Small-trained S37 ep1060 checkpoint is evaluated on Large workloads under the new protocol, on seeds 242006, 242009 and 242010, along with queue-aware EFT/full frequency. These are old-protocol known failures, so recovery is a diagnostic only, not an unbiased ESP estimate.

Several environment components changed together, including workflow count and DAG mixture. Any gain demonstrates behavior in a different environment, **not a trained algorithm improvement**. Formal comparisons require all methods to share the new generator and to be trained/evaluated consistently; old and new tables must remain separate.

With 50 workflows, an ESP@90 episode passes at 45/50 or better (at most five misses). The old 20-workflow episode passed at 18/20 (at most two misses). Even at the same underlying per-workflow success probability, changing the episode size can change ESP. The three deliberately selected failure seeds are not a representative test block.

## Completed diagnostic

All six rollouts finished: the frozen S37 model and queue-aware EFT/full frequency each achieved 50/50 on-time workflows on each of the three selected seeds, with reliability satisfaction 100%. The same-workload FCFS/isolated-HEFT reference ratio had median 1.6794. Holding the new-model trajectories fixed and applying hypothetical isolated-HEFT deadlines gives only 29/50, 34/50 and 24/50 for D3QN+SAC; this is an offline threshold check, not a rerun with those deadlines. Recovery therefore cannot be called an algorithmic generalization improvement.

Full raw data and interpretation: `results/generalization_diagnostics/S37_sec_job_protocol_20261007/REPORT.md`. No new training was started.

## Current experiment transition (2026-10-07)

At the user's request, the old Pegasus sampling/isolated-HEFT generation body has been removed from the active entry. Direct `PegasusWorkflowGenerator.generate` calls now route to the same sec-job generator; its constructor defaults are 50 workflows, `data/sec_job_reference`, `sec_job_exponential`, and 0.03/s. Explicit old arrival settings fail instead of silently running the retired protocol. XML parsing and physical workload ranges are retained. A recoverable copy of the previous generator/factory is under `old/generation_pre_sec_job_20261007`; no result or checkpoint was deleted. The isolated-HEFT helper remains available for custom/Alibaba workflows and HEFT baselines/tests, but the three new Pegasus training configs do not call it for deadline construction.

Current configs are `configs/v3_ss_37.json`, `configs/v3_sm_74.json`, and `configs/v3_sl_111.json`. The earlier single diagnostic config remains solely for reproducing the frozen-model diagnostic. See `docs/V3_GENERALIZATION_PROTOCOL_20261007.md` for the full training/evaluation matrix and commands. This changes the experiment, not the D3QN/SAC algorithm. No full training has been started.

Subsequent user-requested evaluation change: these three training configs now evaluate every episode on exactly one fixed seed, 542000 (`eval_interval=1`, `eval_episodes=1`, no secondary block). Periodic saving stays every 50 episodes; constrained best-checkpoint selection stays enabled. The earlier diagnostic config still retains its original two-block/five-seed settings. Formal multi-seed confirmation and independent testing remain separate from this single-instance screen.

## Complete six-case Medium-deadline diagnostic

The same frozen S37 ep1060 checkpoint was then evaluated on Small/Medium/Large workloads crossed with 26/37 servers, five consecutive seeds 242000–242004 in every case. All 30 rollouts completed (1500 workflows), without filtering seeds or changing the checkpoint. ESP@90: S26 100%, S37 100%, M26 20%, M37 100%, L26 0%, L37 100%; reliability satisfaction was 100% in all cases. The network has only 37 server slots, so S53/M53/L53 were not run, not resized, and not scored as empirical failures.

Report and exact per-seed/workflow CSVs: `results/generalization_diagnostics/S37_sec_job_medium_matrix_5seed_20261007/REPORT.md`. This confirms that the earlier three-seed success on L37 must not be generalized to all workload/resource transfers. Entry point: `tools/eval_sec_job_generalization.py`, reusing the production reliability-aware checkpoint loader and deterministic rollout collector. No new training was started.
