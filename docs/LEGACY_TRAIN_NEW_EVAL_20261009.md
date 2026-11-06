# Legacy training / new-protocol evaluation

## Model identities

| Model | Training workflows | Resources | Source / training config |
|---|---|---|---|
| SS | Small | 28 edge + 9 cloud | `results/v2_small_28edge_9cloud_dvfs_v3_future_esp/20260924_160451/best_d3qn_checkpoint.pt`, ep1060 |
| SM | Small | 56 edge + 18 cloud | `configs/v3_sm_74_legacy_train.json` |
| SL | Small | 84 edge + 27 cloud | `configs/v3_sl_111_legacy_train.json` |

SS SHA256: `b9f717aec0bde883c4f74d4c2be88aa2906a14dd057fdd25c2002079aea3fb5e`.
Do not substitute the October SS ep1060: it is a different model.

## Training and selection (old protocol)

- Explicit generator profile: `cai_2025_heft`.
- Original root `data/` Pegasus templates, original cumulative Small/Middle/Large sampling and shared Python RNG sequence.
- 20 workflows per training and validation episode.
- System-wide exponential arrivals, rate **0.1 workflows/s**, mean interarrival **10 s**; first arrival at 0.
- Relative deadline: 2 or 3 times the individual workflow's full-frequency HEFT reference makespan, independent of competing workflows.
- Medium deadline during training: equal probabilities for the 2x and 3x multipliers.
- Resource types / physical model: `resources_edge_cloud_dvfs_v3.json`, unchanged.
- Algorithm/reward/dual settings copied from the verified old SS checkpoint configuration, not from the new sec-job SS run.
- 1200 training episodes, fresh initialization, no warm start.
- Every 20 episodes: two fixed validation blocks, `542000–542019` and `642000–642019`, on the **old** generator.
- Checkpoint feasibility: each block ESP@90 = 100% and reliability attainment = 100%; minimize energy among feasible candidates. If none is feasible, trainer fallback stays explicitly infeasible; do not describe it as a feasible model.
- Two evaluation workers per training job; no selection based on new-protocol test results.
- Outputs: `results/v3_sm_74_legacy_train_20wf/<timestamp>/` and `results/v3_sl_111_legacy_train_20wf/<timestamp>/`.

## Frozen-model evaluation (new protocol)

Use **evaluation-only** configurations:

- SM: `configs/v3_sm_74_new_eval_50wf.json`.
- SL: `configs/v3_sl_111_new_eval_50wf.json`.

These use `sec_job_fcfs_v1`, imported templates under `data/sec_job_reference`, 50 workflows, rate **0.03/s**, independent random streams, and shared-load FCFS/first-idle reference deadlines. Server count stays fixed for each trained model. Evaluate Small/Medium/Large workflows under Loose/Medium/Tight deadlines: 9 evaluation conditions per model, 27 across SS/SM/SL.

Do not train with the evaluation-only configs; do not evaluate the new protocol with the legacy training configs. Explicitly specify the frozen checkpoint in each evaluation command. Use independent, predetermined test seeds shared by all methods; never replace failed seeds after observing results. This is cross-protocol transfer, not an identical-distribution experiment.

## Training commands (PowerShell, project root)

```powershell
& 'D:\Miniconda\envs\py310\python.exe' tools/train_coupled_hierarchical_d3qn_sac.py --config configs/v3_sm_74_legacy_train.json
& 'D:\Miniconda\envs\py310\python.exe' tools/train_coupled_hierarchical_d3qn_sac.py --config configs/v3_sl_111_legacy_train.json
```

Run in separate terminals only if both are intended concurrently. No SS retraining is needed for this protocol. Existing `configs/v3_sm_74.json` and `configs/v3_sl_111.json` remain new-protocol configs and are **not** the training configs above.

Formal background launch was not performed: the host launch approval was rejected. The commands above are ready for manual execution. The two-episode integration runs below are disposable checks, not formal training runs.

## Verification and stopped jobs

- `tests/test_legacy_generation_restore.py`: exact generated-data comparison against `old/generation_pre_sec_job_20261007/pegasus_workflow_generator.py` at both resource sizes and seeds 42 / 542000; also checks configs against the old SS snapshot.
- Combined legacy/new XML and generation tests: 13 passed. Output: `tmp/legacy_restore_checks/tests.log`.
- Full-resource two-episode integration checks **both exited 0**, including the two-block parallel evaluation: `tmp/legacy_restore_checks/smoke_sm_host.log` and `smoke_sl_host.log`; reduced to 2 workflows only for these disposable checks, not formal training. These are execution checks, not convergence evidence.
- Sandbox integration checks were stopped by temporary-directory permissions; host checks are the authoritative reruns.
- User requested cancellation of the checkpoint screen and new-protocol SL training. Both job trees were stopped; saved checkpoints and partial results were retained. A partial screening report is **not** a final selected model.

The legacy adapter reuses the current physical XML parser; default/new-protocol routing is unchanged. This restoration does not claim bitwise equivalence of historical neural training or guarantee transfer success.
