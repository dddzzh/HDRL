# Four-stage D3QN-SAC training

The staged entry point is `tools/train_staged_hierarchical_d3qn.py`. It uses
the existing simulator, state encoders, actions, rewards, replay learners, and
evaluation code. It changes only which policy is allowed to learn in each
phase.

This curriculum intentionally disables the shared Deadline Lagrange
multiplier through the staged configuration override. Each level keeps its own
multiplier: the lower multiplier is updated in phases 1 and 3, the upper
multiplier is updated in phase 2, and both are updated in phase 4. A shared
multiplier would keep changing the frozen level's reward scale and would also
carry HEFT-induced constraint pressure into D3QN training.

| Phase | Server selector | DVFS policy | Purpose |
| --- | --- | --- | --- |
| 1 | Fixed HEFT | Train SAC | Learn an initial energy/deadline DVFS policy under a stable server-selection distribution. |
| 2 | Train D3QN | Freeze deterministic SAC | Train server placement while the lower-level environment remains stationary. |
| 3 | Freeze deterministic D3QN | Fine-tune SAC | Adapt DVFS to the server distribution produced by D3QN. |
| 4 | Train D3QN and SAC | Train D3QN and SAC | Jointly fine-tune both policies using smaller learning rates and fewer gradient updates. |

The default schedule contains 300 + 600 + 300 + 300 episodes. Phase 4 uses a
learning rate of `6e-5` and two replay updates per learner per episode. All
values can be edited in `configs/hierarchical_d3qn_sac_staged.json`.

Run with the requested `py310` environment:

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_staged_hierarchical_d3qn.py --config configs\hierarchical_d3qn_sac_staged.json
```

Each phase writes to a separate subdirectory. The best checkpoint is passed to
the next phase, while replay buffers are intentionally fresh because the
server-selection distribution changes between phases. The root output folder
contains `staged_training_manifest.json`, which records phase boundaries,
input and output checkpoints, and key evaluation metrics.

Training workflow seeds use a continuous global episode index for data
generation. Lagrangian warm-up uses a separate active-training
episode counter for each learner: HEFT episodes do not consume D3QN warm-up,
and frozen phases do not consume the frozen learner's warm-up. Evaluation keeps
the fixed seed specified by the base configuration so phase results remain
directly comparable.

When a phase loads the previous full checkpoint, its configured learning rates
are explicitly reapplied to the restored optimizer states. Thus phase 3 really
uses the SAC learning rate `1e-4`, and phase 4 really uses `6e-5` for both
learners.

Each train/eval block starts with `upper_status` and `lower_status`. These lines
identify the active policy, deterministic/stochastic mode, whether the reward
is used for replay or diagnostics only, active learner episode, current gradient
updates, actual optimizer learning rate, current Deadline multiplier, and
whether the multiplier updated. `high_epupd` and `low_epupd` are cumulative
counts of episodes that performed at least one learner update; `updates_now` is
the number of gradient updates in the current episode.

The upper constraint always observes the end-to-end workflow miss, preserving
D3QN's responsibility for the system Deadline envelope. The lower constraint
observes only misses causally introduced by DVFS. This lets the upper layer
learn a server placement that remains feasible under the lower policy while
the lower layer learns how far it can reduce frequency.
