# Hierarchical Reward History and Debug Notes

This file records the reward variants used in the hierarchical PPO/P3O and D3QN experiments. Read this file before changing high-level reward again.

## Current Context

- Project: hierarchical edge-cloud workflow scheduling with primary/backup and low-level DVFS.
- High level chooses primary VM/server and backup VM/server.
- Low level chooses continuous DVFS ratio `rho`.
- Deadline constraint is workflow-level: `M_w <= d_w`.
- High rollout stores raw deadline stream in seconds:
  - `deadline_cost = Delta M_w`
  - `deadline_limit = d_w`
  - `partial_makespan = M_w`
- Network/P3O training may normalize deadline internally, but buffer/logging semantics above must stay raw.
- Reliability cost is separate log-risk and should not be changed when only debugging deadline reward.

## Reward Variant A: Old Workflow Makespan Increment Reward

This was the earlier high-level reward before the task-regret experiment.

Formula:

```text
Delta M_w,t = max(0, M_w,t - M_w,t-1)

r_high =
- high_makespan_weight * Delta M_w,t / d_w
+ high_low_reward_weight * low_energy_saving_ratio / N_w
```

With common config:

```text
high_makespan_weight = 0.8 or 1.0
high_low_reward_weight = 0.2 or 0.0
```

Meaning:

```text
Penalize each step only when the workflow partial response time increases.
The penalty is normalized by the same workflow relative deadline.
```

Strengths:

```text
Directly aligned with workflow deadline: sum Delta M_w / d_w = M_w / d_w.
Better deadline_ok in staged high-only rho=1 experiments.
```

Weaknesses:

```text
Sparse for non-critical-path tasks because many tasks do not increase partial makespan.
Weak local load/queue feedback.
May push toward fast/high-energy machines because the easiest way to reduce response time is to use fast servers.
```

Observed behavior:

```text
Old stage1_best, same-seed eval:
deadline_ok ~= 0.75
avg_response ~= 2.20
energy ~= 15471
shield_relaxed ~= 5.3
```

Staged summary best from `results/hierarchical_staged/20260723_214824`:

```text
stage1_best deadline_ok ~= 0.94
stage1_best avg_response ~= 2.235
stage1_best energy ~= 15701
stage1_best shield_relaxed ~= 2.2
```

## Reward Variant B: Task Complete-Time Regret Reward

This is the newer experiment implemented in `src/ppo/hierarchical_objective.py` and `src/D3QN/reward.py`.

Formula:

```text
T_actual = final_aft - task_ready_time

T_best = min_s(estimated_finish_time_s - task_ready_time)

time_regret = max(0, T_actual - T_best) / max(T_best, eps)

r_high =
- high_makespan_weight * time_regret
+ high_low_reward_weight * low_energy_saving_ratio / N_w
```

Important definition:

```text
T_best is NOT pure execution time.
T_best is the fastest complete finish duration on a server, including communication/data ready, queue waiting, and execution.
Use decision_context.server_safety_metrics[server_id]["estimated_finish_time"] when available.
Fallback: estimated_ast + execution_time_rho_1.
```

Strengths:

```text
Dense per-task reward.
Directly observes local queue/load through complete finish time.
Can reduce energy indirectly because it no longer blindly favors fastest/high-power servers.
```

Weaknesses:

```text
Local task regret is not the same as workflow deadline safety.
It does not reserve slack for remaining DAG critical path.
It can make locally reasonable decisions while hurting workflow-level deadline_ok.
```

Observed behavior:

```text
New stage1_best, same-seed eval:
deadline_ok ~= 0.48
avg_response ~= 2.51
energy ~= 10332
shield_relaxed ~= 15.3
```

Staged summary best from `results/hierarchical_staged/20260724_095825`:

```text
stage1_best deadline_ok ~= 0.54
stage1_best avg_response ~= 2.571
stage1_best energy ~= 10781
stage1_best shield_relaxed ~= 14.4
```

Interpretation:

```text
Energy got lower, but not because energy reward was active.
In that run config had:
high_makespan_weight = 1.0
high_low_reward_weight = 0.0

Energy decreased because local regret changed VM-selection preference, not because high-level energy bonus was used.
Deadline got worse because workflow-level deadline alignment became weaker.
```

## D3QN Extra Deadline Penalty

High D3QN replay additionally used:

```text
replay_reward = buffer_reward - deadline_cost_reward_weight * deadline_cost / deadline_limit
```

With config:

```text
deadline_cost_reward_weight = 1.0
```

This is a normalized deadline penalty, not energy reward.
Do not confuse it with `high_low_reward_bonus`.

## Low-Level Reward Context

Low-level reward currently remains energy saving:

```text
low_reward = primary_energy_ref_rho1 - primary_energy_actual
```

Low-level buffer also stores normalized deadline cost for low deadline critic/PPO terms.
Forced max-frequency samples should not update low actor.

Do not change low-level reward/loss when only comparing high-level reward variants unless explicitly requested.

## Recommended Next Reward: Hybrid Time Reward

Combine the old global deadline alignment and new local queue/load feedback.

Recommended first version:

```text
r_high =
- workflow_weight * Delta M_w,t / d_w
- task_regret_weight * time_regret
+ high_low_reward_weight * low_energy_saving_ratio / N_w
```

Suggested starting weights when high energy is disabled:

```text
workflow_weight = 0.7
task_regret_weight = 0.3
high_low_reward_weight = 0.0
```

If using total high time weight:

```text
workflow_weight + task_regret_weight = high_makespan_weight
```

Why this is preferred:

```text
Delta M_w / d_w keeps workflow deadline direction.
time_regret gives dense per-task queue/load feedback.
Both terms are normalized and roughly comparable.
```

Potential further addition if deadline remains poor:

```text
task_deadline_pressure = max(0, final_aft - task_latest_finish_time) / d_w

r_high_time =
- workflow_weight * Delta M_w,t / d_w
- task_regret_weight * time_regret
- pressure_weight * task_deadline_pressure
```

Only add this after testing the simpler hybrid reward.

## Self-Prompt for Future Codex Work

Before changing reward again, do the following:

1. Read `docs/reward_history.md`.
2. Check current `configs/hierarchical_safe_ppo.json` values:
   - `objective.high_makespan_weight`
   - `objective.high_low_reward_weight`
   - `agent.high.deadline_cost_reward_weight`
3. Confirm whether the run is PPO/P3O, D3QN, staged stage1, staged stage2, or ablation.
4. Do not assume lower energy means energy reward was active. Verify `high_low_reward_weight` and replay penalty.
5. Keep these semantics unchanged unless explicitly requested:
   - deadline buffer values are raw seconds: `Delta M`, `d`, `M`
   - reliability cost is log-risk
   - low-level reward/loss/state are not part of high reward experiments
6. When evaluating a new reward, compare on the same seeds and report:
   - `deadline_ok`
   - `total_deadline_cost`
   - `avg_response`
   - `global_ms`
   - `energy`
   - `shield_relaxed`
   - `rho_mean`
   - `forced_max`
   - selection counts
7. If staged stage2 with HEFT is unchanged across reward experiments, do not attribute changes to low PPO.
8. Prefer one controlled change at a time.

## Known Runs

Old reward staged run:

```text
results/hierarchical_staged/20260723_214824
```

New task-regret reward staged run:

```text
results/hierarchical_staged/20260724_095825
```

Same-seed comparison used:

```text
seed = 282000
episodes = 20
old checkpoint = results/hierarchical_staged/20260723_214824/stage1_best_checkpoint.pt
new checkpoint = results/hierarchical_staged/20260724_095825/stage1_best_checkpoint.pt
```

## Reward Variant C: Multiplicative Workflow-Local Ratio Reward

This is the current recommended and implemented high-level time reward after Variant B degraded deadline.

Formula:

```text
Delta M_w,t = max(0, M_w,t - M_w,t-1)
normalized_makespan_increment = Delta M_w,t / d_w

T_actual = final_aft - task_ready_time
T_best = min_s(estimated_finish_time_s - task_ready_time)
task_complete_time_ratio = T_actual / max(T_best, eps)

r_high_time =
- high_makespan_weight
  * normalized_makespan_increment
  * task_complete_time_ratio
```

Full high reward remains:

```text
r_high =
r_high_time
+ high_low_reward_weight * low_energy_saving_ratio / N_w
```

With current config:

```text
high_makespan_weight = 1.0
high_low_reward_weight = 0.0
```

So the active reward is:

```text
r_high = - (Delta M_w,t / d_w) * (T_actual / T_best)
```

Why this design uses fewer hyperparameters:

```text
It does not introduce alpha/beta mixing weights.
Workflow deadline remains the base objective through Delta M_w / d_w.
Task local queue/load only multiplies the penalty through a natural dimensionless ratio.
```

Behavior:

```text
If Delta M_w,t = 0, time penalty is 0.
If T_actual = T_best, it reduces to the old reward: -Delta M_w,t / d_w.
If T_actual > T_best, it penalizes worse-than-fastest complete-time choices more strongly.
```

Diagnostics added to history/logs:

```text
task_complete_time_ratio
mean_task_complete_time_ratio
max_task_complete_time_ratio
normalized_makespan_increment
```

Do not clip `T_actual / T_best` before seeing its empirical distribution.
If ratio explodes in training, inspect `max_task_complete_time_ratio` first, then decide whether a mathematically justified robust transform is needed.

## Reward Variant D: Post-Step Slack Ratio Reward (Current)

This version directly tests whether upper-level VM selection improves remaining workflow deadline slack.

Formula:

```text
M_w,t = current partial workflow response time
d_w = relative workflow deadline
N_w = workflow task count

slack_ratio_w,t = (d_w - M_w,t) / d_w

r_high_time =
high_makespan_weight
* slack_ratio_w,t
/ N_w
```

Full high reward remains:

```text
r_high =
r_high_time
+ high_low_reward_weight * low_energy_saving_ratio / N_w
```

With current config:

```text
high_makespan_weight = 1.0
high_low_reward_weight = 0.0
```

So the active reward is:

```text
r_high = ((d_w - M_w,t) / d_w) / N_w
```

Reason for dividing by `N_w`:

```text
Without division, workflows with more tasks produce more repeated positive slack rewards.
Dividing by task count keeps one workflow's total slack reward roughly comparable across DAG sizes.
```

Expected behavior:

```text
Positive reward while the workflow still has deadline slack.
Near-zero reward when partial response time is near deadline.
Negative reward after the workflow exceeds deadline.
```

Diagnostics:

```text
current_slack_ratio
mean_current_slack_ratio
min_current_slack_ratio
high_slack_reward_raw
```

Main risk:

```text
This is less incremental than Variant A/C.
It may improve slack awareness, but early tasks can still receive similar positive rewards unless the selected VM changes M_w,t enough.
Use eval deadline_ok, slack_min, and deterministic server concentration to judge it.
```
