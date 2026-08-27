# Hierarchical Reward History and Debug Notes

This file records the reward variants used in the hierarchical PPO/P3O and D3QN experiments. Read this file before changing high-level reward again.

## Reliability-Aware Low SAC Lagrangian (2026-08-25)

Enabled only in the reliability-aware configuration. The existing low SAC
reliability violation component is included in replay training as
`-lambda_reliability * reliability_violation`. The multiplier starts at 1.0,
uses dual learning rate 0.001, zero violation budget and maximum 100.0, and is
updated once per episode from workflow-level sums of low-level reliability
violation. Deadline lambda remains a separate controller. SAC actor, critic,
state, action and replay schema are unchanged.

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

## Reliability-Aware High D3QN (standalone, 2026-08-25)

This is a new standalone high-level extension; the original joint trainer and
low-level SAC are unchanged. It adds `Q_reliability(s,a)` for cumulative
reliability-constraint violation cost. The terminal workflow cost is
`max(0, log(R_required) - log(R_workflow))`, stored on the workflow's final
high transition. `lambda_reliability` starts at 1.0 and is updated once per
completed workflow with dual learning rate 0.001 and maximum 100.0. Energy,
deadline and action-mask semantics remain those of the production dual-head
D3QN. This profile is diagnostic until a real workflow run is completed.

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

## Reward Variant F: Pre-Z-Score D3QN Baseline (Archived 2026-08-10)

This is the exact D3QN upper reward immediately before introducing the
replay-buffer z-score reward.  Keep it as the rollback baseline.

```text
C_w,t = current partial workflow makespan after scheduling task t
D_w   = workflow relative deadline
N_w   = workflow task count

slack_ratio_t = (D_w - C_w,t) / D_w

energy_ratio_t =
  (median_candidate_energy_at_rho1 - selected_primary_actual_energy)
  / median_candidate_energy_at_rho1

reliability_penalty_t =
  5 * reliability_crossed_t
  + 2 * reliability_violation_delta_t

buffer_reward_t =
  0.4 * slack_ratio_t / N_w
  + 0.6 * energy_ratio_t / N_w
  - reliability_penalty_t

replay_reward_t =
  buffer_reward_t
  - 1.5 * Delta_C_w,t / D_w
```

Configuration and environment at archive time:

```text
high_makespan_weight = 0.4
high_energy_weight = 0.6
deadline_cost_reward_weight = 1.5
backup duplication = disabled
upper action mask = structural only; all 26 real servers selectable
upper state = 4 task features + 12 features per server
```

The lower SAC reward was not part of this upper-reward formula and remains
the absolute rho=1 energy saving plus its existing time/deadline/reliability
terms.

## Reward Variant G: Deadline-Distance + Absolute-Energy Replay Z-Score (Archived 2026-08-10)

This is the replacement requested after Variant F was archived.

```text
A_w     = workflow arrival time
AFT_t   = actual finish time of the current task
C_w,t   = current partial workflow makespan
D_w     = workflow relative deadline

deadline_distance_t = D_w - (AFT_t - A_w)          [seconds]
energy_saving_t      = E_selected,rho=1 - E_actual [joules]

z_D = (deadline_distance_t - replay_mean_D) / replay_std_D
z_E = (energy_saving_t      - replay_mean_E) / replay_std_E

P_t = 0  if C_w,t <= D_w
P_t = -1 if C_w,t >  D_w

r_high_t = z_D + z_E + P_t
```

Both raw components are stored separately in the upper D3QN replay buffer.
The mean and standard deviation are recomputed from the populated replay
buffer.  No division by workflow task count is used.  The previous
`1.5 * Delta_C_w,t / D_w` replay penalty and the upper reliability penalty
are not part of Variant G.  The lower SAC reward remains unchanged.

## Reward Variant H: Workflow-Slack + Violation-Severity Deadline Reward

Variant H replaces only the upper-level deadline component and deadline
penalty.  The upper absolute-energy component and the lower SAC reward remain
unchanged.

```text
C_w,t = current partial workflow makespan after scheduling task t
D_w   = workflow relative deadline

s_t = (D_w - C_w,t) / D_w
V_t = max(0, (C_w,t - D_w) / D_w)

z_s = (s_t - replay_mean_s) / replay_std_s
z_E = (energy_saving_t - replay_mean_E) / replay_std_E

r_high,t = z_s + z_E - 2 * V_t
```

When the lower frequency is fixed at `rho=1`, `energy_saving_t=0` and `z_E=0`,
so the reward is exactly:

```text
r_high,t = z_s - 2 * V_t
```

Crossing is retained only as a diagnostic metric and does not enter the
reward.  The normalized violation term remains active while the workflow is
late.  Terminal patching does not apply the violation penalty a second time.

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

## Reward Variant I: Deadline-Only Incremental Reward (Current Experiment)

This diagnostic version disables upper-level energy and lower-level learning so
the D3QN server selector is trained only for workflow Deadline satisfaction.

```text
Delta C_w,t = C_w,t - C_w,t-1

V_w,t = 0                                      if C_w,t <= D_w
V_w,t = 1 + (C_w,t - D_w) / D_w               if C_w,t > D_w

Delta V_w,t = V_w,t - V_w,t-1

r_high = -Delta C_w,t / D_w - Delta V_w,t
```

Properties:

```text
sum_t Delta C_w,t / D_w = C_w,final / D_w
sum_t Delta V_w,t = V_w,final

R_w = -C_w,final / D_w - V_w,final
```

The unit jump in `V_w,t` creates a clear Deadline boundary without applying a
full violation penalty repeatedly after the workflow is already late. Neither
term is divided by `N_w`, because both sums telescope to one final workflow
quantity. Neither term is replay-z-score normalized because both are already
dimensionless and the Deadline boundary must stay fixed.

Active configuration:

```text
agent.high.reward.mode = scalar
objective.reward_mode = deadline_incremental_only
objective.deadline_violation_weight = 1.0
```

Diagnostics:

```text
normalized_makespan_increment
high_deadline_violation
high_deadline_violation_delta
high_deadline_crossed
high_deadline_penalty
```

## Reward Variant J: Frozen-High Slack-Aware SAC Reward (Current)

This lower-level reward is used while the validated upper D3QN policy is
frozen.  It replaces the old deadline-normalized local time increase and the
z-scored Deadline violation term.

```text
C_before = workflow partial makespan before the current task
D_w      = workflow relative deadline
S_t      = D_w - C_before

Delta T_t = max(0, AFT_rho - AFT_rho1)
g_t       = Delta T_t / max(S_t, 0.05 * D_w)

V(C) = 0                              if C <= D_w
V(C) = 1 + (C - D_w) / D_w           if C > D_w
Delta V_t = max(0, V(C_after) - V(C_before))

r_low =
  1.0 * z(E_rho1 - E_rho)
  - 1.0 * z(g_t)
  - 5.0 * Delta V_t
```

`z(x)=(x-replay_mean)/replay_std` is the ordinary replay-buffer z-score.
Energy and `g_t` are clipped to `[-3,3]` after normalization.  `Delta V_t` is
not normalized: its unit jump gives Deadline crossing a fixed penalty of at
least `5`.  Therefore there is no additional Deadline crossing penalty.  The
`0.05 * D_w` term is only a denominator floor; it is not a new constraint.
Reliability is still calculated and logged as a diagnostic, but it does not
enter this lower-level reward.

Initial configuration:

```text
energy_weight = 1.0
slack_time_weight = 1.0
deadline_violation_weight = 5.0
slack_floor_ratio = 0.05
energy_std_floor = 0.05
slack_time_std_floor = 0.005
```

## Shared State/Reward Interface for Comparison Algorithms

The reusable adapter is implemented in
`src/shared/hierarchical_state_reward.py`.  A comparison algorithm can call the
same current upper-state encoder and the same action-local DVFS reward without
copying D3QN/SAC internals.  This avoids silently changing the problem while
leaving each baseline's policy and action space intact.

For a joint server-frequency baseline such as OO actor-critic:

```text
r_high = -Delta C_w,t / D_w - Delta V_w,t

r_low =
  z(E_same_server,rho1 - E_same_server,rho)
  - z(Delta T_t / max(S_t, 0.05 * D_w))
  - 5 * Delta V_w,t

r_joint = r_high + 0.1 * r_low
```

The two z-scores use that algorithm's own replay buffer; only the formula and
units are shared.  The energy reference is the selected server at `rho=1`, not
the median energy of other servers.  Reliability remains a reported constraint
diagnostic and is not part of this reward version.  OO actor-critic still uses
its paper-specific binary actor, Top-s ordinal search, and discrete
`server x {0.2,0.4,0.6,0.8,1.0}` action.

Current reusable identifiers:

```text
state  = d3qn_high_347_v1
reward = hierarchical_deadline_dvfs_v1
```

## Lower SAC Energy-Weight Ablation

Starting from the same frozen upper D3QN checkpoint, the next controlled run
changes only the lower energy coefficient:

```text
energy_weight = 2.0      # previously 1.0
slack_time_weight = 1.0
deadline_violation_weight = 5.0
```

The reward is therefore:

```text
r_low = 2 * z(E_rho1 - E_rho) - z(g_t) - 5 * Delta V_t
```

State, upper policy, fixed alpha, seeds, and all other reward terms remain
unchanged so this run isolates the strength of the lower energy signal.

## Incremental Deadline + Restored Upper Energy (Current)

The old upper absolute-energy ratio is inserted directly into the incremental
Deadline reward.  Per request, the energy term is not divided by workflow task
count:

```text
r_high =
  - Delta C_w,t / D_w
  - Delta V_w,t
  + 0.6 * (E_task_median - E_selected_actual) / E_task_median
```

The incremental Deadline terms, 347-dimensional state, and lower SAC reward
remain unchanged.  `high_energy_weight=0.6` is explicit in the experiment
config.  A missing coefficient still defaults to zero for old
`deadline_incremental_only` checkpoints, preserving their evaluation behavior.

## OOAC Best Non-Lagrangian Compatibility Run (2026-08-20)

OOAC keeps its paper-specific binary actor, ordinal-optimization Top-s search,
and joint `server x 5 discrete rho levels` action. For this controlled run,
only its observation and scalar reward are aligned with the validated
non-Lagrangian D3QN model saved at
`results/hierarchical_d3qn_sac/20260806_111214/best_d3qn_high.pt` (episode 1040).

```text
state = d3qn_high_580_best_non_lagrangian_v1

slack_t = (D_w - C_w,t) / D_w
energy_ratio_t = (median_candidate_energy_rho1 - selected_actual_energy)
                 / median_candidate_energy_rho1

r_buffer = 0.4 * slack_t / N_w
           + 0.6 * energy_ratio_t / N_w
           - 5 * reliability_crossed_t
           - 2 * reliability_violation_delta_t

r_replay = r_buffer - 1.5 * Delta_C_w,t / D_w
```

There is no Lagrange multiplier, dual update, or replay reward recomposition by
lambda in this OOAC mode.

## D3QN-SAC ep740 Reward Restoration (2026-08-21)

The default proposed-method configuration restores the reward and independent
dual-update semantics saved in `20260817_130319/checkpoint_ep000740.pt`:

```text
r_high = ((E_task_median_rho1 - E_task_actual) / E_task_median_rho1) / N_w
         - lambda_H * (Delta C_w,t / D_w)

r_low  = 2 * clip(z(E_same_server_rho1 - E_actual), -3, 3)
         - clip(z(slack_time_cost), -3, 3)
         - lambda_L * counterfactual_deadline_cost
```

The upper multiplier is updated from rolling binary workflow misses with a
`0.01` miss budget and learning rate `0.001`. The lower multiplier uses its
counterfactual DVFS Deadline costs, zero budget, and learning rate `0.01`.
The two multipliers are independent. Team-energy credit and terminal binary
miss redistribution remain available under `cooperative_causal_v2`, but are
disabled by the restored default profile.

## Reward Variant K: Per-Workflow Sub-Deadline Lagrangian (2026-08-22, Archived Failed Experiment)

Version identifiers and entry points:

```text
objective.reward_profile = sub_deadline_lagrangian_v1
low.reward.mode           = sub_deadline_lagrangian
high.constraint_metric    = sub_deadline_violation
low.constraint_metric     = sub_deadline_violation
config                    = configs/hierarchical_d3qn_sac.json
trainer                   = tools/train_hierarchical_d3qn.py
checkpoint_dir            = results/hierarchical_d3qn_sac_subdeadline
```

This version keeps each workflow's existing global Deadline.  A maximum-
frequency HEFT reference schedule distributes that Deadline to its tasks:

```text
d_i = D_w * AFT_i_HEFT / M_w_HEFT
```

The exit task therefore has `d_exit = D_w`, and precedence order is preserved.
For a task finishing at relative time `F_i`, define the local constraint cost:

```text
v_i(F_i) = 0                                             if F_i <= d_i
v_i(F_i) = (1 + (F_i - d_i) / d_i) / N_w               if F_i > d_i
```

Upper D3QN reward:

```text
e_i_H = ((E_scale_i - E_actual_i) / E_scale_i) / N_w
r_i_H = e_i_H - lambda_H * v_i(F_i_actual)
```

Lower SAC uses a same-server, maximum-frequency counterfactual.  It is charged
only for extra local-Deadline damage caused by DVFS:

```text
e_i_L = ((E_rho1_i - E_rho_i) / E_rho1_i) / N_w
c_i_L = max(0, v_i(F_i_rho) - v_i(F_i_rho1))
r_i_L = e_i_L - lambda_L * c_i_L
```

Both energy and time components are dimensionless and divided by `N_w`.
Replay z-score normalization is not used in this profile.  The division makes
the sum of task rewards comparable across workflows of different sizes; it is
not an additional objective weight.

The multipliers remain independent and update from per-workflow accumulated
costs:

```text
c_w_H = sum_i v_i(F_i_actual)
c_w_L = sum_i c_i_L

lambda_H <- clip(lambda_H + eta_H * (mean_w(c_w_H) - 0), 0, 100)
lambda_L <- clip(lambda_L + eta_L * (mean_w(c_w_L) - 0), 0, 100)
```

Initial settings use `lambda_H=lambda_L=1`, `eta_H=eta_L=0.01`, zero budgets,
and 50 warm-up episodes. Final workflow Deadline success is still reported as
the main evaluation metric, but it is not the dual-update input in this
variant.

### Rejected K0 Ablation: Removing Both `1 / N_w` Factors

This ablation was analyzed but is not active. On the small-data runs the mean
workflow contained about 27.8 tasks. Removing both factors would therefore
increase task rewards by roughly 28 times and increase the dual constraint
input by a similar factor. With the same dual learning rate, `lambda` would
grow roughly 28 times faster and larger workflows would dominate replay.
Running this ablation later requires a separate config and a correspondingly
smaller dual learning rate; it must not silently replace Variant K.

## Reward Variant K-C3C: Counterfactual Sub-Deadline Decomposition (2026-08-22, Archived)

Version identifiers and entry points:

```text
objective.reward_profile = sub_deadline_lagrangian_v1
objective.c3c_reward_profile = c3c_sub_deadline_residual_v1
config                    = configs/hierarchical_d3qn_sac_c3c_subdeadline.json
trainer                   = tools/train_c3c_d3qn_sac.py
checkpoint_dir            = results/hierarchical_d3qn_sac_c3c_subdeadline
```

This uses exactly the same realized upper objective as Variant K, but splits
it into a maximum-frequency placement baseline and a DVFS residual:

```text
r_i_base = ((E_scale_i - E_rho1_i) / E_scale_i) / N_w
           - lambda_H * v_i(F_i_rho1)

r_i_C3C  = ((E_rho1_i - E_actual_i) / E_scale_i) / N_w
           - lambda_H * max(0, v_i(F_i_actual) - v_i(F_i_rho1))

r_i_base + r_i_C3C
    = ((E_scale_i - E_actual_i) / E_scale_i) / N_w
      - lambda_H * v_i(F_i_actual)
```

D3QN learns the `rho=1` placement baseline, C3C learns the cross-level DVFS
residual, and SAC retains the Variant-K lower reward. The upper and lower dual
controllers consume the same per-workflow accumulated costs returned by the
common collector as the no-C3C run. The old C3C configurations and reward
profiles remain available and are not overwritten.

## Reward Version Recording Rule

Every future reward modification must append a new section here before a
formal run. Each entry must include:

1. A unique immutable version identifier.
2. Complete upper and lower equations.
3. Normalization, clipping, and replay-statistics rules.
4. Constraint definitions, budgets, and multiplier-update equations.
5. Config, trainer, output directory, seed policy, and checkpoint source.
6. Whether it is active, archived, or only an ablation proposal.

Existing entries must not be silently rewritten when a new experiment starts;
rollback must be performed by selecting the recorded profile/config.

## Rollback Record R1: Restore ep740 Global-Deadline Reward (2026-08-22, Active)

Variant K was rejected after the no-C3C run
`results/hierarchical_d3qn_sac_subdeadline/20260822_102654`. Across fixed
20-seed evaluations, its best workflow Deadline success was `31.75%` at
episode 660 and the final episode reached only `8.50%`. The local HEFT-scaled
sub-Deadlines were too strict for dynamically interleaved workflows, were not
part of the policy state, and caused the zero-budget upper multiplier to grow
from `1.0` to `24.08`. This is an archived negative result, not the active
proposed reward.

The default no-C3C config now selects the already recorded
`ep740_lagrangian_v1` equations without changing their implementation:

```text
config                    = configs/hierarchical_d3qn_sac.json
objective.reward_profile  = ep740_lagrangian_v1
high.constraint_metric    = workflow_miss_rate
high.deadline_budget      = 0.01
high.lambda_init          = 1.0
high.dual_learning_rate   = 0.001
low.reward.mode           = lagrangian_deadline
low.energy_weight         = 2.0
low.slack_time_weight     = 1.0
low.constraint_metric     = workflow_miss_rate
low.deadline_budget       = 0.0
low.lambda_init           = 10.0
low.dual_learning_rate    = 0.01
checkpoint_dir            = results/hierarchical_d3qn_sac
```

Sub-Deadline generation, reward profiles, tests, and their dedicated configs
remain in the repository solely for reproducibility. They are disabled in the
active default configuration. For the previous non-sub-Deadline C3C-v2
experiment, use `configs/hierarchical_d3qn_sac_c3c_v2.json`; do not use
`configs/hierarchical_d3qn_sac_c3c_subdeadline.json`.

## C3C Variant A1: Always-On Relative Safe Coordination (2026-08-23, Experimental)

This is a separate experimental coordination variant. It does not overwrite
the active no-C3C reward or the archived globally gated C3C-v2 run.

```text
version_id                = c3c_always_on_relative_v1
base_reward_profile       = ep740_lagrangian_v1
config                    = configs/hierarchical_d3qn_sac_c3c_always_on.json
trainer                   = tools/train_c3c_always_on_d3qn_sac.py
checkpoint_dir            = results/hierarchical_d3qn_sac_c3c_always_on
training_seed             = 42
fixed_eval_seeds          = 242000..242019
status                    = experimental ablation
```

The upper and lower environment rewards, normalization, replay statistics,
constraint costs, budgets, and independent multiplier updates remain exactly
those of `ep740_lagrangian_v1`:

```text
r_high = ((E_task_median_rho1 - E_task_rho1) / E_task_median_rho1) / N_w
         - lambda_H * (Delta C_rho1 / D_w)

r_low  = 2 * clip(z(E_same_server_rho1 - E_actual), -3, 3)
         - clip(z(slack_time_cost), -3, 3)
         - lambda_L * counterfactual_deadline_cost
```

C3C continues to learn the DVFS residual decomposition:

```text
Q_E^C -> discounted residual energy reward
Q_D^C -> discounted residual Deadline cost
A_C(j) = min(Q_E,1^C(j), Q_E,2^C(j))
         - lambda_H * max(Q_D,1^C(j), Q_D,2^C(j))
```

The global 97/95-percent binary gate is disabled. For the D3QN greedy baseline
server `j_H`, C3C forms a relative, conservative candidate advantage:

```text
Delta A_C(j) = A_C(j) - A_C(j_H)
               - lambda_H * max(0, Q_D^C(j) - Q_D^C(j_H))
```

The correction is multiplied by twin-critic confidence and by a continuous
transition-count schedule (`beta_final=0.10`, no hard warm-up, 100000-
transition anneal), aligned to the D3QN candidate-Q standard deviation, and
clipped to `0.25 * std(Q_H)`. The correction of `j_H` is exactly zero.

## DQN-Edge ep740 Upper-Component Adapter (2026-08-24, Pending Formal Run)

This entry records the DQN-Edge comparison adapter reward alignment requested
on 2026-08-24. It does not change the DQN-Edge network, VM-only action,
two-objective DQN structure, attention fusion, `rho=1.0`, or backup-disabled
execution.

```text
version_id                = dqn_edge_ep740_upper_components_v1
base_algorithm            = adapted_dqn_edge
config                    = configs/dqn_edge.json
trainer                   = tools/train_dqn_edge.py
checkpoint_dir            = results/dqn_edge
fixed_eval_seeds          = 242000..242019
status                    = pending formal run
```

Only the scalar definitions fed to DQN-Edge's separate objective Q-networks are
aligned with the current proposed method's upper ep740 components:

```text
r_time   = -Delta M_w / D_w
r_energy = ((E_task_scale - E_task_actual) / E_task_scale) / N_w
```

Because DQN-Edge remains VM-only with fixed `rho=1.0`, `E_task_actual` is the
realized energy of the selected VM at maximum frequency. No SAC lower policy,
DVFS residual, Lagrange multiplier update, or checkpoint Deadline target is
added to DQN-Edge.

## State Ablation S1: Shared High State for SAC (2026-08-23, Experimental)

This experiment changes state only. The active reward remains the recorded
`ep740_lagrangian_v1` profile without any equation, normalization, Lagrange,
queue, action, or training-schedule change.

```text
config         = configs/hierarchical_d3qn_sac_shared_state.json
trainer        = tools/train_hierarchical_d3qn.py
checkpoint_dir = results/hierarchical_d3qn_sac_shared_state
C3C            = disabled
status         = experimental ablation; not the default
```

The upper D3QN keeps the existing 347-dimensional state. The lower SAC uses
the exact same 347-dimensional vector plus a 26-dimensional one-hot vector
identifying the server selected by D3QN:

```text
s_high = shared_high_state_347
s_low  = concat(shared_high_state_347, selected_server_one_hot_26)
dim(s_low) = 373
```

The selected-server identity is required because the same shared environment
state can lead to different DVFS decisions after different upper actions. The
legacy 9-dimensional local SAC state remains the default in
`configs/hierarchical_d3qn_sac.json` and can be selected with
`low.state_mode=local_9`.

### State Ablation S1-B: Freeze ep580 D3QN and Continue SAC

The paired follow-up freezes the highest-Deadline checkpoint from S1 and
continues the same SAC network for 500 additional episodes:

```text
config             = configs/hierarchical_d3qn_sac_shared_state_freeze_high_ep580.json
initial checkpoint = results/hierarchical_d3qn_sac_shared_state/20260823_192649/checkpoint_ep000580.pt
frozen             = upper D3QN policy and upper lambda
trainable          = lower SAC and lower lambda
episode/seed start = 581
checkpoint rule    = Deadline >= 99%, then minimum energy
reward             = unchanged ep740_lagrangian_v1
C3C                = disabled
status             = experimental continuation; not the default
```

The checkpoint did not save replay, so continuation restores both network and
optimizer states but rebuilds the SAC replay statistics from the new episodes.

## State/Replay Ablation S2: Server-Local Actor + Global Twin Critic (2026-08-24, Experimental)

This experiment changes the lower observation, existing SAC critic inputs,
and replay sampling only. It does not add C3C/HCC or any third network. The
upper/lower rewards, replay z-score statistics, constraint costs, independent
Lagrange multipliers, queue rule, actions, and direct continuous rho remain the
recorded `ep740_lagrangian_v1` profile.

```text
version_id       = server_ctde_responsibility_replay_v1
config           = configs/hierarchical_d3qn_sac_server_ctde.json
trainer          = tools/train_hierarchical_d3qn.py
checkpoint_dir   = results/hierarchical_d3qn_sac_server_ctde
training_seed    = 42
fixed_eval_seeds = 242000..242019
C3C/HCC          = disabled
checkpoint_rule  = Deadline >= 99%, then minimum energy
status           = experimental ablation; not the default
```

The parameter-shared actor receives the existing selected-server local DVFS
state plus three already-computed D3QN server features: queued processing
time, queued-processing EMA, and recent primary assignments. The two existing
SAC critics additionally receive the five global features and
permutation-invariant mean/max pools of all valid server features:

```text
o_actor  = concat(local_9, selected_server_features[0, 9, 10])  # 12D
g_server = concat(mean(valid_server_features), max(valid_server_features))
s_critic = concat(o_actor, global_features, g_server)            # 41D
```

All servers use the same actor, twin critics, and replay buffer. The replay
priority uses the existing rho=1 counterfactual fields and raises sampling
priority only when rho=1 was feasible but the actual rho crossed the workflow
Deadline:

```text
p_i = clip((abs(TD_i) + eps)^alpha
           * (1 + beta_cross * I_counterfactual_cross
                + beta_severity * V_counterfactual_cross), eps, p_max)
P(i) = 0.4 * p_i / sum(p) + 0.6 / replay_size
w_i = normalize((replay_size * P(i))^(-importance_beta))
```

The default constants are `alpha=0.6`, `importance_beta=0.4`,
`beta_cross=2.0`, `beta_severity=1.0`, `eps=1e-4`, and `p_max=20`. Samples
where rho=1 already misses are not marked as lower responsible and receive no
responsibility multiplier. Importance weights are applied to both SAC critic
losses; the actor loss and the registered reward are unchanged.

## Reward Variant L: CTDE Placement Energy + Cumulative DVFS Credit (2026-08-24, Experimental)

This is a new recoverable experiment. It does not overwrite
`ep740_lagrangian_v1` or `server_ctde_responsibility_replay_v1`.

```text
version_id       = ctde_placement_cumulative_v2
config           = configs/hierarchical_d3qn_sac_server_ctde_v2.json
trainer          = tools/train_hierarchical_d3qn.py
checkpoint_dir   = results/hierarchical_d3qn_sac_server_ctde_v2
training_seed    = 42
fixed_eval_seeds = 242000..242019
checkpoint_rule  = Deadline >= 99%, then minimum energy
status           = experimental; not the default
```

The upper reward no longer contains energy saved by the lower SAC action. For
task `t` in workflow `w`, the upper energy term is based only on the selected
server at maximum frequency:

```text
r_E,t^H = (E_t^scale - E_t,selected^(rho=1)) / (N_w E_t^scale)
c_w^H   = I(C_w > D_w)
r_t^H   = r_E,t^H - lambda_H c_t^H
```

At workflow completion, `c_w^H` is distributed across its upper transitions
in proportion to the recorded non-negative makespan increments, so their sum
is exactly the binary workflow miss. The upper dual remains a workflow-miss
constraint with budget `0.01`, but its learning rate changes from `0.001` to
`0.01`:

```text
lambda_H <- clip(lambda_H + 0.01 * (mean_w(c_w^H) - 0.01), 0, 100)
```

The lower base reward and CTDE state are unchanged:

```text
r_base,t^L = 2 z_replay(E_rho1 - E_rho) - z_replay(g_t)
o_actor     = concat(local_9, selected_server_features[0, 9, 10])
s_critic    = concat(o_actor, global_5, mean(server_12), max(server_12))
```

The lower Deadline responsibility is now assigned after workflow completion.
For each lower transition, let `delta_t^L` be the extra workflow partial
makespan caused by its actual rho relative to rho=1. Define:

```text
Delta_w^L       = sum_t delta_t^L
C_w^cf          = max(0, C_w - Delta_w^L)
c_w^L           = I(C_w > D_w and Delta_w^L > 0 and C_w^cf <= D_w)
c_t^L           = c_w^L * delta_t^L / Delta_w^L
r_t^L           = r_base,t^L - lambda_L c_t^L
lambda_L        <- clip(lambda_L + 0.01 * mean_w(c_w^L), 0, 100)
```

Thus an upper placement that still misses after removing all directly observed
DVFS makespan delay is not charged to SAC. Transitions with positive `c_t^L`
are marked responsible for replay priority; their severity is `c_t^L`. The
same shared local actor, global twin critics, direct continuous rho action,
queue rule, and all other hyperparameters remain unchanged.

## Upper Ablation U1: Dual-Head Constrained D3QN at Fixed Rho=1 (2026-08-24, Experimental)

This is a standalone upper-only experiment. It does not overwrite the original
D3QN, SAC, CTDE, C3C/HCC, or their configs/checkpoints.

```text
version_id       = upper_dual_head_cmdp_v1
config           = configs/hierarchical_dual_head_d3qn.json
trainer          = tools/train_dual_head_d3qn.py
checkpoint_dir   = results/hierarchical_dual_head_d3qn
training_seed    = 42
fixed_eval_seeds = 242000..242004 during training
formal_eval      = 242000..242019 after training
rho              = fixed 1.0; no SAC network or SAC replay
checkpoint_rule  = Deadline >= 99%, then minimum energy
status           = experimental ablation; not the default
```

The 347-dimensional upper state, 26-server primary-only action, original queue
rule, and structural-only padding mask are unchanged. Two independent dueling
Double-DQN heads learn placement energy reward and workflow-Deadline cost:

```text
r_E,t = (E_t^scale - E_t,selected^(rho=1)) / (N_w E_t^scale)
c_w   = I(C_w > D_w)
c_t   = c_w * DeltaC_t / sum_j DeltaC_j
sum_(t in w) c_t = c_w

a_t = argmax_a [Q_E(s_t,a) - lambda_H Q_D(s_t,a)]
y_E = r_E,t + gamma_E Q_E^target(s_next,a_next)
y_D = c_t   + gamma_D Q_D^target(s_next,a_next)
a_next = argmax_a [Q_E^online(s_next,a) - lambda_H Q_D^online(s_next,a)]
```

`Q_E` and `Q_D` have separate Huber losses and optimizers. Lambda is not part
of either Bellman label; it only combines the two heads for current and next
action selection. The workflow-level multiplier remains:

```text
lambda_H <- clip(lambda_H + 0.001 * (mean_w(c_w) - 0.01), 0, 100)
```

The energy head samples uniformly. The Deadline head requests 30% positive-cost
transitions when available and applies inverse class-quota weights, so sparse
miss samples are seen more often without changing the expected replay loss.

## Upper Ablation U1.1: Dense-Deadline Dual-Head D3QN (2026-08-24, Experimental)

This is a recoverable derivative of U1. The original U1 configuration keeps
`deadline_head_cost_mode=binary_miss_credit` by default and is not overwritten.

```text
version_id       = upper_dual_head_dense_deadline_v1_1
config           = configs/hierarchical_dual_head_d3qn_dense_deadline.json
trainer          = tools/train_dual_head_d3qn.py
checkpoint_dir   = results/hierarchical_dual_head_d3qn_dense_deadline
training_seed    = 42
fixed_eval_seeds = 242000..242004 during training
formal_eval      = 242000..242019 after training
rho              = fixed 1.0; no SAC network or SAC replay
status           = experimental ablation; not the default
```

The energy head, two-network architecture, action, state, queue rule, replay
size, optimizer settings, exploration, and binary workflow-miss dual update
remain identical to U1. Only the Deadline-head immediate Bellman cost changes:

```text
g_D,t = max(0, Delta C_w,t) / D_w
sum_(t in w) g_D,t = C_w / D_w
y_D   = g_D,t + gamma_D Q_D^target(s_next, a_next)

a_t = argmax_a [Q_E(s_t,a) - lambda_H Q_D(s_t,a)]
lambda_H <- clip(lambda_H + 0.001 * (mean_w(I(C_w>D_w)) - 0.01), 0, 100)
```

Thus successful workflows retain a dense Deadline-margin signal, while the
dual controller still targets a 1% binary workflow miss budget. Diagnostic-only
logging adds the valid-action standard deviations of both heads and
`lambda_H * std(Q_D) / std(Q_E)`; these diagnostics do not alter action choice.

## Upper Ablation U1.2: Dense-Deadline + Crossing Dual-Head D3QN (2026-08-24, Experimental)

This is a recoverable single-change derivative of U1.1. U1 and U1.1 remain
available and are not overwritten.

```text
version_id       = upper_dual_head_dense_crossing_v1_2
config           = configs/hierarchical_dual_head_d3qn_dense_crossing.json
trainer          = tools/train_dual_head_d3qn.py
checkpoint_dir   = results/hierarchical_dual_head_d3qn_dense_crossing
training_seed    = 42
fixed_eval_seeds = 242000..242004 during training
formal_eval      = 242000..242019 after training
rho              = fixed 1.0; no SAC network or SAC replay
status           = experimental ablation; not the default
```

State, action, queue rule, energy head, network sizes, optimizers, replay,
exploration, and binary workflow-miss dual update are identical to U1.1. The
only Bellman-cost change is a one-off penalty at the first Deadline crossing:

```text
I_cross,t = I(C_w,t-1 <= D_w < C_w,t)
beta_cross = 1.0
g_D,t = max(0, Delta C_w,t) / D_w + beta_cross * I_cross,t
sum_(t in w) g_D,t = C_w / D_w + beta_cross * I(C_w > D_w)

a_t = argmax_a [Q_E(s_t,a) - lambda_H Q_D(s_t,a)]
lambda_H <- clip(lambda_H + 0.001 * (mean_w(I(C_w>D_w)) - 0.01), 0, 100)
```

The crossing term is applied exactly once per missed workflow. The initial
ablation runs for 600 episodes so it covers the U1.1 best region near episode
420 and the beginning of its later degradation. Logging adds the per-episode
crossing count and configured crossing coefficient.

## Upper Ablation U1.3: Dense-Deadline + Distributed Miss Credit (2026-08-24, Experimental)

This is a recoverable derivative of U1.1. U1, U1.1, and U1.2 are preserved.

```text
version_id       = upper_dual_head_dense_miss_credit_v1_3
config           = configs/hierarchical_dual_head_d3qn_dense_miss_credit.json
trainer          = tools/train_dual_head_d3qn.py
checkpoint_dir   = results/hierarchical_dual_head_d3qn_dense_miss_credit
training_seed    = 42
fixed_eval_seeds = 242000..242004 during training
formal_eval      = 242000..242019 after training
rho              = fixed 1.0; no SAC network or SAC replay
status           = experimental ablation; not the default
```

For workflow `w`, let `Delta C_w,t=max(0,C_w,t-C_w,t-1)` and let the terminal
miss label be `m_w=I(C_w>D_w)`. The Deadline head receives:

```text
beta_miss = 1.0
omega_w,t = Delta C_w,t / sum_(k in w) Delta C_w,k
g_D,t = Delta C_w,t / D_w + beta_miss * m_w * omega_w,t
sum_(t in w) g_D,t = C_w/D_w + beta_miss*m_w

a_t = argmax_a [Q_E(s_t,a) - lambda_H Q_D(s_t,a)]
lambda_H <- clip(lambda_H + 0.001 * mean_w(I(C_w>D_w)), 0, 100)
```

If all increments are numerically zero, `omega_w,t` falls back to a uniform
distribution. The extra unit miss cost is therefore distributed over the
workflow according to each transition's positive makespan increment rather
than assigned only to the first crossing transition. The required Deadline
success is set to 100%: `deadline_budget=0.0` and
`checkpoint_deadline_target=1.0`. State, action, energy head, network sizes,
queue rule, optimizers, exploration, and fixed `rho=1` remain unchanged.

## U1.3-L1: Frozen Dual-Head U1.3 + Fresh SAC (2026-08-24, Experimental)

```text
version_id       = dual_head_u13_frozen_high_low_sac_v1
config           = configs/hierarchical_dual_head_d3qn_sac_freeze_high.json
trainer          = tools/train_dual_head_d3qn_sac.py
checkpoint_dir   = results/hierarchical_dual_head_d3qn_sac_freeze_high
high_checkpoint  = results/hierarchical_dual_head_d3qn_dense_miss_credit/20260824_200638/best_d3qn_checkpoint.pt
high_episode     = 500
high_training    = frozen and deterministic
low_training     = fresh SAC only
training_seed    = 42
fixed_eval_seeds = 242000..242004 during training
formal_eval      = 242000..242019 after training
status           = experimental ablation; not the default
```

The upper U1.3 state, action, two Q heads, distributed miss-credit cost, and
loaded multiplier are unchanged. The low SAC directly outputs continuous
`rho in [0.2,1]` after the upper server decision. Its replay reward is the
existing recorded global-Deadline profile:

```text
r_L = 2*clip(z(E_rho1-E_rho),-3,3)
      - clip(z(slack_time_cost),-3,3)
      - lambda_L*c_D,L

lambda_L <- clip(lambda_L + 0.01*mean_workflow_miss, 0, 100)
deadline_budget_L = 0
alpha = 0.2 fixed; automatic entropy tuning disabled
```

The fresh actor is initialized conservatively with deterministic mean
`rho=0.95` and `log_std=-2.0`. This changes only the initial policy, not the
action interval or reward. It prevents the default zero-mean tanh actor from
starting near `rho=0.6`, which in a one-episode integration test caused severe
queue delay and negative energy saving before SAC had learned the environment.

`c_D,L` attributes only additional Deadline damage caused by DVFS relative to
the same upper placement at `rho=1`; an upper-level miss is not charged again
to SAC. The fixed-rho upper checkpoint evaluated at `97.25%` Deadline success,
`242.12 kJ`, and `36.94 s` over seeds `242000..242019`. Therefore this run is
an integration experiment, not evidence of a 100% hard guarantee.

### U1.3-L1 Fine-tuning Protocol: Frozen SAC, Trainable Dual Head

This protocol changes no state, action, reward, Deadline cost, or multiplier
formula. It resumes the complete U1.3-L1 episode-600 checkpoint, keeps the SAC
actor/critics frozen with deterministic actions, and adapts only the two upper
D3QN heads to the queue distribution induced by the trained SAC policy.

```text
config           = configs/hierarchical_dual_head_d3qn_finetune_high_frozen_sac.json
trainer          = tools/finetune_dual_head_d3qn_with_frozen_sac.py
source           = results/hierarchical_dual_head_d3qn_sac_freeze_high/20260824_210601/checkpoint_ep000600.pt
episodes         = 300
energy_head_lr   = 1e-4
deadline_head_lr = 1e-4
updates/episode  = 5
SAC              = frozen and deterministic
eval             = 10 fixed seeds during training; 20 paired seeds after training
checkpoint_rule  = Deadline >= 99%, then minimum system energy
status           = experimental adaptation protocol; reward profile unchanged
```

## Upper Ablation U2: Feasible-Set Energy D3QN + Workflow Risk Head (2026-08-24, Experimental)

This is a new standalone upper-only experiment. It preserves U1 and all
original D3QN/SAC implementations and checkpoints.

```text
version_id       = upper_feasible_workflow_risk_v2
config           = configs/hierarchical_dual_head_feasible_d3qn.json
trainer          = tools/train_dual_head_feasible_d3qn.py
checkpoint_dir   = results/hierarchical_dual_head_feasible_d3qn
training_seed    = 42
fixed_eval_seeds = 242000..242004 during training
formal_eval      = 242000..242019 after training
rho              = fixed 1.0; no SAC network or SAC replay
checkpoint_rule  = Deadline >= 99%, then minimum energy
status           = experimental ablation; not the default
```

The 347-dimensional upper state, 26-server primary-only action, original queue
rule, structural-only padding mask, and placement-energy signal are unchanged:

```text
r_E,t = (E_t^scale - E_t,selected^(rho=1)) / (N_w E_t^scale)
y_E   = r_E,t + gamma_E Q_E^target(s_next, a_next)
```

The U1 Deadline Bellman head, Delta-C credit split, and Lagrange multiplier are
removed.  A completed workflow supplies one terminal binary label to all of
its actually selected state/action pairs:

```text
y_w      = I(C_w > D_w)
p_D(s,a) = sigmoid(z_D(s,a))
L_D      = weighted_BCE_with_logits(z_D(s_t,a_t), y_w)
```

Risk replay stores a completed workflow as one group.  A minibatch first
samples workflows and then one step from each selected workflow, so every
workflow has equal expected gradient weight regardless of task count.  When a
30% positive-workflow quota is available, inverse class-quota weights restore
the uniform-workflow BCE objective.  Focal loss is deliberately disabled to
preserve probability calibration.

The feasibility threshold uses a curriculum and is activated only after 500
completed training workflows:

```text
tau(e) = 0.20                                      e <= 100
tau(e) = linear interpolation from 0.20 to 0.01   100 < e < 500
tau(e) = 0.01                                      e >= 500
A_f(s) = {a in structural actions : p_D(s,a) <= tau(e)}
```

The deterministic action and the Double-DQN next action use the same rule:

```text
a = argmax_(a in A_f) Q_E(s,a)   if A_f is non-empty
a = argmin_(valid a) p_D(s,a)    otherwise
```

Training exploration remains epsilon-greedy over every structurally valid
server and therefore crosses the learned feasibility boundary.  Diagnostics
record Brier score, ECE with fine bins around 0.01, positive recall, selected
risk, selected-risk rank, feasible action count, fallback rate, threshold, and
gate state.  The objective mode `feasible_deadline_risk` keeps the recorded
upper placement-energy component but applies no scalar Deadline penalty and no
dual update.

## Baseline B2: Fixed-Lambda Deadline Crossing + Reliability (2026-08-26)

This version is used by the tuned OOAC and DReaM comparison runs. Their
published network/action structures remain unchanged. The common normalized
reward is extended with an explicit unit-jump Deadline miss cost:

```text
r_t = (E_t^scale - E_t^actual) / (N_w E_t^scale)
      - 10 * Delta_C_t / D_w
      - 5 * Delta_V_t
      - c_R,t

V_t       = 0                         if C_t <= D_w
            1 + (C_t - D_w) / D_w    otherwise
Delta_V_t = max(0, V_t - V_(t-1))
c_R,t     = max(0, V_R(rho_t) - V_R(rho=1))
```

The reliability term is a same-server counterfactual cost, so it charges only
the additional reliability damage caused by DVFS and does not charge an
already-infeasible upper placement twice. Both baselines use an effective
fixed reliability coefficient of `1.0`.

```text
OOAC config       = configs/oo_actor_critic.json
OOAC exact top-s  = 30 (was 10); candidate generator and network unchanged
DReaM config      = configs/dream_dqn.json
DReaM gamma       = 0.30 (paper setting; avoids max-Q overestimation in 182 actions)
DReaM batch       = 256
DReaM replay      = 50000
DReaM target sync = every 500 environment steps
DReaM epsilon     = epsilon * 0.99999, floor 0.05
checkpoint rule   = maximum Deadline success, then reliability, then energy
status            = tuned comparison baseline; requires fresh training
```
