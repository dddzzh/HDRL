# 2026-07-24 Staged Training Notes

## Current Run

Output directory:

```text
results/hierarchical_staged/20260724_151900
```

This run used high-level D3QN staged training:

```text
stage1: high D3QN + fixed low rho=1.0
stage2: HEFT high + low PPO
```

Current active high-level reward is Variant D:

```text
slack_ratio_w,t = (d_w - M_w,t) / d_w

r_high_time =
high_makespan_weight
* slack_ratio_w,t
/ N_w
```

Current config:

```text
high_makespan_weight = 1.0
high_low_reward_weight = 0.0
```

So active reward is:

```text
r_high = ((d_w - M_w,t) / d_w) / N_w
```

## Main Results

Stage1 high D3QN with fixed `rho=1.0`:

```text
best eval episode      560
best deadline_ok       0.90
best avg_response      2.2399
best total_deadline    22.3992
best total_energy      14306.0
best global_makespan   14.9766
best slack_mean        0.5077
best slack_min        -0.0634
best shield_relaxed    1.4
```

Stage1 final eval:

```text
episode                1500
deadline_ok            0.68
avg_response           2.1007
total_deadline         21.0069
total_energy           13085.3
global_makespan        19.5273
slack_mean             0.4682
slack_min             -0.1315
shield_relaxed         7.0
```

Stage1 last-10 eval average:

```text
deadline_ok            0.542
avg_response           2.4659
total_energy           13121.8
global_makespan        20.9001
slack_mean             0.3835
slack_min             -0.1962
shield_relaxed         16.9
```

Stage2 low PPO with HEFT high:

```text
best eval episode      1440
best deadline_ok       0.44
best avg_response      3.6399
best total_energy      2004.6
best global_makespan   24.0244
best slack_mean        0.1542
best slack_min        -1.7840
best shield_relaxed    31.2
best rho_mean          0.7084
best low_save_ratio    0.2815
```

Stage2 final eval:

```text
episode                1500
deadline_ok            0.32
avg_response           4.5202
total_energy           2281.7
global_makespan        20.5808
slack_mean            -0.0293
slack_min             -2.1251
shield_relaxed         47.2
rho_mean               0.7472
low_save_ratio         0.2456
```

## Reward Variant Comparison

Historical staged high-only best eval:

```text
Variant A old workflow makespan increment:
deadline_ok            0.94
avg_response           2.2353
energy                 15701.5
shield_relaxed         2.2

Variant B task regret:
deadline_ok            0.54
avg_response           2.5710
energy                 10780.6
shield_relaxed         14.4

Variant C multiplicative ratio:
deadline_ok            0.80
avg_response           2.3339
energy                 11253.6
shield_relaxed         7.2

Variant D slack ratio:
deadline_ok            0.90
avg_response           2.2399
energy                 14306.0
shield_relaxed         1.4
```

Interpretation:

```text
Variant D is a good direction for deadline.
It recovers most of Variant A's deadline performance while using less energy than Variant A.
It is much better than Variant B and better than Variant C on deadline_ok.
```

But:

```text
Variant D is still unstable.
Best eval reaches 0.90, but later eval drops to 0.68 final and 0.542 last-10 average.
The policy quality degrades as epsilon becomes lower.
```

## Why Train And Eval Differ

Stage1 training uses epsilon-greedy D3QN:

```text
train: epsilon-greedy
eval: epsilon = 0 deterministic argmax
```

At the end of stage1:

```text
high_epsilon ~= 0.2505
```

So training still has about 25% random exploration. This can spread VM choices and reduce queue collapse. Eval removes exploration, so deterministic argmax exposes server/action collapse.

Train and eval also use different workload seeds:

```text
train seed = seed + episode
eval seed  = seed + 100000 + episode
```

Conclusion:

```text
The train/eval gap is not caused by low-level DVFS in stage1.
Stage1 eval uses fixed rho=1.0.
The likely issue is deterministic D3QN action collapse or overfitting/instability as epsilon decays.
```

## Stage2 Interpretation

Stage2 result is almost identical across reward variants because stage2 uses:

```text
HEFT high + low PPO
```

It does not depend on stage1 high reward.

Stage2 has very low energy:

```text
energy ~= 2000 - 2400
```

But deadline is poor:

```text
deadline_ok ~= 0.32 - 0.44
```

This means low PPO currently saves too much energy and consumes too much deadline slack:

```text
rho_mean ~= 0.71 - 0.75
slack_min often around -2.0
shield_relaxed around 31 - 47
```

The low-level policy needs a stricter slack/deadline tradeoff before joint training can work.

## Code Changes Made Today

High-level deadline cost stream was already changed earlier to raw seconds:

```text
deadline_cost = Delta M
deadline_limit = d
partial_makespan = M
```

P3O network loss internally normalizes deadline by deadline limit to avoid critic scale explosion.

High-level reward variants tested/implemented:

```text
Variant C:
-(Delta M / d) * (T_actual / T_best)

Variant D current:
((d - M_t) / d) / N_w
```

Added diagnostics:

```text
global_makespan
global_makespan_workflow_id
global_makespan_task_id
global_makespan_server_id
task_complete_time_ratio
mean_task_complete_time_ratio
max_task_complete_time_ratio
current_slack_ratio
mean_current_slack_ratio
min_current_slack_ratio
```

Checkpoint saving:

```text
stage1 saves full and high-only checkpoints
stage2 saves low-only checkpoints
```

Low-level network:

```text
low PPO has deadline cost critic
forced/fixed rho samples are masked from low actor updates where applicable
```

## Tests Run

Use conda py310:

```text
D:\miniconda\envs\py310\python.exe -m unittest tests.test_hierarchical_safe_ppo
```

Result:

```text
Ran 28 tests
OK
```

D3QN smoke:

```text
D:\miniconda\envs\py310\python.exe tools\train_hierarchical_d3qn.py --config configs\hierarchical_safe_ppo.json --smoke-test --episodes 2
```

Result:

```text
completed successfully
```

## Current Diagnosis

Most likely issues in order:

```text
1. High D3QN deterministic argmax still unstable as epsilon decays.
2. Low PPO saves too aggressively and consumes deadline slack.
3. Stage2 low is trained under HEFT high, not under learned high, so it does not test true coordination.
```

Not the main issue for stage1:

```text
low DVFS, because stage1 uses fixed rho=1.0
```

## Recommended Next Steps

Step 1:

```text
Run deterministic eval on selected stage1 checkpoints, especially episode 560, 600, 800, 1000, 1500.
Use the same eval seed and more episodes, e.g. 20 or 50.
```

Purpose:

```text
Confirm whether episode 560 is genuinely best or just eval noise from 5 episodes.
```

Step 2:

```text
Add stochastic eval for high D3QN with epsilon = 0.05 or 0.10.
```

Purpose:

```text
If stochastic eval is much better than deterministic eval, the issue is argmax/action collapse.
```

Step 3:

```text
For low PPO, add a stricter deadline/slack penalty before pursuing energy.
```

Candidate low reward direction:

```text
low_reward =
energy_saving
- penalty_for_slack_consumption
```

Do not allow low PPO to reduce rho when available slack is already small.

Step 4:

```text
After high stage1 is stable and low no longer destroys deadline, train low under frozen learned high instead of HEFT high.
```

Purpose:

```text
Reduce distribution mismatch between staged training and joint deployment.
```

## Tomorrow Prompt

Read this file first:

```text
docs/20260724_staged_training_notes.md
```

Then inspect:

```text
results/hierarchical_staged/20260724_151900/stage1_high_eval.jsonl
results/hierarchical_staged/20260724_151900/stage2_low_eval.jsonl
```

Continue from this question:

```text
Can Variant D high D3QN be made deterministic-stable near 100% deadline_ok, and how should low PPO be constrained so it only uses slack that high-level VM selection leaves behind?
```

## Additional 10-Seed Stage1 Check

Command style:

```text
Evaluate stage1 D3QN high with fixed rho=1.0.
Compare current Variant D checkpoint against old Variant A checkpoint.
Use seeds 292000-292009.
For each seed, regenerate resources and workflows, then run one eval episode.
```

Result files:

```text
results/hierarchical_ablation/ten_seed_stage1_compare_20260724_1813/raw.json
results/hierarchical_ablation/ten_seed_stage1_compare_20260724_1813/summary.json
```

Current Variant D slack checkpoint:

```text
checkpoint = results/hierarchical_staged/20260724_151900/stage1_best_checkpoint.pt

deadline_ok_mean       0.78
deadline_ok_std        0.147
deadline_ok_values     [0.5, 0.9, 0.7, 0.7, 1.0, 0.9, 0.6, 0.8, 0.8, 0.9]
avg_response_mean      2.2740
energy_mean            14551.9
global_makespan_mean   14.9618
slack_mean             0.4612
slack_min_mean        -0.1040
shield_relaxed_mean    3.4
```

Old Variant A checkpoint:

```text
checkpoint = results/hierarchical_staged/20260723_214824/stage1_best_checkpoint.pt

deadline_ok_mean       0.77
deadline_ok_std        0.110
deadline_ok_values     [0.7, 0.7, 0.9, 0.7, 0.9, 0.9, 0.6, 0.7, 0.7, 0.9]
avg_response_mean      2.2963
energy_mean            15887.6
global_makespan_mean   15.1310
slack_mean             0.4611
slack_min_mean        -0.1303
shield_relaxed_mean    5.6
```

Interpretation:

```text
Across regenerated resources/workflows, Variant D is not worse than Variant A on deadline.
Variant D has slightly better deadline_ok, lower response time, lower energy, less negative worst slack, and fewer relaxed shields.
The old run's historical 0.94 best was not reproduced under this 10-seed robustness check.
```

Separate fixed-resource 10-workload check:

```text
seed_start = 292000
resources fixed from seed 292000
workflows = seed_start ... seed_start + 9

Variant D deadline_ok = 0.75
Variant A deadline_ok = 0.84
```

Interpretation:

```text
On one fixed resource configuration, old Variant A is better.
Across 10 regenerated resource/workflow seeds, Variant D is slightly better and more energy-efficient.
More seeds are needed before claiming a decisive win.
```
