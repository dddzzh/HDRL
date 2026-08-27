# Adapted HIRL baseline

This baseline implements the algorithm described in *HIRL: Hierarchical
Reinforcement Learning for Coordinated Resource Management in Heterogeneous
Edge Computing* while keeping this project's simulator interface unchanged.

## Mapping used for a fair comparison

- State: the same shared high-level state encoder used by the other adapted
  learning baselines.
- Action: TD3 selects continuous `rho`; DDQN selects one of the existing
  servers. The environment receives the existing `server_id + rho` action.
- Coordination: DDQN observes the shared state plus the rho selected by TD3,
  preserving the paper's continuous-control-before-discrete-allocation order.
- Reward: the paper's completion-minus-energy upper reward and
  latency-energy-failure lower reward, divided by the existing task deadline
  and energy reference so the two physical units remain numerically stable.
- Replay: both tiers use failure-aware prioritized sampling based on
  `abs(TD error) * (1 + failure_priority * failure_indicator)`.
- DDQN learning rate: load-adaptive Eq. (19),
  `1e-4 * (1 + 0.5 * load)`.
- Backups: disabled, matching the primary-only comparison setting.

The public HIRL repository does not include the TD3/DDQN source files. It only
contains environment code and compiled reinforcement-learning bytecode, so
this is a paper-faithful adaptation rather than a claim of line-by-line source
reproduction.

The paper-explicit training setup is 800 episodes, DDQN/TD3 replay capacities
of 600,000/5,000, batch size 256, DDQN hard target update every 1,000 steps,
TD3 soft update `tau=0.001`, DDQN learning rate `1e-4`, and TD3 actor/critic
learning rates `1e-3`/`5e-4`. Parameters not reported by the paper are listed
separately under `training.parameter_provenance` in the JSON configuration.

## Commands

Smoke test:

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_hirl.py --smoke-test --device cpu
```

Full training:

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_hirl.py --config configs\hirl_td3_ddqn.json
```

Unified evaluation after training:

```powershell
D:\Miniconda\envs\py310\python.exe tools\eval_all_baselines.py `
  --modes d3qn_sac,hirl_td3_ddqn `
  --hirl-checkpoint results\hirl_td3_ddqn\RUN_ID\best_checkpoint.pt
```
