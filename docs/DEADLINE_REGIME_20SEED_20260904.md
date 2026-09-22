# Small/Medium × Tight/Medium/Loose 20-seed evaluation

Date: 2026-09-04

## Protocol

- Tight: `P(K=2)=0.8`, `P(K=3)=0.2`
- Medium: `P(K=2)=0.5`, `P(K=3)=0.5`
- Loose: `P(K=2)=0.2`, `P(K=3)=0.8`
- Workflow deadline: `D_w = K * T_HEFT`
- Small seeds: `242000-242019`
- Medium seeds: `742000-742019`
- The same checkpoint is evaluated under all three deadline regimes; no checkpoint is reselected per regime.
- `ESP@90` is the fraction of seeds whose workflow Deadline satisfaction rate is at least 90%.
- `Joint@90` additionally requires 100% reliability satisfaction in that seed.

## Small instance

| Deadline | Method | Energy (kJ) ↓ | Mean Deadline ↑ | ESP@90 ↑ | Mean Reliability ↑ | Joint@90 ↑ | Min Deadline |
|---|---|---:|---:|---:|---:|---:|---:|
| Tight | D3QN+SAC (ours) | **233.23** | **97.75%** | **95% (19/20)** | 100.00% | **95% (19/20)** | 85% |
| Tight | OOAC | 253.73 | 87.00% | 50% (10/20) | 100.00% | 50% (10/20) | 60% |
| Tight | DReaM | 335.94 | 93.25% | 85% (17/20) | 99.75% | 85% (17/20) | 75% |
| Tight | DQN-Edge | 296.49 | 91.25% | 75% (15/20) | 100.00% | 75% (15/20) | 65% |
| Medium | D3QN+SAC (ours) | **227.51** | **99.00%** | **100% (20/20)** | **100.00%** | **100% (20/20)** | 90% |
| Medium | OOAC | 242.92 | 90.75% | 65% (13/20) | 100.00% | 65% (13/20) | 65% |
| Medium | DReaM | 333.07 | 93.25% | 80% (16/20) | 94.50% | 35% (7/20) | 75% |
| Medium | DQN-Edge | 288.96 | 95.25% | 95% (19/20) | 100.00% | 95% (19/20) | 80% |
| Loose | D3QN+SAC (ours) | **221.56** | **99.50%** | **100% (20/20)** | 99.75% | 95% (19/20) | 95% |
| Loose | OOAC | 258.51 | 97.50% | 90% (18/20) | 100.00% | 90% (18/20) | 85% |
| Loose | DReaM | 322.68 | 98.50% | 90% (18/20) | 100.00% | 90% (18/20) | 85% |
| Loose | DQN-Edge | 280.78 | 98.50% | 95% (19/20) | 100.00% | 95% (19/20) | 85% |

## Medium instance

| Deadline | Method | Energy (kJ) ↓ | Mean Deadline ↑ | ESP@90 ↑ | Mean Reliability ↑ | Joint@90 ↑ | Min Deadline |
|---|---|---:|---:|---:|---:|---:|---:|
| Tight | D3QN+SAC (ours) | 323.60 | **89.00%** | **65% (13/20)** | 100.00% | **65% (13/20)** | 60% |
| Tight | OOAC | 330.04 | 66.75% | 35% (7/20) | 100.00% | 35% (7/20) | 20% |
| Tight | DReaM | **315.66** | 38.00% | 0% (0/20) | 100.00% | 0% (0/20) | 0% |
| Tight | DQN-Edge | 402.92 | 79.50% | 35% (7/20) | 100.00% | 35% (7/20) | 35% |
| Medium | D3QN+SAC (ours) | **316.90** | 93.75% | 80% (16/20) | 100.00% | 80% (16/20) | 65% |
| Medium | OOAC | 424.02 | 89.75% | 65% (13/20) | 100.00% | 65% (13/20) | 45% |
| Medium | DReaM | 419.00 | 90.00% | 70% (14/20) | 100.00% | 70% (14/20) | 65% |
| Medium | DQN-Edge | 475.46 | **94.75%** | **90% (18/20)** | 100.00% | **90% (18/20)** | 70% |
| Loose | D3QN+SAC (ours) | **305.80** | **98.00%** | **95% (19/20)** | 100.00% | **95% (19/20)** | 85% |
| Loose | OOAC | 325.73 | 86.00% | 60% (12/20) | 100.00% | 60% (12/20) | 35% |
| Loose | DReaM | 310.55 | 74.00% | 15% (3/20) | 100.00% | 15% (3/20) | 50% |
| Loose | DQN-Edge | 397.02 | 94.00% | 85% (17/20) | 100.00% | 85% (17/20) | 70% |

## Checkpoints

- Small ours: `results/v2_seed_checkpoints/ep300_sac460_lambdaD2.pt`
- Small OOAC: `results/oo_actor_critic/20260826_144528/best_oo_actor_critic_checkpoint.pt`
- Small DReaM: `results/dream_dqn/20260826_175606/best_checkpoint.pt`
- Small DQN-Edge: `results/dqn_edge/20260826_103433/best_checkpoint.pt`
- Medium ours: `results/v2_medium_28edge_9cloud_dvfs_v3/20260901_083951/checkpoint_ep001140.pt`
- Medium OOAC: `results/oo_actor_critic_medium_28edge_9cloud/20260902_150804/best_oo_actor_critic_checkpoint.pt`
- Medium DReaM: `results/dream_dqn_medium_28edge_9cloud/20260902_190435/best_checkpoint.pt`
- Medium DQN-Edge: `results/dqn_edge_medium_28edge_9cloud/20260902_150801/best_checkpoint.pt`

## Direct conclusion

- On Small, ours has the lowest energy and the highest or tied-best Deadline robustness in all three regimes.
- On Medium/Medium, ours saves 33.35% energy versus DQN-Edge (`316.90` vs `475.46` kJ), but ESP@90 is 80% versus 90%.
- On Medium/Tight, all methods are weak; ours is best on Deadline/ESP among the four, but only reaches 89.00% mean Deadline and 65% ESP@90.
- On Medium/Loose, ours is best overall: lowest energy, 98.00% mean Deadline, and 95% ESP@90.
- DReaM and OOAC are not monotonic from Medium to Loose because their policies consume Deadline-related state; changing the regime changes their actions, and their out-of-distribution action choices can offset the looser numerical deadline.
