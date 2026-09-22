# DQN-Edge reproduction notes

This baseline adapts Wen et al., "Deep reinforcement learning for energy-efficient workflow scheduling in edge computing," Computer Networks 274 (2026) 111790, to the repository's common simulator. It exposes two explicit variants so the paper reproduction is not silently mixed with the fair-comparison adapter.

## Variants

- `paper`: the paper state, upward-rank task dispatcher, Eq. 28 reward, VM
  action, and fixed `rho=1`.
- `paper_normalized`: the paper state, upward-rank task dispatcher, two Eq. 28
  objective rewards, VM action, and fixed `rho=1`. Only the two costs are made
  dimensionless: `r_l=-(FT_i-AT_w)/D_w` and
  `r_e=-E_t(t_i)/E_scale,t`, where `E_scale,t` is the action-independent
  median rho=1 task energy across candidate servers.
- `common_state_paper_reward`: the current common upper state and simulator
  task order, while retaining the paper's separate Eq. 28 latency and energy
  rewards. The two objective DQNs and learned attention fusion remain exactly
  separated; this is the default fair-comparison variant.
- `common_reward`: the paper state, upward-rank task dispatcher, two objective
  Q-networks, attention, and VM-only action are unchanged. Only the time and
  energy reward components are replaced by the project's fixed-coefficient
  common upper reward. No Lagrange multiplier is used.
- `common_upper`: the current 347-dimensional D3QN upper state, the simulator's
  common current-task dispatcher, and the fixed-coefficient common upper
  Deadline/energy reward. It retains DQN-Edge's two objective Q-networks,
  context attention, and VM-only action. It does **not** instantiate or update
  a Lagrange multiplier.

`configs/dqn_edge_small_20edge_6cloud.json` selects `paper_normalized` by default. Its latency and
energy components are written to separate replay fields and learned by the
paper's separate objective Q-networks. The normalization denominators are the
same for every candidate action in a state, so they correct scale without
changing Eq. 28's within-state server ordering. Set `agent.variant` to `paper`
for the strict raw-scale paper reward, or `common_upper` for the previous
347-dimensional common-state/common-reward adapter.

The default fair-comparison config uses the paper state, paper task dispatcher,
paper dual-objective reward structure, network, and training hyperparameters.
Only the two reward units are normalized to prevent raw seconds and joules from
creating an artificial scale imbalance: replay capacity `10,000`, epsilon
decay `1e-5` per task decision, learning rate `5e-6`, discount factor `0.9`,
batch size `128`, target copy every `100` steps, and epsilon range `0.3` to
`0.01`.

## Implemented paper components

- Enhanced upward ranking before allocation (Eq. 25-26), including mean execution time, dependency transfer time, and a critical-path bonus.
- One discrete action per VM; `rho=1.0` and backup duplication are fixed off.
- Eq. 27 state: normalized task ID, task workload, critical-path indicator, and two resource-availability values per VM.
- Separate latency and energy online/target DQNs, two 64-unit ReLU hidden layers, Double-DQN targets, replay capacity 10,000, batch size 128, target copy every 100 steps, and Adam/MSE.
- Three-layer context attention. For every candidate VM it consumes
  `[Q_latency, Q_energy, average_latency, cumulative_energy]`, produces two
  softmax objective weights, and applies EMA smoothing before Q fusion.
- Eq. 28 rewards: `r_l=-(FT_i-AT)` and `r_e=-E_t(t_i)`.

## Explicit reproduction assumptions

The paper is not specification-complete. This implementation makes the following assumptions rather than hiding them:

1. Table 4 overrides conflicting Algorithm 1 values: gamma 0.9 and initial epsilon 0.3. Epsilon decreases by `1e-5` per step to 0.01 as described in Section 5.1.2.
2. Section 4.3.1's two 64-neuron layers override Section 5.1.2's conflicting single 50-neuron layer.
3. The paper says each `R_i` contains remaining CPU cycles and available memory but does not define them. Here they are represented by `exp(-VM queue delay / workflow deadline)` and normalized VM memory.
4. The attention ground-truth weights required by Eq. 45 are not defined. Targets here are softmax-normalized observed latency and energy pressure relative to exponential running scales. A larger observed cost receives more attention.
5. The paper's latency/energy context is represented by log average completed-task latency and log cumulative energy.
6. The critical-path coefficient beta is unspecified and defaults to 1.0. It is configurable.
7. The repository's own common execution, communication, energy, deadline,
   reliability, workflow arrival, and resource models are deliberately used.
   Paper-side environment parameters are not imported. This isolates the
   scheduling algorithm under the same simulator as the proposed method and is
   therefore an adapted reproduction, not an author-code-identical reproduction.

## Commands

Smoke test:

```powershell
python tools/train_dqn_edge.py --smoke-test --device cpu
```

Full common-state/adapted-two-objective-reward training:

```powershell
python tools/train_dqn_edge.py --device cuda
```

To run the paper variant, copy the config and set:

```json
{"agent": {"variant": "paper"}}
```

Fixed 20-seed evaluation:

```powershell
python tools/eval_dqn_edge.py results/dqn_edge/<run>/best_checkpoint.pt --seed-start 242000 --num-seeds 20 --device cpu
```
