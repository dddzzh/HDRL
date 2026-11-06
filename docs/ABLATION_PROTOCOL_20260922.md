# V2 消融实验协议（2026-09-22）

本协议以 `configs/v2_small_20edge_6cloud_dvfs_v3_future_esp.json` 和
`configs/v2_medium_28edge_9cloud_dvfs_v3_future_esp.json` 为 Full。实验目标是
**Deadline 与可靠性达标时最小化总能耗**；不可行行的低能耗不算胜出。

## 固定项

- 资源表、工作流生成、到达过程、队列规则、Deadline 档位和可靠性要求与 Full 相同。
- 每个训练型变体使用相同的训练轮数（1200）和配对训练种子；主表至少 3 个训练种子。
- 542000–542019 与 642000–642019 只用于 checkpoint 选择；742000–742019 仅用于最终验收。每档 Loose/Medium/Tight 均使用相同的验收种子，不按档位重选 checkpoint。
- 选择规则沿用 Full：两选择块均要求 ESP@90 = 100% 且可靠性达标；可行者选最低能耗。若无可行点，明确标记回退，不把它写成可行解。
- 主表报告 ESP@90、可靠性达标率、总能耗均值±标准差、相对 Full 的能耗变化和可行性；只在均可行时比较能耗。

## 主表：先回答分层是否有用

| 变体 | 上层 | 下层 | 是否重训 | 回答的问题 |
| --- | --- | --- | --- | --- |
| Full | 三头 D3QN | SAC | 已有；补齐训练种子 | 参照 |
| D3QN + ρ=1 | 同一 Full checkpoint | 固定满频 | 仅评估 | SAC 对能耗和约束的净作用；这是诊断，不称为独立训练消融 |
| HEFT + SAC | 在线队列感知 HEFT | SAC | **重训 SAC** | 学习型服务器选择相对经典启发式放置的价值 |
| HEFT + ρ=1 | 在线队列感知 HEFT | 固定满频 | 仅评估 | 无学习参照 |

HEFT + SAC 必须重新训练：Full 的 SAC 回放来自 D3QN 的服务器分布，直接接到 HEFT 会产生分布偏移。四行先跑 Small，再在 Medium 复验。

## 机制消融：只改一个因素

1. **上层 Deadline 头**：先在 `ρ=1` 的上层训练中比较 Full 上层与去 Deadline 头；两者都重新训练。这样不让 SAC 替上层补偿，能直接测 Deadline 头的贡献。去头不仅是把 λ 设为 0，还须确保动作选择和学习都不使用该头；实施前单独核对代码路径。
2. **下层 Deadline/可靠性约束**：固定同一个 Full 上层 checkpoint，分别重新训练 Full SAC、去下层 Deadline 约束 SAC、去下层可靠性约束 SAC。上层保持冻结，隔离下层机制。
3. **队列与未来安全信息**：Full 联合训练中分别去 `deadline_queue_share` 和 `future_critical_path_safety`，从零重训。状态维度若改变，禁止从 Full checkpoint 热启。

先完成主表，只有主表证明 HEFT + SAC 与 Full 的差异后再扩展机制消融。固定 λ、稀疏信用、PPO 等不是当前核心 claim，放附录或暂不做，避免训练成本掩盖主问题。

## Small 训练入口

所有变体都由 `tools/train_small_ablation.py` 从当前
`v2_small_20edge_6cloud_dvfs_v3_future_esp.json` 深拷贝后只修改消融开关，
避免复制配置后发生漂移。

- 第一组唯一新增训练：`heft_sac`。
- 第二组上层：`high_full_rho1` 与 `high_no_deadline_rho1` 均从零训练。
- 第二组下层：`low_full`、`low_no_deadline`、`low_no_reliability` 使用同一个
  上层 checkpoint，冻结上层并分别从头训练 SAC。

## 当前完成状态（2026-09-25）

- 已完成并正式用 742000–742019 验收：Full、Full checkpoint 固定 ρ=1、独立上层固定 ρ=1、去上层 Deadline 头、HEFT+SAC、HEFT固定ρ=1。
- HEFT+SAC 使用 `results/ablations/small_future_esp/heft_sac/20260924_084635/best_heft_checkpoint.pt`（episode 340）。
- 汇总表和图位于 `results/ablation_results/small_main_ablation/`；HEFT+SAC 与 HEFT固定ρ=1 的原始20-seed结果分别位于 `results/ablation_results/small_main_ablation_heft_final/` 和 `results/ablation_results/small_main_ablation_heft_rho1_final/`。
- 本轮约定的正式消融范围是主结构消融与上层 Deadline 头消融，共6个变体，现已全部评估完成。`low_full`、`low_no_deadline`、`low_no_reliability` 保留为后续扩展机制实验，不计入本轮六变体主表。
- 当前各训练型变体只有一个训练种子；20个固定种子是最终测试种子而不是训练种子。用于论文主表前，仍应补齐至少3个训练种子或明确将其标为单次训练结果。
