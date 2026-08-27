# Hierarchical Triple-Head D3QN–SAC：当前问题与排查任务

## 1. 项目目标

当前算法用于动态 Edge–Cloud 科学工作流调度：

- 上层三头 D3QN 选择主服务器；三个价值头分别学习能耗收益、Deadline 成本和可靠性成本。
- 下层 SAC 为被选服务器输出连续 DVFS 频率比例 `rho`。
- 优化目标是最小化完整系统中全部工作流的总计算能耗。
- 硬约束目标是：Deadline 成功率至少 99%，可靠性成功率 100%（当前工作流可靠性要求为 0.9）。
- 当前不启用备份副本，不能通过修改对比算法、数据集、服务器参数或 Deadline 定义来让本算法看起来更好。

请先检查现有实现和历史数据，确定根因后再修改。不要直接重写整个框架，也不要仅凭经验增大网络或惩罚系数。

## 2. 已确认的统一 20-seed 结果

统一条件：Small 数据集、20 个动态到达工作流、种子 `242000-242019`、相同服务器、相同 `smallest_workload` ready-task 顺序、关闭备份。

| 算法 | 能耗 | Deadline | 可靠性 | 平均响应时间 | rho |
|---|---:|---:|---:|---:|---:|
| 当前三头 D3QN + SAC（冻结上层 ep300，SAC best ep460） | **213.69 kJ** | **97.00%** | **100%** | 37.90 s | 0.962 |
| OOAC best ep680 | 242.92 kJ | 90.75% | 100% | 37.45 s | 0.946 |
| DQN-Edge best ep499 | 288.96 kJ | 95.25% | 100% | 35.62 s | 1.000 |
| DReaM best ep999 | 333.07 kJ | 93.25% | 94.50% | 35.70 s | 0.937 |

当前方法已经明显降低能耗，但还未达到 Deadline >= 99% 的硬约束，因此不能仅凭平均指标宣布问题解决。

## 3. 需要排查的核心问题

### 3.1 上层本身尚未守住 Deadline

使用同一个三头上层 ep300：

- 固定 `rho=1`：215.44 kJ，Deadline 96.75%，可靠性 100%。
- 接入训练后的 SAC：213.69 kJ，Deadline 97.00%，可靠性 100%。

这说明即使所有任务满频执行，上层服务器选择也不能稳定达到 99%。请优先检查上层，而不是把全部违约归因于 SAC。

### 3.2 上层后期出现明显策略退化

- ep300 的独立 20-seed Deadline 为 96.75%。
- 原训练自动选出的 best ep1160 能耗进一步下降，但独立 20-seed Deadline 只有 91.25%。
- 三个价值头的 loss 始终有限，没有数值发散。
- 训练过程中 `Q_E`、`Q_D` 均平稳增长，Deadline 拉格朗日乘子只从约 1.15 增至 1.20。

因此目前更像是约束策略漂移、Replay 分布变化、TD 目标/信用分配或 checkpoint 泛化问题，而不是简单的网络容量不足。

### 3.3 原 checkpoint 选择存在小样本过拟合

原三头配置使用：

```json
"eval_episodes": 5,
"checkpoint_deadline_target": 0.95
```

这会让在少量固定验证工作流上刚超过 95% 的低能耗模型成为 best，但换成独立 20 seeds 后可能明显退化。新的实验配置已经改成 20 个固定评估种子、Deadline 目标 0.99、可靠性目标 1.0。请保留这一严格选择逻辑。

### 3.4 Deadline 问题集中在困难尾部

当前三头 + SAC 的 20 个种子中：

- 14 个种子 Deadline 为 100%；
- 15 个种子 Deadline 至少为 95%；
- 5 个种子低于 95%；
- 最差种子 `242009` 只有 85%。

失败种子为 `242009、242010、242014、242015、242017`。请检查这些种子的工作流、服务器选择、队列积压、关键任务和 Q 值排序，判断失败是由少量服务器集中、错误的时间成本归因、对尾部风险不敏感，还是状态缺少预测未来排队所需的信息造成。

### 3.5 SAC 的局部节能与系统级节能不一致

下层日志中的同轨迹反事实计算显示 SAC 平均节省约 3.72% 的任务计算能耗；但与独立 `rho=1` 完整系统轨迹配对比较：

- 系统平均只节省 1.759 kJ，约 0.82%；
- 95% CI 为 `±2.712 kJ`，包含 0；
- 20 个种子中 12 个能耗下降，8 个能耗反而上升。

可能原因是降频改变任务完成时间和队列状态，即使上层参数冻结，后续上层输入状态和服务器选择轨迹仍然改变，局部节能被额外排队或服务器重分配抵消。请检查训练和评估是否把“同任务同服务器的局部反事实收益”错误等同于“整个系统真实收益”。

### 3.6 三头版本缺少动作区分度诊断

双头父类原本记录：

```text
Std_a(Q_E)
Std_a(Q_D)
lambda_D * Std_a(Q_D) / Std_a(Q_E)
```

可靠性感知三头类覆盖 `select_assignment` 后没有继续记录这些指标，已有评估文件中相关字段为 0。只看 Q 值均值和 lambda 大小无法判断 Deadline 头是否真正改变服务器动作排序。

请恢复并扩展动作维度诊断，至少记录：

```text
Std_a(Q_E)
Std_a(Q_D)
Std_a(Q_R)
lambda_D * Std_a(Q_D) / Std_a(Q_E)
lambda_R * Std_a(Q_R) / Std_a(Q_E)
三个头各自的 greedy action
最终 combined greedy action
各头对最终动作排序的改变率
```

先利用这些数据定位问题，再决定是否调整 lambda、价值尺度或动作选择公式。

### 3.7 可靠性约束在当前场景中几乎不活跃

- 当前算法 20-seed 可靠性成功率为 100%。
- 平均工作流可靠性约 0.935，高于要求 0.9。
- `lambda_R` 基本保持 1.0。
- 可靠性头 loss 已降至约 `1e-6`。

可靠性头目前没有明显的约束压力。需要检查它是真的正确识别并规避了危险服务器，还是因为当前环境本身太容易。修改后必须做相同种子的双头/三头消融，不能仅凭 100% 成功率声称可靠性头有效。

### 3.8 检查工作流边界和约束信用是否串扰

请重点审计上层 Replay 写入和 TD target：

- 环境中的多个工作流是交错调度的。
- Deadline 和可靠性是工作流级约束。
- 当前 transition 的 `next_state`、`done`、终端可靠性 cost 是否正确对应同一工作流或全局 MDP 语义？
- 工作流完成后，其终端 cost 是否通过 bootstrap 混入后续其他工作流的约束回报？
- Deadline miss credit 按 `Delta C` 分配是否会错误惩罚非关键任务？
- Replay 中安全样本、违约样本及不同工作流长度是否造成严重不平衡？
- `deadline_positive_fraction=0.3` 是否真正作用在三头采样上？

不要预先假设这是 bug；请结合 transition 顺序和实际 Replay 数据验证。

### 3.9 Makespan 指标命名需要核对

当前输出中：

```text
episode_makespan ≈ 20 * average_workflow_response_time
```

它更像所有工作流响应时间之和，而不是传统的全局 schedule makespan。请区分并统一：

- average workflow response/makespan；
- maximum workflow makespan；
- global schedule makespan；
- sum of workflow response times。

算法训练目标、日志和论文表格不能把这些量混用。

## 4. 建议优先查看的数据

### 当前三头上层训练

```text
results/hierarchical_dual_head_d3qn_reliability_aware/20260826_091038/train_history.json
results/hierarchical_dual_head_d3qn_reliability_aware/20260826_091038/eval_history.json
results/hierarchical_dual_head_d3qn_reliability_aware/20260826_091038/eval_ep300_rho1_20seed.json
results/hierarchical_dual_head_d3qn_reliability_aware/20260826_091038/eval_ep300_sac_20seed.json
results/hierarchical_dual_head_d3qn_reliability_aware/20260826_091038/eval_best_rho1_20seed.json
results/hierarchical_dual_head_d3qn_reliability_aware/20260826_091038/eval_best_sac_20seed.json
```

### 冻结三头上层训练 SAC

```text
results/hierarchical_reliability_aware_freeze_high_ep300_train_sac/20260826_142653/train_history.json
results/hierarchical_reliability_aware_freeze_high_ep300_train_sac/20260826_142653/eval_history.json
results/hierarchical_reliability_aware_freeze_high_ep300_train_sac/20260826_142653/eval_ep460_rho1_20seed.json
results/hierarchical_reliability_aware_freeze_high_ep300_train_sac/20260826_142653/eval_ep460_sac_20seed.json
```

### 最新三个对比算法的统一评估

```text
results/comparisons/20260827_latest_baselines_small_20seed/comparison_summary.md
results/comparisons/20260827_latest_baselines_small_20seed/comparison_summary.json
results/comparisons/20260827_latest_baselines_small_20seed/comparison_rows.jsonl
```

### 对比算法训练历史

```text
results/dqn_edge/20260826_103433/history.csv
results/dqn_edge/20260826_103433/eval_history.csv
results/oo_actor_critic/20260826_144528/train_history.json
results/oo_actor_critic/20260826_144528/eval_history.json
results/dream_dqn/20260826_175606/history.csv
results/dream_dqn/20260826_175606/eval_history.csv
```

注意：DQN-Edge 完整训练到 500 轮；OOAC 实际只到 ep748；DReaM 实际只到 ep1195。后两者使用当前已保存的最佳 checkpoint 比较，不能描述成完整 1500 轮实验。

## 5. 建议优先审计的代码

```text
tools/train_reliability_aware_hierarchical_d3qn.py
tools/train_hierarchical_d3qn.py
tools/eval_dual_head_d3qn.py
src/D3QN/reliability_aware_high.py
src/D3QN/dual_head_constrained.py
src/D3QN/deadline_lagrangian.py
src/sac/hierarchical_low_sac.py
src/sac/deadline_lagrangian.py
src/sac/reliability_lagrangian.py
src/shared/hierarchical_state_reward.py
```

当前相关配置：

```text
configs/hierarchical_dual_head_d3qn_reliability_aware.json
configs/hierarchical_reliability_aware_freeze_high_ep300_train_sac.json
configs/hierarchical_reliability_aware_joint_ep460_1200.json
```

## 6. 修改要求

1. 先给出证据链：哪个数据异常、对应哪段代码、为什么会造成当前现象。
2. 优先做最小修改，不要同时更换 state、reward、网络和训练流程，否则无法归因。
3. 保留现有代码和配置，新增独立版本，不能覆盖已经能复现的结果。
4. 不修改对比算法、服务器资源、工作流生成、Deadline 或可靠性要求来制造优势。
5. 修改后先做短程诊断和相同种子消融，再决定是否训练 1200/1500 轮。
6. 验收必须使用独立固定 20 seeds，并按以下顺序选择 checkpoint：
   - Deadline >= 99%；
   - Reliability = 100%；
   - 满足约束后总系统能耗最低。
7. 必须分别报告：固定 `rho=1` 的上层结果、接入 SAC 后的完整系统结果、SAC 同轨迹局部反事实节能，以及与 `rho=1` 独立轨迹的系统配对节能。

## 7. 最终需要回答的问题

请在修改前后明确回答：

1. 上层为什么在 ep300 后逐渐牺牲 Deadline 换取能耗？
2. Deadline 头是否真的改变了候选服务器排序？
3. 五个困难种子的违约由哪些任务、服务器和队列状态造成？
4. SAC 为什么局部节能约 3.72%，系统级却只有约 0.82% 且不显著？
5. 可靠性头在当前场景中是否有可测量的边际贡献？
6. 修改后是否在独立 20 seeds 上同时达到 Deadline >= 99%、Reliability = 100%，并保持能耗低于三个对比算法？
