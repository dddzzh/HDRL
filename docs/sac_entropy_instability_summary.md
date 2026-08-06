# SAC 自动熵调节失稳总结

## 实验信息

- 训练目录：`results/hierarchical_d3qn_sac/20260728_110649`
- 训练长度：1500 episodes
- 上层：D3QN
- 下层：离散 SAC，11 个 DVFS 动作
- 原配置：`alpha=0.2`、`automatic_entropy_tuning=true`、`gamma=0.99`

## 现象

训练前 300～500 episodes 的性能较好，之后逐渐退化，约 episode 600 后失稳开始变得明显。

| 指标 | 训练前期 | 训练末期 |
|---|---:|---:|
| SAC alpha | 约 0.24 | 约 8.02（最后单轮约 8.94） |
| SAC Q 均值 | 约 0.93 | 约 266.82（最后单轮约 300.86） |
| SAC critic loss | 约 2.46 | 约 363.39（最后单轮约 563.10） |
| 训练平均 rho | 约 0.88 | 约 0.70 |
| 训练 deadline 成功率 | 约 68.6% | 约 5.7% |
| 训练节能率 | 约 4.7% | 接近 0 |

评估的最佳综合点在 episode 320，deadline 成功率为 68%；最终 episode 1500 下降到 36%。可靠性成功率一直为 100%，不是这次失稳的来源。

## 因果链

### 1. 目标熵设置得过高

11 个离散动作的最大熵为：

\[
H_{max}=\log 11\approx 2.398
\]

原代码把目标熵设置成最大熵的 98%：

\[
H_{target}=0.98\log 11\approx 2.349
\]

这会要求策略长期接近均匀随机选择频率。

### 2. alpha 持续增大

实际策略熵长期低于目标熵，自动熵优化器因此不断提高 `alpha`。实验中 `alpha` 从 0.2 上升到 8 以上，没有上限约束。

### 3. 熵项抬高 Q target

离散 SAC 的目标状态价值为：

\[
V(s')=\sum_a\pi(a|s')\left[Q(s',a)-\alpha\log\pi(a|s')\right]
=\mathbb{E}[Q]+\alpha H
\]

当 `alpha≈8`、`H≈2.25` 时，单步熵奖励约为 18。再叠加 `gamma=0.99` 的长时域累计，Q target 被持续抬高，导致 Q 值、critic loss 和梯度一起增长。

### 4. 熵奖励压过约束惩罚

下层 deadline/reliability 的固定硬惩罚为 5。训练后期单步熵奖励已经明显大于 5，策略为了维持高熵会继续随机选择较低频率，硬约束惩罚难以起到主导作用。

### 5. 下层失稳传递给上层

低频动作增多后，任务完成时间和 deadline 违反增加；同时上层 reward 包含下层节能分量。不断变化的 SAC 策略使 D3QN 面对非平稳环境，D3QN 的 TD 误差也随之上升，最终上下层共同退化。

## 当前修改

配置已改为：

```json
"alpha": 0.2,
"automatic_entropy_tuning": false
```

此时：

- `alpha` 固定为 0.2；
- 不创建 `log_alpha` 和 alpha optimizer；
- actor 和 critic 仍然保留 SAC 熵正则，但熵强度不再自动膨胀；
- reward 各项权重继续使用当前固定值，暂不做自适应。

## 后续自适应方向

若后续重新设计自适应熵系数，至少需要同时满足：

1. 目标熵显著低于最大熵，例如 `0.3～0.5 × log(action_dim)`；
2. 给 `alpha` 设置上下限，例如 `[1e-4, 0.5]`；
3. 监控熵项 `alpha × entropy` 与 deadline/reliability 固定惩罚的比例；
4. 当 Q 值、critic loss 或 deadline crossing 持续上升时停止增加 alpha；
5. 在 SAC 稳定后，再考虑 reward 权重的自适应，避免两个自适应机制相互干扰。

## 下一轮判断标准

建议先训练 300～500 episodes，并确认：

- `low_alpha` 始终为 0.2；
- critic loss 不再长期单调上升；
- Q 均值不再持续膨胀；
- deadline crossing 不随训练持续增加；
- 训练与确定性评估的 rho 差距缩小。
