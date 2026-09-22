# 上层对偶改为 episode 级 ESP 目标（2026-09-11）

只改**拉格朗日对偶的更新目标**。三头 D3QN 的 Deadline 头仍学原来的稠密信用
（`ΔC/D + β_q·queue + miss credit`），SAC 的 workflow 反事实归因（`low_dual_attribution =
workflow_counterfactual`）以及下层 `low.lagrangian` 全部不动。

## 为什么要改

论文报的是 **ESP@0.92**：每个种子自己的 Deadline 成功率是否达标，不是把所有工作流
汇总后求平均（见 `memory/report-esp-not-mean-dsr.md`）。旧的对偶盯的是**汇总**的
per-workflow miss rate 的 CVaR，这个量看不见 ESP 关心的区别：

* 4 个 episode 各违约 1 个工作流 → ESP 满分；
* 4 个 episode 是 (0, 0, 0, 4) → 丢掉 1/4 的 run。

两者的汇总 miss rate **完全相同**。所以对偶在优化一个和验收指标不一致的量。

第二个问题是对偶根本没在调节。实测（`20260910_111924` / `20260910_221916`，ep200 之后）
λ **100% 的 episode 都贴在 `lambda_max` 上**，积分项饱和。此时无论对偶测的是什么，
策略看到的都是一个常数——**约束指标是个 no-op**。原因是
`λ ← [λ + η(cost − budget)]₊` 在约束不可行时恒正，必然 clip：窗口 miss rate 实测
0.108–0.128，budget 0.08。

## 改了什么

`src/D3QN/deadline_lagrangian.py` 新增 `constraint_metric = "episode_deadline_failure"`：

```
c_e = 1{ DSR_e < episode_deadline_target }      # 20 个工作流、目标 0.92 ⇒ 至多违约 1 个
λ  ← [ λ_int + K_p·(mean(c_e) − budget) ]₊ ,  λ_int ← [ λ_int + η(mean(c_e) − budget) ]₊
```

`mean(c_e)` 在训练分布上就是 `1 − ESP`。要点：

* 该 cost 是 0/1，**CVaR 对它是阶跃函数、没有梯度**，所以构造函数强制
  `constraint_statistic = "mean"`，配 cvar 直接报错。
* 阈值用 DSR 比较并留 `1e-9` 容差：`2/20` 是精确值而 `1 − 0.90` 是
  `0.09999999999999998`，否则刚好达标的 run 会被判失败。
* 窗口 60 个 episode（二值率的采样方差比连续 miss rate 大），并要求至少 10 个样本
  才开始更新。
* `episode_deadline_target` 必须等于 `training.checkpoint_deadline_target`，否则训练
  按一个阈值优化、验收按另一个阈值挑 checkpoint。已有单测钉住。

## 超参怎么定的

λ 只在动作选择处进入（`src/D3QN/reliability_aware_high.py:118`，
`combined = Q_E − λ_D·Q_D − λ_R·Q_R`），所以可以在冻结的 checkpoint 上自由扫。
在 `β_q-only ep1180` 上扫两块不相交种子：

| λ | 542000 选择块 ESP@0.92 | 642000 验证块 ESP@0.92 | 两块 min | 平均能耗 kJ |
|---:|---:|---:|---:|---:|
| 3 | 0.80 | 0.60 | 0.60 | 226.9 |
| 4 | 0.80 | 0.80 | 0.80 | 256.5 |
| 5 | 0.95 | 0.80 | 0.80 | 284.9 |
| **6** | **0.90** | **0.90** | **0.90** | **309.2** |
| 7 | 0.90 | 0.95 | 0.90 | 327.7 |
| 8 | 0.85 | 0.85 | 0.85 | 343.7 |

λ=6 是两块 min 的最优点，且在并列的 0.90 里能耗最低（比 λ=7 少 18.5 kJ）。
故 `lambda_max: 4.0 → 6.0`。

| 键 | 旧 | 新 | 理由 |
|---|---:|---:|---|
| `constraint_metric` | `workflow_miss_rate` | `episode_deadline_failure` | 对齐 ESP |
| `constraint_statistic` | `cvar` | `mean` | 二值 cost 的 CVaR 无梯度 |
| `episode_deadline_target` | — | 0.92 | = `checkpoint_deadline_target` |
| `deadline_budget` | 0.08 | 0.10 | ESP@0.92 ≥ 0.90 的补集 |
| `lambda_max` | 4.0 | 6.0 | 见上表 |
| `dual_learning_rate` | 0.2 | 0.1 | 见下 |
| `proportional_gain` | 15.0 | 3.0 | 见下 |
| `window_episodes` | 40 | 60 | 二值率的方差 |

`K_p = 15` 配 budget 0.10 时，单是比例项就有 `15 × (0.50 − 0.10) = 6.0`，等于整个
`lambda_max`——λ 第一次更新就贴顶。`K_p = 3` 时比例项的调节区间是 `[−0.3, +2.1]`。
`η = 0.1` 让积分项在 warmup 结束后约 200 个 episode 内走到上限（旧的 0.05 要 1100 个，
大半个训练都耗在 λ≈1.5 上，而扫描显示 λ=1 的 ESP 只有 0.15）。

## 预期行为（对偶回路仿真，`np.random` 驱动的 c_e 序列）

| 场景 | 结果 |
|---|---|
| c_e 稳定在 0.44（约束不可行） | λ 在 ep250 前升到 6.0 并保持 |
| c_e 降到 0.10（刚好达标） | λ 在 5.7–6.0 之间小幅回退 |
| c_e 降到 0.02（远超目标） | λ 衰减到 <2.0，把余量还给能耗 |

第一种情况**不是失败**：λ 贴在 6.0 等价于"用扫描出来的最优 λ 固定训练"，是安全下界；
旧版的问题是贴在 **4.0**（低于最优点）且贴的理由与 ESP 无关。第三种情况是旧设计
永远到不了的——那才是能耗真正下降的路径。

升降**故意不对称**（最大上升 `0.1×0.9`，最大下降 `0.1×0.1`）：对称回路会在约束刚满足
时把 λ 拉到 0，策略立刻丢掉刚买到的余量，两者进入极限环。

## 落盘的新字段

`train_history.json(l)` 每个 episode 新增：`deadline_episode_target`、
`deadline_episode_success_rate`、`deadline_episode_failed`、
`deadline_episode_failure_rate`、`deadline_episode_failure_count`。
`[D3QN-H-LAG]` 日志行新增 `c_e=` 与 `esp_fail=`。
`tools/train_coupled_hierarchical_d3qn_sac.py` 的 drift guard 现在也覆盖
`constraint_metric` / `episode_deadline_target` / `proportional_gain` / `window_episodes`。

## 验证

* `tests/test_episode_deadline_failure_dual.py`（16 个用例）
* 全量 `tests/`：249 passed + 20 subtests（`test_all_baseline_eval.py` 除外，
  它有 2 个先前就存在的 `KeyError: 'deadline_type'` 失败）
* 端到端：`tools/train_coupled_hierarchical_d3qn_sac.py` 短跑确认
  `metric=episode_deadline_failure`、`updated=1`、λ 从 1 → 3.79 →（每 ep +0.09）

## 2026-09-14：Deadline 头加入 ESP 附加信用

纯 `episode_failure_counterfactual_credit` 实验把一个失败 episode 的总成本压成 1，
再分给约 785 个任务。`20260914_184242` 到 ep280 时 `Q_D=-0.059`，两块评估的
Deadline 成功率仍仅 0–0.75%，因此该纯模式保留作失败消融，不再作为正式配置。

正式实验改用 `dense_miss_plus_episode_failure_credit`：

```
c_t^D = c_t^{dense+workflow-miss}
      + I(DSR_e < 0.92) * regret_t / sum_k(regret_k)
```

其中 `regret_t` 是所选服务器与候选服务器中最早预测工作流完成时间之差，旧稠密成本、
逐工作流 miss credit 和原有 λ 更新均保持不变。配置同时启用
`nonnegative_deadline_q`：Deadline cost-to-go 在动作选择和 target bootstrap 时下界为 0，
避免负 `Q_D` 在 `Q_E - λ_D Q_D` 中变成奖励。环境继续启用
`future_critical_path_safety`，从当前 Deadline 余量中预留 HEFT 剩余关键路径时间。

## 2026-09-15：上层预计完成时间损失改为稠密信用

`ep1160` 在 `642017` 上固定 `rho=1` 仍只有 80% DSR，而在线 HEFT 达到 95%。
逐任务诊断显示 850 个任务中有 526 个发生等待，其中 154 次等待时存在空闲、至少同速且
计算能耗不更高的服务器。与此同时 Deadline 头已在 97.8% 的决策上改变能耗头动作，
`lambda_D Q_D` 的动作区分尺度约为能耗头的 2.42 倍，因此不再继续放大乘子。

环境原本已经计算：

```
r_t^F = max(0, F_hat_chosen^wf - min_j(F_hat_j^wf)) / D_w
```

旧版只在整个 episode 的 DSR 低于 0.92 时，用该量分摊总量为 1 的 ESP 信用；新版同时把
它作为小的逐任务稠密项加入 Deadline 头：

```
c_t^{D,new} = c_t^{D,old} + beta_F * r_t^F,  beta_F = 0.1
```

最后 201 个训练 episode 中，旧 Deadline 成本总量均值为 18.09，`r_t^F` 总量均值为
29.58，因此 `beta_F=0.1` 使新增项约占合并成本的 14.1%。网络、动作、能耗头、可靠性头、
SAC 与拉格朗日更新均不变；`beta_F=0` 精确恢复旧目标。
