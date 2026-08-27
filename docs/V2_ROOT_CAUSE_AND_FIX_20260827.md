# 三头 D3QN + SAC：根因证据链与最小修改（2026-08-27）

回应 `docs/CLAUDE_HANDOFF_CURRENT_ALGORITHM_ISSUES_20260827.md`。

所有诊断由 `tools/diagnose_triple_head_dvfs.py` 产生，输出在
`results/diagnostics/20260827/`。该工具在同一 20 seeds（242000–242019）上复现出
**213.69 kJ / 97.00%**（ep300 + SAC ep460）与 **209.52 kJ / 91.25%**（best ep1160，ρ=1），
与文档记录逐位一致，因此下列数据可信。

---

## 0. 环境兼容性（非算法问题，但阻塞一切复现）

`tmp/py310_site_packages` 垫片在当前 Python 3.13 下会让**所有** `tools/*.py` 崩溃
（`OSError: [WinError 126] ... torch_python.dll`），因为那个 3.10 环境已不存在。
已在全部 17 个文件中把条件改为 `sys.version_info[:2] == (3, 10) and ...`，
3.10 下行为逐字不变。

`tests/test_dqn_edge.py::test_common_reward_keeps_normalized_time_and_energy_objectives_separate`
在本次工作开始前就已失败（`src/D3QN/reward.py` 在初始 `git status` 中已是 modified，
且该测试属于对比算法 DQN-Edge，按要求不修改）。其余 198 个测试全部通过。

---

## 1. 证据链

### E1 · 拉格朗日乘子实际上是冻结的 → §3.2、问题 1

| 项 | 值 |
|---|---|
| 配置 `dual_learning_rate` | 0.001 |
| 实测 `deadline_excess` | 0.05 – 0.08 |
| 预测 Δλ/episode | ≈ 6e-5 |
| 1200 episodes 预测总增量 | +0.06 |
| **实测 λ_D** | **1.1419 → 1.1986（+0.057）** |

代码：`src/D3QN/deadline_lagrangian.py:107`。
对偶回路时间常数约 **10⁴ episodes**，在 1200 episodes 内 λ_D 等价常数 1.15，
**约束反馈从未闭合**。算法实际在优化固定标量化 `Q_E − 1.15·Q_D`；
Q_E 的梯度信号密集一致，Q_D 稀疏，策略必然单向漂向能耗。

**实测漂移**（同一上层，ρ=1）：

| | ep300 | ep1160 |
|---|---:|---:|
| 能耗 | 215.45 kJ | 209.52 kJ（−2.8%） |
| 平均响应时间 | 37.12 s | 40.76 s（**+9.8%**） |
| Deadline | 96.75% | 91.25% |

→ **用 10% 的时间余量换 2.8% 的能耗。** 这就是问题 1 的答案。

### E2 · 三个头都在改变排序，§3.6 的怀疑不成立 → 问题 2

原 eval 文件中这些字段全为 0，是因为 `ReliabilityAwareHighLevelD3QNAgent.select_assignment`
（`src/D3QN/reliability_aware_high.py:252`）覆写了父类却没写回诊断字段。
探针重算 11134 次决策：

| 指标 | ep300 | ep1160 |
|---|---:|---:|
| Std_a(Q_E) | 0.02016 | 0.02291 |
| Std_a(Q_D) | 0.02221 | 0.02476 |
| Std_a(Q_R) | 0.00283 | 0.00138 |
| λ_D·Std(Q_D)/Std(Q_E) | **1.354** | 1.320 |
| λ_R·Std(Q_R)/Std(Q_E) | 0.144 | 0.060 |
| Deadline 头改变贪心动作 | **88.1%** | 87.7% |
| 可靠性头改变贪心动作 | **18.9%** | 6.6% |

Deadline 头对排序的影响比能耗头更大，88% 的决策被它改写。
**问题不在"Deadline 头没起作用"，而在"λ_D 停在错误水平"**，矛头指回 E1。

同时暴露一个结构隐患：`mean(Q_D)=0.881` 而动作间极差仅 `0.090`，
**Q_D 有 90% 是与动作无关的共模项**（见 E7）。

### E3 · 物理可用降频空间 22.6%，SAC 只拿到 3.6% → 用户核心关切

`src/simulator/resource.py:344`：`E(ρ) = (L/f_max)·(P_s/ρ + P_d·ρ²)`，
在 `(0,1]` 上**有内点极小值** `ρ* = (P_s/2P_d)^{1/3}`。

| 服务器类型 | P_s | P_d | ρ* | E(ρ*)/E(1) | ep300 任务数 | 能耗 | 可省 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cloud_fast | 50 | 382 | **0.405** | **0.431** | 308 | 544.0 kJ | 309.7 |
| edge_efficient | 9 | 42 | **0.476** | **0.557** | 7780 | 1272.7 kJ | 563.4 |
| cloud_standard | 116 | 359 | 0.553 | 0.673 | 157 | 268.2 kJ | 87.8 |
| edge_legacy | 178 | 121 | 0.903 | 0.989 | 845 | 1284.8 kJ | 13.7 |
| edge_lowend | 86 | 31 | 1.000 | 1.000 | 2044 | 939.1 kJ | 0 |

**无约束**的纯 DVFS 可及空间 = **974.6 kJ / 20 seeds = 48.7 kJ/seed = 系统能耗的 22.6%**。
SAC 实际 1.76 kJ/seed = 0.82% 系统能耗 = 无约束空间的 3.6%。

> **重要修正**：这 22.6% 忽略了可靠性约束。计入之后可及空间约为 **9.9%**，见 E12。
> 按可行空间算，SAC 拿到的是 **8.3%**。

### E4 · SAC 不降频的直接原因：z-score 归一化的自增强坍缩

`src/sac/hierarchical_low_sac.py`：`base_reward = w_E·z(energy_saving_J) − w_T·z(slack_time_cost)`，
两项都用 replay 滚动 std 归一化。换算成"延迟 1 秒值多少焦耳"的物理汇率
`(w_T/σ_T)/(w_E/σ_E)/slack`：

| episode | ρ_mean | σ_E (J) | σ_T | slack (s) | **隐含汇率** |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9488 | 25.07 | 0.00696 | 36.8 | 48.9 J/s |
| 101 | **0.7573** | 144.63 | 0.29187 | 36.2 | **6.9 J/s** |
| 301 | 0.9480 | 87.95 | 0.01190 | 37.8 | 97.7 J/s |
| 600 | **0.9600** | 60.13 | 0.00707 | 48.2 | **88.2 J/s** |

真实物理边际率 `dE/dt = P_s − 2P_d·ρ³`，在 ρ=1：
cloud_fast **−714 J/s**、edge_efficient **−75 J/s**、edge_legacy −64 J/s、edge_lowend **+24 J/s**。

对照 SAC 学到的 ρ：cloud_fast 0.873（714 ≫ 88，该降）、edge_efficient 0.956（75 < 88，不该降）、
edge_lowend 0.986。**SAC 对它拿到的奖励是最优的——错的是奖励的汇率，偏了 7–14 倍且在漂移。**

**正反馈坍缩**：ρ→1 ⟹ slack_time_cost→0 ⟹ σ_T→0 ⟹ 时间单价→∞ ⟹ ρ→1。
轨迹吻合：0.949 → 0.757（ep101）→ 单调爬回 0.960。

### E5 · 局部 3.72% ≠ 系统 0.82% 的真正原因：两层解耦 → 问题 4

强制按服务器 ρ*（按环境自身 `rho_safe` 截断）重放冻结的 ep300 上层：

| | ρ=1 | + SAC | + 解析 ρ* |
|---|---:|---:|---:|
| 系统能耗 | 215.45 kJ | 213.69 kJ | **318.52 kJ (+48%)** |
| Deadline | 96.75% | 97.00% | **40.00%** |
| ρ_mean | 1.000 | 0.962 | 0.742 |
| edge_efficient | 7780 任务 / 1272.7 kJ | 7585 / 1189.9 | **4689 / 463.1 kJ (−64%)** |
| cloud_fast | 308 / 544.0 | 301 / 471.4 | 409 / **414.6 (−24%)** |
| edge_legacy | 845 / 1284.8 | 929 / 1391.1 | **2275 / 3260.6 kJ** |
| edge_lowend | 2044 / 939.1 | 2155 / 987.5 | **3387 / 1763.3 kJ** |

**每台服务器上的局部节能全部兑现（edge_efficient −64%），但系统能耗涨 48%。**
降频撑大队列，冻结的上层被挤出 edge_efficient，2275 个任务落到 edge_legacy（299 W）、
3387 个落到 edge_lowend（119 W）。

**这就是"不像真正的分层强化学习"的根**：上层选服务器**全部基于 ρ=1 的估计**
（能耗参考 `primary_energy_ref_rho1`，状态特征 `estimated_execution_time` /
`slack_after_assign` / `rho_safe` 也都在 ρ=1 下算）。上层承诺服务器时看不见下层会怎么调频；
下层只能在既定服务器上"制造延迟"，而延迟被局部惩罚。
**这个分层有一个"下层什么都不做"的不动点**，当前策略正落在上面。

### E6 · `deadline_positive_fraction=0.3` 是死配置，但**不是**病因 → §3.8

`src/D3QN/reliability_aware_high.py:128` 的 `learn()` 覆写父类，三个头共用 `_uniform_indices()`；
父类的 `_deadline_indices_and_weights()`（30% 正样本配额 + Horvitz–Thompson 权重）**从未被调用**。

**但**实测在 `dense_normalized_makespan_miss_credit` 下 replay 中 deadline cost > 0 的
**天然比例是 0.648**。接回父类逻辑反而会把正样本**压到 0.30**。
这是应当清理的死配置项，**不应作为改进手段**。

父类的 `no_valid` 保护在三头版也缺失，但 `next_mask` 只在 `done` 时全零，而 `done` 行已被置零，
功能等价，不是 bug。

### E7 · 跨工作流 bootstrap 使 Q_D/Q_R 90% 是共模项 → §3.8

`src/D3QN/reliability_aware_high.py:314`：`done` 只在 buffer 最后一条为真；
工作流结束时其终端可靠性 cost 写在最后一条上，但该条 `done=False`，
`next_state` 属于**另一个**工作流。γ=0.99，episode 约 557 步。
任一步的 Q_D 是此后**所有工作流** deadline cost 的折扣和。
实测 `mean(Q_D)=0.881` 而动作极差 `0.090`。

### E8 · 五个困难种子的违约归因 → 问题 3

20 seeds 共 12 个违约工作流：

- **C/D 均值 1.062，最大 1.194** —— 全部是**边缘性违约**。
- **排队主导**：违约工作流内 Σ排队 / 响应时间 = **1.336**（排队 41–128 s vs 执行 78–148 s）。
- **12 例中 9 例最坏排队任务在 edge_efficient 上**，排队 6–12 s 而执行仅 1.7–5.9 s
  （**等待是执行的 1.5–4 倍**）。
- 违约工作流内服务器构成：edge_efficient 143/321 = **45%**。

→ 违约是**上层对 7 台 edge_efficient 的羊群式过度集中**造成的拥塞
（全局 70% 任务派给它们），不是错误的时间成本归因，也不是状态缺信息
（`wait_time` / `queue_count` / `queue_load` 及 EMA 都在状态里）——
是 λ_D 停在 1.15 让这些特征"被看到但不被重视"。

### E9 · Makespan 命名 → §3.9

ep300 / ρ=1，20 seeds 实测四个量互不相同：

| 量 | 值 |
|---|---:|
| average workflow response/makespan | 37.12 s |
| **sum of workflow response times**（现 `episode_makespan` 报的就是这个） | 742.37 s |
| maximum workflow makespan | 63.85 s |
| **global schedule makespan** | 232.62 s |

诊断工具已把四者分开命名输出。

### E10 · 拉格朗日超参数被 checkpoint 静默覆盖（新发现）

`DeadlineLagrangianController.load_state_dict`（高层与低层同构）会从 checkpoint 恢复
`dual_learning_rate` / `deadline_budget` / `lambda_max` / `warmup_episodes` / `window_workflows`。
**任何 warm-start 的实验都会静默继承 checkpoint 的对偶超参数并忽略配置。**

实测：把 S1 的 `dual_learning_rate` 从 0.001 改成 0.2 后，前 100 episodes 的 λ_D 仍是
1.142 → 1.144（与 0.001 一致）。这解释了为什么本项目历次分阶段重启都继承了坏掉的对偶学习率。

值得注意的是，**两个可靠性控制器本来就是对的**
（`src/sac/reliability_lagrangian.py`、`src/D3QN/reliability_aware_high.py` 只恢复 λ 和滑窗）；
只有两个 Deadline 控制器恢复了调参。这个不一致本身就是佐证。

#### 修复（已改根因）

先核实影响面：逐一比对所有 warm-start 配置与其 checkpoint 中保存的对偶超参，
**全部既有配置都与 checkpoint 一致**，只有新增的 `v2_*` 配置存在差异。
因此这个修复不改变任何已有可复现结果。

在两个 Deadline 控制器的 `load_state_dict` 中把 **状态** 与 **超参** 分开：

| 从 checkpoint 恢复（状态） | 从配置读取（超参） |
|---|---|
| `lambda_deadline` | `dual_lr`、`budget`、`lambda_max` |
| `workflow_cost_window` | `warmup_episodes`、`window_workflows`、`lambda_init` |

并且在二者不一致时**打印出来**，不再静默：

```
[deadline-lagrangian] config wins: dual_learning_rate checkpoint=0.001 config=0.2
[deadline-lagrangian] config wins: deadline_budget checkpoint=0.0 config=0.01
[deadline-lagrangian] config wins: lambda_max checkpoint=100.0 config=20.0
```

**乘子本身不重置**，warm start 仍从已达到的 λ 继续；
配置的 `window_workflows` 变小时只保留最近的若干条；
配置的 `lambda_max` 变小时对恢复的 λ 做裁剪；
`constraint_metric` 不匹配时仍保持原有的"丢弃语义不兼容的乘子"行为。

验证（原生 production trainer，完全不经过 v2 代码，warmup 结束后）：

| episode | λ_before | λ_after | excess | 实际步长 | 期望 `0.2×excess` |
|---:|---:|---:|---:|---:|---:|
| 51 | 1.14187 | 1.14987 | 0.040 | 0.00800 | 0.00800 |
| 53 | 1.15787 | 1.17387 | 0.080 | 0.01600 | 0.01600 |
| 55 | 1.19187 | 1.21187 | 0.100 | 0.02000 | 0.02000 |

**6 个 episode 走了 +0.082；修复前是 1200 个 episode 走 +0.057。**

回归测试：`tests/test_lagrangian_config_precedence.py`（8 个用例 / 14 个 subtest）。
`tests/test_hierarchical_sac.py` 中原有的
`test_low_lagrangian_controller_keeps_warmup_and_restores_state` 曾断言
`restored.dual_lr == 0.1`（即断言了这个缺陷本身），已改为断言修正后的契约。
新 trainer 里的 `ConfiguredLagrangianMixin` 从"静默重新施加"改为
"发现漂移就抛异常"的**校验守卫**。

### E11 · 低层对偶的约束信号恒为 0（新发现）

低层对偶约束的是 `_low_causal_deadline_misses`：只有当工作流违约**且**某任务在
ρ=1 参考可行时被实际 ρ 越界，才计一次。而该参考是在**已被撑大的队列**上重算的，
所以一旦 DVFS 拥塞了服务器，之后每次违约看起来都像上层放置错误。

已发布 SAC 运行实测：

| ep | 系统违约率 | 低层因果 cost | λ_low |
|---:|---:|---:|---:|
| 1 | 0.00 | 0.000000 | 10.0000 |
| 81 | **0.35** | **0.003546** | 10.0029 |
| 201 | 0.25 | 0.000014 | 10.0135 |
| 401 | 0.00 | 0.000000 | 10.0159 |
| 600 | 0.00 | 0.000001 | 10.0185 |

**系统 35% 违约时低层 cost 是 0.0035，λ_low 600 episodes 只走了 +0.0185。
低层对偶结构上无法调节 ρ。**

### E12 · 可靠性才是 DVFS 的真正约束（新发现，改写问题 5 的答案）

`src/simulator/reliability.py:28`：`λ(f) = λ_max · 10^(d·(f_max − f)/(f_max − f_min))`，
`f_min = 0.1·f_max`，`d = 1`。降频同时**放大失效率**并**拉长暴露时间**，
任务风险按 `10^((1−ρ)/0.9) / ρ` 增长。以实测基线（工作流可靠性 0.94192，
风险 0.0598；要求 0.9，预算 0.1054，允许增长 **1.761×**）投影：

| ρ | 风险倍数 | 投影工作流可靠性 | 满足 0.9？ |
|---:|---:|---:|:--:|
| 1.00 | 1.000 | 0.9419 | ✓ |
| 0.95 | 1.196 | 0.9309 | ✓ |
| 0.90 | 1.435 | 0.9177 | ✓ |
| **0.85** | 1.727 | **0.9018** | ✓（临界） |
| 0.80 | 2.085 | 0.8827 | ✗ |
| 0.70 | 3.078 | 0.8318 | ✗ |
| 0.50 | 7.188 | 0.6505 | ✗ |

**可行频率下界约 ρ ≈ 0.85**，而能耗最优 ρ* 是 0.40–0.48。
按 ρ=0.85（edge_lowend 保持 ρ=1，因为那里降频反而耗电）重算可及空间：

| 服务器类型 | E(0.85)/E(1) | 能耗 | 可省 |
|---|---:|---:|---:|
| edge_efficient | 0.8026 | 1272.7 kJ | 251.1 |
| cloud_fast | 0.7750 | 544.0 kJ | 122.4 |
| cloud_standard | 0.8335 | 268.2 kJ | 44.7 |
| edge_legacy | 0.9927 | 1284.8 kJ | 9.4 |
| edge_lowend | 1.0563（更差） | 939.1 kJ | 0（保持 ρ=1） |

**可靠性可行的 DVFS 空间 ≈ 427.6 kJ / 20 seeds = 21.4 kJ/seed = 系统能耗的 9.9%。**

实验证据：所有采用物理量纲低层奖励的臂在低层可靠性对偶被修好之前，
都冲到 ρ ≈ 0.70–0.83，可靠性成功率崩到 **1%–8%**：

| 实验臂 | ep | ρ | 平均工作流可靠性 | 可靠性成功率 |
|---|---:|---:|---:|---:|
| S2（物理奖励） | 200 | 0.7376 | — | **0.0100** |
| S3（从零耦合） | 420 | 0.7583 | 0.8523 | **0.0125** |
| S2ctl（z-score） | 380 | 0.9817 | — | 1.0000 |
| S3ctl（从零 z-score） | 620 | 0.9816 | 0.9441 | 1.0000 |

低层可靠性对偶 `LowReliabilityLagrangianController` 的
`reliability_dual_learning_rate` 出厂值是 **0.001** —— 与 E1/E11 同一个"冻结对偶"病灶的
**第三处**。在 ρ=1 时它无害（可靠性 0.937 > 0.9，成本恒为 0），
**一旦下层真的开始降频，它就是决定工作点的那个约束，却完全不动。**

改为 `reliability_lambda_init 5.0 / dual_lr 0.5 / lambda_max 200` 后，
λ_R^low 在 40 episodes 内从 5.0 升到 14.6，ρ 稳在 0.95–0.96，可靠性成功率回到 1.00；
同时**高层可靠性乘子 λ_R 首次开始移动**（1.0000 → 1.0682）。

### 关于问题 5（可靠性头的边际贡献）

推理期把 λ_R 置 0（等价于两头）在同 20 seeds 上：

| 指标 | λ_R=1（三头） | λ_R=0（两头） |
|---|---:|---:|
| Deadline | 0.96750 | 0.96750 |
| Reliability 成功率 | 1.00000 | 1.00000 |
| 平均工作流可靠性 | 0.94192 | 0.94190 |
| 能耗 | 215445.09 J | **215184.71 J（低 0.12%）** |
| 响应时间 | 37.118 s | 37.197 s |

尽管它改变了 18.9% 的贪心动作，**可靠性头没有任何可测收益，还多花 0.12% 能耗**。
λ_R 恒为 1.0（实际可靠性 0.937 > 要求 0.9，`reliability_cost` 恒为 0，
`src/D3QN/reliability_aware_high.py:49` 的对偶更新量恒为 0），
`Std_a(Q_R)` 从 ep300 的 14%·Std(Q_E) 衰减到 ep1160 的 6%，loss 已降到 1e-6。
**在 ρ=1 的轨迹上，可靠性约束不活跃，第三个头是一个正在消失的、略有害的扰动项。**

**但这个结论只在"下层不工作"的前提下成立。** 由 E12：一旦下层真的降频，
可靠性立刻成为**决定工作点的那个约束**（可行 ρ ≈ 0.85，而能耗最优 ρ* = 0.40–0.48），
它比 Deadline 更早生效。也就是说：

> 可靠性头之所以看起来没用，正是因为下层 SAC 没在工作。
> 它是这个分层结构在**真正开始节能之后**才被激活的约束。

这对论文是一个更强也更诚实的论述：三头不是"多加一个头"，
而是"能耗–时限–可靠性"三者在 DVFS 维度上真实竞争。
（训练期的双头/三头消融见第 3 节。）

---

## 2. 零训练成本的关键验证：λ_D 扫描

λ_D 只进入动作选择 `Q_E − λ_D·Q_D`，不参与训练。在**同一冻结的 ep300 网络**上扫描
（20 seeds，242000–242019）：

| λ_D | Deadline | 能耗 (kJ) | 响应 (s) | λ_D·Std(Q_D)/Std(Q_E) | 最差种子 | 100% 种子数 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0275 | 122.25 | 276.27 | 0.000 | 0.00 | 0 |
| **1.1516**（现状） | 0.9675 | 215.45 | 37.12 | 1.354 | 0.90 | 9 |
| **2.0** | **0.9950** | **229.85** | 34.98 | 2.227 | 0.95 | 18 |
| 3.0 | 0.9975 | 239.56 | 33.85 | 3.233 | 0.95 | 19 |
| 5.0 | **1.0000** | 252.36 | 32.82 | 5.217 | 1.00 | 20 |
| 8.0 | 1.0000 | 262.77 | 32.40 | 8.147 | 1.00 | 20 |
| 12.0 | 1.0000 | 264.50 | 32.37 | 12.130 | 1.00 | 20 |
| 20.0 | 1.0000 | 270.75 | 31.89 | 20.173 | 1.00 | 20 |

两个结论：

1. **Deadline ≥ 99% 用现有 ep300 网络就能达到，只需 λ_D ≈ 2。** 对偶回路从未走到那里，
   因为它的学习率小了 200 倍。这完整验证 E1，并回答问题 1。
   在 λ_D = 2 时三项验收条件同时满足：
   Deadline 99.50% ≥ 99% ✓、Reliability 100% ✓、能耗 229.85 kJ < 242.92 kJ（最好的对比算法 OOAC）✓。
2. **λ_D = 0（纯能耗头）时 Deadline 只有 2.75%、响应时间 276 s。**
   Deadline 头承担了几乎全部的可行性维持工作。

加上下层：

| λ_D | ρ 模式 | Deadline | 能耗 (kJ) | ρ_mean |
|---:|---|---:|---:|---:|
| 2.0 | ρ=1 | 0.9950 | 229.85 | 1.0000 |
| 2.0 | SAC ep460 | 0.9900 | 227.51 | 0.9603 |
| 2.0 | 解析 ρ* | 0.4725 | 329.63 | 0.7409 |
| 3.0 | ρ=1 | 0.9975 | 239.56 | 1.0000 |
| 3.0 | SAC ep460 | 0.9950 | 236.22 | 0.9592 |
| 3.0 | 解析 ρ* | 0.4875 | 335.81 | 0.7402 |

即使把上层调到可行区，现有 SAC 仍只贡献 2.3 kJ（1.0%），而无耦合的激进降频依然灾难性。

---

## 3. 最小修改与消融设计

全部新增，不覆盖任何已有配置或结果；不改动对比算法、服务器资源、工作流生成、
Deadline 定义或可靠性要求。

### 新增代码

| 文件 | 内容 |
|---|---|
| `tools/diagnose_triple_head_dvfs.py` | 只读诊断：三头动作维度指标、DVFS 空间、困难种子归因、四种 makespan、λ 覆盖扫描 |
| `src/sac/physical_dvfs_sac.py` | S2：`PhysicalDVFSLowSACAgent`，去掉 z-score，固定物理量纲 |
| `src/D3QN/dvfs_aware_state.py` | S3：`DVFSAwareHighStateEncoder`，每服务器 +3 维 DVFS 选项特征（347 → 425 维） |
| `tools/train_coupled_hierarchical_d3qn_sac.py` | 新 trainer：S2/S3 开关、`ConfiguredLagrangianMixin`（修 E10）、`dvfs_delay_share` 低层归因（修 E11） |
| `tools/accept_v2_checkpoints.py` | 双种子块选择/验收 |

### 对现有代码的改动（全部行为保持）

| 文件 | 改动 | 是否改变已有行为 |
|---|---|---|
| 17 个 `tools/*.py` + 1 个 test | py310 垫片加 `sys.version_info[:2] == (3, 10)` 守卫 | 否（3.10 下逐字不变） |
| `src/sac/hierarchical_low_sac.py` | 把 base reward 抽成可覆写的 `_base_reward()`，两处调用点合并 | 否（逐位等价，198 个测试通过） |
| `src/D3QN/hierarchical_d3qn_agent.py` | 加 `build_state_encoder()` / `build_low_sac_agent()` 两个工厂钩子 | 否（默认返回原类） |

### S1：让对偶回路真正工作（纯配置 + E10 修复）

`configs/v2_s1_active_dual.json`：`dual_learning_rate` 0.001 → 0.2，
`deadline_budget` 0.0 → 0.01（正好对应 99%），`lambda_max` 100 → 20。
其余与原配置逐字相同，含同一个 `initial_full_checkpoint`。

### S2：物理量纲的低层奖励

`r = w_E·(E(1)−E(ρ))/E(1) − w_T·Δt/max(slack, 0.05·D) − λ_low·deadline_cost − λ_R·rel_cost`

无任何 replay 统计量。`w_E = w_T = 1` 时驻点为
`ρ³ = [P_s + (w_T/w_E)·τ·(P_s+P_d)] / (2P_d)`，`τ = T_exec(1)/slack`，
在 slack 充裕时回到物理最优 ρ*，在 edge_lowend 上正确给出 ρ=1。

上层冻结在**原始 ep300**（与已发布 SAC 运行同一上层），干净隔离。
对照组 `v2_s2control_zscore_low_reward` 与 S2 **仅差 `agent.physical_low_reward` 一个开关**。
两臂都启用 E11 的 `dvfs_delay_share` 低层归因（受控）。

### S3：让上层看见 DVFS 选项（真正的分层）

每服务器 +3 维：

1. `ρ_econ = clip(ρ*_j, ρ_safe_j, 1)`
2. `tanh(E_j(ρ_econ)/E_median)` —— 与现有第 8 维 `tanh(E_j(1)/E_median)` 直接可比，二者之差就是服务器 j 的 DVFS 优势
3. `tanh(T_exec_j(1)·(1/ρ_econ − 1)/D_w)`

**上层奖励不变**：`reward_profile='ep740_lagrangian_v1'` 下
`energy_actual_for_reward` 已经是下层动作之后的**实际**任务能耗，
Q_E 头本来就收到 DVFS 信用——缺的只是观测。

选项语义：上层的动作是一个 option（服务器 + 其 DVFS 工作点），
其价值必须在**将要执行它的下层策略**之下评估。这是让两层互为最优响应的最小改动。
S3 与对照 `v2_s3control_scratch` 均从零训练、同超参，构成 2×2。

### 验收协议

- **选择块** 542000–542019，按 Deadline ≥ 0.99 → Reliability = 1.0 → 总能耗最低排序；
- **验收块** 242000–242019，与选择块**不相交**（原 §3.3 的配置中二者重合，仍是选择集偏差）；
- 每个 checkpoint 报告：ρ=1 上层结果、接入 SAC 的完整系统结果、
  SAC 同轨迹局部反事实节能、与 ρ=1 独立轨迹的**配对**系统节能及 95% CI。

结果见 `docs/V2_RESULTS_20260827.md`（训练完成后填写）。
