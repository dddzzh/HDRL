# Proposed Method: current implementation cross-check

核对日期：2026-10-06。方法章节符号已按同日作者提供的最新稿统一。
本文档是给作者核对实现用的，不是论文正文。
本次只写论文与核对说明，没有修改训练代码、配置或已有论文。

## 1. 锁定版本

训练入口：`tools/train_coupled_hierarchical_d3qn_sac.py`。

当前完整方法使用以下 future_esp 配置：

- S26：`configs/v2_small_20edge_6cloud_dvfs_v3_future_esp.json`
- S37：`configs/v2_small_28edge_9cloud_dvfs_v3_future_esp.json`
- M37：`configs/v2_medium_28edge_9cloud_dvfs_v3_future_esp.json`
- M53：`configs/v2_medium_40edge_13cloud_dvfs_v3_future_esp.json`

配置选择的实际类是 `ConfiguredLagrangianReliabilityAgent`，不是入口文件中同样存在的 `CoupledDVFSHierarchicalAgent`。
原因：`physical_low_reward=false`、`dvfs_aware_high_state=false`、`enforce_config_lagrangian=true`。
这不影响未来关键路径与全局紧迫负载特征的启用；这两类信息已在当前公共状态管道中。

不属于本次方法：Multi-Critic、C3C/HCC、子 Deadline 主约束、物理比例下层奖励、CTDE 全局下层 Critic、冻结分阶段训练。

## 2. 正文结构与真实实现

| 论文小节 | 真实路径 | 论文应当表达的内容 |
| --- | --- | --- |
| A 分层框架 | `tools/train_coupled_hierarchical_d3qn_sac.py`、`tools/train_hierarchical_d3qn.py` | 当前任务先选服务器，再选连续频率；同一批交互轨迹训练两层 |
| B Deadline 状态 | `src/D3QN/state.py`、`src/ppo/hierarchical_state.py`、`src/simulator/edge_cloud_env.py` | 剩余关键路径预算、候选完成时间和全局紧迫负载进入观测 |
| C 约束放置 | `src/D3QN/reliability_aware_high.py`、`src/D3QN/dual_head_constrained.py` | 三套独立 Dueling Q；能耗、Deadline、可靠性分别学习 |
| D 责任频率控制 | `src/sac/hierarchical_low_sac.py`、`src/sac/reward.py`、`tools/train_hierarchical_d3qn.py` | 普通 SAC 双软 Q；工作流固定调度反事实信用，包括队列外部性 |
| E 联合更新 | 上述训练器、`src/D3QN/deadline_lagrangian.py`、`src/sac/deadline_lagrangian.py` | 完整轨迹反馈后更新；上下层 Deadline 统计不同，可靠性更新也不同 |

## 3. 上层到底训练什么

上层 action 只有服务器选择，不选择任务、不输出频率。任务顺序由所有算法共用的 `smallest_workload` 队列规则给出。

“三头”是概念上的三类价值估计，代码是三套独立网络、三套优化器和对应目标网络，不是一个共享主干接三个输出头。论文不能声称共享表示带来的收益。

动作得分：

\[
Q_E-\mu_D^H[Q_D]_+-\mu_R^H Q_R.
\]

只有 Deadline Q 在选择和目标值中做非负截断，可靠性 Q 没有这个截断。真实服务器都可选；动作 mask 不是 Deadline 可行性过滤。

### 能耗头

\[
r_E^H=\frac{\bar E_i-E_i}{n\bar E_i},
\qquad\bar E_i=\operatorname{median}_m\widehat E(i,m,1).
\]

参考尺度是当前任务在所有真实服务器上满频能耗的中位数；实际项是 SAC 决定频率后执行的能耗。
因此上层并非完全忽视下层。该归一化奖励是学习代理指标，不能声称其累计值与物理总能耗完全等价。

### Deadline 头

先形成：

\[
d_t=\Delta T_{w,t}/D_w+\beta_q q_t/(nD_w).
\]

再加工作流最终超时信用和整次轨迹 ESP 失败信用：

\[
g_{D,t}^H=d_t+\beta_{\mathrm{miss}}\mathbb I(T_w>D_w)\omega_t
                +\beta_{\mathrm{miss}}\mathbb I(\mathrm{SR}_e<\Gamma)\nu_t.
\]

- `q_t` 是执行开始减去所选服务器的基础数据就绪时间，反映服务器竞争等待，不是整个 `WT(i)`，也不是执行时间。
- 工作流失败信用 `omega` 按该工作流的 dense 成本比例分配；队列项也进入分配权重。
- ESP 失败信用 `nu` 按“所选服务器的预测完成时间 − 最快候选预测完成时间”分配，预测使用满频执行与剩余关键路径。
- 两个失败项共用 `deadline_miss_penalty`，当前默认 1.0。`objective.deadline_violation_weight=3.0` 不能直接当成这个系数。
- 当前 `deadline_future_regret_weight` 默认 0，因此没有额外的稠密未来 regret 项；但 regret 仍用于 ESP 失败信用分配。
- 没有启用子 Deadline 项。
- `reward.py` 的标量 `high_training_reward` 不是三套网络实际共用的训练奖励。真正的头目标在 `remember_from_high_buffer` 中重新组装。

队列系数由初始校准窗口估计后固定，目标 share 为 0.20，不是每轮重新调 beta，也不能保证后续占比一直为 20%。

### 可靠性头

最后结果为：

\[
\ell_w^R=[\log R_w-\log\mathcal R_w]_+.
\]

该成本挂到该工作流最后一个任务的 transition；其他任务当前可靠性成本为零。
它不是成功/失败分类头，也不是逐任务耗尽可靠性预算的稠密头。

### 回放链

保持整个系统交互的时间顺序，下一状态可能是其他工作流的任务。
只在整个 rollout 最后一个 transition 设置 done，不是每个工作流最后一个任务都 done。
这样保留不同工作流共享服务器队列的关联。
旧稿里“按工作流重建上下层 next_state”不能沿用。

## 4. 当前 State 的实质

上层原始数值布局为 4 个任务特征、每服务器 12 个特征、服务器 mask 和 8 个全局特征；固定服务器数量 M 下是 `13*M+12`。
其中一个未使用的历史特征恒零，不应写成论文贡献。

下层配置中沿用 `local_9` 名称，但实际编码器来自 `src/ppo/hierarchical_state.py`，现在输出 **11 维**：原先 9 个局部特征，加全局平均与最严重的关键路径压力。
Actor 和两个 Critic 都使用这 11 维，不是“局部 Actor + 全局 Critic”。

剩余关键路径安全状态：用 upward rank 减去当前任务平均执行时间，保留后续计算/通信长度，从工作流总 Deadline 中扣掉这部分后再看当前任务执行预算。
`rho_safe` 是按该预算估计的频率提示。它进入 State，但当前 action 不被强制提升到 `rho_safe`。

全局紧迫负载：对所有已经到达但尚未完成的工作流，算“剩余关键路径时长 / 剩余 Deadline”，再提取总体压力和最严重压力。
这和只看服务器忙碌比例不同，也不是 Reward 项。

## 5. 下层到底怎么承担责任

下层是一个随机 Actor + 两个普通 SAC 软 Q，不是能耗/Deadline/可靠性三类下层 Critic。
两个 Q 都学习同一个合成回报，只是取较小值抑制高估。

### 真实动作

Actor 先在全局 `[0.2,1]` 产生绝对频率比例，再由执行端截断到每台服务器的能效下界。
不是 residual SAC，不输出 `rho_base+delta`，也不是直接在每服务器 `[rho_ee,1]` 上仿射生成动作。

**需要关注的一处实现差异：** 回放 action 是截断后的执行 rho，Actor/目标策略采样与 log-prob 用的是截断前的 proposal。
本文正文明确区分两者，没有写成已经实现了逐服务器可行区间重参数化。
若下界经常被触发，这个差异可能影响 SAC 学习；本次没有改动它，也没有运行绑定率诊断。

### 当前奖励

\[
r^L=Z_E(E_i(1)-E_i(\rho_i))-Z_h(h_t)
     -\mu_D^L g_D^L-\mu_R^L g_R^L.
\]

能耗差和局部延迟项分别按 Replay 均值/标准差归一化并截断；两类约束成本不做这套 z-score。
所以当前不是“只有能耗奖励，加两个拉格朗日惩罚”，还存在权重 1.0 的固定局部时间 shaping。
它不是论文优化目标增加了第二个目标，但不能从方法公式中删掉。

### Deadline 反事实

完成后固定真实 placement、服务器执行顺序、通信延迟和原决策释放时刻，重放两类场景：

1. 所有工作流恢复满频，计算每个受害工作流可归于 DVFS 的超时严重度差；限制到 `[0,1]`。
2. 每次只恢复一个来源工作流满频，量它能减轻各个受害者多少超时。

受害者成本按各来源的边际减轻程度分配，再按该来源各任务的执行 slowdown 分给 SAC transitions。
若单来源边际都为零，退回来源总 slowdown 比例分配，避免交互效应导致成本消失。
因此来源 A 的降频拖慢来源 B 的任务时，A 可以收到外部责任。

注意：反事实仍保留原决策释放时刻，所以是固定调度条件下的责任代理量，不是全局最优因果贡献，也不是 Shapley 值。

### 可靠性反事实

当前使用的是“同一任务在实际频率与满频下的**投影工作流可靠性违约差**”，其他已安排任务与剩余任务估计不变。
不是 `reliability_budget_share_spent`：后者确实存在于 info，但在 `physical_low_reward=false` 的当前奖励中没有进入 SAC 惩罚。
二者不能混写，否则会虚构“每次降频都有可靠性梯度”。

## 6. 四个 multiplier 的更新并不相同

| 乘子 | 实际观察量 | 当前更新规则与预算 |
| --- | --- | --- |
| 上层 Deadline | 最近 60 次 rollout 的 `1{SR<0.90}` 均值 | PI；预算 0.10；积分步长 0.1，比例增益 3，上限 6 |
| 下层 Deadline | 每次 rollout 各受害工作流 DVFS-attributable lateness 的均值，再取最近 40 次的最差 25% 均值 | PI；预算 0.08；积分步长 0.2，比例增益 25，上限 8 |
| 上层 Reliability | 最终 workflow log-reliability 正缺口 | 每个完成工作流增加 `0.001*缺口`；上限 100；不下降 |
| 下层 Reliability | 最近 100 个完成工作流的真实可靠性违约指示量均值 | 积分更新，减预算 0.005，步长 0.5，上限 50；满足预算时可下降 |

下层配置名字虽为 `workflow_miss_rate`，当前实际喂入 Deadline controller 的是反事实严重度差，不是原始二元 miss rate。
更新使用受害者成本，SAC 回放使用分配给来源的成本；两套数组不能混为一谈。

上层 PI 追求的训练条件是平均轨迹失败频率不超过 10%，即训练目标 ESP@90 >=90%。
Checkpoint 规则要求两块验证的 ESP@90 都达到 100%，可靠性达标率 100%，再选能耗低的。
训练预算和模型选择要求不是同一数值，论文不能写成当前对偶直接以 ESP=100% 为训练约束。

## 7. 更新流程

1. 收集完整环境交互；每次先 D3QN 选服务器，再 SAC 选频率。
2. 获取最终 workflow 结果，补齐时间增量与可靠性成本。
3. 校准队列系数（仅校准窗口），组装上层 dense/miss/ESP 成本。
4. 固定调度反事实重放，组装下层 source credit 和 victim dual cost。
5. 写入各自 Replay，上层 3 个 Q 更新 5 次，下层 Actor/双 Q 更新 10 次，并软更新目标网络。
6. 两层 Deadline PI 与下层可靠性对偶基于最终统计更新，供后续决策使用；上层可靠性更新在上层存储完成成本时执行。

这是同一环境联合学习，不是分阶段冻结，也不是新增第三个协同网络。

## 8. 论文写作边界

保留：状态设计、独立价值分解、稠密 + 最终 + ESP 的信用、固定调度反事实外部性、各乘子的观察量与公式。

移到实验设置或附录：网络宽度、Replay 容量、学习率、训练/验证种子、checkpoint 策略、校准起止轮数。

不写入正文：历史失败版本、模块文件名、日志字段、配置兼容别名、没有启用的分支。

不声称：严格硬安全保证、理论全局最优、奖励与物理总能耗严格等价、投影可靠性成本处处稠密、统一四个乘子机制、SAC 对 ESP 提升的未经消融证实的效果。

引用键沿用 `docs/references.bib` 的 `wang2016dueling`、`vanhasselt2016`、`haarnoja2018`、`stooke2020`；它们分别支持标准 Dueling、Double-Q、SAC 与 PI/PID 对偶控制背景，不用来证明本文自有机制有效。

## 9. 与作者最新稿的符号统一

基准：附件 `86944893-58fa-431f-9111-7d76e8d9c34a/已粘贴的文本.txt`。
附件原文未改动；本次只修改方法章节和本核对说明，不改变算法。

| 含义 | 统一后的符号 | 处理 |
| --- | --- | --- |
| 工作流任务及数量 | `\mathcal N`、`n` | 沿用正文局部任务编号，移除额外 `n_w`；服务器求和不再用 n 作索引 |
| 实际时间、能耗、频率 | `ET_i`、`RT(i)`、`WT(i)`、`ST(i)`、`FT(i)`、`E_i`、`f_i=\rho_i F_{m_i}` | 实际量不加候选服务器索引 |
| 候选满频时长 | `\widehat{ET}(i,m)=L_i/F_m` | 与实际执行时长区分 |
| 候选能耗 | `\widehat E(i,m,\rho)` | 是原能耗公式在候选 m、rho 下的评价，不是新能耗模型 |
| 工作流完成时间 | `T_w` | 不使用 C_w |
| 可靠性要求与实际值 | `R_w`、`\mathcal R_w` | R_w 不再表示实际可靠性 |
| 硬件最低比例 | `\underline\rho_m` | 完全沿用作者定义 |
| 能效最优比例 | `\rho_m^{\mathrm{ee}}` | 截断闭式解放在 `POWER_ENERGY_MODEL_CURRENT.tex` 的能耗模型；方法只引用其含义 |
| 单次调度成功率与服务目标 | `\mathrm{SR}_e`、`\Gamma`、`\mathrm{ESP}_\Gamma`、`\eta` | SR_e 仅是正文 SR 的逐次调度取值，不另造 S_k 或 xi_e |
| 故障率与灵敏度 | `\lambda_m(f)`、`\kappa_m` | 保留给物理模型；不用于对偶或 CVaR 参数 |
| 对偶与折扣 | `\mu_D^H,\mu_R^H,\mu_D^L,\mu_R^L`、`\delta_h,\delta_L` | 不占用故障率 lambda 与成功率门槛 Gamma |
| 尾部比例 | `\vartheta` | 原 kappa 改名，避免混淆故障率灵敏度 |
| 选择损失、失败系数、标准化截断 | `\Delta_t^{\mathrm{sel}}`、`\beta_{\mathrm{miss}}`、`b_Z` | 不占用到达时间 a、服务器索引 m、通信时间 c |

下层可靠性投影现在写为频率的函数 `\widehat{\mathcal R}_{w,t}(f)`，实际/满频对比为 `f_i` 与 `F_{m_i}`，与作者故障率模型的输入一致。

### 2026-10-06 方法组织调整

参考多智能体分层论文的逐层组织方式，而非借用其三智能体结构。当前正文顺序为：

1. POMDP 元组及逐项定义，再衔接分层框架；按作者提供图片的格式，保留两层条件式动作，不复用任务集合 N、服务器集合 M、可靠性 R 或成功率门槛 Gamma。
2. 上层：State → Action → Reward/constraint costs → Double-Q targets/Huber loss。
3. 下层：State/action → Reward/responsibility credit → SAC targets/actor loss。
4. 联合训练与四个乘子的更新。

能效闭式解移到能耗模型，文献沿用作者第一篇的 Xie et al. (2017) 和 Zhu--Aydin (2006)，BibTeX 已通过 DOI 记录核对补入 `references.bib`。本次仅改论文文件，不修改训练代码、状态维度、奖励、动作或配置。

### 最新稿本身的三处衔接问题（仅指出，未改附件）

1. 删除重复的 `\subsection{Frequency-Dependent Reliability}`：上一节已完整定义相同模型。重复段重新引入未定义的 `R_{wi}`、`m_{wi}`、`\rho_{wi}`、`\tau^{\mathrm{exe}}_{wi}`、`\mathcal V_w`，且把 R_w 改成实际可靠性，与前文冲突。
2. 符号表最后一行仍是 `\xi_e,\mathrm{ESP}_{\gamma}`，应改为 `\mathrm{SR},\Gamma,\mathrm{ESP}_{\Gamma},\eta`，对应最终 Problem Formulation。eta 是达到所需成功率 Gamma 的最低服务满足概率，需要文字定义。
3. Power Model 的频率式后补硬件范围及最低频率对应关系：

```latex
\begin{equation}
    f_i=\rho_i F_{m_i},\qquad
    \underline\rho_{m_i}\leq\rho_i\leq1.
\end{equation}
```

并注明 `F_m^{\min}=\underline\rho_m F_m`。删除重复可靠性小节后，这个关系不能随之丢失。
