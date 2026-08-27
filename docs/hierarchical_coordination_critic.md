# D3QN-SAC 层次协同评价网络（HCC）

## 1. 要解决的问题

原框架按照以下顺序决策：

1. 上层 D3QN 选择服务器 `k`；
2. 下层 SAC 只针对已选服务器输出连续频率 `rho`；
3. 环境执行联合动作 `(k, rho)`。

最终能耗、排队时间和 Deadline 是否满足由两个动作共同决定，但原来的
`Q_H(s, k)` 无法显式区分“该服务器在 SAC 调频后是否仍然合适”。HCC
因此学习联合成本：

\[
Q_C(s,k,\rho)=\left(Q_C^E(s,k,\rho),Q_C^D(s,k,\rho)\right).
\]

## 2. 网络选择

HCC 使用两套独立的双头 MLP critic，每套网络均为：

- 输入：347 维上层状态 + 26 维服务器 one-hot + 1 维 `rho`，共 374 维；
- 共享骨干：两层 256 单元 MLP；
- 能耗头：预测归一化累计能耗成本；
- Deadline 头：预测累计工作流违约成本；
- 两套 critic 取逐项较大值，防止单个网络过度乐观。

能耗和 Deadline 不放进同一个输出头，以避免二者尺度变化时互相覆盖。

## 3. 训练目标

对真实执行的联合动作 `(k_t, rho_t)`，HCC 从现有 episode buffer 收集：

\[
c_t^E=E_t/1000,
\qquad
c_t^D=\text{workflow-terminal causal miss credit}.
\]

两个成本头分别使用 TD 目标：

\[
y_t^j=c_t^j+\gamma(1-d_t)Q_{C^-}^j
(s_{t+1},k_{t+1},\rho_{t+1}),\quad j\in\{E,D\}.
\]

损失使用 Huber loss，目标网络采用软更新。HCC 不改写 D3QN 或 SAC 的
环境 reward，也不把梯度传入二者，第一版先避免三个学习器形成循环反馈。

## 4. 动作协商

在需要选择服务器时，当前 SAC 对每个结构上有效的候选服务器确定性输出
一个候选频率 `rho_hat_k`。HCC 计算：

\[
q_k^C=-Q_C^E(s,k,\hat\rho_k)
-\lambda_H Q_C^D(s,k,\hat\rho_k).
\]

最终使用以下分数重排候选服务器：

\[
q_k^{final}=(1-\beta)z(Q_H(s,k))+\beta z(q_k^C).
\]

正式配置先收集 20,000 条 transition，只训练 HCC，不干预 D3QN；随后
在 40,000 条 transition 内把 `beta` 从 0 线性提高到 0.2。

## 5. 推荐训练方式

使用四阶段配置：

1. HEFT 固定，训练 SAC，同时预训练 HCC；
2. 冻结 SAC，训练 D3QN，HCC 向 D3QN提供联合动作评价；
3. 冻结 D3QN，微调 SAC，HCC适应新的频率分布；
4. 两层小学习率、低频率联合微调。

正式运行：

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_staged_hierarchical_d3qn.py --config configs\hierarchical_d3qn_sac_hcc_staged.json
```

## 6. 必须做的消融

使用相同训练 seed、固定 20 个 eval seed 比较：

- 原四阶段 D3QN-SAC；
- HCC 四阶段，`beta_final=0.1`；
- HCC 四阶段，`beta_final=0.2`；
- 仅能耗头；
- 仅 Deadline 头。

主要看 Deadline 成功率、系统总能耗、`rho`、HCC 改选率，以及改选动作相对
D3QN 原动作的实际能耗和 Deadline 差异。只有在固定种子评估中稳定改善，
HCC 才能被认为真正有效。
