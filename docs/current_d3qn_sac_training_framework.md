# 当前 D3QN + SAC 分层训练框架说明

> 文档依据当前仓库实际生效的代码与 `configs/hierarchical_d3qn_sac.json` 编写。用途是准确描述方法，并交给其他模型检索和筛选公平的对比算法。

## 1. 一句话概括

当前方法是一个两层、联合训练、off-policy 的分层调度框架：

- 上层 D3QN：为环境指定的当前 ready task 选择一个主执行服务器。
- 下层连续 SAC：在上层选定服务器后输出连续 DVFS 动作，决定归一化运行频率。
- 优化重点：降低系统总能耗，同时通过 reward 惩罚 deadline 和 reliability 违反。
- 当前不使用安全强化学习：没有拉格朗日乘子、CPO/P3O 约束更新、安全 shield 或按约束过滤动作。
- 当前不使用主备复制：`enable_backup_duplication=false`，每个任务只在一个主服务器执行。
- 当前所有服务器共用一个下层 SAC 网络，并不是每台 edge/cloud 各自维护一个独立 SAC。

总体决策过程：

```text
环境产生 ready tasks
        |
        v
环境按最小 workload 排序，取 ready_tasks[0]
        |
        v
上层 D3QN 选择主服务器 k
        |
        v
共享下层 SAC 根据“当前任务 + 服务器 k”输出 u in [0,1]
        |
        v
u 映射为实际频率比例 rho，环境执行任务
        |
        +--> 上层 reward：时间/能耗/可靠性
        +--> 下层 reward components：绝对节能量/时间增加/deadline/reliability
        |
        v
episode 结束后，上下层分别从 replay buffer 更新 10 次
```

## 2. 当前默认实验设置

| 项目 | 当前值 |
|---|---:|
| 工作流数据 | Pegasus |
| 每个 episode 工作流数 | 10 |
| 每个工作流任务数 | 20 |
| 每个 episode 决策步数 | 约 200 |
| 移动设备数 | 5 |
| 到达过程 | multi-device Poisson |
| 到达率 | 0.05 |
| 工作流可靠性要求 | `Rreq=0.9` |
| deadline 类型 | medium |
| edge 数量 | 20 |
| cloud 数量 | 6 |
| 服务器总数 | 26 |
| backup duplication | 关闭 |
| 训练方式 | D3QN + SAC 联合训练 |
| 更新方式 | episode-end replay |
| 总 episode | 1500 |

medium deadline 的生成方式为：

\[
D_w = \kappa_w M_w^{\mathrm{HEFT}},
\qquad \kappa_w\in\{2,3\}.
\]

其中 `medium` 情况以 0.5 概率选择 2 倍 HEFT reference makespan，否则选择 3 倍。绝对 deadline 为：

\[
d_w^{abs}=a_w+D_w,
\]

其中 \(a_w\) 是工作流到达时间。

## 3. 环境中的任务顺序

上层 D3QN 当前不是同时选择“任务 + 服务器”。环境先收集所有 ready tasks，然后按以下顺序排序：

1. workload 从小到大；
2. arrival time；
3. workflow id；
4. task id。

上层始终处理排序后的 `ready_tasks[0]`，只决定这个任务放到哪个服务器。因此当前问题准确地说是：

> 固定 ready-task dispatch rule 下的服务器选择 + DVFS 控制。

如果某个对比算法同时优化任务优先级和服务器，那么它的动作空间比当前方法更强，不能直接作为完全公平的同设置对比。

## 4. 上层 D3QN

### 4.1 上层 state

默认 26 台服务器时，上层扁平 state 维度为：

\[
3 + 26\times21 + 26 + 5 = 580.
\]

组成如下：

#### A. 当前任务特征：3 维

| 序号 | 特征 | 处理方式 |
|---:|---|---|
| 1 | workload cycles | \(\tanh(L/10^{10})\) |
| 2 | 剩余 deadline 比例 | \((d^{abs}-t)/D\)，截断到 `[-10,10]` |
| 3 | 工作流任务数 | \(N/\text{max_ready_tasks}\)，当前分母为 32 |

编码器内部会构造最多 32 个 ready task 的特征和 task mask，但 D3QN 的最终扁平向量只放入 `ready_tasks[0]` 的 3 维任务特征。其他 ready tasks 只通过全局统计间接体现。

#### B. 每台服务器特征：21 维

每台服务器相对于当前任务计算一组 task-server pair 特征：

| 序号 | 特征 | 归一化/含义 |
|---:|---|---|
| 1 | 当前等待时间 | \(\tanh(wait/D)\) |
| 2 | 队列任务数 | \(\tanh(q/32)\) |
| 3 | queue load | \(\tanh(load/D)\) |
| 4 | queued processing time | \(\tanh(T_q/D)\) |
| 5 | \(\rho=1\) 执行时间 | \(\tanh(T_{exec}^{1}/D)\) |
| 6 | 预计开始等待 | \(\tanh((AST-t)/D)\) |
| 7 | 预计完成响应时间 | \(\tanh((AFT-a_w)/D)\) |
| 8 | 分配后的 slack | \((d^{abs}-AFT)/D\)，截断 `[-10,10]` |
| 9 | safe execution budget | 除以 \(D\)，截断 `[-10,10]` |
| 10 | communication time | \(\tanh(T_{comm}/D)\) |
| 11 | 诊断量 `rho_safe` | 截断到 `[0,1]` |
| 12 | `low_action_feasible` | 0/1 |
| 13 | bandwidth | \(\tanh(B/10000)\) |
| 14 | 最大频率 | 当前服务器 \(f_{max}\) / 所有服务器最大 \(f_{max}\) |
| 15 | 最大总功率 | \(\tanh((P_s+P_d)/250)\) |
| 16 | 基础 failure rate | \(\tanh(\lambda/10^{-3})\) |
| 17 | wait-time EMA | \(\tanh(wait_{ema}/D)\) |
| 18 | queued-time EMA | \(\tanh(T_{q,ema}/D)\) |
| 19 | queue-count EMA | \(\tanh(q_{ema}/32)\) |
| 20 | recent primary assignments | \(\tanh(n_p/32)\) |
| 21 | recent backup assignments | \(\tanh(n_b/32)\)；当前 backup 关闭，通常为 0 |

说明：当前环境没有把一个独立的 `communication_time` 字段直接写入 server state，因此第 10 项通常为 0；传输影响仍会通过 `estimated_ast` 和 `estimated_finish_time` 间接进入 state。

#### C. server mask：26 维

- 真实服务器位置为 1；
- padding 位置为 0；
- 当前 26 台服务器时不存在 padding。

这个 mask 只用于屏蔽不存在的 padding 动作，不按 deadline、reliability、服务器相似性或其他规则过滤真实服务器。

#### D. 全局特征：5 维

| 序号 | 特征 |
|---:|---|
| 1 | ready task 数 / 32 |
| 2 | 当前服务器数 / max_servers |
| 3 | 所有 ready task 的平均剩余 deadline 比例 |
| 4 | 所有 ready task 的最小剩余 deadline 比例 |
| 5 | busy-server ratio + mean server wait ratio |

### 4.2 上层 action

上层离散动作是：

\[
a_t^H=k,\qquad k\in\{0,1,\ldots,25\},
\]

即选择一个主服务器索引。动作不包含：

- task index；
- backup server；
- DVFS frequency；
- offload/不 offload 的额外二元变量。

训练时使用 epsilon-greedy：

- 初始 \(\epsilon=1.0\)；
- 最小 \(\epsilon=0.1\)；
- 每次成功执行一次梯度更新后减少 0.00018；
- eval 时 \(\epsilon=0\)，使用最大 Q 值动作。

当前 Q 输出头固定为 26 个位置相关动作，因此 checkpoint 不能直接零样本扩展到超过 26 台服务器。

### 4.3 D3QN 网络

- 输入维度：580；
- 两层 MLP，每层 256，ReLU；
- dueling head：一个 value head 和一个 26 维 advantage head；
- Q 值：

\[
Q(s,a)=V(s)+A(s,a)-\frac{1}{|A|}\sum_{a'}A(s,a').
\]

- Double-DQN target：online network 选择下一动作，target network 评价该动作；
- target network 软更新，\(\tau=0.005\)；
- replay size：50000；
- batch size：256；
- learning rate：0.0003；
- discount：0.99；
- gradient clipping：1.0；
- 每 episode 最多更新 10 个 minibatch。

### 4.4 上层 reward

设当前任务属于工作流 \(w\)，工作流任务数为 \(N_w\)，相对 deadline 为 \(D_w\)。当前已经完成/调度任务形成的 partial makespan 为：

\[
M_{w,t}=\max_i AFT_i-a_w.
\]

#### 时间部分

代码中的 `high_makespan_reward` 实际是当前 slack reward：

\[
r_{time,t}^H=
w_T\frac{D_w-M_{w,t}}{D_wN_w},
\qquad w_T=0.5.
\]

它在 deadline 内通常为正，超过 deadline 后变成负数。

#### 能耗部分

先计算当前任务在所有服务器、\(\rho=1\) 时的能耗，取其中位数：

\[
E_{scale,t}=\operatorname{median}_k E_{t,k}^{\rho=1}.
\]

上层能耗奖励为：

\[
r_{energy,t}^H=
w_E\frac{E_{scale,t}-E_{actual,t}}{E_{scale,t}N_w},
\qquad w_E=0.5.
\]

这里的 \(E_{actual,t}\) 是上层服务器选择和下层 SAC 频率共同产生的实际主任务能耗。因此上层训练目标会随下层 SAC 策略变化，存在联合训练的非平稳性。

#### reliability penalty

工作流当前可靠性乘积为 \(R_{w,t}\)，使用 log-risk：

\[
L_{w,t}=-\log R_{w,t},\qquad B_w=-\log R_w^{req}.
\]

归一化违反程度：

\[
v_{w,t}=\frac{\max(0,L_{w,t}-B_w)}{B_w}.
\]

可靠性惩罚为：

\[
p_{rel,t}^H=
5\cdot\mathbf{1}[v_{w,t}>0, v_{w,t-1}=0]
+2\max(0,v_{w,t}-v_{w,t-1}).
\]

写入 high rollout buffer 的基础 reward：

\[
r_{buffer,t}^H=r_{time,t}^H+r_{energy,t}^H-p_{rel,t}^H.
\]

#### 写入 D3QN replay 时的 deadline cost

partial makespan 增量为：

\[
\Delta M_{w,t}=\max(0,M_{w,t}-M_{w,t-1}).
\]

真正用于 D3QN TD target 的 reward 还会减去：

\[
p_{deadline,t}^H=
1.0\frac{\Delta M_{w,t}}{D_w}.
\]

因此最终上层学习 reward 是：

\[
\boxed{
r_t^H=
0.5\frac{D_w-M_{w,t}}{D_wN_w}
+0.5\frac{E_{scale,t}-E_{actual,t}}{E_{scale,t}N_w}
-p_{rel,t}^H
-\frac{\Delta M_{w,t}}{D_w}
}
\]

需要注意：上层 deadline 项不是“首次越过 deadline 才给固定惩罚”，而是每一步都惩罚归一化 partial-makespan 增量。

## 5. 下层连续 SAC

### 5.1 当前部署方式

当前代码只创建一个共享的下层 SAC：

```text
20 edge + 6 cloud
        |
        +--> 共享同一个 actor
        +--> 共享 critic 1 / critic 2
        +--> 共享 replay buffer
```

服务器差异通过选中服务器的 state 特征输入。同一个网络学习对所有 edge/cloud 的频率控制策略。Server 类内部没有独立 agent 字段。

### 5.2 下层 state：9 维

| 序号 | 特征 | 处理方式 |
|---:|---|---|
| 1 | workload | \(\tanh(L/10^{10})\) |
| 2 | \(\rho=1\) 执行时间 | \(T_{exec}^{1}/D_w\)，截断 `[0,10]` |
| 3 | 预计开始前等待 | \((AST-t)/D_w\)，截断 `[0,10]` |
| 4 | 当前剩余 deadline | \((d_w^{abs}-t)/D_w\)，截断 `[-10,10]` |
| 5 | safe execution budget | 除以 \(D_w\)，截断 `[-10,10]` |
| 6 | `rho_safe` | 截断 `[0,1]` |
| 7 | static power | \(\tanh(P_s/100)\) |
| 8 | max dynamic power | \(\tanh(P_d/250)\) |
| 9 | workflow reliability requirement | 当前通常为 0.9 |

当前下层 state 没有显式包含：server id、server type、bandwidth、基础 failure rate、当前工作流已经累计的可靠性乘积。`rho_safe` 当前主要由 deadline 可行性计算，不包含 reliability 所需的最低频率。

### 5.3 下层 action

SAC actor 输出一维连续动作：

\[
u_t=\frac{\tanh(z_t)+1}{2}\in[0,1].
\]

它是归一化控制量，不直接等于最终频率比例。实际执行的 DVFS 比例为：

\[
\rho_t=\rho_{min}^{hw}+(1-\rho_{min}^{hw})u_t.
\]

当前 simulation 的全局 `rho_min=0.2`，因此通常：

\[
\rho_t=0.2+0.8u_t\in[0.2,1].
\]

实际频率：

\[
f_t=\rho_t f_{max,k}.
\]

虽然 state 中包含 `rho_safe`，但当前 reward-only 版本不会执行：

\[
\rho\leftarrow\max(\rho,\rho_{safe}).
\]

所以 deadline/reliability 不会在动作执行前得到硬保证。

### 5.4 下层 reward 原始分量

所有 reference 都是在“上层已经选定相同服务器”的前提下，把频率设为 \(\rho=1\) 重新估计得到。因此下层 reward 主要评价 DVFS，不评价服务器选择。

#### A. 绝对节能量

\[
x_E=E_{\rho=1}-E_{\rho}.
\]

单位是 J。当前用于训练的是绝对节能量，不是节能比例。

日志仍额外记录：

\[
\text{saving ratio}=\frac{E_{\rho=1}-E_{\rho}}{E_{\rho=1}}.
\]

#### B. 时间增加量

\[
x_T=
\frac{\max(0,AFT_{\rho}-AFT_{\rho=1})}{D_w}.
\]

这是当前任务最终 AFT 相对同服务器满频 reference 的增加量，再除以工作流 deadline；不是整个 episode global makespan 的增加量。

#### C. deadline violation 增量

任务的最晚完成时间直接受工作流 deadline 约束；IoT 入口上传和 exit-task 最终下载均位于当前调度边界之外。定义：

\[
v_D(AFT)=\frac{\max(0,AFT-task\_latest\_finish)}{D_w}.
\]

下层使用动作相对 \(\rho=1\) reference 新增加的违反程度：

\[
x_D=\max(0,v_D(AFT_{\rho})-v_D(AFT_{\rho=1})).
\]

首次由 reference 可行变成实际动作违反时：

\[
c_D=\mathbf{1}[v_D(AFT_{\rho})>0,\ v_D(AFT_{\rho=1})=0].
\]

#### D. reliability violation 增量

预测工作流最终 log reliability 时使用：

- 已执行任务的真实 reliability；
- 当前候选任务在该 \(\rho\) 下的 reliability；
- 未执行任务采用 `remaining_task_reliability_estimate=0.999`。

设预测 log reliability 为 \(\log \hat R_w\)，要求为 \(R_w^{req}\)，归一化违反程度为：

\[
v_R=\frac{\max(0,\log R_w^{req}-\log\hat R_w)}{-\log R_w^{req}}.
\]

下层分量：

\[
x_R=\max(0,v_R^{actual}-v_R^{ref}).
\]

首次从 reference 满足变成实际动作不满足时 \(c_R=1\)。

### 5.5 replay-buffer z-score 与最终下层 reward

Replay buffer 存储四个原始分量：

\[
(x_E,x_T,x_D,x_R).
\]

每个 episode 更新前，使用当前整个下层 replay buffer 分别计算：

\[
z_j=\operatorname{clip}\left(
\frac{x_j-\mu_j}{\max(\sigma_j,0.05)},-3,3
\right).
\]

最终 SAC reward 为：

\[
\boxed{
r_t^L=
z_E-z_T
-2\operatorname{ReLU}(z_D)
-2\operatorname{ReLU}(z_R)
-5c_D-5c_R
}
\]

固定权重：

| 分量 | 权重 |
|---|---:|
| 绝对节能量 | +1 |
| 时间增加 | -1 |
| deadline severity | -2 |
| reliability severity | -2 |
| deadline crossing | -5 |
| reliability crossing | -5 |

当前权重不是自动调节的。z-score 均值和标准差会随 replay buffer 数据变化，但 reward 权重保持固定。

### 5.6 SAC 网络与更新

- actor：两层 128 MLP，输出 Gaussian mean 和 log-std；
- `log_std` 截断到 `[-5,2]`；
- tanh-squashed Gaussian，再映射到 `[0,1]`；
- twin critics；
- twin target critics；
- learning rate：0.0003；
- discount：0.99；
- target soft-update：\(\tau=0.005\)；
- replay size：50000；
- batch size：128；
- gradient clipping：1.0；
- 每 episode 更新 10 个 minibatch；
- entropy coefficient：固定 \(\alpha=0.2\)；
- `automatic_entropy_tuning=false`。

Critic target：

\[
y=r+\gamma(1-d)\left[
\min(Q_{1,target}(s',a'),Q_{2,target}(s',a'))
-\alpha\log\pi(a'|s')
\right].
\]

Actor objective：

\[
J_\pi=\mathbb E\left[
\alpha\log\pi(a|s)-\min(Q_1(s,a),Q_2(s,a))
\right].
\]

## 6. 能耗、执行时间和可靠性模型

### 6.1 执行时间

\[
T_{exec}=\frac{L}{\rho f_{max}}.
\]

任务总时间还包括依赖数据传输和服务器排队：

\[
T_{task}=T_{communication}+T_{waiting}+T_{exec}.
\]

当前通信边界采用 Edge–Cloud DAG 模型：IoT 设备只生成工作流到达事件，不作为计算资源；入口任务在 $A_k$ 时刻直接就绪，出口任务完成即工作流完成。只有父子任务位于不同服务器时才计算依赖边传输：

\[
T_{p\rightarrow i}^{comm}=
\begin{cases}
0, & r_p=r_i,\\
\dfrac{8D_{p,i}}{B_{\ell_p,\ell_i}}, & r_p\ne r_i,
\end{cases}
\]

其中 $D_{p,i}$ 的单位为 MB，$B_{\ell_p,\ell_i}$ 的单位为 Mbps。默认链路为 Edge–Edge、Edge–Cloud 和 Cloud–Cloud；IoT–Edge Shannon 链路不参与当前训练。

### 6.2 功率和能耗

\[
P(\rho)=P_{static}+P_{dynamic}^{max}\rho^3,
\]

\[
E(\rho)=P(\rho)T_{exec}(\rho).
\]

当前系统总能耗是所有实际任务事件能耗之和。backup 关闭，因此没有 backup energy。

### 6.3 频率相关可靠性

服务器在最大频率的 failure rate 为 \(\lambda_{max}\)，频率下降后：

\[
\lambda(\rho)=\lambda_{max}
10^{d_k\frac{f_{max}-f}{f_{max}-f_{min}}}.
\]

单任务可靠性：

\[
R_i=\exp[-\lambda(\rho)T_{exec}].
\]

一个工作流的可靠性：

\[
R_w=\prod_{i\in w}R_i.
\]

工作流满足可靠性的判断为：

\[
R_w\ge R_w^{req}.
\]

最终日志中的 `reliability_success_rate` 是“满足可靠性要求的工作流比例”：

\[
\frac{\#\{w:R_w\ge R_w^{req}\}}{\#\{w\}}.
\]

它不是把所有工作流可靠性再次相乘得到一个系统可靠性。

## 7. 约束处理的准确性质

当前日志会显示：

```text
constraints=reward_only
high_action_mask=structural_only
```

准确含义是：

- deadline 和 reliability 都进入 reward；
- 真实服务器不会因为约束风险而被 mask；
- 下层动作不会被 `rho_safe` 强制修正；
- 没有 constraint critic；
- 没有拉格朗日乘子；
- 没有安全策略投影；
- 违反约束的动作允许被执行和进入 replay。

因此论文中应表述为“reward-penalty constrained optimization”或“constraints incorporated into reward”，不应表述为严格保证约束的 safe RL。

## 8. 联合训练流程

### 8.1 每个 episode 的数据收集

默认一个 episode 有 10 个工作流、每个 20 个任务，大约 200 次调度决策：

1. 用 `seed + episode` 生成新的工作流和到达时间；
2. 环境选择 `ready_tasks[0]`；
3. D3QN 通过 epsilon-greedy 选择服务器；
4. 共享 SAC 随机采样连续动作 \(u\)，映射为 \(\rho\)；
5. 环境计算通信、排队、执行、能耗、reliability；
6. 生成上下层 reward 信息；
7. 上层 transition 暂存在 high rollout buffer；
8. 下层原始 reward components 直接写入 SAC replay；
9. episode 结束后进行网络更新。

### 8.2 上层 replay transition

上层 transition 按全局环境决策顺序连接：

\[
(s_t,a_t^H,r_t^H,s_{t+1},done).
\]

只有整个 episode 的最后一个 transition 设置 `done=true`。不同工作流可以在全局序列中交错，这对应整个 edge-cloud 系统的连续 MDP，而不是为每个工作流建立独立 episode。

### 8.3 下层 replay transition

下层 replay 存储：

\[
(s_t^L,u_t,x_E,x_T,x_D,x_R,c_D,c_R,s_{t+1}^L,done).
\]

当前 `next_state` 是下一次可执行 SAC 动作的低层 state。所有服务器和工作流共用一条全局 transition 序列，并不是按服务器分别维护轨迹。

### 8.4 episode-end 更新顺序

每个 episode 数据收集完成后：

1. 将 high buffer 转存到 D3QN replay；
2. D3QN 从 replay 最多更新 10 次；
3. SAC 从 replay 最多更新 10 次；
4. 保存 latest checkpoint；
5. 到 eval/save interval 时进行额外评估和保存。

虽然配置写的是 `training_mode=joint`，但每个 episode 内不是逐 step 更新，而是先完整收集 episode，再依次更新上层和下层。

## 9. 评估和 checkpoint 选择

默认设置：

- 每 20 episodes 评估一次；
- 每次评估 10 个固定 seed；
- `eval_seed=242000`，每轮评估都复用同一组工作流种子；
- D3QN 确定性选择 argmax Q；
- SAC 使用 actor mean；
- 每 20 episodes 保存周期 checkpoint；
- 每个 episode 保存 latest；
- 训练结束保存 final；
- 默认不保存 replay buffer。

best checkpoint 使用以下字典序：

1. deadline success rate 越高越好；
2. reliability success rate 越高越好；
3. total energy 越低越好；
4. average response time 越低越好。

这意味着模型选择逻辑是“先满足约束，再优化能耗”，与主要目标为能耗最小化并不矛盾，但不是单纯按最低能耗选 best。

## 10. 已实现的诊断/消融模式

### A. D3QN + SAC

```powershell
python tools/train_hierarchical_d3qn.py `
  --config configs/hierarchical_d3qn_sac.json `
  --high-algorithm d3qn `
  --low-algorithm sac
```

### B. HEFT + 相同 SAC

HEFT 替代上层服务器调度，下层仍使用完全相同的共享 SAC，用于判断问题主要来自上层还是下层：

```powershell
python tools/train_hierarchical_d3qn.py `
  --config configs/hierarchical_d3qn_sac.json `
  --high-algorithm heft `
  --low-algorithm sac
```

当前 HEFT 的 upward rank 使用平均执行时间和 DAG 边的平均跨服务器通信时间；online EFT 使用父任务实际服务器到候选服务器的链路带宽。同服务器通信为 0，不包含 IoT 上传或最终下载。HEFT 会先从全部 ready tasks 中选择最高 rank 任务，再按 online earliest-finish-time 选择服务器；训练代码随后把 HEFT 选中的任务移到 `ready_tasks[0]`。因此 HEFT 同时拥有任务排序和服务器选择权限，而 D3QN 只有服务器选择权限。`HEFT + same SAC` 很适合诊断，但不是动作权限完全相同的严格公平基线。

### C. D3QN + 固定满频率

用于隔离下层 DVFS 的影响：

```powershell
python tools/train_hierarchical_d3qn.py `
  --config configs/hierarchical_d3qn_sac.json `
  --high-algorithm d3qn `
  --fixed-low-rho-one
```

## 11. 为寻找对比算法必须说明的边界

### 11.1 当前问题属于什么类型

- 多工作流 DAG 调度；
- edge-cloud 异构服务器；
- 随机工作流到达；
- 上层离散服务器选择；
- 下层连续 DVFS；
- 能耗最小化；
- deadline/reliability reward penalties；
- 两层 off-policy hierarchical DRL；
- 无 backup；
- 固定环境任务排序。

### 11.2 公平对比应尽量保持一致

- 相同 Pegasus workflows；
- 相同到达种子；
- 相同 20 edge + 6 cloud；
- 相同时间、通信、功率与 reliability 模型；
- 相同 `Rreq=0.9` 和 deadline；
- 相同是否允许 DVFS；
- 相同是否允许算法选择 task priority；
- 相同 backup 关闭设置；
- 至少报告 total energy、deadline success、reliability success 和 response time。

### 11.3 当前实现的重要局限

1. 上层输出头固定为 26 个服务器动作，不能直接泛化到更多 VM。
2. 上层动作依赖服务器位置编号，不是 permutation-invariant 的 per-server scorer。
3. 上层不选择任务，只选择 `ready_tasks[0]` 的服务器。
4. 当前 HEFT 基线同时选择 task 和 server，比 D3QN 多出 task-priority 权限。
5. 下层是一个共享 SAC，不是每服务器独立 SAC。
6. 下层 state 没有基础 failure rate 和当前累计 workflow reliability，可靠性决策信息不完整。
7. `rho_safe` 仅作为 state/诊断量，不执行硬约束。
8. 上层 reward 包含下层实际能耗，联合训练时目标会随 SAC 改变。
9. 下层 z-score 使用所有服务器、任务混合后的全 replay 统计；绝对能耗量纲差异可能被任务/服务器异质性影响。
10. violation severity 使用 `ReLU(z_violation)`；低于 replay 均值但仍大于 0 的违反可能没有 severity penalty，不过首次 crossing 仍有固定 -5。
11. 增加 `num_workflows` 主要增加序列长度，不必然增加单位时间并发负载；真正的负载泛化还应改变 arrival rate。

## 12. 可以直接交给 GPT 的检索描述

```text
请为下面的论文实验寻找合适且可复现的对比算法，并区分传统启发式、单层 DRL、分层 DRL 和连续控制算法：

问题：多工作流 DAG 在异构 edge-cloud 服务器上的在线调度与 DVFS。工作流随机到达；任务存在依赖、传输时间、服务器排队。环境预先按最小 workload 选择当前 ready task，算法不选择任务优先级。

动作：上层从 26 台服务器中选择一个主服务器；下层输出一维连续 DVFS 动作 u in [0,1]，映射到 rho in [0.2,1]。不使用 backup。

目标：主要最小化系统总能耗，同时要求工作流 deadline 和 workflow reliability product >= 0.9。当前方法把 deadline/reliability 作为 reward penalty，不使用 safe RL 或硬动作投影。

方法：上层 Dueling Double DQN，下层共享 continuous SAC，二者都是 off-policy，联合训练并在 episode 结束后从 replay buffer 更新。

请优先寻找：
1. 可用于 DAG edge-cloud scheduling 的传统算法；
2. 离散 server selection + continuous resource/DVFS control 的混合动作或分层 DRL；
3. DQN/DDQN/Dueling DQN、PPO、SAC、TD3/DDPG 等可公平改造成相同动作权限的基线；
4. 同时考虑 energy、deadline、reliability 的 MEC/edge-cloud 调度论文；
5. 有公开代码、明确状态/动作/reward、且能在相同模拟器中复现的算法。

请对每个候选算法说明：论文、年份、是否有代码、原始 state/action/reward、需要怎样改造成当前实验设置，以及是否会获得比当前方法更大的任务选择或资源控制权限。
```

## 13. 当前代码对应关系

| 内容 | 文件 |
|---|---|
| 默认配置 | `configs/hierarchical_d3qn_sac.json` |
| 训练入口和 episode 循环 | `tools/train_hierarchical_d3qn.py` |
| 上层 D3QN 和分层 agent | `src/D3QN/hierarchical_d3qn_agent.py` |
| Dueling Double DQN 网络/更新 | `src/D3QN/D3QN.py` |
| 上层 reward | `src/D3QN/reward.py` |
| 上下层 state encoder | `src/ppo/hierarchical_state.py` |
| 下层 continuous SAC | `src/sac/hierarchical_low_sac.py` |
| 下层 reward components | `src/sac/reward.py` |
| 环境、时间和决策 context | `src/simulator/edge_cloud_env.py` |
| 能耗/频率模型 | `src/simulator/resource.py` |
| reliability 模型 | `src/simulator/reliability.py` |
| deadline 生成 | `src/simulator/deadline.py` |
| episode 指标与 best 排序 | `tools/hierarchical_common.py` |
