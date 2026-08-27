# D3QN–SAC 当前奖励函数与拉格朗日约束设计说明

> **2026-08-21 ep740 回滚：** 默认配置已恢复验证过的
> `ep740_lagrangian_v1`。上层重新使用 SAC 执行后的实际任务能耗，
> 上下层恢复独立 Deadline 乘子并在 episode 末更新；后续的
> workflow-miss 分摊和 team-energy credit 不再用于该配置。
>
> 当前上层：
> \[
> r_t^H=\frac{1}{N_w}
> \frac{E_t^{\mathrm{scale}}-E_t^{\mathrm{actual}}}
> {E_t^{\mathrm{scale}}}
> -\lambda_H\frac{\Delta C_t}{D_w}.
> \]
>
> 当前下层：
> \[
> r_t^L=2\,\operatorname{clip}(z_E(E_{t,j}^{\rho=1}-E_t),-3,3)
> -\operatorname{clip}(z_g(g_t),-3,3)
> -\lambda_L c_{D,t}^{L}.
> \]
>
> \(\lambda_H\) 使用工作流 miss rate 和 0.01 预算更新；
> \(\lambda_L\) 使用下层反事实 Deadline cost 和 0 预算更新。
> ep740 参数为 \(\eta_H=0.001\)、\(\eta_L=0.01\)、
> \(\lambda_H^{(0)}=1\)、\(\lambda_L^{(0)}=10\)。

> **历史记录（2026-08-19 分阶段版本，现已停用）：** 该版本
> `lagrangian_deadline` 模式的上层能耗信号使用当前任务在所有候选服务器
> \(\rho=1\) 能耗的中位数作为参考：
> \(r_{E,t}^H=(E_{t,\mathrm{median}}^{\rho=1}-E_{t,j}^{\rho=1})/(N_wE_{t,\mathrm{median}}^{\rho=1})\)。
> 它不做 replay z-score、不包含下层 SAC 的 DVFS 节能，
> 直接与 Deadline 拉格朗日成本组合。这里只保留作消融对照。
> 本文后续关于旧 `deadline_incremental_only` 公式的章节仅保留为历史说明。

> 更新时间：2026-08-13  
> 对应配置：`configs/hierarchical_d3qn_sac.json`  
> 对应上层模式：`deadline_incremental_only`  
> 本文首先记录**当前代码实际执行的奖励函数**，然后给出拉格朗日约束改造建议。二者不能混为一谈。

## 1. 当前分层决策过程

对于工作流 \(w\) 的当前任务 \(t\)：

1. 上层 D3QN 选择主服务器 \(j_t\)；
2. 下层 SAC 在已选服务器上输出连续归一化频率 \(\rho_t\in[\rho_{\min},1]\)；
3. 环境执行任务，得到实际完成时间、能耗和可靠性；
4. 分别构造一个上层 transition 和一个下层 transition；
5. 当前训练配置为 `joint`，上下层都在每个 episode 结束后从各自 Replay Buffer 更新。

当前关闭备份：

\[
E_t=E_t^{\mathrm{primary}},\qquad E_t^{\mathrm{backup}}=0.
\]

## 2. 符号定义

| 符号 | 含义 |
|---|---|
| \(D_w\) | 工作流 \(w\) 的相对 Deadline |
| \(N_w\) | 工作流 \(w\) 的任务数 |
| \(C_w^{(t-1)}\) | 执行当前任务前，工作流已完成任务的部分 makespan |
| \(C_w^{(t)}\) | 执行当前任务后，工作流已完成任务的部分 makespan |
| \(\Delta C_t\) | 部分 makespan 增量，\(\max(0,C_w^{(t)}-C_w^{(t-1)})\) |
| \(E_t^{\mathrm{scale}}\) | 当前任务在所有候选服务器上以 \(\rho=1\) 执行时能耗的中位数 |
| \(E_t\) | 当前任务在 D3QN 所选服务器和 SAC 所选频率下的实际能耗 |
| \(E_{t,j}^{\rho=1}\) | 当前任务在已选服务器 \(j\) 上以 \(\rho=1\) 执行的参考能耗 |
| \(AFT_t^{\rho=1}\) | 当前任务在已选服务器上以 \(\rho=1\) 的参考完成时间 |
| \(AFT_t^{\rho}\) | 当前任务在 SAC 频率 \(\rho\) 下的实际完成时间 |
| \(R_w^{(t)}\) | 当前已经执行部分的工作流可靠性乘积 |
| \(R_w^{\mathrm{req}}\) | 工作流最低可靠性要求，当前为 0.9 |

所有 makespan 均以工作流到达时间为零点，因此最终：

\[
C_w=C_w^{(N_w)}=\text{工作流响应时间}.
\]

---

## 3. 当前上层 D3QN 奖励

### 3.1 当前完整公式

当前配置为：

```json
"objective": {
  "reward_mode": "deadline_incremental_only",
  "deadline_violation_weight": 3.0,
  "high_energy_weight": 1.0
}
```

代码实际使用的逐任务上层奖励为：

\[
\boxed{
r_t^H=
-\frac{\Delta C_t}{D_w}
+\frac{\lambda_E^H}{N_w}
\frac{E_t^{\mathrm{scale}}-E_t}{E_t^{\mathrm{scale}}}
-\lambda_D^H\Delta\widetilde V_t
}
\]

当前：

\[
\lambda_E^H=1.0,\qquad \lambda_D^H=3.0.
\]

### 3.2 时间项

\[
r_{t,\mathrm{time}}^H=-\frac{\Delta C_t}{D_w}.
\]

其中：

\[
\Delta C_t=\max\left(0,C_w^{(t)}-C_w^{(t-1)}\right).
\]

这一项没有除以 \(N_w\)，原因是它具有望远镜求和性质：

\[
\sum_{t=1}^{N_w}\Delta C_t=C_w,
\]

所以：

\[
\sum_{t=1}^{N_w}r_{t,\mathrm{time}}^H=-\frac{C_w}{D_w}.
\]

它最终等价于最小化归一化工作流响应时间，但存在一个信用分配盲点：如果当前任务没有刷新部分 makespan，则 \(\Delta C_t=0\)，即使本次服务器选择明显较慢，也不会立即受到时间惩罚。

### 3.3 能耗项

\[
r_{t,\mathrm{energy}}^H=
\frac{\lambda_E^H}{N_w}
\frac{E_t^{\mathrm{scale}}-E_t}{E_t^{\mathrm{scale}}}.
\]

其中 \(E_t^{\mathrm{scale}}\) 不是已选服务器的满频能耗，而是：

\[
E_t^{\mathrm{scale}}
=\operatorname{median}_{j\in\mathcal S}
E_{t,j}^{\rho=1}.
\]

因此该项同时评价：

- 上层是否选择了节能服务器；
- 下层是否通过 DVFS 降低了能耗。

若 \(E_t<E_t^{\mathrm{scale}}\)，该项为正；若选择的服务器或频率导致 \(E_t>E_t^{\mathrm{scale}}\)，该项为负。

该项除以 \(N_w\)，是为了让不同任务数量的工作流拥有近似可比的能耗 return：

\[
\sum_{t=1}^{N_w}r_{t,\mathrm{energy}}^H
=\frac{\lambda_E^H}{N_w}
\sum_{t=1}^{N_w}
\frac{E_t^{\mathrm{scale}}-E_t}{E_t^{\mathrm{scale}}}.
\]

### 3.4 Deadline unit-jump 违反量

先定义工作流在部分 makespan \(C\) 下的违反函数：

\[
\widetilde V_D(C,D_w)=
\begin{cases}
0,&C\le D_w,\\[4pt]
1+\dfrac{C-D_w}{D_w},&C>D_w.
\end{cases}
\]

逐任务增量违反为：

\[
\Delta\widetilde V_t=
\max\left(
0,
\widetilde V_D(C_w^{(t)},D_w)
-\widetilde V_D(C_w^{(t-1)},D_w)
\right).
\]

当前惩罚为：

\[
P_{D,t}^H=3.0\,\Delta\widetilde V_t.
\]

其行为分为三种情况：

1. 当前任务后仍未超期：\(\Delta\widetilde V_t=0\)；
2. 当前任务第一次越过 Deadline：产生至少为 1 的跳变，因此惩罚至少为 3；
3. 工作流已经超期：之后只惩罚进一步增加的归一化超期程度。

若部分 makespan 单调增加，则整个工作流上的增量违反也会望远镜求和：

\[
\sum_t\Delta\widetilde V_t
=\widetilde V_D(C_w,D_w).
\]

### 3.5 上层可靠性当前只记录、不惩罚

当前代码仍然计算可靠性诊断量。令：

\[
L_w^{(t)}=-\log R_w^{(t)},\qquad
B_R=-\log R_w^{\mathrm{req}}.
\]

增量 log-risk 为：

\[
c_{R,t}=\max(0,L_w^{(t)}-L_w^{(t-1)}).
\]

归一化可靠性违反为：

\[
V_R^{(t)}=
\frac{\max(0,L_w^{(t)}-B_R)}{B_R}.
\]

但在当前 `deadline_incremental_only` 模式下：

\[
P_{R,t}^H=0.
\]

也就是说，`reliability_cost`、`reliability_violation` 和 `reliability_crossed` 会被记录，但不会进入当前上层 reward。

### 3.6 上层 Replay Buffer 中真正训练的 reward

当前高层配置是：

```json
"high": {
  "reward": {
    "mode": "scalar"
  }
}
```

因此上层不会在 Replay Buffer 采样时重新对时间或能耗分量做 z-score，D3QN 训练使用的就是上述标量奖励。

高层 Agent 还支持额外项：

\[
-w_{\mathrm{extra}}\frac{\Delta C_t}{D_w},
\]

对应 `deadline_cost_reward_weight`。当前配置没有设置该字段，默认：

\[
w_{\mathrm{extra}}=0,
\]

所以不存在重复扣除 \(\Delta C_t/D_w\) 的情况。

### 3.7 工作流结束时的 terminal patch

任务执行过程中记录的部分 makespan 可能尚未完全等于最终响应时间。工作流完成时，代码计算：

\[
\Delta C_{w,\mathrm{terminal}}
=\max(0,C_w-C_w^{\mathrm{last}}),
\]

并将相应的：

\[
-\frac{\Delta C_{w,\mathrm{terminal}}}{D_w}
-3\Delta\widetilde V_{w,\mathrm{terminal}}
\]

补到该工作流最后一个上层 transition 中。

代码还会检查：

\[
\sum_t\Delta C_t=C_w,
\]

防止遗漏或重复计算最终 makespan。

---

## 4. 当前下层 SAC 奖励

### 4.1 当前配置

```json
"low": {
  "reward": {
    "energy_weight": 2.0,
    "slack_time_weight": 1.0,
    "deadline_violation_weight": 10.0,
    "slack_floor_ratio": 0.05,
    "component_clip": 3.0,
    "energy_std_floor": 0.05,
    "slack_time_std_floor": 0.005
  }
}
```

训练时实际使用：

\[
\boxed{
r_t^L=
2\,\operatorname{clip}(z_E(E_{t,j}^{\rho=1}-E_t),-3,3)
-\operatorname{clip}(z_g(g_t),-3,3)
-10\Delta\widetilde V_t^L
}
\]

### 4.2 下层能耗节省量

下层使用同一台已选服务器上的满频执行作为参考：

\[
\Delta E_t^L=E_{t,j}^{\rho=1}-E_{t,j}^{\rho_t}.
\]

这与上层的候选服务器能耗中位数不同。下层该项只评价 SAC 在既定服务器上调频后节省了多少绝对能耗。

可能出现：

\[
\Delta E_t^L<0,
\]

例如实际执行事件包含的能耗高于参考估计、排队或能耗估计口径不同、数值扰动等。代码不会把负节能量截断为 0，而是如实存入 Replay Buffer。

### 4.3 下层时间成本

SAC 相对同服务器满频执行带来的完成时间增加为：

\[
\Delta T_t^L=
\max\left(0,AFT_t^{\rho_t}-AFT_t^{\rho=1}\right).
\]

执行前工作流剩余 slack 为：

\[
S_t=D_w-C_w^{(t-1)}.
\]

为防止 slack 接近 0 时分母爆炸，定义：

\[
S_t^{\mathrm{eff}}
=\max\left(S_t,\epsilon_sD_w\right),
\qquad \epsilon_s=0.05.
\]

下层时间成本为：

\[
g_t=\frac{\Delta T_t^L}{S_t^{\mathrm{eff}}}.
\]

它具有提前预警性质：同样的降频时间增加，在剩余 slack 越小时，惩罚越大。

### 4.4 下层 Deadline 违反量

下层执行前后部分 makespan 分别为：

\[
C_{w,L}^{\mathrm{before}}=C_w^{(t-1)},
\]

\[
C_{w,L}^{\mathrm{after}}
=\max\left(
C_w^{(t-1)},
AFT_t^{\rho_t}-A_w
\right),
\]

其中 \(A_w\) 是工作流到达时间。

随后复用与上层相同的 unit-jump 函数：

\[
\Delta\widetilde V_t^L=
\max\left(
0,
\widetilde V_D(C_{w,L}^{\mathrm{after}},D_w)
-\widetilde V_D(C_{w,L}^{\mathrm{before}},D_w)
\right).
\]

当前惩罚为：

\[
P_{D,t}^L=10\Delta\widetilde V_t^L.
\]

需要注意：尽管系数 10 很大，但在工作流真正越过 Deadline 之前，\(\Delta\widetilde V_t^L\) 通常为 0。因此它是强烈的事后 crossing 惩罚，提前控制主要依赖 \(g_t\)。

### 4.5 Replay Buffer z-score

下层 Replay Buffer 存储的是原始分量：

\[
(\Delta E_t^L,\ g_t,\ \Delta\widetilde V_t^L,\ \Delta V_{R,t}^L).
\]

每次 episode 结束更新时，使用**当前整个 SAC Replay Buffer** 中对应分量的均值和标准差：

\[
z_E(x)=\frac{x-\mu_E}{\max(\sigma_E,0.05)},
\]

\[
z_g(x)=\frac{x-\mu_g}{\max(\sigma_g,0.005)}.
\]

之后进行截断：

\[
\operatorname{clip}(z,-3,3).
\]

Deadline unit-jump 违反量具有固定语义边界，因此代码明确不对它做 z-score：

\[
-10\Delta\widetilde V_t^L.
\]

这意味着“10.0”不会再被 Replay Buffer 标准差缩放。

### 4.6 下层可靠性当前不进入 reward

代码计算了相对满频参考动作增加的预计可靠性违反：

\[
\Delta V_{R,t}^L
=\max(0,V_{R,t}^{\mathrm{actual}}-V_{R,t}^{\rho=1}).
\]

该分量和 `reliability_crossed` 会存入 Replay Buffer，但当前 `_compose_rewards()` 没有使用第四个可靠性分量。因此：

\[
P_{R,t}^L=0.
\]

当前上下层都还没有真正训练可靠性约束。

---

## 5. 当前奖励的整体性质

### 5.1 上层和下层能耗参考不同

上层：

\[
E_t^{\mathrm{scale}}
=\operatorname{median}_j E_{t,j}^{\rho=1},
\]

同时评价服务器选择和 DVFS。

下层：

\[
E_{t,j}^{\rho=1},
\]

只评价同一服务器上的 DVFS 节能。

### 5.2 上层 reward 不做 z-score，下层部分分量做 z-score

| 层级 | 时间/能耗归一化 | Deadline 违反归一化 |
|---|---|---|
| 上层 D3QN | 公式自身用 \(D_w\)、\(N_w\) 和能耗比例归一化；Replay 不做 z-score | unit-jump 原值 |
| 下层 SAC | 能耗和 slack-time 使用整个 Replay Buffer 的 z-score | unit-jump 原值，不做 z-score |

### 5.3 当前不是拉格朗日约束

当前的 3.0 和 10.0 都是人工固定 penalty weight：

\[
\lambda_D^H=3,\qquad \lambda_D^L=10.
\]

它们不会根据约束满足情况更新，所以当前仍属于固定加权奖励，不是拉格朗日或 primal-dual 方法。

---

## 6. 推荐的拉格朗日 CMDP 定义

拉格朗日设计最好先把“优化目标”和“约束成本”彻底分开。

### 6.1 原始优化目标：最小化系统总能耗

可以将任务级基础奖励定义为：

\[
r_{E,t}=-\frac{E_t}{E_{\mathrm{norm}}},
\]

或继续使用节能形式：

\[
r_{E,t}=\frac{E_t^{\mathrm{ref}}-E_t}{E_t^{\mathrm{ref}}}.
\]

如果论文最终目标明确是最小化总焦耳数，建议使用第一种绝对能耗目标，并选择一个固定或缓慢更新的能耗尺度 \(E_{\mathrm{norm}}\)。使用相对节能率时，不同服务器的参考能耗会改变优化含义。

### 6.2 Deadline 约束成本

Deadline 原始约束为：

\[
C_w\le D_w.
\]

最适合拉格朗日的逐任务可加成本是：

\[
c_{D,t}=\frac{\Delta C_t}{D_w}.
\]

因为：

\[
J_D(w)=\sum_{t=1}^{N_w}c_{D,t}
=\frac{C_w}{D_w}.
\]

所以约束可以精确写成：

\[
J_D(w)\le1.
\]

这比 unit-jump 更适合拉格朗日：

- 每个使工作流 makespan 增长的任务都有信号；
- 不需要等到 crossing 后才产生约束成本；
- 累积成本严格对应最终 \(C_w/D_w\)；
- 不需要人为设计“超期从 1 开始”的固定跳变。

### 6.3 可靠性约束成本

工作流可靠性约束是乘积型：

\[
R_w=\prod_t r_t\ge R_w^{\mathrm{req}}.
\]

取负对数后变成可加约束：

\[
-\log R_w=\sum_t-\log r_t
\le-\log R_w^{\mathrm{req}}.
\]

定义归一化逐任务可靠性成本：

\[
c_{R,t}=\frac{-\log r_t}{-\log R_w^{\mathrm{req}}}.
\]

则：

\[
J_R(w)=\sum_t c_{R,t}
=\frac{-\log R_w}{-\log R_w^{\mathrm{req}}}
\le1.
\]

这和当前代码内部的 `partial_log_risk`、`reliability_cost`、`reliability_limit` 是一致的，可以复用现有计算。

### 6.4 工作流级拉格朗日目标

对能耗回报 \(J_E\)，Deadline 成本 \(J_D\) 和可靠性成本 \(J_R\)，定义：

\[
\mathcal L(\theta,\lambda_D,\lambda_R)
=J_E(\theta)
-\lambda_D\left(J_D(\theta)-1\right)
-\lambda_R\left(J_R(\theta)-1\right),
\]

其中：

\[
\lambda_D\ge0,\qquad\lambda_R\ge0.
\]

Agent 最大化 \(\mathcal L\)，对偶变量最小化约束违反对应的对偶目标。

任务级训练 reward 可以写成：

\[
r_t^{\lambda}
=r_{E,t}
-\lambda_D c_{D,t}
-\lambda_R c_{R,t}.
\]

策略更新时，常数 \(+\lambda_D+\lambda_R\) 不影响固定乘子的动作选择；但更新乘子时必须保留预算 1。

### 6.5 对偶变量更新

每批已完成工作流计算：

\[
\overline J_D=\frac{1}{B}\sum_{w=1}^{B}\frac{C_w}{D_w},
\]

\[
\overline J_R=\frac{1}{B}\sum_{w=1}^{B}
\frac{-\log R_w}{-\log R_w^{\mathrm{req}}}.
\]

使用投影梯度上升更新乘子：

\[
\lambda_D\leftarrow
\left[\lambda_D+\eta_\lambda(\overline J_D-1)\right]_+,
\]

\[
\lambda_R\leftarrow
\left[\lambda_R+\eta_\lambda(\overline J_R-1)\right]_+.
\]

其中：

\[
[x]_+=\max(0,x).
\]

建议加入上界避免早期发散：

\[
\lambda_D,\lambda_R\in[0,\lambda_{\max}].
\]

例如初步可以设置：

```json
"lagrangian": {
  "enabled": true,
  "deadline_budget": 1.0,
  "reliability_budget": 1.0,
  "deadline_lambda_init": 1.0,
  "reliability_lambda_init": 1.0,
  "dual_learning_rate": 0.001,
  "lambda_max": 100.0,
  "update_interval": "episode_end",
  "warmup_episodes": 50
}
```

这些只是推荐初值，不是当前代码已经存在的配置。

---

## 7. 分层系统采用拉格朗日时的建议

### 7.1 推荐共享系统级乘子

Deadline 和可靠性都是最终工作流约束，不属于单独某一层。因此优先建议共享：

\[
\lambda_D^H=\lambda_D^L=\lambda_D,
\qquad
\lambda_R^H=\lambda_R^L=\lambda_R.
\]

乘子使用最终完成工作流的 \(C_w/D_w\) 和 log-reliability 更新，而不是分别用上层、下层的局部代理指标独立更新。

这样可以避免出现：

- 上层认为约束已经满足而下层继续降频；
- 上下层两个乘子相互竞争；
- 同一个 Deadline 违反被重复放大。

### 7.2 两层的即时信用分配可以不同

共享乘子不等于两层必须使用完全相同的局部 cost。

上层服务器选择直接影响排队和完成时间，适合使用：

\[
c_{D,t}^H=\frac{\Delta C_t}{D_w}.
\]

还可以增加一个不参与乘子更新的局部 shaping：

\[
G_t^H=\max\left(
0,
\frac{AFT_t^{\mathrm{selected}}-\min_jAFT_{t,j}}{D_w}
\right).
\]

下层 DVFS 只控制相对满频增加的时间，适合保留：

\[
g_t=\frac{\Delta T_t^L}{S_t^{\mathrm{eff}}}
\]

作为提前风险 shaping，但不要把 \(g_t\) 同时当作系统 Deadline 预算，因为：

\[
\sum_t g_t\ne\frac{C_w}{D_w}.
\]

推荐区分：

- `constraint cost`：严格用于更新 \(\lambda_D\) 的 \(\Delta C_t/D_w\)；
- `shaping cost`：改善任务级信用分配的 \(G_t^H\) 或 \(g_t\)；
- `terminal metric`：最终用于更新乘子的 \(C_w/D_w\)。

### 7.3 推荐的两层形式

上层：

\[
r_t^H=
r_{E,t}^H
-\lambda_D c_{D,t}^H
-\lambda_R c_{R,t}^H
-\beta_HG_t^H.
\]

下层：

\[
r_t^L=
z_E(\Delta E_t^L)
-\lambda_D c_{D,t}^L
-\lambda_R c_{R,t}^L
-\beta_Lz_g(g_t).
\]

其中 \(\beta_H\) 和 \(\beta_L\) 是固定 shaping 权重，不作为拉格朗日乘子。

### 7.4 不建议直接把当前 3 和 10 当作两个初始乘子

当前：

\[
-3\Delta\widetilde V_t^H,
\qquad
-10\Delta\widetilde V_t^L
\]

是两个不同层级、不同尺度的固定事后惩罚。若直接让它们分别自适应，会让相同的工作流超期在两个层级重复产生对偶压力。

更干净的做法是：

1. 只保留一个系统级 \(\lambda_D\)；
2. 使用最终工作流 \(C_w/D_w-1\) 更新它；
3. 给上下层 transition 分配各自的增量 Deadline cost；
4. unit-jump 可以暂时作为辅助 shaping，也可以在拉格朗日稳定后删除；
5. 可靠性使用同样结构的 \(\lambda_R\)。

---

## 8. 建议记录的训练指标

改成拉格朗日后，每次 eval 至少输出：

| 指标 | 含义 |
|---|---|
| `lambda_deadline` | 当前 Deadline 对偶变量 |
| `lambda_reliability` | 当前可靠性对偶变量 |
| `mean_C_over_D` | 批次工作流平均 \(C_w/D_w\) |
| `p95_C_over_D` | \(C_w/D_w\) 的 95 分位数 |
| `deadline_excess` | \(\overline J_D-1\) |
| `mean_log_risk_ratio` | 平均 \((-\log R_w)/(-\log R_{req})\) |
| `reliability_excess` | \(\overline J_R-1\) |
| `base_energy_reward` | 未扣约束前的能耗目标 |
| `deadline_lagrangian_penalty` | \(\lambda_Dc_D\) |
| `reliability_lagrangian_penalty` | \(\lambda_Rc_R\) |
| `deadline_success_rate` | 最终工作流成功比例 |
| `reliability_success_rate` | 最终工作流可靠性成功比例 |
| `total_energy` | 系统所有工作流总能耗 |

必须同时保存 base reward、constraint cost 和 multiplier。只记录合成后的 reward，后续无法判断是策略变好还是 \(\lambda\) 变大。

---

## 9. 当前实现与拉格朗日改造对照表

| 内容 | 当前实现 | 拉格朗日建议 |
|---|---|---|
| 主目标 | 时间 + 能耗 + 固定 Deadline penalty | 能耗作为主目标 |
| Deadline | unit-jump 增量，固定权重 3/10 | \(\sum\Delta C/D\le1\)，自适应 \(\lambda_D\) |
| 可靠性 | 只诊断，不进 reward | log-risk 可加约束，自适应 \(\lambda_R\) |
| 上层标准化 | 比例归一化，无 Replay z-score | constraint cost 保留物理语义，不做 z-score |
| 下层标准化 | 能耗、slack cost 做 Replay z-score | 能耗可标准化，约束 cost 不建议 z-score |
| 对偶更新 | 无 | 用已完成工作流批次在 episode 末更新 |
| 两层乘子 | 固定 3 和 10 | 优先共享系统级 \(\lambda_D,\lambda_R\) |
| 提前信用 | \(\Delta C/D\)、\(g_t\) | 保留 shaping，但不用于对偶预算定义 |

## 10. 对应代码位置

| 功能 | 文件 |
|---|---|
| 上层当前奖励 | `src/D3QN/reward.py` |
| 上层公用组合公式 | `src/shared/hierarchical_state_reward.py` |
| D3QN Replay reward | `src/D3QN/hierarchical_d3qn_agent.py`、`src/D3QN/D3QN.py` |
| 下层原始 reward 分量 | `src/sac/reward.py` |
| SAC Replay z-score 与合成 reward | `src/sac/hierarchical_low_sac.py` |
| 任务执行与 transition 收集 | `tools/train_hierarchical_d3qn.py` |
| 当前配置 | `configs/hierarchical_d3qn_sac.json` |
| 能耗中位数尺度 | `src/simulator/edge_cloud_env.py` |

## 11. 最简总结

当前上层：

\[
\boxed{
r_t^H=
-\frac{\Delta C_t}{D_w}
+\frac{1}{N_w}\frac{E_t^{\mathrm{scale}}-E_t}{E_t^{\mathrm{scale}}}
-3\Delta\widetilde V_t
}
\]

当前下层：

\[
\boxed{
r_t^L=
2\operatorname{clip}(z_E(E_{t,j}^{\rho=1}-E_t),-3,3)
-\operatorname{clip}(z_g(g_t),-3,3)
-10\Delta\widetilde V_t^L
}
\]

当前可靠性：上下层都计算诊断量，但都没有进入实际 reward。

推荐拉格朗日约束：

\[
\boxed{
\sum_t\frac{\Delta C_t}{D_w}
=\frac{C_w}{D_w}\le1
}
\]

\[
\boxed{
\sum_t\frac{-\log r_t}{-\log R_w^{\mathrm{req}}}
=\frac{-\log R_w}{-\log R_w^{\mathrm{req}}}\le1
}
\]

这两项分别对应 Deadline 和乘积可靠性的精确可加约束。
