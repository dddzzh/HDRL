# DRLEC C3C-v2 工作交接（2026-08-21）

## 1. 当前目标与判定顺序

本项目的优化目标不是简单加权的“能耗 + Deadline”，而是：

\[
\min E_{\mathrm{system}}
\quad \text{s.t.}\quad
P(C_w\le D_w)\ge 0.99.
\]

因此模型和 checkpoint 的判定顺序必须是：

1. 先筛选固定 20-seed 评估中 `deadline_success_rate >= 0.99` 的 checkpoint；
2. 再在可行 checkpoint 中选择系统总能耗最低者；
3. 没有 checkpoint 达到 99% 时，只能报告“当前不可行”，不能因为能耗低而称为最优。

可靠性目前很容易满足，PB/备份也仍关闭；本轮只解决 D3QN、SAC 和 C3C 的 Deadline–能耗协同问题。

## 2. 已确认的基线与 C3C-v1 结果

相同的小规模数据配置、固定 20-seed 评估下：

| 实验 | checkpoint/episode | Deadline | 系统能耗 | 平均 rho | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 旧 D3QN+SAC | `20260817_130319`, ep740 | 99.00% | 237.647 kJ | 0.930 | 当前可行基准 |
| 旧 D3QN+SAC | `20260818_092437`, ep460 | 98.75% | 242.129 kJ | 0.932 | 接近可行，但未到 99% |
| C3C-v1 | `20260821_170346`, ep560 | 88.50% | 230.574 kJ | 0.907 | 能耗低，但不满足约束 |
| C3C-v1 | `20260821_170346`, ep1160 | 54.25% | 246.908 kJ | 0.843 | 后期同时损失 Deadline 和能耗 |

C3C-v1 的 critic loss 始终有限，奖励分解误差接近零，所以它不是 NaN、梯度爆炸或奖励计算不连续造成的退化。主要证据如下：

| 训练区间 | 每 episode 平均违约工作流 | 其中由降频跨过 Deadline 边界 | rho=1 放置参考能耗 | 实际能耗 | Cloud 选择比例 | 平均 rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ep501–600 | 3.22 | 0.04 | 272.859 kJ | 228.903 kJ | 10.02% | 0.881 |
| ep901–1000 | 6.35 | 0.22 | 312.443 kJ | 247.598 kJ | 13.72% | 0.852 |
| ep1001–1100 | 7.51 | 0.34 | 336.750 kJ | 263.527 kJ | 15.75% | 0.841 |

结论：大多数违约在 `rho=1` 下已经发生，首先是上层服务器选择逐渐变差；SAC 后期降频更激进是次要问题。继续单纯增大下层 `lambda_L` 或直接把 C3C 的 `lambda_H` 换成 `lambda_L`，不能修复上层漂移。

## 3. C3C 的奖励分解

对任务 t，设：

- `E_scale`：任务能耗尺度；
- `E_1`：当前服务器在 rho=1 时的参考能耗；
- `E_rho`：SAC 实际 rho 下的能耗；
- `N_w`：当前工作流任务数；
- `C_before`：调度前工作流部分完成时间；
- `C_1`：rho=1 反事实下调度后的部分完成时间；
- `C_rho`：实际 rho 下调度后的部分完成时间。

上层 rho=1 基础部分：

\[
r^H_{E,t}=\frac{E_{scale}-E_1}{N_w E_{scale}},
\qquad
c^H_{D,t}=\frac{\max(0,C_1-C_{before})}{D_w},
\]

\[
r^H_t=r^H_{E,t}-\lambda_H c^H_{D,t}.
\]

C3C 学习 SAC 相对 rho=1 产生的残差：

\[
r^C_{E,t}=\frac{E_1-E_\rho}{N_w E_{scale}},
\qquad
c^C_{D,t}=\frac{\max(0,C_\rho-C_{before})}{D_w}-c^H_{D,t},
\]

\[
A^C_t=r^C_{E,t}-\lambda_H c^C_{D,t}.
\]

两部分严格满足：

\[
r^H_t+A^C_t
=\frac{E_{scale}-E_\rho}{N_wE_{scale}}
-\lambda_H\frac{\max(0,C_\rho-C_{before})}{D_w}.
\]

这里 C3C 使用 `lambda_H` 是因为它的修正最终作用于上层服务器选择；下层 SAC 仍由自己的 `lambda_L` 和自己的 replay z-score 奖励更新。两个 lambda 对应的 cost 和尺度不同，数值不能直接互换。

## 4. C3C-v1 为什么会拖坏 D3QN

### 4.1 C3C 修正与 D3QN Q 值不在同一尺度

v1 直接使用：

\[
Q_{select}(s,a)=Q_H(s,a)+\beta\,\mathrm{conf}(s,a)A_C(s,a,\rho).
\]

即使 C3C loss 很小，只要 C3C correction 相对于候选服务器间的 `Q_H` 差值过大，它就会变成第三个调度器，逐步把 D3QN 推向更差的服务器分布。

### 4.2 C3C 在 Deadline 尚未稳定时就参与控制

critic 只要完成 warmup，v1 correction 就会参与动作和 Bellman target；此时 D3QN/SAC/critic 都在变化，形成非平稳的相互追逐。模型可能通过更低 rho 暂时获得残差节能，却破坏尚未稳定的上层可行策略。

### 4.3 v1 直接给 SAC actor 额外梯度

SAC 本来就在优化 replay 中经过 z-score 的能耗/Deadline reward；C3C 又用原始残差尺度对 actor 施加安全梯度。两条尺度不同的梯度会竞争，而且 v1 的绝大多数 Deadline 违约根本不是 SAC 降频造成的。

## 5. 已实现的 C3C-v2 改动

新配置：`configs/hierarchical_d3qn_sac_c3c_v2.json`。旧的 C3C-v1 配置和旧训练入口未被替换。

### 5.1 关闭 C3C 对 SAC 的直接 actor guidance

配置：

```json
"low_guidance_enabled": false,
"low_guidance_weight_final": 0.0,
"low_guidance_updates": 0
```

SAC 继续使用自己的低层奖励和 `lambda_L` 学习。C3C 只预测“这个服务器与 SAC rho 配对以后会产生什么残差”，再把该信息反馈给 D3QN。这保留了跨层协同，但避免第三个网络直接和 SAC critic 抢 actor 更新方向。

### 5.2 Deadline-first 约束门控

滚动窗口记录训练工作流是否违约：

\[
\hat p_D=1-\frac{1}{W}\sum_{w=1}^{W}\mathbb I(C_w>D_w).
\]

当前配置：

- 窗口 500 个工作流（20 workflow/episode 时约 25 episode）；
- `success >= 0.97` 才激活 C3C correction；
- `success < 0.95` 立即关闭 correction；
- 激活后经过 1000 个工作流线性升到完整强度；
- 0.97/0.95 构成迟滞区，避免每一轮反复开关。

门控只暂停 C3C 的“控制权”，不会停止 C3C critic 收集数据和学习。门控同时用于在线动作选择和 D3QN/C3C 的 next-action Bellman target，避免行为策略与训练 target 不一致。

### 5.3 将 C3C correction 对齐到 D3QN 候选 Q 尺度

对有效服务器候选，计算：

\[
\tilde A_C(a)=A_C(a)-\operatorname{mean}_{a'}A_C(a'),
\]

\[
\Delta Q_C(a)=
g\,\beta_{eff}(a)
\frac{\tilde A_C(a)}{\max(\sigma_A,\epsilon_A)}
\max(\sigma_Q,\epsilon_Q),
\]

并截断：

\[
\Delta Q_C(a)\in[-0.5\sigma_Q,\;0.5\sigma_Q].
\]

其中 `g` 是约束门控因子。这样 C3C 可以改变候选服务器排序，但不能用任意绝对尺度覆盖 D3QN。若 C3C 对所有候选给出几乎相同的 advantage，则 correction 直接置零。

### 5.4 增加必要诊断日志

训练/评估现在记录：

- gate 是否启用、是否激活、强度、窗口成功率；
- D3QN 候选 `Q` 标准差；
- C3C advantage 标准差；
- correction 平均绝对值及其相对 `Q` 标准差；
- C3C energy/deadline Q 和 TD absolute error；
- SAC 实际训练 reward、基础 reward、拉格朗日惩罚；
- 上层/下层/C3C 参数是否真的更新。

控制台中的 `reward=` 会打印当前配置的 profile，不再把 v2 错标为 v1。

### 5.5 checkpoint 保存约束门控状态

保存并恢复滚动 miss 窗口、active 状态和 anneal 进度，避免恢复后门控突然重置。

## 6. 修改文件

- `src/coordination/c3c.py`：C3C critic、Q 尺度对齐、门控后的 Bellman next action。
- `src/coordination/c3c_agent.py`：在线候选选择、门控状态机、checkpoint 状态。
- `tools/train_c3c_d3qn_sac.py`：门控更新、三网络/奖励/TD 诊断日志。
- `configs/hierarchical_d3qn_sac_c3c_v2.json`：独立 v2 配置。
- `tests/test_c3c.py`：奖励分解、三网络更新、checkpoint、门控、尺度截断测试。

## 7. 已完成验证

1. 四个修改文件均通过 `py_compile`；配置 JSON 可解析。
2. `tests/test_c3c.py` 中 5 个测试函数全部通过：
   - 奖励严格分解；
   - D3QN/SAC/C3C 三网络均更新；
   - checkpoint 往返一致；
   - 门控迟滞与退火正确；
   - correction 不超过 0.5 倍候选 Q 标准差。
3. v2 进行了 2-episode 冒烟训练，三网络都在 RTX 5070 CUDA 上运行并更新，奖励非零且无 NaN/Inf。

说明：项目的 `pytest` 当前未安装在 py310 环境，所以使用等价的直接函数执行完成了上述 5 项测试。PyTorch CUDA 包位于项目 `tmp/py310_site_packages`，C3C 训练脚本会自动加载；不要因为裸 `python -c "import torch"` 报错而重复安装 PyTorch。

## 8. 推荐下一步实验（不要直接先跑 1500）

先跑 600 episode 诊断：

```powershell
D:\Miniconda\envs\py310\python.exe tools/train_c3c_d3qn_sac.py --config configs/hierarchical_d3qn_sac_c3c_v2.json --episodes 600
```

通过条件：

1. ep300 以后固定 20-seed Deadline 不应持续下降；
2. gate 激活前，结果应接近普通 D3QN+SAC，而不是 v1 的快速漂移；
3. gate 激活后，`c3c_selection_correction_to_q_std_mean <= 0.5`；
4. 若成功率掉到 95% 以下，下一轮 gate 应回到 0；
5. 固定评估存在 `Deadline >= 99%` 的 checkpoint 后，才比较能耗是否低于 237.647 kJ。

只有 600 episode 诊断达到以上条件，才跑完整 1500：

```powershell
D:\Miniconda\envs\py310\python.exe tools/train_c3c_d3qn_sac.py --config configs/hierarchical_d3qn_sac_c3c_v2.json
```

## 9. 若 v2 仍不理想，严格按此顺序消融

不要同时调多个超参数。固定训练 seed、固定 20-seed eval、相同数据配置，依次比较：

1. 普通 D3QN+SAC（无 C3C）；
2. C3C-v1；
3. v2 仅约束门控，关闭 Q 尺度对齐；
4. v2 仅 Q 尺度对齐，关闭门控；
5. 完整 v2。

判断规则仍是先 `Deadline >= 99%`，再比较能耗。如果完整 v2 能守住 Deadline 但无法降能耗，优先尝试将 `correction_q_std_clip` 从 0.5 调到 0.25，而不是增大 lambda；如果 gate 几乎从不激活，先查普通 D3QN+SAC 本身是否能在训练滚动窗口达到 97%，不要下调底线掩盖不可行策略。

## 10. 当前已知注意事项

- 正在运行的 C3C-v1 进程在启动时已加载旧 Python 代码，本次文件修改不会动态改变该进程；v2 必须新开训练。
- `save_replay_buffer=false` 时，resume 不能视为统计意义上的严格续训，因为 replay 未恢复；正式对比应从 episode 1 独立训练。
- 不要把旧 C3C-v1 checkpoint 直接加载进 v2 后声称是完整 v2 训练结果；门控可兼容恢复，但 replay、策略分布和实验定义都不同。
- 当前资源配置存在若干 edge-efficient 服务器在频率、功率、可靠性、带宽上同时支配大量服务器的警告。这会让策略天然集中到少数服务器，也是后续资源建模需要单独做的敏感性实验，但不是本轮 C3C 退化的直接代码错误。
