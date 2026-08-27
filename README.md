# DRLEC: 动态边云工作流调度

当前主线是面向 Deadline 和可靠性约束的分层强化学习：

- 上层三头 D3QN 选择主服务器：能耗价值头、Deadline 代价头和可靠性代价头。
- 下层 SAC 输出连续 DVFS 频率比 `rho`，在约束底线下降低计算能耗。
- 上下层分别使用 Deadline 和可靠性拉格朗日乘子。
- 支持 Small/Middle Pegasus 科学工作流、泊松到达、异构 Edge–Cloud 资源和 20-seed 确定性评估。

## 主要入口

```text
tools/train_coupled_hierarchical_d3qn_sac.py    当前三头 D3QN＋SAC 训练入口
tools/eval_dual_head_d3qn.py                    D3QN 系列 checkpoint 评估
tools/eval_all_baselines.py                     对比算法统一评估
tools/accept_v2_checkpoints.py                  按 Deadline/可靠性门槛筛选 checkpoint
```

## 环境

推荐使用已验证的 Conda `py310` 环境：

```powershell
D:\Miniconda\envs\py310\python.exe -m pip install -r requirements.txt
```

GPU 检查：

```powershell
D:\Miniconda\envs\py310\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 训练最新版本

Small（20 Edge＋6 Cloud）：

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_coupled_hierarchical_d3qn_sac.py --config configs\v2_small_20edge_6cloud.json
```

Middle（28 Edge＋9 Cloud）：

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_coupled_hierarchical_d3qn_sac.py --config configs\v2_medium_28edge_9cloud.json
```

两个配置默认都是上下层联合训练，不会固定 `rho=1`，也不冻结任一层。

## 对比算法

保留 DQN-Edge、DReaM、OO Actor-Critic 和 HIRL：

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_dqn_edge.py --config configs\dqn_edge.json
D:\Miniconda\envs\py310\python.exe tools\train_dream.py --config configs\dream_dqn.json
D:\Miniconda\envs\py310\python.exe tools\train_oo_actor_critic.py --config configs\oo_actor_critic.json
D:\Miniconda\envs\py310\python.exe tools\train_hirl.py --config configs\hirl_td3_ddqn.json
```

## 目录

```text
configs/    最新 Small/Middle 配置、资源参数和对比算法配置
data/       Pegasus 科学工作流 XML
docs/       当前 V2 结果、根因分析和对比算法说明
src/        算法、数据集、调度器和仿真环境
tests/      核心模块与对比算法测试
tools/      训练、评估、诊断与数据工具
results/    本地训练输出（Git 默认忽略）
```

## 当前文档

- `docs/V2_RESULTS_20260827.md`：已验证实验结果。
- `docs/V2_ROOT_CAUSE_AND_FIX_20260827.md`：上下层约束与 DVFS 问题的根因分析。

训练 checkpoint、replay buffer、生成图表和本地论文 PDF 均不提交到 Git，避免仓库膨胀。
