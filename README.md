# DRLEC: 动态边云工作流调度

## 当前论文数据入口（2026-10-31整理）

- 正式实验数据、统计证据和14个保留模型：[results/paper_current/README.md](results/paper_current/README.md)。
- 历史结果及清理清单：[old/paper_cleanup_20261031/README.md](old/paper_cleanup_20261031/README.md)。旧评估数据和失败记录未删除，未采用模型与中间checkpoint已按授权删除。
- 数据核验：`python -B -X utf8 tools/verify_paper_current.py`。绘图命令见正式数据目录说明。
- `tests/`回归测试目录已按授权移除；恢复包存放在历史归档内。数据集、训练与评估代码保持保留。

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

## 当前实验配置

论文主线使用 `configs/resources_edge_cloud_dvfs_v3.json`。请在命令中显式传入下列配置；部分旧脚本的默认配置仅供历史实验复现。

| 场景 | 本文方法 | DQN-Edge | DReaM-DQN | OOAC |
| --- | --- | --- | --- | --- |
| Small | `configs/v2_small_20edge_6cloud_dvfs_v3_future_esp.json` | `configs/dqn_edge_small_20edge_6cloud.json` | `configs/dream_dqn.json` | `configs/oo_actor_critic_small_20edge_6cloud.json` |
| Medium | `configs/v2_medium_28edge_9cloud_dvfs_v3_future_esp.json` | `configs/dqn_edge_medium_28edge_9cloud.json` | `configs/dream_dqn_medium_28edge_9cloud.json` | `configs/oo_actor_critic_medium_28edge_9cloud.json` |

旧版配置已清理；`beta_sd` 配置仅为历史实验报告保留，不作为当前主线。

## 训练当前版本

Small（20 Edge＋6 Cloud）：

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_coupled_hierarchical_d3qn_sac.py --config configs\v2_small_20edge_6cloud_dvfs_v3_future_esp.json
```

Middle（28 Edge＋9 Cloud）：

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_coupled_hierarchical_d3qn_sac.py --config configs\v2_medium_28edge_9cloud_dvfs_v3_future_esp.json
```

两个配置默认都是上下层联合训练，不会固定 `rho=1`，也不冻结任一层。

## 对比算法

当前论文表格使用 DQN-Edge、DReaM-DQN、OOAC：

```powershell
D:\Miniconda\envs\py310\python.exe tools\train_dqn_edge.py --config configs\dqn_edge_small_20edge_6cloud.json
D:\Miniconda\envs\py310\python.exe tools\train_dream.py --config configs\dream_dqn.json
D:\Miniconda\envs\py310\python.exe tools\train_oo_actor_critic.py --config configs\oo_actor_critic_small_20edge_6cloud.json
```

Medium 场景使用上表对应的三份 `medium_28edge_9cloud` 配置。已完成的训练结果位于本地 `results/`，不随 Git 上传。

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
- `docs/ABLATION_PROTOCOL_20260922.md`：当前版本的配对消融协议与需重训的变体。

训练 checkpoint、replay buffer、生成图表和本地论文 PDF 均不提交到 Git，避免仓库膨胀。
