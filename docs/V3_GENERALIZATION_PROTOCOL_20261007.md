# V3: three Small-trained models, 27 evaluation conditions

## 1. 本轮采用的实验定义

最小资源设为 37 台。参考师兄实验的 **1:2:3 资源数量比例**，不迁移他的 MI/MIPS、Host/VM 功耗模型。保留 DRLEC 的 cycles/Hz、通信、能耗、可靠性模型。

| 训练模型 | 工作流 | 资源级别 | 边缘 + 云 | 总服务器 | 训练 Deadline | 当前配置 |
|---|---|---|---:|---:|---|---|
| SS | Small | Small | 28 + 9 | 37 | Medium | `configs/v3_ss_37.json` |
| SM | Small | Medium | 56 + 18 | 74 | Medium | `configs/v3_sm_74.json` |
| SL | Small | Large | 84 + 27 | 111 | Medium | `configs/v3_sl_111.json` |

三份配置仅服务器数量、输出目录不同。资源类型与参数范围来自同一个 `configs/resources_edge_cloud_dvfs_v3.json`，使用现有资源生成器，类型数量存在整数取整；不是逐台完全复制的资源池。

**这是 3 个模型泛化到 27 个测试条件，不是训练 9 个模型。** 当前网络使用固定服务器槽位，因此每份模型在测试时保持其训练服务器数量。不把 SS 的 37 槽网络改成 111 槽，也不载入不同容量的 checkpoint。

## 2. 测试矩阵与模型映射

第一个字母表示工作流组合；第二个表示资源级别。每一行均测试 Loose / Medium / Tight，且不再训练。

| 测试场景 | 工作流组合 | 资源数量 | 使用模型 |
|---|---|---:|---|
| SS | Small | 37 | SS 的唯一选定 checkpoint |
| MS | Small + Medium | 37 | 同一个 SS checkpoint |
| LS | Small + Medium + Large | 37 | 同一个 SS checkpoint |
| SM | Small | 74 | SM 的唯一选定 checkpoint |
| MM | Small + Medium | 74 | 同一个 SM checkpoint |
| LM | Small + Medium + Large | 74 | 同一个 SM checkpoint |
| SL | Small | 111 | SL 的唯一选定 checkpoint |
| ML | Small + Medium | 111 | 同一个 SL checkpoint |
| LL | Small + Medium + Large | 111 | 同一个 SL checkpoint |

9 行 × 3 档 = **27 个测试条件**。训练点为 SS-Medium、SM-Medium、SL-Medium，因此 27 个条件包含 3 个同分布评估点与 24 个迁移条件；不能全部称为跨场景泛化。

工作流采样采用当前生成器的真实定义：

- Small：五类小 DAG，均匀采样，任务数量为 24/25/29/30。
- Medium（配置值 `middle`）：小 DAG 组占 30%，中 DAG 组占 70%。
- Large（配置值 `large`）：小/中/大 DAG 组占 20%/30%/50%；不是每个工作流都是 100 个任务。
- 每个 episode 都是 50 个工作流，组比例是抽样概率，不强行固定每个 episode 的数量。

## 3. 当前到达与 Deadline 协议

所有新训练与测试均使用 `generator.profile=sec_job_fcfs_v1`：

1. 到达率 0.03/s，指数间隔均值 33.33 秒；累加生成到达时间，第一条也在正时间到达，不是全批同时到达。
2. 到达、模板、任务物理负载、Deadline 倍率使用独立随机流。给定工作流规模与种子，换资源数量或 Deadline 档位不会改变 DAG/负载/到达序列。
3. 在 **当前实际资源池** 上，给同批 50 个工作流做一次共享负载 FCFS / first-idle / 满频参考调度，包含通信和争用。
4. 每个工作流的参考响应时间为最后任务完成时间减到达时间。相对 Deadline = 参考响应时间 × 2 或 3；绝对 Deadline = 到达时间 + 相对 Deadline。
5. Tight/Medium/Loose 取 2 倍的概率分别为 0.8/0.5/0.2，否则取 3 倍。训练只用 Medium。

换资源规模会重新计算参考 Deadline。这是**平台归一化的相对 Deadline 难度**，不是固定绝对 Deadline 下增加服务器的因果实验。不能把资源更多后的成功率变化单独归因为容量；参考 Deadline 也变了。

当前 Pegasus 工厂默认已经切为新协议；直接调用旧类名 `PegasusWorkflowGenerator.generate` 也会转到同一实现。显式旧 profile 或旧到达配置会报错，不会偷偷走旧路径。

删除的是旧 Pegasus 的采样/孤立 HEFT Deadline 生成主体。XML 解析、HEFT 对比调度器、自定义/Alibaba 工作流所需的孤立 HEFT 工具不删除，它们不是三份配置的 Deadline 生成入口。删除前的源文件可在 `old/generation_pre_sec_job_20261007` 恢复。旧输出和 checkpoints 未删除。

## 4. 算法和训练设置

复用 `tools/train_coupled_hierarchical_d3qn_sac.py`，仍是三头 D3QN + SAC 联合训练。没有改变网络机制、奖励公式、频率动作范围或拉格朗日更新算法。

- 上层 Deadline：episode failure 目标，`episode_deadline_target=0.90`，预算 0.1；注意这允许训练窗口有 10% 的失败 episode，不是训练过程保证 ESP=100%。
- 下层保留 workflow 反事实责任归因、Deadline CVaR 对偶及可靠性对偶。
- 队列信用、剩余关键路径安全状态、全局负载感知等仍使用当前 future 版本的实现。
- 1200 episodes；每 20 轮评估一次，选择块 542000–542019 与验证块 642000–642019 各 20 个固定种子。周期 checkpoint 每 20 轮保存，CPU 线程 1。
- 本地评估默认 2 个进程，仅并行不同种子的冻结模型 rollout。训练、回放更新和 λ 更新仍在主进程；无新增依赖，暂不部署服务器。
- 不冻结上下层，不设 `--fixed-low-rho-one`，不加载旧 checkpoint，从零训练。
- 当前训练 seed=42；本轮是每种资源一份训练模型，**20 个测试种子不等于 20 个训练种子**。

训练期间的候选选择：每个种子至少 45/50 工作流按时完成才算通过；**两块各 20 个种子全部通过（各自 ESP@90=100%），且可靠性全部达标**，才写 best checkpoint。符合条件的候选按两块平均能耗最低选择；不再用平均 Deadline 成功率作闸门。没有通过候选时仍保存周期/final checkpoint，但不写正式可行 best，不宣称它们可行。

2026-10-08 恢复多种子约束筛选口径：每 20 轮两块各 20 个种子，1200 轮共 2400 个评估 rollout（60×40）。这不是 20 个训练种子，也不是把任务更新改成每 20 轮一次。多进程减少评估等待时间，不减少评估数量，不保证固定倍数提速。**仅恢复评估/选择协议，工作流到达及 Deadline 仍采用第三节的新生成协议。**

正式报告前，将候选用选择块 542000–542019 与验证块 642000–642019（各 20 个种子）确认，并在训练分布 Small+Medium Deadline 上锁定每个资源级别的一份模型。不得在 Large/Tight 上重新挑模型。若没有满足选择条件的候选，明确报告不可行，不换种子、不放宽阈值。

正式测试统一预留 **1042000–1042019**，三份模型、三个工作流规模、三档 Deadline 与未来对比算法均使用同一整块，不替换失败种子。报告 ESP@90、可靠性达标率、能耗均值 ± 样本标准差（J）。50 个工作流中至少 45 个按时完成才算该种子 Deadline 通过。

旧生成协议的结果单独保留。新协议、旧协议的能耗与 ESP 不直接拼在同一主表。若用新版结果写正式对比，baseline 也需要在同一新协议上训练/测试；本次未修改它们的算法或训练配置。

## 5. 训练命令（按顺序运行，不必三个同时开）

默认已启用本地双进程评估，运行入口仍是 coupled，不加恢复旧 checkpoint 参数。若显存/内存不足，在运行前设置 `$env:DRLEC_EVAL_WORKERS="1"` 回到串行；要试 4 进程设置为 `"4"`，但先验证机器余量。每个评估进程独立持有冻结模型（包括 GPU 上的模型），因此进程越多不一定越快。

SS：

```powershell
& D:\Miniconda\envs\py310\python.exe tools\train_coupled_hierarchical_d3qn_sac.py --config configs\v3_ss_37.json *> train_v3_ss_37.log
```

SM：

```powershell
& D:\Miniconda\envs\py310\python.exe tools\train_coupled_hierarchical_d3qn_sac.py --config configs\v3_sm_74.json *> train_v3_sm_74.log
```

SL：

```powershell
& D:\Miniconda\envs\py310\python.exe tools\train_coupled_hierarchical_d3qn_sac.py --config configs\v3_sl_111.json *> train_v3_sl_111.log
```

日志用于保存队列校准信息。要实时看进度，可在另一终端运行 `Get-Content train_v3_ss_37.log -Tail 30 -Wait`，另两份相应换日志名。

输出根目录依次为 `results/v3_ss_37_sec_job_fcfs`、`results/v3_sm_74_sec_job_fcfs`、`results/v3_sl_111_sec_job_fcfs`；训练器在其下面新建时间戳目录。

## 6. 锁定 checkpoint 后的泛化评估入口

复用 `tools/eval_all_baselines.py` 的 `d3qn_sac` 模式；不需要再写一个训练器或 27 份重复配置。示例为 SS 模型的 9 个条件，checkpoint 路径必须换成最终锁定的那一份：

```powershell
& D:\Miniconda\envs\py310\python.exe tools\eval_all_baselines.py --config configs\v3_ss_37.json --d3qn-checkpoint "实际锁定的SS完整checkpoint路径" --modes d3qn_sac --scales small,medium,large --deadline-types loose,medium,tight --seed-start 1042000 --num-seeds 20 --output-dir results\v3_generalization\ss_37
```

SM/SL 相应使用 `v3_sm_74.json` / `v3_sl_111.json`、各自锁定 checkpoint 与独立输出目录 `sm_74` / `sl_111`。三次评估一共 27 条件 × 20 种子 = 540 个完整 rollout。不要使用旧诊断脚本的硬编码 ep1060/37 槽检查来评估这些新模型。

评估中网络与 λ 冻结，确定性动作，不训练，不 resize，不按 Deadline 档位换 checkpoint。当前仅跑本文模式时，评估器不再加载无关 baseline 的旧默认配置文件。

## 7. 已完成检查与仍未验证的内容

2026-10-08 本地多进程检查：`tests.test_parallel_evaluation`、`tests.test_checkpoint_selection`、`tests.test_sec_job_generation` 共15个测试通过。37/74/111台服务器的CPU测试及37台服务器的CUDA测试中，双进程与串行的完整指标字典一致，主进程探索率、λ和随机状态未改变；每种测试仅用2个种子、每种子2个工作流，不代表完整训练结果。coupled入口的1轮smoke检查亦通过；smoke每块仅1个种子，实际训练仍按配置每块20个。未测试服务器部署，未启动新的1200轮训练。

复查本次改动：

```powershell
& D:\Miniconda\envs\py310\python.exe -m unittest tests.test_parallel_evaluation tests.test_checkpoint_selection tests.test_sec_job_generation -q
```

以下为2026-10-07首次准备配置时的检查记录：

- 已确认之前 `S37_sec_job_medium_matrix_5seed_20261007` 用了新协议，不是旧 HEFT Deadline；那份是冻结旧 S37 网络的诊断，不是新训练模型结果。
- 已检查三个配置除资源数量/输出目录外一致。
- 已检查 37/74/111 台资源实际生成数量，三种工作流规模均能生成完整 Deadline。
- 已检查工厂默认与直接调用一致、旧协议会拒绝、随机流独立、共享参考不重叠。
- 已创建三个实际容量的 coupled agent，完成上/下层动作选择与环境一步交互，无状态/动作维度错误。
- 未启动完整训练，未宣称泛化 ESP 必为 100%。5 个种子诊断没有这种证明力。

可复查：

```powershell
& D:\Miniconda\envs\py310\python.exe -m unittest tests.test_sec_job_generation tests.test_pegasus_xml_generator tests.test_checkpoint_selection -q
```
