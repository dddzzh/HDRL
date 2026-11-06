# SM74 两项消融：配置与手动启动（2026-10-14）

仅准备配置与指令；未启动训练，也未运行训练 smoke test。

## 场景与固定口径

- 本项目 SM = Small 工作流 / Medium 资源，74 台 = 56 edge + 18 cloud。
- 不把 SM 改成 Medium 工作流；Medium 工作流 / 74 台在九宫格里叫 MM。
- 训练沿用当前 Full SM 的旧口径：`cai_2025_heft`、20 workflows、0.1/s、Medium deadline、可靠性要求 0.9。
- 训练 seed 42、1200 episodes、每 20 episodes 验证/保存；两个验证块分别从 542000、642000 开始，各 20 seeds。
- checkpoint 选择仍沿用现有验证规则（ESP@90 target=1、可靠性 target=1，满足约束后优先最低能耗）；无可行 checkpoint 时保留真实回退状态。不能用最终测试来重新选模型。
- Torch CPU threads=1，验证 workers=2，device=auto，均与 Full 保持一致。两个独立终端可同时跑；不要重复启动同一项。
- 基础文件：[v3_sm_74_legacy_train.json](../configs/v3_sm_74_legacy_train.json)，SHA256 `b494950ce079c495d445d175fa45bd86d66e4cdd36fd2534deface7093d7ec78`。
- 已只读核对该基础配置与当前 Full checkpoint 内 `config` 完全一致：`results/v3_sm_74_legacy_train_20wf/20261009_115939/best_d3qn_checkpoint.pt`，SHA256 `32764207f74371ae00c0c3cfdf9c16bd760c0a3143be560dd7f72f2ec1c652ea`。

## 1. HEFT + SAC

配置：[ablation_sm_74_heft_sac_legacy_train.json](../configs/ablation_sm_74_heft_sac_legacy_train.json)。

只把上层服务器/任务选择替换为项目现有在线队列感知 HEFT；SAC、环境与其余配置不变。SAC 从零训练，不加载 Full 的旧 SAC，不训练 HEFT。复用现有 HEFT 实现，而非另写一个启发式。

在第一个 PowerShell 终端粘贴：

```powershell
Set-Location 'D:/研究生期间/第二篇/DRLEC'
& 'D:/Miniconda/envs/py310/python.exe' -u 'D:/研究生期间/第二篇/DRLEC/tools/train_coupled_hierarchical_d3qn_sac.py' --config 'D:/研究生期间/第二篇/DRLEC/configs/ablation_sm_74_heft_sac_legacy_train.json' 2>&1 | Tee-Object -FilePath ('D:/研究生期间/第二篇/DRLEC/train_sm74_ablation_heft_sac_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
```

模型输出：`results/ablations/sm_74_legacy_train_20wf/heft_sac/<时间戳>/`。
选定文件：`best_heft_checkpoint.pt`。实际是否可行要检查验证记录，不能只看文件名。

## 2. 去上层 Deadline Head，固定 rho=1

配置：[ablation_sm_74_high_no_deadline_rho1_legacy_train.json](../configs/ablation_sm_74_high_no_deadline_rho1_legacy_train.json)。

`agent.high.deadline_head_enabled=false`：Deadline head 不参与动作评分、TD 动作选择或学习，相关上层 Deadline dual 停用；保留能耗与可靠性头及环境约束。不是仅把 lambda 设成 0。沿用现有消融实现，底层仍可能分配兼容性的 Deadline 网络对象，不能据此宣称参数量/FLOPs 已减少。

**必须保留启动参数 `--fixed-low-rho-one`**：生产训练入口以该参数控制实际频率、是否启用 SAC 和验证频率；单独在 JSON 中设 `use_low_agent=false` 或 `default_rho=1` 不足以替代该参数。此项从零训练上层，既不加载 checkpoint，也不训练 SAC。

在第二个 PowerShell 终端粘贴：

```powershell
Set-Location 'D:/研究生期间/第二篇/DRLEC'
& 'D:/Miniconda/envs/py310/python.exe' -u 'D:/研究生期间/第二篇/DRLEC/tools/train_coupled_hierarchical_d3qn_sac.py' --config 'D:/研究生期间/第二篇/DRLEC/configs/ablation_sm_74_high_no_deadline_rho1_legacy_train.json' --fixed-low-rho-one 2>&1 | Tee-Object -FilePath ('D:/研究生期间/第二篇/DRLEC/train_sm74_ablation_no_deadline_rho1_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
```

模型输出：`results/ablations/sm_74_legacy_train_20wf/high_no_deadline_rho1/<时间戳>/`。
选定文件：`best_d3qn_checkpoint.pt`。运行日志应显示固定 rho=1 / lower disabled，lower update count 应保持 0。

## 对照边界与后续评估

- 两条命令均不带 `--resume-checkpoint` 或 `--high-checkpoint`，不得自动热启旧模型。
- 不使用 `tools/train_small_ablation.py`：该旧入口硬编码 20 edge + 6 cloud，不适用于 74 台。本次直接用已有生产入口，无须改旧入口或策略。
- HEFT+SAC 可与当前 SM Full 做结构比较，并补 HEFT rho=1、Full checkpoint rho=1 的纯评估对照。
- 去 Deadline Head 的严格单因素对照需要同资源、同训练口径、同种子的**独立训练完整上层 rho=1**。当前联合训练 Full checkpoint 改成 rho=1 只是诊断对照，不能声称隔离了唯一变量。本次按用户要求仅准备上述两项，未另外准备/启动完整上层 rho=1 的第三项。
- 每项目前只配置训练 seed 42，最终 20 个测试 seeds 不是 20 次独立训练；若仅一轮训练，报告为单训练种子结果。
- 训练结束后再单独评估：固定模型与 74 台资源，使用 `configs/v3_sm_74_new_eval_50wf.json`，50 workflows、`sec_job_fcfs_v1`、0.03/s。SM 子场景为 Small 测试工作流；MM/LM 可作为额外泛化测试。三个 deadline 档不分别重选 checkpoint。
- 正式未筛选测试建议预先固定 742000–742019，保留所有失败；若使用已按结果筛选的配对名单，单独标为条件样本，不与未筛选结果混报。
- 不覆盖原训练、原九宫格 CSV/JSONL 或已完成的新配对结果；训练完成不等于新协议评估完成。

## 已执行的检查（没有训练）

`tests/test_sm74_ablation_configs.py`：检查配置只有指定消融开关及输出标签变化；用 mock 阻断 `run_training` 后检查上述参数能正确传到生产入口。另保留旧 Small 配置及 Deadline head 行为回归检查。未通过实际训练 rollout 测试性能/收敛。
