# SM74 Deadline责任反馈消融：接口与训练指令

日期：2026-10-22。仅新增独立入口、三份配置、测试和本说明；不修改旧训练代码、网络、配置、论文或历史结果。本文提供启动指令，不代表已启动1200轮正式训练。

## 1. 版本定义与复用边界

新入口：`tools/train_deadline_credit_ablation.py`。三个配置均逐字段复制现有 `configs/v3_sm_74_legacy_train.json`，只增加 `deadline_credit_ablation.variant`、训练阶段名称和独立输出根目录。

令原有反事实函数输出的受影响者增量代价为d，其总量为C。

| 配置中的variant | SAC工作流责任G | 下层Deadline乘子输入 |
|---|---|---|
| cf_self | G=d，把增量损失给受影响工作流自身 | 原d |
| cf_uniform | C平均给累计执行增量>1e-8秒的来源工作流；其他为0 | 原d |
| raw_self | 截断到[0,1]的自身归一化实际迟到，不扣满频基准 | 同一实际迟到向量 |

- 三组均保留完整上层三Q网络、Deadline Head、SAC、局部耗时项、可靠性机制、频率能效下限。
- 从头联合训练，不加载旧checkpoint，不冻结上层，不固定rho=1。
- 复用原 `_workflow_counterfactual_low_deadline_credit` 计算增量参考，不重写调度回放；复用原collector把工作流责任按任务正执行增量分摊，零分母时按原规则均分。
- 新入口仅在本进程训练调用期间临时替换工作流责任函数；正常/异常退出均恢复原引用。其他进程和旧入口不受影响。不支持在同一Python进程的不同线程中同时运行这些入口；独立终端/进程可以。
- 冻结验证没有训练transition，且不学习责任奖励，故完整复用旧验证路径。原双块20seed验证、每20轮评估、选模门槛和并列规则不变；eval_workers=2不变。
- 继续调用 `coupled.run_training`，不另造训练循环。特别保留旧coupled入口对可靠性乘子输入的处理：按最终`reliability_met`生成二元失败指标；即时可靠性代价与SAC奖励公式也不变。不能改为直接使用裸collector的另一种可靠性输入，否则不再是单因素对照。
- 为复用并核对原参考，本实现仍计算Full反事实信息；不要据此声称控制组减少了回放计算开销。细节中的外部分配量对Uniform只是分账结果，不是证实的因果外部影响。
- CF-Self和CF-Uniform保持同轨迹总成本守恒。Raw-Self有意改变成本定义及对应乘子输入；预算数值仍为0.08，不宣称比较了各自最优调参结果。

## 2. 固定训练设置

- SM：Small科学工作流，74台服务器（56 edge + 18 cloud）。不是Medium工作流。
- 每轮20工作流，训练到达率0.1/s，Medium Deadline，旧`cai_2025_heft`生成器。
- 每次1200轮；默认模型随机种子42；三组模型/优化器/更新次数与原配置一致，Torch CPU线程1。
- 复用旧逐轮seed规则：第e轮输入seed=训练seed+e，因此默认43到1242。生成器使用独立的`random.Random(seed)`，不是探索随机数流；新增测试核对全局探索随机数变化不改变相同输入seed的任务/DDL指纹。
- 验证池仍为542000—542019和642000—642019。验证门槛：ESP@90=100%、Rel=100%，可行后选择最低验证能耗。
- 全程无可行验证checkpoint时，不伪造best、不自动加轮数。保留`final_checkpoint.pt`用于后续明确标注的不可行诊断。

## 3. VSCode终端启动

工作目录：`D:/研究生期间/第二篇/DRLEC`。先选好 `D:/Miniconda/envs/py310/python.exe` 对应的环境，使终端`python`使用该解释器。

CF-Self：

```powershell
python -u tools/train_deadline_credit_ablation.py --config configs/ablation_sm_74_cf_self_legacy_train.json --episodes 1200 --seed 42
```

CF-Uniform：

```powershell
python -u tools/train_deadline_credit_ablation.py --config configs/ablation_sm_74_cf_uniform_legacy_train.json --episodes 1200 --seed 42
```

Raw-Self：

```powershell
python -u tools/train_deadline_credit_ablation.py --config configs/ablation_sm_74_raw_self_legacy_train.json --episodes 1200 --seed 42
```

三个命令分别使用对应配置，不要把这些新配置交给旧训练脚本：旧入口不认识新顶层variant，会按Full运行。可在独立终端分别运行，但不要重复启动同一个版本同一seed。入口不自动批量启动、不停止其他进程、不自动重启。

如需仅检查配置，在对应命令后加`--check`，不会创建训练目录，也不会训练。

后续正式独立重复可把`--seed 42`统一换成43、44；本次没有自动安排或启动这些重复。历史Full可用于初步参考，不等于已完成同冻结代码、多训练重复的完整四组证据。

## 4. 输出与审计

每组写入独立目录：

```text
results/ablations/sm_74_deadline_credit_20wf/<variant>/seed42_<时间与PID>/
```

每次新建唯一目录，拒绝覆盖同名已有目录。保留原训练器的：

- `train_history.jsonl`及其原有汇总文件；
- `eval_history.jsonl`及双验证记录；
- `latest_checkpoint.pt`、周期checkpoint、`final_checkpoint.pt`；
- 仅验证可行时的`best_d3qn_checkpoint.pt`，以及原有组件checkpoint。

额外审计文件：

- `launch_config.json`：实际启动配置；checkpoint内也保存variant。
- `credit_ablation_manifest.json`：版本、实际seed、轮数、逐轮输入seed表、验证池、关键旧代码/资源配置SHA256、运行/异常/完成状态及是否得到可行验证模型。
- `workload_inputs.jsonl`：复用已有`tools.eval_alibaba_sl111.identity`记录输入的任务/到达/可靠性/DDL指纹。正式默认并行验证在子进程生成输入，故此文件主要记录父进程训练输入；不能当成完整验证输入清单。
- `deadline_credit_history.jsonl`：仅训练轨迹的来源责任、受影响者代价和分配详情；验证不混入此文件。其行序与训练轮序对应。

正式不支持本入口resume：原配置不保存replay，故不把重新加载权重声称为无缝续训。意外中断先保留文件并报告，再决定后续处理。

## 5. 检查范围

新增测试覆盖A拖慢B而C无关的分配差异、Full原路径等价、成本守恒、Raw基准迟到、无迟到、缺失计划拒绝、输入非法拒绝、异常后引用恢复、旧coupled可靠性处理保留、配置差异白名单、逐轮输入配对，以及每个版本一轮缩小模型/资源的端到端训练、双验证和checkpoint保存。

缩小资源/模型的一轮测试不是科学实验或正式预检信号证据。当前交付不宣称已验证1200轮收敛、外部责任信号强度或任何消融性能优势；后续正式结果须保留全部成败，不替换测试种子。

2026-10-22检查结果：新增12项测试与相关旧回归合计107项通过；另有原串行/并行验证一致性测试1项通过，共108项。三份配置的实际CLI `--check`均返回1200轮、seed42、`training_started=False`。开始前234个既有Python/JSON文件的SHA256逐一核对不变。未启动正式训练；端到端测试的临时checkpoint在测试结束时清理，不混入实验结果。

## 6. 2026-10-23：区分日志中的两种反事实参照

仅修改新增入口的诊断详情，不修改奖励、乘子输入、网络、配置、回放安排或旧训练函数。新训练记录明确包含：

| 字段 | 含义 |
|---|---|
| `all_full_frequency_response_time` | 所有工作流恢复满频后，该工作流的响应时间（秒） |
| `all_full_frequency_violation` | 上述情况下的归一化迟到，可大于0，不截断 |
| `self_restored_response_time` | 只恢复该工作流自身至满频，其他工作流保持实际频率时的响应时间 |
| `self_restored_violation` | 上述只恢复自身情况下的归一化迟到，不截断 |
| `counterfactual_reference` | 明确说明原`counterfactual_*`字段仍指“只恢复自身”，保留原字段兼容性 |
| `deadline_cost_basis` | CF两组标记`all_full_frequency_incremental_lateness`；Raw-Self标记`raw_actual_lateness` |

例如A拖慢B：B实际11秒、全部满频6秒、只恢复B仍11秒，两种参照会分别记录，不再混淆。CF-Self的处罚仍是0.375，没有改成0。

新增两项回归测试，覆盖三组的参照区分以及满频后仍迟到的正基准；端到端测试同时核对这些字段实际写入JSONL。对两条轨迹×三个版本逐项比较，修复前后的责任G、乘子成本d以及所有既有诊断字段完全一致。旧日志不回填，已运行进程不会自动获得新字段；不需要为了这项日志修正重新训练。

修复后检查：相关回归合计110项全部通过；三条实际CLI配置检查全部通过（1200轮、seed42，不启动训练）。239个Python/JSON文件的修改前后SHA256比较仅发现本次新增入口与其测试文件变化，所有旧生产代码和三份消融配置均未修改；另更新本说明。本次没有启动正式训练。
