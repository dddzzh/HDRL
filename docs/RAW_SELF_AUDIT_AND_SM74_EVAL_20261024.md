# Raw-Self实现核对与74台全场景评估

## 结论

2026-10-24核对：相对原budget0.08 Full，Raw-Self的算法改动集中于下层Deadline反馈，不是更换网络或禁用约束。但不能说只是删掉一个开关：它同时改变工作流代价定义、工作流间归属，以及送给下层拉格朗日控制器的代价输入。任务内分摊规则、PI/CVaR控制器公式及预算没有改。

实测Raw日志1200轮、24000工作流，source_credit和victim_dual_cost逐项等于clip(actual_violation,0,1)，最大差异0；external_responsible_cost逐项为0。

## 配置和模型身份

- 与原Full实际检查点 `results/v3_sm_74_legacy_train_20wf/20261009_115939/best_d3qn_checkpoint.pt` 中的config比较，agent/objective/generator/simulation/resources均完全一致。
- 与原训练配置比较，只存在variant=raw_self、输出目录、phase_name、run_name等差异；原下层budget仍为0.08，不是昨天新Full的0.01。
- 原科学工作流采样、隔离HEFT训练Deadline、训练seed42、20工作流/轮、1200轮和两块验证池不变。
- CF-Self/CF-Uniform/Raw-Self三组训练输入文件SHA256完全相同：5c425fa65f437a40ed60fc5b32a46a9ba53087f4c0da92e278704bf0f552dd94。
- Raw run：`results/ablations/sm_74_deadline_credit_20wf/raw_self/seed42_20261023_093836_720976_27828`。
- 按原双验证排序选中第1200轮best_d3qn_checkpoint.pt；checkpoint SHA256：d3911fdb917f0ece732aa776af46b2ddd34d0a3d1817264d4aac6609eed1f046。
- 训练完成及60双验证轮已核验；manifest记录的源码哈希无漂移。

以上“没变”指代码路径、配置和输入协议，不是训练后的网络权重、探索动作、经验缓冲区、乘子轨迹或最终策略相同。反馈改变后，这些学习结果自然会变化。

## 实际代码链路

1. tools/train_deadline_credit_ablation.py 的deadline_credit(raw_self)令G=d=clip(actual_violation,0,1)，不减去all-full参考、不使用来源分摊。
2. installed_credit只在该训练进程临时替换生产credit函数；正常和异常退出均恢复。训练仍调用coupled.run_training；冻结评估不学习Raw代价。
3. tools/train_coupled_hierarchical_d3qn_sac.py继续使用原agent类选择及可靠性二元违约输入，不改可靠性规则。
4. tools/train_hierarchical_d3qn.py仍按工作流内time_increase_seconds占比分摊G；分母为零时均分。不是给每个任务重复施加完整G。
5. src/sac/hierarchical_low_sac.py仍用当前lambda乘代价组成奖励；PI更新、窗口/CVaR、奖励标准化、rho能效下限不变。
6. _apply_cooperative_workflow_credit的上层实际迟到反馈和奖励公式不依赖Raw的G；相关low_causal_workflow_miss诊断可能变化，但不是上层损失公式修改。

注意：为了复用和记录诊断，Raw入口仍调用完整反事实计算。它不使用反事实结果构造Raw实际迟到代价，但不能据此声称当前实现已经省掉反事实运行开销。生产代码部分历史注释/字段名仍带counterfactual，不能仅凭名字判断实际代价。

## 评估范围修正

原主表27档来自37/74/111三份对应模型，不是一份模型跨全部资源。当前Raw只有74槽位；超过max_servers的上下文会拒绝，不能当111台模型使用。37台即便可用padding输入，也是另一个跨资源迁移实验，不能冒充37台独立训练的Raw模型。

用户已选择：先看74台全部结果，不启动37/111台训练。

- SM：Small，74台，三档期限；复用已有Raw的60条结果。
- MM：Medium混合工作流，74台，三档期限；新增60条。
- LM：Large混合工作流，74台，三档期限；新增60条。
- 每档20个未筛选连续种子742000—742019，每种子50工作流，到达率0.03/s，共180条。
- 测试仍用sec_job_fcfs_v1共享FCFS期限，不是修改后的共享HEFT训练参考。
- 固定模型初始化seed742000、资源seed42、两个评估worker、每worker Torch线程1；不重选checkpoint、不换失败seed、不训练网络、不覆盖旧结果。

## 文件入口

- 新工具：[eval_raw_sm74_generalization.py](D:/研究生期间/第二篇/DRLEC/tools/eval_raw_sm74_generalization.py)
- 汇总生成位置：`results/raw_sm74_generalization_20261024/REPORT.md`
- 合并的逐种子CSV/JSONL同目录；SM60条来自旧Raw结果，MM/LM120条为新评估，来源在frozen_plan.json中。
- 完成判定：completion.json包含180条验证，日志raw-generalization-complete且进程正常退出；只有部分行时不报告为完成。

本次没有修改Raw训练/策略实现，只新增评估编排和审计材料。新增3项验证加复用6项评估选择/输入回归已通过。
