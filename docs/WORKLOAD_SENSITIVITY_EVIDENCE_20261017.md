# 工作负载敏感性：论文写入与证据索引

已写入当前稿件 `paper/manuscript/main2.tex` 的 `Sensitivity to Workflow Count and Arrival Rate`，标签 `subsec:workload_sensitivity`；10行联合数据表 `tab:workload_sensitivity`。旧稿docs/manuscript_revised_20261006没有混入新数据。

## Evidence Records

### ER-20261017-workload-count-01

- Source：results/workflow_count_sm74_20261017/frozen_plan.json、summary.json及四法n50至n250原始comparison_rows.jsonl。
- Type：experiment artifact，完整400条，20个统计单元。
- Supports：固定0.03/s时，50至250工作流中D3QN+SAC平均计算能耗均最低；相对最低能耗基线DReaM降低22.15%–22.40%；单位工作流能耗7.49–7.57kJ；四法ESP@90和批次可靠性达标均100%。
- Limitation：数量增加延长到达时域，DDL按批次重算，不等同增加瞬时负载；单训练模型，不是多训练seed。
- Claim strength：observed；没有对本组进行显著性检验，不写“统计显著”。

### ER-20261017-workload-arrival-02

- Source：results/arrival_rate_sm74_20261017/frozen_plan.json、summary.json、REPORT.md及逐工作流workflow_details；代码tools/eval_sm74_arrival_rate.py，协议docs/ARRIVAL_RATE_SM74_20261017.md。
- Type：experiment artifact，完整400条，20个统计单元，4条真实DDL门槛未达标记录保留。
- Supports：固定50工作流与0.03/s生成的相对预算，扫描0.01/0.03/0.06/0.10/0.20/s；D3QN+SAC五点能耗最低、ESP和可靠性达标均100%。0.20/s DQN-Edge/DReaM的ESP各90%，仍满足论文90%经验服务门槛，不能写为不可行；OOAC100%。最高点ours比OOAC低9.02%、比DReaM低15.70%。
- Congestion evidence：ours平均任务等待1.001→2.211s；活动窗口平均利用率1.68%→24.24%，表明时间竞争增加，不证明全系统饱和。
- Reliability evidence：全四法所有工作流最低R约0.93415；ours最低约0.99804，门槛0.9。100%是经验门槛事件，不是R=1或独立可靠性模块收益。
- Limitation：有限批次、Small/74/Tight、固定训练模型；0.03点80条是数量N50点的明确复现门控，不是独立新增样本。
- Claim strength：observed。

## Allowed / forbidden wording

允许：固定模型在所测批次规模和固定相对预算的到达强度下维持节能优势；两组是互补的单因素敏感性证据。

禁止：任意高负载可行、达到饱和容量、数量×到达率联合鲁棒、可靠性为1、可靠性模块的独立因果贡献、多训练seed稳定或显著优于所有方法。

相对节能百分比用未舍入组均值计算；表格kJ均值±样本SD按两位小数显示。表中仅显示百分比，不显示分子/分母。论文原始主表和筛选实验保持分离，没有修改原实验数据或策略代码。
