# SM 工作流数量敏感性：冻结协议与 Research Question Card

日期：2026-10-17（Asia/Shanghai）。用户授权在 SM 扫描50、100、150、200、250个工作流并绘图。

## Research Question Card

- Question：固定 Small 科学工作流来源、74台服务器、到达率和模型，有限评估批次的工作流数量变化如何影响能耗与约束达标？
- Type：探索性的批次规模敏感性，不是到达率扫描、训练重复或新模型选参。
- Hypothesis：总能耗随工作量改变；单位工作流能耗、共享队列及约束达标可能变化，不预设单调或全部成功。
- Current evidence：原始未筛选 SM 四法已有50工作流的三档测试；当前没有100至250工作流的本组配对结果。
- Missing evidence：五档完整配对原始记录、归一化能耗、任务数量与到达时域、期限预算变化。
- Support / falsification：如指标随规模变化则报告幅度；如单位能耗与达标率稳定，则如实报告所测范围内的稳定性，不能人为改变故障率或筛选种子制造曲线。
- Minimal next action：检查前缀嵌套与全部样本指纹、50工作流 Full 复现门控，再完成四法评估与能耗图。
- Decision：执行首组 Tight 期限；未授权扩展三档期限笛卡尔积、阿里实验或新训练。

## 固定设计

- SM = Small 科学工作流，74台（56 edge +18 cloud），资源seed42。
- 沿用 `configs/v3_sm_74_new_eval_50wf.json`，只覆盖 generator.num_workflows 与 Tight 测试档。
- 数量依次50、100、150、200、250；到达率固定0.03/s，sec_job_fcfs_v1、smallest_workload、原有故障率与rho下限均不改。
- 四法原固定 SM checkpoint/loader，来源由已冻结 panel_plan.json 提取；只借用模型身份，**不用其筛选种子或结果**。
- 每档20个原始未筛选seed742000–742019，四法同名单；400条完整rollout，不删除失败，不按本实验选择新模型。
- 每个数量档重新从相同checkpoint和初始化742000建立策略；档内按seed升序连续执行。DQN-Edge保持这20条的attention EMA连续，不并行拆种子，不盲目续跑partial；不宣称其N=50等于原九档连续序列中的Tight状态。
- 四个算法进程并发，每进程单worker、Torch线程1；不停止其他进程、不启动训练、不自动重启。
- 相同seed大批次的DAG/计算量/边/到达应包含小批次的完全相同物理前缀。DDL按每个批次原有共享FCFS/first-idle/rho1参考重新计算，可能随批次变化，不强行固定或伪称前缀DDL不变。
- 50工作流 Full 的20条与原始 SM Small/Tight逐条比较；未通过则停止该模型扫描，保留异常，不改检查点或容差掩盖差异。

## 指标与输出

总计算能耗J均值±样本SD；每工作流能耗J；ESP@90（每批按时比例>=0.9的seed比例）；可靠性达标率（每批全部工作流R>=0.9的seed比例）。保存实际任务数、到达时域及参考相对DDL预算均值。

数量增加首先延长有限到达时域，不等于增加瞬时到达强度。严格的“全部工作流可靠性达标”事件也会随批大小变化；不能把它等同于平均单工作流成功概率。

输出独立 `results/workflow_count_sm74_20261017/`，原始结果不覆盖。最终图横轴Number of workflows、纵轴Total energy (J)，四条均值曲线与样本SD，框架配色，D3QN+SAC图例最后。精确能耗/单位能耗/约束指标另存JSON、CSV和REPORT，不重复画成功率图。

## Evidence Records / 允许结论

- ER-20261017-count-source-01：原始SM四法checkpoint_provenance与原始JSONL、panel_plan模型清单；支持身份/原协议，不支持新规模结果。
- ER-20261017-count-generator-02：SecJobWorkflowGenerator独立到达RNG、按index构造工作流；支持检验嵌套物理样本。DDL用完整批次参考，不能由物理前缀相同推断DDL相同。
- 允许：固定四个模型在所测批次数量上的能耗与经验达标表现。禁止：证明任意负载可行、多训练seed稳定、未筛选总体100%或实际阿里线上节能。
