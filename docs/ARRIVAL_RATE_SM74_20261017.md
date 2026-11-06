# SM 到达率敏感性：冻结协议

## Research Question Card

- Question：固定工作流任务、资源、相对DDL预算和四法模型后，到达流压缩是否改变能耗优势、队列拥塞和约束达标？
- Type：探索性有限批次到达率敏感性；不是训练选参或稳态容量证明。
- Hypothesis：提高到达率会加强工作流间竞争，可能改变能耗、等待、DDL裕量与可靠性；不预设所有方法成功或单调变化。
- Current evidence：SM/Tight数量扫描400条已完整核验，但固定到达率不能检验到达强度变化。
- Missing evidence：相同工作流/相同相对预算下的密集到达、实际队列与可靠性裕量证据。
- Support/falsification：报告真实曲线及失败；若没有明显拥塞或约束下降，则结论限于本次负载范围，不加种子或改参数制造失败。
- Minimal next action：预检全部100组输入、测试预算与物理身份不变，0.03/s四法逐行复现门控，再完成预注册扫描。
- Decision：执行，不启动训练，不更换checkpoint，不用按结果筛选的种子。

## 固定设计

- SM = Small科学工作流，74台（56 edge +18 cloud），每批50工作流，Tight，资源seed42。
- 到达率0.01/0.03/0.06/0.10/0.20 workflow/s，对应期望间隔100/33.33/16.67/10/5秒。
- 每点四法各20个未筛选seed742000–742019，共400条rollout。四模型沿用原SM检查点及各自训练loader配置。
- 模型初始化742000，每个到达率重新加载同一模型；档内20种子升序连续，尤其DQN-Edge attention EMA不能拆种子、partial重载或拼接。
- 执行顺序先0.03再0.01/0.06/0.10/0.20；图表按到达率升序。0.03的四法80条逐条与已完成数量扫描N50比对能耗/两类成功指标。这是明确的复现门控，不合并旧行充数。
- 四算法进程并发、每模型workers1/Torch线程1；无其他任务时独立运行，不暂停其他进程、不自动重启。

## 关键控制：固定相对DDL预算

每个seed先按原0.03/s、50工作流、Tight、共享FCFS/first-idle/rho1生成基准。冻结每个工作流的DAG、cycles、边数据、可靠性门槛、reference_makespan、DDL倍率及relative_deadline。

仅将每个arrival和inter_arrival按0.03/new_rate缩放，再设置absolute_deadline = new_arrival + frozen_relative_deadline。相同底层指数随机变量保证成对到达流；原生成器首次到达是正指数间隔，保持这一约定，不平移到0。

**不**在每个新到达率重新用拥塞的参考调度放宽DDL。absolute_deadline随arrival平移，不能同时固定绝对时间并声称只变到达率。保留基准reference_makespan的含义，不伪称它是新负载下的参考响应。

这是独立的fixed-relative-budget压力协议，非原主表各档重算参考DDL的替代；不覆盖原始config或结果。simulation故障率、可靠性门槛0.9、rho物理下限/能效下限、调度/网络/能耗模型不变。

## 指标、审计与输出

- 总计算能耗J均值±样本SD、每工作流能耗、ESP@90、每批全部工作流可靠性达标率、两约束联合达标率；失败也进入能耗汇总，不把未达标的低能耗称为有效节能。
- 保存每个工作流的arrival/finish/相对与绝对DDL、response、R与门槛；保存seed级等待时间、活动利用率、响应时间、C/D的p95/最大值、最小可靠性裕量、最大活动工作流数、wall-clock执行时间。时间为带诊断的评估耗时，不是严格隔离的在线推理延迟benchmark。
- 每批完成50工作流；四法能耗与两类比例独立从环境明细核对。所有行/样本和预算指纹、模型/config/代码SHA、manifest初始化与顺序均核验。
- 不把20次全部达标解释为总体100%保证；不等同所有工作流按时，也不等同R=1。利用率不高时不称饱和压力；有限50工作流不称稳态容量上限。
- 输出独立results/arrival_rate_sm74_20261017，原始数据不覆盖；JSONL/CSV、逐工作流JSON、summary.json/csv、REPORT.md。主图仅能耗，约束/拥塞与裕量另存表；不自动改论文或主场景汇总。

## Evidence / Claim boundary

- ER-20261017-arrival-source-01：workflow_count_sm74_20261017冻结模型清单、N50原始80行及源SHA；只支持模型身份与基准表现。
- ER-20261017-arrival-generator-02：SecJobWorkflowGenerator到达、模板和workload独立随机源；预检验证缩放到达并保持任务与相对预算，不能由到达率增加推定已经拥塞。
- 允许：固定模型在所测到达强度、固定相对预算下的有限批次表现。禁止：任意负载可靠性保证、多训练种子稳定性或实际硬件故障率证明。
