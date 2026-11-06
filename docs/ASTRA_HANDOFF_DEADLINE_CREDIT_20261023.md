# Astra交接：Deadline归因消融的负结果、乘子归零与最小干预方案

更新日期：2026-10-23。

## 0. 先读：任务范围与执行边界

用户希望解决当前CAHDRL反事实Deadline归因消融的问题，尽可能利用已有训练与测试数据，少重跑。本文件是事实、代码入口与待办交接，不是要求Astra立即启动训练。

- 工作目录：D:/研究生期间/第二篇/DRLEC
- Python解释器：D:/Miniconda/envs/py310/python.exe；用户使用VSCode，给启动指令时偏好以python开头。
- 本次仅新增本MD；没有改动生产代码、配置、checkpoint、论文或实验结果，没有启动训练/环境评估。
- 用户最后明确要求：把现存问题和代码位置写清楚，交给Astra处理。请先独立复核并提出方案；实际代码修改、正式重训与评估按新聊天中用户明确授权执行。
- 范围是当前Deadline反馈四版本消融，不是整仓、整篇论文已经完成的新一轮全面审计。
- 保留全部正负结果；不要为了让Full获胜而更换测试种子、按测试结果重选checkpoint、删失败样本、改指标或挑选有利场景。
- 不修改旧结果或覆盖原配置。优先新增独立配置/诊断工具，复用已有训练与回放函数；必要修改先说明影响范围。
- 不动论文main2，除非用户另行授权；不启动旧候选筛查、旧评估或自动重复任务；不停止其他进程。

## 1. 问题概览：事实、解释与假设分开

| 问题 | 已确认事实 | 尚不能认定 |
|---|---|---|
| 反事实机制优势未被完整支持 | 当前Raw-Self在五个测试条件的平均能耗均低于历史Full；压力条件下迟到也更少 | 不能写“Full全面优于原始自身反馈”；也不能据单训练种子断言机制普遍无效 |
| 下层Deadline惩罚后期归零 | Full最后有非零采样平均Deadline惩罚的训练轮为428；429—1200该项为0 | 尚未证明这就是Full不及Raw的原因 |
| 代价与预算的解释 | 实际下层观察量是连续归一化增量迟到；预算0.08不是8%工作流迟到率 | 不能仅看到字段workflow_miss_rate就把输入解释成二元失败指标 |
| PI控制器的作用 | 长期低于预算会降低积分项；负比例项可在积分项仍正时把输出截为0 | 这符合当前代码，不应未经论证认定为程序bug |
| 四版本证据强度 | 新训练三控制组都只有训练seed42；Full是历史训练模型和历史未筛选测试数据 | 20测试种子不等于20次独立训练；不是四版本同冻结源码新训、多训练重复 |
| 日志命名覆盖 | 训练行的low_deadline_constraint_cost_mean最终可被SAC采样均值覆盖 | 它不能直接当成整轮工作流代价均值；详见第6节 |
| 方案尚未执行 | 曾建议Full仅把下层预算0.08改为0做诊断 | 这不是已验证修复，不是原训练协议，不保证Full胜出 |

核心目标：先判断“反馈代价尺度/预算与控制器相互作用，是否削弱了归因消融的可识别性”，不要把任务设成“想办法让消融必胜”。

## 2. 方法与四个版本到底区别在哪里

### 2.1 共用结构

完整上层三个独立dueling Double-Q估计器与下层共享SAC；保留Deadline Head、可靠性反馈、局部能耗/时间项、频率能效下限规则。不是每个工作流一个独立SAC。

本消融只处理下层Deadline责任反馈，不能单独验证可靠性机制或整个上下层差异化框架。

### 2.2 代价与分配

对工作流v定义：

- ell_actual(v) = max(response_time(v) / relative_deadline(v) - 1, 0)。
- ell_full(v)：固定排程回放中所有工作流恢复全频后的同类归一化迟到。
- d(v) = clip(ell_actual(v) - ell_full(v), 0, 1)。
- Delta(u,v)：仅恢复来源工作流u至全频后，受影响工作流v归一化迟到的非负减少量。

| 版本 | 工作流责任G | 下层Deadline乘子观察向量 |
|---|---|---|
| Full | 对每个v，将d(v)按Delta(u,v)的归一化比例分配给来源u | d |
| CF-Self | G(v)=d(v)，把同一增量代价给受影响者自身 | d |
| CF-Uniform | 将sum(d)均分给累计正执行时间增量大于1e-8秒的来源工作流 | d |
| Raw-Self | G(v)=clip(ell_actual(v),0,1)，不扣全频基线 | 同一个Raw向量 |

补充边界：

- Full零边际分母时，原函数使用来源工作流累计执行减速量分摊；不是Shapley值。
- 四版本均复用原任务级分配：工作流责任按正的任务执行时间增量分到transition；零分母沿用均分。
- Full、CF-Self、CF-Uniform在同一给定轨迹上守恒总成本，乘子输入相同；独立训练产生不同轨迹后，不能要求它们实际乘子曲线相同。
- Raw同时改变代价定义和相应乘子输入，不是纯“来源责任分配”单因素；原协议仍使用相同数值预算0.08，未对每种定义独立调到最优。
- CF-Self不是“只看当前任务的即时局部奖励”，而是工作流自身终局增量迟到。
- 所有回放保留服务器放置、服务顺序、依赖/通信和记录的决策释放约束；不重新调度。解释只能限定为该固定回放模型中的归因，不能直接提升为真实系统因果效应。
- 控制组三个入口仍调用Full回放计算参考信息，不能据此声称Raw计算开销更低。

## 3. 代码地图（行号是交接时定位，后续编辑可能变化）

| 内容 | 文件与定位 |
|---|---|
| 固定排程模型、回放 | [train_hierarchical_d3qn.py:1026](D:/研究生期间/第二篇/DRLEC/tools/train_hierarchical_d3qn.py:1026)，回放函数1104 |
| Full增量代价d、边际Delta、责任G | [train_hierarchical_d3qn.py:1177](D:/研究生期间/第二篇/DRLEC/tools/train_hierarchical_d3qn.py:1177) |
| collector取得G和d | [train_hierarchical_d3qn.py:637](D:/研究生期间/第二篇/DRLEC/tools/train_hierarchical_d3qn.py:637) |
| G分配到任务并写replay | [train_hierarchical_d3qn.py:705](D:/研究生期间/第二篇/DRLEC/tools/train_hierarchical_d3qn.py:705)，写deadline_constraint_cost在751 |
| 下层learn先于本轮dual更新 | [train_hierarchical_d3qn.py:2305](D:/研究生期间/第二篇/DRLEC/tools/train_hierarchical_d3qn.py:2305)，dual调用2346 |
| 上层cooperative信用补丁 | [train_hierarchical_d3qn.py:1324](D:/研究生期间/第二篇/DRLEC/tools/train_hierarchical_d3qn.py:1324)；上层不是直接用下层G作为自己的系统Deadline代价 |
| 当前乘子重组replay奖励 | [hierarchical_low_sac.py:885](D:/研究生期间/第二篇/DRLEC/src/sac/hierarchical_low_sac.py:885)，奖励合成1067 |
| 采样诊断字段及Deadline惩罚 | [hierarchical_low_sac.py:1022](D:/研究生期间/第二篇/DRLEC/src/sac/hierarchical_low_sac.py:1022) |
| PI乘子控制器、CVaR窗口、投影 | [deadline_lagrangian.py:134](D:/研究生期间/第二篇/DRLEC/src/sac/deadline_lagrangian.py:134)；excess170，积分172，输出179 |
| 三控制组接口和隔离hook | [train_deadline_credit_ablation.py:31](D:/研究生期间/第二篇/DRLEC/tools/train_deadline_credit_ablation.py:31)，installed_credit121 |
| 新入口配置检查与训练调用 | [train_deadline_credit_ablation.py:171](D:/研究生期间/第二篇/DRLEC/tools/train_deadline_credit_ablation.py:171)，run_training193 |
| 必须保留的coupled入口语义 | [train_coupled_hierarchical_d3qn_sac.py:275](D:/研究生期间/第二篇/DRLEC/tools/train_coupled_hierarchical_d3qn_sac.py:275)，install_low_dual_attribution301，run_training340 |
| 冻结评估及验证选模核对 | [eval_deadline_credit_ablation.py:40](D:/研究生期间/第二篇/DRLEC/tools/eval_deadline_credit_ablation.py:40)，prepare94，run233，report263 |
| 原配置下层预算等参数 | [v3_sm_74_legacy_train.json:116](D:/研究生期间/第二篇/DRLEC/configs/v3_sm_74_legacy_train.json:116)，deadline_budget122 |
| 当前论文贡献第二项 | [main2.tex:120](D:/研究生期间/第二篇/DRLEC/paper/manuscript/main2.tex:120) |
| 论文固定回放、边际、责任、奖励、PI | [main2.tex:780](D:/研究生期间/第二篇/DRLEC/paper/manuscript/main2.tex:780)，相关label见下方 |

论文相关label：eq:method_counterfactual_replay、eq:method_counterfactual_marginal、eq:method_source_credit、eq:method_low_deadline_credit、eq:lagrangian_low、eq:method_pi_dual、eq:method_reliability_duals。本交接不修改论文。

### 必须避免的入口误用

1. 新消融配置的顶层deadline_credit_ablation.variant只有新入口负责解释。把它交给旧训练入口可能静默运行Full，而不是控制组。
2. deadline_credit(..., variant="full")函数内部支持Full，但新入口VARIANTS及validate_config只允许cf_self/cf_uniform/raw_self。不要假设CLI已支持full。
3. Full若只试新预算，可新增原Full配置副本、复用coupled入口；若要统一四组日志/manifest，Astra应明确提出最小接口扩展并测试，不要复制整个训练循环。
4. 不要用裸production.run_training替代coupled.run_training。后者将下层可靠性dual输入替换为最终reliability_met的二元失败指标；奖励侧仍走已有稠密代理代价。换入口会混入可靠性机制变化。

## 4. 已完成的训练、模型身份和选模

### 4.1 共同设置

- SM是Small科学工作流、74台服务器（56 edge + 18 cloud），不是Medium工作流。
- 训练20工作流/轮、到达率0.1/s、Medium Deadline、旧cai_2025_heft生成口径。
- 1200轮、训练seed42；逐轮输入seed43—1242。
- 每20轮双块验证：542000—542019与642000—642019，各20seed。
- 沿用原rank_rounds/combine_checkpoint_metrics/metric_rank：两块ESP@90、Rel均达标后，优先最低验证能耗；没有可行点才使用预先约定的final不可行诊断，不按测试重选。
- 三新训练都有1200训练行和60轮双验证；三份workload_inputs.jsonl共用SHA256：5c425fa65f437a40ed60fc5b32a46a9ba53087f4c0da92e278704bf0f552dd94。
- 三新组输入配对不等于历史Full已有同格式逐轮输入指纹。历史Full与Raw的实质配置此前比较一致，仅variant/运行命名和输出元数据不同；首轮energy、rho、成功率及输入seed也完全一致。这不能证明全部运行环境一致，但不能凭负结果宣称选错配置。

### 4.2 精确模型

以下目录均相对工作目录；每个目录内使用best_d3qn_checkpoint.pt。本次交接再次读取并核对了四份checkpoint的SHA256。

| 版本 | 训练目录 | 选中episode |
|---|---|---:|
| 历史Full | results/v3_sm_74_legacy_train_20wf/20261009_115939 | 1200 |
| CF-Self | results/ablations/sm_74_deadline_credit_20wf/cf_self/seed42_20261023_093737_282001_49884 | 1200 |
| CF-Uniform | results/ablations/sm_74_deadline_credit_20wf/cf_uniform/seed42_20261023_093810_352809_46864 | 1180 |
| Raw-Self | results/ablations/sm_74_deadline_credit_20wf/raw_self/seed42_20261023_093836_720976_27828 | 1200 |

~~~text
Full       32764207f74371ae00c0c3cfdf9c16bd760c0a3143be560dd7f72f2ec1c652ea
CF-Self    6851c8a3a1461228aaac35a9acd2f58102706382c7dfde9b9a0009e7de7991ec
CF-Uniform 7857daa9ad5a719e9d68d09d9932fe3c6c03950032cbc728ed2b58cc542032e9
Raw-Self   d3911fdb917f0ece732aa776af46b2ddd34d0a3d1817264d4aac6609eed1f046
~~~

重点：CF-Uniform是1180，不要换成final1200。以上均为完整高低层联合checkpoint，不是只加载high/low组件。Full配置保存在checkpoint内部，不要凭目录里没有resolved_config.json就换另一份配置。

原三份配置：

- [CF-Self配置](D:/研究生期间/第二篇/DRLEC/configs/ablation_sm_74_cf_self_legacy_train.json)
- [CF-Uniform配置](D:/研究生期间/第二篇/DRLEC/configs/ablation_sm_74_cf_uniform_legacy_train.json)
- [Raw-Self配置](D:/研究生期间/第二篇/DRLEC/configs/ablation_sm_74_raw_self_legacy_train.json)

## 5. 已有评估结果：负结果不能省略

### 5.1 协议与文件

总目录：[deadline_credit_ablation_sm74_20261023](D:/研究生期间/第二篇/DRLEC/results/deadline_credit_ablation_sm74_20261023)

- [REPORT.md](D:/研究生期间/第二篇/DRLEC/results/deadline_credit_ablation_sm74_20261023/REPORT.md)：最终四版本结果。
- [frozen_plan.json](D:/研究生期间/第二篇/DRLEC/results/deadline_credit_ablation_sm74_20261023/frozen_plan.json)：模型、源hash、输入指纹、条件和顺序。
- [validation_selection.json](D:/研究生期间/第二篇/DRLEC/results/deadline_credit_ablation_sm74_20261023/validation_selection.json)：选中模型与全部验证候选。
- [summary.json](D:/研究生期间/第二篇/DRLEC/results/deadline_credit_ablation_sm74_20261023/summary.json)：all_complete=true。
- 根comparison_rows.jsonl/.csv是100条Full历史参考；三个子目录各有100条新评估、workflow_details、completion.json、experiment_manifest.json。
- 三个新评估已完成且此前观察到session正常exit0。旧session 28327、10039、12201不要复用或重启。
- prepare/run已执行完，不得再次prepare或重复run。report会写汇总，不是纯只读；若只审查，直接读已有文件。

冻结评估：Small、74台、50工作流/批、新sec_job_fcfs_v1；五条件分别是0.03/s三DDL和0.10、0.20/s的Tight。每条件20个未筛选连续seed742000—742019；冻结策略，不学习，不固定rho=1。高到达率只改变到达时刻，保留DAG、任务量和相对Deadline预算，输入指纹已核对。

Full不是新跑：0.03三档来自results/legacy_train_new_eval_20261009/sm_74；0.10、0.20来自results/arrival_rate_sm74_20261017/d3qn_sac/rate_0p10和rate_0p20，模型hash与样本指纹已配对。

### 5.2 能耗（J，均值±测试种子间样本SD）

| 到达率/s | DDL | Full历史参考 | CF-Self | CF-Uniform | Raw-Self |
|---:|---|---:|---:|---:|---:|
| 0.03 | Loose | 340859.56 ± 13517.62 | 412342.40 ± 9633.65 | 372281.81 ± 8271.10 | 337919.45 ± 9443.28 |
| 0.03 | Medium | 358003.88 ± 15724.41 | 416296.77 ± 9722.69 | 377442.12 ± 8836.72 | 352249.04 ± 9818.50 |
| 0.03 | Tight | 378345.50 ± 14164.90 | 417851.46 ± 7062.79 | 381784.04 ± 8880.20 | 364071.21 ± 9502.11 |
| 0.10 | Tight | 396275.87 ± 15814.23 | 394717.12 ± 10121.19 | 397846.87 ± 9904.44 | 378002.26 ± 12600.79 |
| 0.20 | Tight | 453437.56 ± 25922.15 | 427934.75 ± 31430.47 | 434122.05 ± 19298.12 | 426714.80 ± 23054.42 |

所有格子Rel=100%。ESP@90除CF-Self在0.20/Tight为85%外，其余都是100%。CF-Self失败批次742004、742011、742013，各自批内按时率0.78、0.80、0.84。

注意：ESP@90=100%表示20批各至少45/50按时，不表示1000个工作流全部按时。Rel=100%是每批全部工作流的模型可靠性达到0.9，不是测到“现实无故障”。

### 5.3 压力条件的连续违约量（本次交接再次从逐工作流记录复算）

0.20/s、Tight，每版本20批×50=1000工作流：

| 版本 | 迟到工作流数 | 全样本平均迟到秒 | 全样本平均归一化迟到% | 每批最差10%平均归一化迟到，再跨批平均% | 最大归一化迟到% |
|---|---:|---:|---:|---:|---:|
| Full历史参考 | 3 | 0.0062026644 | 0.00702076 | 0.07020761 | 3.82444035 |
| CF-Self | 38 | 0.2762963284 | 0.39426560 | 3.26517814 | 34.11312149 |
| CF-Uniform | 9 | 0.0287367478 | 0.04370491 | 0.43704906 | 11.46029817 |
| Raw-Self | 1 | 0.0002738632 | 0.00027706 | 0.00277061 | 0.27706054 |

归一化迟到=max(response_time/relative_deadline-1,0)，均值包含按时样本的0；尾部指标是每批50个中取最差5个再平均，不是把1000个合并后取最差100个。

数据路径：

- Full：results/arrival_rate_sm74_20261017/d3qn_sac/rate_0p20/workflow_details/seed_742000.json至seed_742019.json，JSON对象内workflows列表。
- 控制组：当前总目录下各variant/workflow_details/rate_0p20_tight_seed742000.json至742019，JSON顶层直接是工作流列表。

Raw相对Full在压力条件平均能耗低约5.89%，连续迟到也更少。不能通过换成上述连续指标把Raw的优势“解释掉”。在单训练模型层面，这是真实观察到的反例。

## 6. 已确认的归零链条与日志陷阱

### 6.1 原参数与确切公式

下层配置：deadline_budget=0.08，deadline_lambda_init=1，dual_learning_rate=0.2，lambda_max=8，proportional_gain=25，warmup_episodes=20，constraint_statistic=cvar，cvar_tail_fraction=0.25，window_episodes=40。

一轮工作流成本取均值；最近40轮的均值中，取最大的25%求均值，得到J。不是直接取最差25%的单个工作流。

在准备好窗口且结束warmup后：

~~~text
e = J - budget
I_new = clip(I_old + 0.2 * e, 0, 8)
lambda_new = clip(I_new + 25 * e, 0, 8)
~~~

只要J低于预算，e为负；积分项逐渐下降，比例项也为负，可以在I仍正时令最终lambda为0。并非输出lambda递推中简单减固定数，也不能说所有版本一到固定轮次都会归零。

SAC采样奖励重组：

~~~text
r = r_base - current_lambda_D * task_deadline_cost
           - current_lambda_R * task_reliability_cost
~~~

奖励的能耗/时间部分走现有归一化处理，约束代价并非自动与它们一起z-score。工作流G分到任务后，任务级代价尺度还受任务数与分配权重影响。这个尺度是否造成学习信号弱，需要诊断，尚不能直接认定是缺陷。

### 6.2 实际惩罚与乘子记录

判定阈值为1e-12，遍历四个run的1200行train_history.jsonl：

| 版本 | 采样平均Deadline惩罚>阈值的轮数 | 最后一轮非零惩罚 | 最后200轮平均Deadline惩罚 |
|---|---:|---:|---:|
| Full | 428 | 428 | 0 |
| CF-Self | 480 | 480 | 0 |
| CF-Uniform | 398 | 410 | 0 |
| Raw-Self | 976 | 976 | 0 |

这里统计的是low_lagrangian_penalty_mean，不是仅凭lambda非零推断惩罚非零。不能把“有惩罚的轮次比例”当成优化步数加权的惩罚占比。

Full示例：

| episode | J | budget | 更新前lambda | 更新后lambda | 当轮采样平均Deadline惩罚 |
|---:|---:|---:|---:|---:|---:|
| 428 | 0.05368857 | 0.08 | 0.01151447 | 0 | 0.00001049302 |
| 429 | 0.05173982 | 0.08 | 0 | 0 | 0 |
| 600 | 0.01148051 | 0.08 | 0 | 0 | 0 |
| 1200 | 0.00139801 | 0.08 | 0 | 0 | 0 |

Full最后200轮J均值约0.0009541081，远小于0.08。这证明当前运行中控制器长期认为约束有余量，不能单凭此证明预算数值“错误”或归零导致了测试表现。

### 6.3 同名字段覆盖与时序（本次读代码新确认）

训练器先构造含result.metrics的row，再追加dual指标，然后在约2507行执行：

~~~python
row.update({f"low_{key}": value for key, value in low_metrics.items()})
~~~

因此：

- collector的low_deadline_constraint_cost_mean本来是整轮工作流责任均值；
- learn返回deadline_constraint_cost_mean，前缀low_后会覆盖同名字段；实际train_history中它通常是SAC采样任务代价均值，且low_agent.update会汇总本轮多次learn；
- low_deadline_constraint_cost_total没有被上述采样mean同步改成同一统计口径，不要再用日志里的total/mean反推样本数；
- 整轮dual输入均值应优先看low_deadline_constraint_batch_mean；滚动CVaR看low_deadline_constraint_value；
- low_lagrangian_penalty_mean只指Deadline项；可靠性是low_reliability_lagrangian_penalty_mean，不能混为总约束惩罚；
- 本轮low.learn发生在episode_end更新乘子之前。low_deadline_lambda_before_update更接近本轮learn所用值；low_deadline_lambda最终又被显式设置成更新后值；不要把after与当轮采样cost相乘解释为真实学习惩罚；
- lambda归零时，当前replay重组也会忽略历史样本存储的非零Deadline代价，而不仅忽略新采集样本。这不等于抹去早期训练对网络的影响。

这是已确认的日志命名/阅读风险，不是已确认的梯度计算bug。若Astra改进诊断，请新增明确字段并保持旧字段兼容，不回填伪造旧日志，不为日志修正强制重训。

### 6.4 从Raw训练日志可做的零重跑代价分解

Raw的deadline_credit_history.jsonl有1200轮、24000工作流，包含actual_violation与all_full_frequency_violation。

- sum(clip(actual_violation,0,1)) = 2495.114269975348。
- 同一轨迹上sum(clip(actual_violation-all_full_frequency_violation,0,1)) = 283.3048947960553。
- 后者比前者小约88.6456%。

它只能说明“在Raw训练轨迹与当前固定回放定义中，基线扣除显著缩小反馈代价”。不能说88.65%真实迟到一定与频率无关，不能证明Full归因更准确或性能更好。这不是Full的独立新实验。

## 7. 如何理解现有证据，不能越过哪些边界

1. Full对比CF-Self/CF-Uniform主要考察增量代价的来源分配；CF-Self对比Raw-Self更接近考察是否扣全频基线。Full对比Raw同时包含两类变化。
2. 当前Full在常规0.03三DDL的平均能耗低于CF两组，且ESP均达标；压力0.20时Full比CF两组迟到少，但能耗更高。不是“所有条件所有指标都赢”。
3. Raw在全部五条件平均能耗更低，压力违约也更少，现有证据不支持“扣除基线必然更好”。
4. 目前仅每版本一个训练模型，Full还是历史参考；不能将20个测试seed之间的差异当成多训练重复的稳定性证明。
5. 相同数值预算作用于不同代价定义，并不自动代表同等有效约束强度。这是方法学局限，不是已经证实的不公平或bug。
6. 队列归因的固定回放正确性测试能证明实现符合定义，不等于证明它一定提升最终调度性能。
7. 主实验曾有按结果筛选种子的独立数据面板。当前归因消融明确使用未筛选样本，selection_is_conditional=false；不得混用results/success_seed_screening_20261013/paired_panel来替换本次Full或控制组。该历史筛选口径不能被当成无条件总体表现。
8. 不应把已有结构消融（HEFT替换、rho=1、去Deadline Head）当成来源归因或基线扣除的直接证据。

## 8. 已有检查与历史文档

实现和日志修正阶段已有回归记录：相关训练测试合计110项通过；冻结评估新增6项测试通过。本次交接没有重新运行全部测试，不能把历史通过等同于当前完整回归新通过。

优先阅读：

- [训练接口与日志字段说明](D:/研究生期间/第二篇/DRLEC/docs/DEADLINE_CREDIT_ABLATION_SM74_LAUNCH_20261022.md)
- [最初四版本研究计划](D:/研究生期间/第二篇/DRLEC/docs/COUNTERFACTUAL_ABLATION_PLAN_20261021.md)
- [训练接口测试](D:/研究生期间/第二篇/DRLEC/tests/test_deadline_credit_ablation.py)
- [选模与评估数据测试](D:/研究生期间/第二篇/DRLEC/tests/test_deadline_credit_evaluation.py)
- [Deadline控制器/队列回放测试](D:/研究生期间/第二篇/DRLEC/tests/test_deadline_lagrangian.py)
- [SAC当前乘子重组测试](D:/研究生期间/第二篇/DRLEC/tests/test_hierarchical_sac.py:479)
- [加载checkpoint后的配置优先级测试](D:/研究生期间/第二篇/DRLEC/tests/test_lagrangian_config_precedence.py)

部分早期计划/启动README仍记录“尚未实现/尚未启动”，属于历史时点描述，不是当前状态依据；以训练manifest、1200行日志、选模审计和本次最终REPORT为准。需要整理时只追加完成说明，不改写旧实验历史。

Windows测试默认临时目录此前存在权限问题，若要运行涉及临时输出的测试，可使用工作区内专用tmp并明确设置tempfile.tempdir；不要因此申请不必要的越权或清理宽泛目录。

## 9. 给Astra的建议处理顺序（尚未执行的方案）

### 阶段A：先零环境重跑诊断

1. 独立核对四版本的d/G、任务分配、duals和reward链；先查实际入口，不只看类名/字段名。
2. 复算第5、6节，画lambda_before、J与budget、实际Deadline/可靠性惩罚曲线，标明warmup和选模点。
3. 诊断成本尺度、非零稀疏性、来源外部责任比例、同名日志字段覆盖；没有存储的数据明确写缺失，不重新构造假装原始观测。
4. 可以用已有每轮观察量离线回放不同budget的控制器响应，但只能解释固定观察序列下的控制器行为，不能预测重训策略或新测试成绩。
5. 先给用户结论：确定的实现问题、参数/目标选择问题、证据不足分别是什么；不要把归零本身自动归为bug。

### 阶段B：若用户同意少量新训练，先一个独立Full诊断

此前建议仅修改新的Full配置副本：

~~~text
agent.low.lagrangian.deadline_budget: 0.08 -> 0.0
~~~

其余维持SM、1200episode、训练seed42、原生成器、原双验证与选模、lambda_init=1、dual_lr=0.2、Kp=25、lambda_max=8；不改上层预算、可靠性机制、rho能效下限、网络或任务分配。

请Astra先复核这个方案是否适合当前目标，而不是机械执行：

- 解释是“不为固定回放中的降频增量迟到保留正预算”，这是更严格的新代理约束，不是原0.08设置的同义替换。
- 对非负代价、从头初始化I=1而言，budget=0使excess非负，现有控制器不会因负偏差把乘子降至0；代价本身为0时惩罚仍应为0。
- 它不保证ESP=100%、真实硬Deadline可行或Full优于Raw，且可能增加能耗。
- 不通过硬编码lambda最小值、直接放大G、同时改Kp/学习率/代价定义等多项变化强行制造正惩罚。
- 用既定验证池判断诊断，先预定义停止/选模规则，不依据已看过的测试结果调参直到获胜。
- 从旧checkpoint恢复权重不是同起点的1200轮重训；原入口不保存完整replay，不能冒称无缝续训。原消融入口也明确拒绝resume。
- 当前CLI没有full variant；按第3节选择原coupled入口，或先提出并测试统一日志的最小扩展。
- 输出新目录、冻结新配置和源码hash，不能覆盖本次结果或悄悄更新论文主表。

### 阶段C：若要形成新正式四版本结论

- 新Full诊断不能与旧预算的三个控制组直接组成“公平新四组表”。
- 需预先冻结共同实验协议；若试统一预算0，四版本都要依该新协议训练，仍需说明Raw代价语义不同，且零实际迟到目标可能更难满足/导致乘子饱和。
- 若要校准不同代价的预算/惩罚强度，只用训练/验证数据，并事先规定对所有版本一视同仁的规则，不能用当前测试集反复调优。
- 条件允许时补独立训练seed；如果仍只有一个，明确是pilot，不夸大稳定性。
- 新确认性测试应预先指定未使用、四组配对的样本，不按结果筛选；已有测试复用只能承认其探索性。
- 若Raw仍更好，收缩“反事实优越性”表述；若要将Raw改成最终方法，算法描述及对应主实验必须一致更新，不能保留Full论文叙述却替换Raw成绩。

## 10. 希望Astra交付什么

先返回中文、面向用户的说明：

1. 当前负结果能否由已确认的实现bug解释？若是，请给位置、最小复现及失败测试；若不是，明确说不是。
2. 0.08预算与连续代价是否语义/尺度合适，PI输出归零是否正常；同名日志覆盖要怎样不破坏兼容地澄清。
3. 哪些结论已有证据，哪些只是“可能”，以及最少需要新增什么证据。
4. 如需改动，列出精确文件、字段、测试、独立输出位置与回退方式；不要直接大改旧代码。
5. 如需训练，先列训练数、预算、选模和评估协议及风险，再按用户明确授权启动；不要承诺“这样一定能让Full赢”。

交接完成时，本文件中的budget=0、日志新字段、正式四版本重训均未执行。现有负结果必须完整保留。
