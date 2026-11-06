# 论文写作交接：代码、实验数据和图表索引

更新日期：2026-10-14。最终状态：北京时间14:11全部12模型配对重评估完成并核验，2160条、真实失败0；这是条件筛选样本，不是未筛选总体成功率。最终核验见results/success_seed_screening_20261013/FINAL_PANEL_VERIFICATION_20261014.json。其后用户明确要求以新结果更新能耗图，已另存新条件筛选版，原始未筛选版保留；使用新图必须披露数据筛选性质。

## 1. 给新聊天的任务边界与第一条消息

项目根目录：`D:/研究生期间/第二篇/DRLEC`。实验Python：`D:/Miniconda/envs/py310/python.exe`。

原聊天负责代码、训练/评估运行、结果核验和图表生成；新聊天负责论文写作、段落修改、图注和表格组织。新聊天不因为写论文而自行启动训练、重新评估、重选模型、换种子或停止现有进程。

可以直接把下面这段发给新聊天：

> 本聊天专门写论文。请先读取 D:/研究生期间/第二篇/DRLEC/docs/PAPER_WRITING_HANDOFF_20261014.md，再读取其中列出的旧稿、方法代码核对说明和相应数据协议。代码与评估由原聊天维护，请不要启动训练或评估、修改策略/配置/checkpoint/原始实验数据或停止现有进程。先确认本次写作使用的是旧主实验、完整原始新泛化评估，还是明确披露的条件筛选实验；不要混用三套数据。论文内容必须和实际启用的实现及原始结果对应，不编造数值、引用或完成状态。

## 2. 之前写过的论文在哪

| 文件 | 用途及注意事项 |
|---|---|
| [main.tex](D:/研究生期间/第二篇/DRLEC/docs/manuscript_revised_20261006/main.tex) | 已整合的IEEE双栏论文入口。包括Introduction、Related Work、System Model、Proposed Method，并通过 `input{experiments}` 引入实验章节。优先从这里继续，不优先用更早的单文件草稿。 |
| [experiments.tex](D:/研究生期间/第二篇/DRLEC/docs/manuscript_revised_20261006/experiments.tex) | 旧实验章节：S26/S37/M37/M53主实验、结构消融、学习动态、Wilcoxon及局限性。尚未替换为最新37/74/111服务器的50工作流评估。 |
| [EXPERIMENTS_README.md](D:/研究生期间/第二篇/DRLEC/docs/manuscript_revised_20261006/EXPERIMENTS_README.md) | 旧稿的实验数据入口、统计定义、历史选择协议差异和编译说明。 |
| [references.bib](D:/研究生期间/第二篇/DRLEC/docs/manuscript_revised_20261006/references.bib) | 该完整论文包的参考文献。此次交接没有重新逐篇核验引用，不把“文件里已有”当成“所有元数据和论断均已验证”。 |
| [data_audit.json](D:/研究生期间/第二篇/DRLEC/docs/manuscript_revised_20261006/experiment_assets/data_audit.json) | 旧实验图表的960条主记录、48组汇总、消融种子和训练曲线来源索引。 |
| [paper_experiments_20261007.zip](D:/研究生期间/第二篇/DRLEC/docs/manuscript_revised_20261006/paper_experiments_20261007.zip) | 之前的论文实验图表包；不是包含本次最新九宫格图的自动更新包。 |
| [DRLEC_MANUSCRIPT_DRAFT.tex](D:/研究生期间/第二篇/DRLEC/docs/DRLEC_MANUSCRIPT_DRAFT.tex) | 更早的草稿。摘要中的旧数值和措辞不能直接作为最新实验结论。 |
| [PROPOSED_METHOD_CURRENT.tex](D:/研究生期间/第二篇/DRLEC/docs/PROPOSED_METHOD_CURRENT.tex) | 方法章节独立片段，可用于对照术语和公式。完整稿入口仍是上面的main.tex。 |
| [PROPOSED_METHOD_CODE_CHECK.md](D:/研究生期间/第二篇/DRLEC/docs/PROPOSED_METHOD_CODE_CHECK.md) | 2026-10-06的方法与实现核对记录；包含启用分支、三套Q、下层状态、信用分配和乘子更新细节。写方法时优先读，参数最终以实际来源run的配置快照为准。 |
| [SECTION_III_A_SYSTEM_SCHEDULING_MODEL.tex](D:/研究生期间/第二篇/DRLEC/docs/SECTION_III_A_SYSTEM_SCHEDULING_MODEL.tex)、[POWER_ENERGY_MODEL_CURRENT.tex](D:/研究生期间/第二篇/DRLEC/docs/POWER_ENERGY_MODEL_CURRENT.tex) | 系统、调度和功率/能耗模型片段。 |
| [CODE_FAITHFUL_SYSTEM_MODEL.md](D:/研究生期间/第二篇/DRLEC/docs/CODE_FAITHFUL_SYSTEM_MODEL.md) | 系统模型对照说明。不要用文档中没有核验过的描述替代真实模拟器。 |

旧稿标题为 Constraint-Aware Hierarchical Deep Reinforcement Learning for Energy-Efficient Dynamic Edge--Cloud Workflow Scheduling。此交接没有改论文正文、没有重新编译整篇论文，也没有替用户确定最终投稿版本。

## 3. 实验数据分成三套，不能混用

### A. 旧稿主实验：20工作流、S26/S37/M37/M53

- 场景：S26=Small/20edge+6cloud；S37=Small/28+9；M37=Medium/28+9；M53=Medium/40+13。
- 主实验测试种子 `242000–242019`，每个种子20工作流，到达率0.1/s；四算法、三deadline，共960条、48个统计组。
- 主实验和消融不是同一个测试块：旧消融使用 `742000–742019`，不能拿消融Full数值替换主实验Full。
- 旧稿说的13.45%–48.93%节能等数字属于这套旧实验，不是最新50工作流泛化实验的自动结论。

| 数据/代码 | 绝对路径 |
|---|---|
| S26/M37逐种子数据 | [formal_comparison_selected_columns.csv](D:/研究生期间/第二篇/DRLEC/docs/figures/formal_comparison_selected_columns.csv) |
| S26/M37统计对照 | [formal_comparison_figure_summary.csv](D:/研究生期间/第二篇/DRLEC/docs/figures/formal_comparison_figure_summary.csv) |
| S37原始数据目录 | [S37_final](D:/研究生期间/第二篇/DRLEC/results/scenario_matrix_evaluation/S37_final)；读取comparison_rows.csv和comparison_summary.csv |
| M53原始数据目录 | [M53_complete_4alg_20260927](D:/研究生期间/第二篇/DRLEC/results/scenario_matrix_evaluation/M53_complete_4alg_20260927)；同样读取逐种子表和汇总 |
| 旧消融 | [small_main_ablation](D:/研究生期间/第二篇/DRLEC/results/ablation_results/small_main_ablation)，入口ablation_summary.csv |
| 旧统计检验/四场景图 | [main_comparison_all_scenarios](D:/研究生期间/第二篇/DRLEC/docs/figures/main_comparison_all_scenarios)，包括WILCOXON_REPORT.md及配对文件 |
| 旧论文图表生成 | [build_experiment_section.py](D:/研究生期间/第二篇/DRLEC/plot/build_experiment_section.py) |
| 旧主实验绘图/统计 | [plot_paper_experiments_20260930.py](D:/研究生期间/第二篇/DRLEC/plot/plot_paper_experiments_20260930.py)、[plot_main_style_options.py](D:/研究生期间/第二篇/DRLEC/plot/plot_main_style_options.py)、[analyze_main_wilcoxon.py](D:/研究生期间/第二篇/DRLEC/plot/analyze_main_wilcoxon.py) |

旧Wilcoxon检验和旧消融比例不能直接迁移到新泛化数据；需要对实际采用的新数据另行计算并说明设计。

### B. 最新完整原始泛化评估：50工作流、37/74/111服务器

这是当前九宫格能耗图的数据来源。全部原始种子保留，包括失败种子；不是成功筛选数据。

权威入口：

- [SCENARIO_MATRIX.md](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/SCENARIO_MATRIX.md)：同一主表内同时有“筛选配对实验段”和“原始结果段”，阅读时必须确认标题。
- [PROTOCOL.md](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/PROTOCOL.md)：模型身份、checkpoint/hash、验证选模、训练和测试协议、DQN原串行口径及历史中断说明。
- [LEGACY_TRAIN_NEW_EVAL_20261009.md](D:/研究生期间/第二篇/DRLEC/docs/LEGACY_TRAIN_NEW_EVAL_20261009.md)：旧生成训练/新生成测试的设计说明；其中早期运行状态不是当前实时状态。

场景缩写：第一字母为测试工作流规模Small/Medium/Large，第二字母为服务器配置Small/Medium/Large。

| 测试工作流 | 37台：28edge+9cloud | 74台：56edge+18cloud | 111台：84edge+27cloud |
|---|---|---|---|
| Small | SS | SM | SL |
| Medium | MS | MM | ML |
| Large | LS | LM | LL |

所有这些模型训练于Small工作流和Medium deadline，训练/验证20工作流、到达率0.1/s、旧 `cai_2025_heft` 生成。冻结模型测试使用50工作流、到达率0.03/s、新 `sec_job_fcfs_v1` 生成，服务器数量对每个模型保持不变。这是跨生成协议/负载的迁移评估，不是同分布测试，也不是同一权重跨37/74/111服务器直接通用。

测试配置：

- [v3_ss_37_new_eval_50wf.json](D:/研究生期间/第二篇/DRLEC/configs/v3_ss_37_new_eval_50wf.json)
- [v3_sm_74_new_eval_50wf.json](D:/研究生期间/第二篇/DRLEC/configs/v3_sm_74_new_eval_50wf.json)
- [v3_sl_111_new_eval_50wf.json](D:/研究生期间/第二篇/DRLEC/configs/v3_sl_111_new_eval_50wf.json)

原始每模型9档×20种子=180条；12个模型共2160条。原始测试块均为 `742000–742019`。下表每目录读取comparison_rows.jsonl或comparison_rows.csv，不从Markdown四舍五入值重建箱线图。

| 资源 | 算法 | 完整原始结果目录 |
|---|---|---|
| 37 | D3QN+SAC | [ss_37](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/ss_37) |
| 37 | DQN-Edge | [baselines/dqn_edge](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/baselines/dqn_edge) |
| 37 | DReaM-DQN | [baselines/dream_dqn](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/baselines/dream_dqn) |
| 37 | OOAC | [baselines/oo_actor_critic](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/baselines/oo_actor_critic) |
| 74 | D3QN+SAC | [sm_74](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/sm_74) |
| 74 | DQN-Edge | [sm_74_dqn_edge](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/sm_74_dqn_edge) |
| 74 | DReaM-DQN | [sm_74_dream_dqn](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/sm_74_dream_dqn) |
| 74 | OOAC | [sm_74_oo_actor_critic](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/sm_74_oo_actor_critic) |
| 111 | D3QN+SAC | [sl_111](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/sl_111) |
| 111 | DQN-Edge | [sl_111_dqn_edge](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/sl_111_dqn_edge) |
| 111 | DReaM-DQN | [sl_111_dream_dqn](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/sl_111_dream_dqn) |
| 111 | OOAC | [sl_111_oo_actor_critic](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/sl_111_oo_actor_critic) |

每个模型的实际checkpoint、来源run、训练配置和hash查PROTOCOL与checkpoint_provenance.json；37台baseline查 [baselines/provenance.json](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/baselines/provenance.json)。也可查下面冻结计划的runs字段，它列出同12个模型的完整路径和hash。

三份本文模型身份：SS来自20260924_160451的ep1060；SM来自v3_sm_74_legacy_train_20wf/20261009_115939的ep1200；SL来自v3_sl_111_legacy_train_20wf/20261009_115957的ep920。不要因数字相同误用另一个October SS模型；SL恢复训练段没有取代原ep920最佳checkpoint。基线各自验证选模、训练轮数和历史例外不能凭“best”文件名统一臆测。

### C. 成功种子筛选后的完整配对实验：另存、已完成核验

用户此前明确授权的用途是非论文成功样本探索。它是在看到原始结果后选择替换种子，不能冒充预先固定、未筛选的泛化测试。若之后用于论文，只能明确披露筛选机制及条件样本性质，不据此声称总体成功率100%。

- [README.md](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/README.md)：完整候选筛查及重评估说明。
- [panel_plan.json](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/panel_plan.json)：冻结的12模型身份、模型/config/hash、原数据hash、旧seed→新真实seed映射和名单；不要重新prepare生成新计划。
- 三份名单：[SS37](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/seed_schedule_ss_37.json)、[SM74](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/seed_schedule_sm_74.json)、[SL111](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/seed_schedule_sl_111.json)。SS替换32个场景种子位置，SL替换3个，SM不替换；同档四算法使用相同20个唯一真实种子。
- 新结果：[paired_panel](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/paired_panel)，每模型目录名为ss_37/sm_74/sl_111加算法后缀，每个最终180条，不能和B套旧行拼接。
- 实时作业状态：[panel_jobs.json](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/panel_jobs.json)；日志在新实验根的 `<id>_panel.log`。每目录completion.json与正常panel-complete日志仅表示该模型完成，还要核验整组。
- 汇总段：[SUCCESS_SEED_REPLACEMENT.md](D:/研究生期间/第二篇/DRLEC/results/legacy_train_new_eval_20261009/SUCCESS_SEED_REPLACEMENT.md)。原始数据仍留在B套目录，没有覆盖。
- 最终状态2026-10-14 14:11：12/12模型complete_verified且全部session exit0，每模型180条，共2160条。各档20唯一种子与四法配对、显式顺序、模型/config/原数据hash、workers1、初始化742000、manifest和completion均核验通过；--report为all_complete=true、failures=[]。每档ESP@90和可靠性达标率均20/20，但只能解释为筛选后的条件名单实际通过。
- [FINAL_PANEL_VERIFICATION_20261014.json](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/FINAL_PANEL_VERIFICATION_20261014.json)：逐模型最终检查记录。完成检查已结束，不再有待完成评估。
- 恢复记录：[suspended_processes.json](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/suspended_processes.json)。前夜挂起保留内存，今日用户授权原地恢复；不得重复Resume、关闭终端或重启/partial重载DQN。

新聊天只读这些结果即可；原聊天已完成运行与最终核验，不需要重复启动任何模型或监控。数据口径选择仍须与用户明确，不自动把条件筛选表替换成未筛选论文证据。

## 4. 实验与算法代码在哪

| 模块 | 源码入口 |
|---|---|
| 本文训练入口/配置分支 | [train_coupled_hierarchical_d3qn_sac.py](D:/研究生期间/第二篇/DRLEC/tools/train_coupled_hierarchical_d3qn_sac.py) |
| rollout、回放、反事实信用和更新流程 | [train_hierarchical_d3qn.py](D:/研究生期间/第二篇/DRLEC/tools/train_hierarchical_d3qn.py)、[train_reliability_aware_hierarchical_d3qn.py](D:/研究生期间/第二篇/DRLEC/tools/train_reliability_aware_hierarchical_d3qn.py) |
| 上层三类Q估计/约束放置 | [reliability_aware_high.py](D:/研究生期间/第二篇/DRLEC/src/D3QN/reliability_aware_high.py)、[dual_head_constrained.py](D:/研究生期间/第二篇/DRLEC/src/D3QN/dual_head_constrained.py) |
| 状态与队列校准 | [state.py](D:/研究生期间/第二篇/DRLEC/src/D3QN/state.py)、[hierarchical_state.py](D:/研究生期间/第二篇/DRLEC/src/ppo/hierarchical_state.py)、[queue_weight_calibrator.py](D:/研究生期间/第二篇/DRLEC/src/D3QN/queue_weight_calibrator.py) |
| 下层SAC与奖励 | [hierarchical_low_sac.py](D:/研究生期间/第二篇/DRLEC/src/sac/hierarchical_low_sac.py)、[reward.py](D:/研究生期间/第二篇/DRLEC/src/sac/reward.py) |
| 两层deadline与下层可靠性对偶 | [上层deadline](D:/研究生期间/第二篇/DRLEC/src/D3QN/deadline_lagrangian.py)、[下层deadline](D:/研究生期间/第二篇/DRLEC/src/sac/deadline_lagrangian.py)、[下层reliability](D:/研究生期间/第二篇/DRLEC/src/sac/reliability_lagrangian.py) |
| 共享环境与物理模型 | [edge_cloud_env.py](D:/研究生期间/第二篇/DRLEC/src/simulator/edge_cloud_env.py)、[resource.py](D:/研究生期间/第二篇/DRLEC/src/simulator/resource.py)、[reliability.py](D:/研究生期间/第二篇/DRLEC/src/simulator/reliability.py)、[deadline.py](D:/研究生期间/第二篇/DRLEC/src/simulator/deadline.py) |
| 环境创建/生成协议路由 | [hierarchical_common.py](D:/研究生期间/第二篇/DRLEC/tools/hierarchical_common.py)、[workflow_factory.py](D:/研究生期间/第二篇/DRLEC/src/datasets/workflow_factory.py) |
| 旧/新数据生成 | [legacy_workflow_generator.py](D:/研究生期间/第二篇/DRLEC/src/datasets/legacy_workflow_generator.py)、[sec_job_workflow_generator.py](D:/研究生期间/第二篇/DRLEC/src/datasets/sec_job_workflow_generator.py)；新模板目录为data/sec_job_reference |
| DQN-Edge基线 | [train_dqn_edge.py](D:/研究生期间/第二篇/DRLEC/tools/train_dqn_edge.py)，实现src/baselines/dqn_edge/，尤其agent.py的smoothed_weights |
| DReaM-DQN基线 | [train_dream.py](D:/研究生期间/第二篇/DRLEC/tools/train_dream.py)，实现src/baselines/dream/ |
| OOAC基线 | [train_oo_actor_critic.py](D:/研究生期间/第二篇/DRLEC/tools/train_oo_actor_critic.py)、[oo_actor_critic.py](D:/研究生期间/第二篇/DRLEC/src/baselines/oo_actor_critic.py) |
| 通用冻结评估 | [eval_all_baselines.py](D:/研究生期间/第二篇/DRLEC/tools/eval_all_baselines.py) |
| 原始泛化验证/主表生成 | [eval_v3_scenarios.py](D:/研究生期间/第二篇/DRLEC/tools/eval_v3_scenarios.py)、[eval_s37_baseline_generalization.py](D:/研究生期间/第二篇/DRLEC/tools/eval_s37_baseline_generalization.py) |
| 条件筛选配对计划/核验/汇总 | [eval_success_seed_panel.py](D:/研究生期间/第二篇/DRLEC/tools/eval_success_seed_panel.py)、[evaluation_seed_schedule.py](D:/研究生期间/第二篇/DRLEC/tools/evaluation_seed_schedule.py) |
| 单种子并行执行支持 | [parallel_evaluation.py](D:/研究生期间/第二篇/DRLEC/tools/parallel_evaluation.py)；不能据文件存在推断DQN可独立种子并行 |
| 回归检查 | tests/test_evaluation_seed_schedule.py、tests/test_success_seed_panel.py、tests/test_parallel_baseline_evaluation.py（均在项目根tests/下） |

写方法时必须保留的实现边界：

1. 上层动作是服务器选择，下层动作是连续DVFS比例；任务顺序是共享smallest_workload规则，不是另一个任务排序智能体。
2. “三头”实际为三套独立Dueling Double-Q估计，不是共享主干三输出；下层是普通SAC双Q，不是三类Multi-Critic。
3. 具体启用类由来源配置决定；已有完整方法核对指向ConfiguredLagrangianReliabilityAgent，不能因为入口名coupled就写入未启用的physical-low等分支。
4. rho_safe是状态提示，不可直接声称执行了保证deadline的硬安全投影。反事实信用是固定调度下的代理归因，不是Shapley值或全局最优因果贡献。
5. 测试时神经网络权重冻结，但DQN-Edge的attention smoothed_weights仍跨种子更新。其顺序必须保持；不能说所有内部状态都冻结，也不能重新加载EMA后拼接部分结果。
6. 当前能耗是活跃计算能耗，不是含网络传输/空闲的总设施电耗。可靠性达标不等于物理可靠性1或零故障。

## 5. 当前图表与配色

| 图/代码 | 最新文件 |
|---|---|
| 详细算法框架图PDF | [d3qn_sac_framework.pdf](D:/研究生期间/第二篇/DRLEC/output/pdf/d3qn_sac_framework.pdf) |
| 框架PNG/SVG与说明 | [method_framework](D:/研究生期间/第二篇/DRLEC/docs/figures/method_framework) |
| 框架生成器 | [draw_method_framework.py](D:/研究生期间/第二篇/DRLEC/tools/draw_method_framework.py) |
| 新条件筛选能耗图PDF | [energy_boxplots_success_selected.pdf](D:/研究生期间/第二篇/DRLEC/output/pdf/energy_boxplots_success_selected.pdf) |
| 新条件筛选能耗图PNG | [energy_boxplots_success_selected.png](D:/研究生期间/第二篇/DRLEC/docs/figures/current_generalization_energy/success_selected/energy_boxplots_success_selected.png) |
| 新条件筛选可编辑SVG | [energy_boxplots_success_selected.svg](D:/研究生期间/第二篇/DRLEC/docs/figures/current_generalization_energy/success_selected/energy_boxplots_success_selected.svg) |
| 新合并完整原始CSV | [paired_panel_full_raw.csv](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/exports/paired_panel_full_raw.csv)，2160条，来源CSV所有字段保留，逐字段核对差异0 |
| 新绘图逐种子列CSV | [paired_energy_rows.csv](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/exports/paired_energy_rows.csv)，2160条，含真实seed、原位置seed及替换标记 |
| 新能耗及服务汇总CSV | [paired_energy_summary.csv](D:/研究生期间/第二篇/DRLEC/results/success_seed_screening_20261013/exports/paired_energy_summary.csv)，108组，每组20种子 |
| 原始未筛选图PDF（保留） | [energy_boxplots_refined_complete.pdf](D:/研究生期间/第二篇/DRLEC/output/pdf/energy_boxplots_refined_complete.pdf) |
| 原始未筛选图PNG（保留） | [energy_boxplots_refined_complete.png](D:/研究生期间/第二篇/DRLEC/docs/figures/current_generalization_energy/energy_boxplots_refined_complete.png) |
| 原始未筛选可编辑SVG（保留） | [energy_boxplots_refined_complete.svg](D:/研究生期间/第二篇/DRLEC/docs/figures/current_generalization_energy/energy_boxplots_refined_complete.svg) |
| 数据与配色溯源 | [refined_figure_provenance.json](D:/研究生期间/第二篇/DRLEC/docs/figures/current_generalization_energy/refined_figure_provenance.json) |
| 箱线图生成器 | [plot_current_generalization_energy.py](D:/研究生期间/第二篇/DRLEC/plot/plot_current_generalization_energy.py) |

上午保留的原始版使用B套2160条；后续用户要求更新的新图使用C套条件筛选配对2160条，图脚注明Outcome-selected seeds，文件名含success_selected。两版都没有成功率热图，不能混用。九个子图为SS/SM/SL、MS/MM/ML、LS/LM/LL，图例在最下方，各子图独立线性纵轴，单位J、乘数10^6。箱体为中位数/IQR、须线为1.5-IQR；小样本点与离群点标记按用户要求隐藏，但各版本来源的全部观测仍参加统计，绘图阶段不额外删除观察。

框架六色：`#E0EAF6`、`#A8C0E4`、`#BBDDB2`、`#C7C5C6`、`#F5AA83`、`#BAE0E3`；中性描边`#303030`。实验四算法当前映射：D3QN+SAC浅蓝`#A8C0E4`、DQN-Edge浅橙`#F5AA83`、DReaM-DQN浅绿`#BBDDB2`、OOAC浅青`#BAE0E3`。后续论文图沿用此映射，不能每张图调换算法颜色。

## 6. 统计口径与写作前核对顺序

- 能耗表：total_energy（J），20个测试种子的均值±样本SD；箱线图不是均值±SD图。
- 新50工作流实验ESP@90：每个种子至少45/50工作流按时完成，再统计20种子中通过比例；旧20工作流实验门槛为18/20。
- 可靠性达标率：每个种子所有工作流达到原可靠性要求0.9的比例。不是平均物理可靠性，也不是平均rho/DSR。
- 20个测试工作流种子不等于20次独立训练；一条训练轨迹不能写成多训练种子稳定性结论。
- 原始泛化测试不能用测试结果重选checkpoint；训练验证能耗不能写成新测试能耗。历史SS基线选择差异应根据来源协议披露。
- 条件筛选样本中100%只描述已筛选名单，不能推断未筛选总体、理论约束保证或独立确认。
- 能耗优势需要同时核对deadline和可靠性，不能因本次图只画能耗就省略可行性证据。
- 此次只做交接索引与现状定位，没有重新审计全文所有公式、参考文献、统计检验或整篇LaTeX排版。

新聊天建议按以下顺序开始：

1. 读本文件、main.tex、experiments.tex、旧EXPERIMENTS_README.md。
2. 与用户确认此次写的是旧主实验章节，还是新增50工作流泛化章节；不要默默用新数据替掉全部旧实验。
3. 读所选数据的PROTOCOL、原始逐种子文件及figure_provenance，而不是凭聊天里的数字写结果。
4. 写方法前读PROPOSED_METHOD_CODE_CHECK.md，再按实际来源配置定位启用源码。
5. 新增/变更引用必须验证元数据和所支撑的原文；未核验处明确标注，不凭记忆造BibTeX。
6. C套现在已完成，但写结果时仍明确按结果筛选的条件样本属性。未来新实验的预计完成时间仍不是完成证据。

本文件是新聊天的入口索引，不替代原始结果、固定模型provenance、实际源码或最终用户选择。
