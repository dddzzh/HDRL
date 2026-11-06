# Alibaba / 111台 / 100工作流 / 三档配对评估

2026-10-15用户授权：先评估一个资源规模、每批100个阿里工作流，并写入场景汇总。未授权新训练，未改变原数据或checkpoint。

## 固定实验

- 配置：`configs/alibaba_sl_111_eval_100wf.json`；生成预检配置保持独立，不把预检当正式评估。
- 资源84 edge+27 cloud，resource_seed=42；每seed100个不重复job；固定742000–742019；Loose/Medium/Tight各20，四算法共240条记录。
- 四算法沿用原场景汇总的SL111模型；不根据Alibaba测试重新选择，所有失败保留，不沿用此前success-selected的替换名单。
- D3QN+SAC：`results/v3_sl_111_legacy_train_20wf/20261009_115957/best_d3qn_checkpoint.pt`，payload920。
- DQN-Edge：`results/dqn_edge_sl_111_legacy_train_20wf/20261011_114737/best_checkpoint.pt`，payload699，即训练第700轮。
- DReaM-DQN：`results/dream_dqn_sl_111_legacy_train_20wf/20261011_114741/best_checkpoint.pt`，payload649，即训练第650轮。
- OOAC：`results/oo_actor_critic_sl_111_legacy_train_20wf/20261010_095213/best_oo_actor_critic_checkpoint.pt`，payload200。
- 全部checkpoint与loader config SHA按原评估provenance核对。DQN-Edge/DReaM的checkpoint.config只保存learner参数，完整环境/loader配置来自同run的resolved_config.json，不能拿两者结构不同误判模型不符。
- 每模型workers=1、Torch线程1；最多四模型并发。初始化742000，顺序Loose→Medium→Tight，每档742000→742019；DQN-Edge在整个60条序列保留原跨seed EMA，不独立seed重载、不partial续跑。
- `scale=small`仅是旧评估器兼容槽位：Alibaba不按Pegasus的Small/Medium/Large模板分组，实际只测试一个阿里来源，不把同一池重复冒充三种工作流规模。

## 物理数据与DDL

复用新统一入口：`source=alibaba_json`、`profile=sec_job_fcfs_v1`。来自独立测试池1万个job，校准参数来自不重叠的4万个job；预先冻结每个seed三档的真实job名单、物理量SHA和DDL SHA，四算法生成后逐条核对相同身份。

到达率0.03/s；真实DAG及相对强度映射0.5–20Gcycles、边1–30MB；共享负载FCFS参考rho=1；绝对DDL=到达+倍率×参考响应，2倍概率Loose=.2/Medium=.5/Tight=.8。前述数值是模拟约束，不是trace原始deadline/通信/能耗测量。

100个工作流适合作为首组跨来源测试。不能将其等同于100并发；参考预检单seed平均每job9.49任务，而科学工作流Small模板约24–30任务，因此需报告实际每批总任务数和负载。此次不根据胜负调整工作流数量、到达率或映射范围。

## 输出与汇总

- 独立实验：`results/alibaba_sl111_100wf_20261015/`。
- `evaluation_plan.json`：固定全部来源/代码SHA、模型、三档20组真实job名单与物理/DDL指纹。
- `seed_schedule.json`：显式种子顺序；每模型一个`<mode>/`保存manifest、run_provenance、原始CSV/JSONL与completion。
- `[alibaba-complete]`、60条严格有序唯一行、manifest workers1与初始化742000、源SHA不变后才视为该模型完成；completion失败清单须与原始行一致。
- 活动进度/结果：`results/legacy_train_new_eval_20261009/ALIBABA_SL111_100WF.md`；现有refresh已支持持久插入主`SCENARIO_MATRIX.md`，原始与筛选实验行全部保留，更新前整张主表另存实验目录以备恢复。
- 只展示能耗J均值±样本SD、ESP@90、可靠性达标率。这里ESP@90为每seed至少90/100按时；可靠性达标要求100个工作流全部达到0.9，不是物理可靠性=1。不用平均DSR代替ESP。
- 每条完成记录自动更新主表，未完整核验均明确“非最终”，不伪称全部成功。异常不自动重启；DQN不能拼接partial。

## 工具

`python tools/eval_alibaba_sl111.py --prepare`仅在新实验首次冻结计划，重复会拒绝。

`python tools/eval_alibaba_sl111.py --run <mode>`只启动一个未运行模型，mode为d3qn_sac / dqn_edge / dream_dqn / oo_actor_critic，已有目录会拒绝重复。

`python tools/eval_alibaba_sl111.py --report`仅从已有原始数据刷新汇总，不开展模型推理或训练。

回归：Alibaba评估顺序/配对/数值守卫、partial不冒充完成、外部章节持久保留、原表行不变，以及旧seed接口/生成接口共21项通过；系统临时目录访问限制按正常审批后执行，未改目录或参数规避权限。

## 启动记录

2026-10-15 14:19:31（北京时间）四个模型分别启动一次；日志、session/PID/创建时间见实验目录 `execution_jobs.json`。12个旧科学场景的原始数据/checkpoint/config SHA未变，旧主表261行表格内容按原顺序完整保留，新Alibaba章节独立插入。冻结的20批阿里样本每批794–1125任务，平均918.05任务；100个workflow仍可能比旧50个科学workflow的任务总量少，不能以workflow个数直接比较负载。

## 完整结束核验

后续CPU加速请求处理期间核对：四模型各60条，共240条完整有序唯一真实记录，全部completion存在、日志alibaba-complete、四个工具session均退出0；check_sources验证所有原始数据、checkpoint、loader/eval配置与冻结代码SHA不变。report返回all_complete=true、failures=[]，主SCENARIO_MATRIX已刷新为完整结果。

这里failures=[]只表示每条记录达到“至少90/100按时且100/100可靠性达到0.9”的门槛；DQN-Edge/DReaM有少量单个工作流未按时，不能改述为所有工作流100%按时。此次没有替换种子、追加候选或重选模型。
