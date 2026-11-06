# Raw37 新训练模型泛化评估

用户于2026-10-25提供完成训练的run，并要求启动多进程评估。本次只新增独立评估入口及测试；未修改原训练、策略、环境、FCFS生成器、旧结果或论文，也未停止111台训练。

## 模型与协议

- run：`results/raw_self_resource_models_20261025/ss_37/seed42_20261025_115202_507189_38724`。
- 1200轮及60次双验证完整，训练manifest的10个来源文件SHA核对无漂移。
- 沿原`combine_checkpoint_metrics + metric_rank`验证规则选择ep1080，不按测试重选，也不直接采用末轮。
- ep1080两验证块ESP@90及可靠性达标均1；合并验证能耗135506.77299457922 J（不是测试值）。
- checkpoint：该run的`best_d3qn_checkpoint.pt`。
- SHA256：`29339da27f5426f074915bc390db8d4cf7b71c923adad8fc5c3086658724783f`。
- checkpoint内config与训练manifest完全一致：Raw-Self、seed42、下层Deadline budget0.08、28 edge+9 cloud、37固定槽位，SAC及上层Deadline Head保留。
- 评估：`configs/v3_ss_37_new_eval_50wf.json`，原`sec_job_fcfs_v1`、50wf/seed、0.03/s、可靠性要求0.9。
- small/medium/large × loose/medium/tight × 连续742000—742019，共180条；对应SS/MS/LS，未筛选种子，所有失败保留。
- 初始化seed742000，资源seed42；8个评估worker，每进程Torch线程1。
- 复用`eval_deadline_credit_ablation`的loader、输入构建、rollout和诊断；未重写策略。Raw37不是DQN-Edge基线，不涉及该基线的跨seed attention EMA限制。
- `frozen_plan.json`记录每个输入的物理/期限/工作量哈希及模型、训练日志、配置、实现文件SHA；每档写入CSV/JSONL和工作流明细。

## 已启动状态

- 入口：`tools/eval_raw_ss37_generalization.py`。
- 输出：`results/raw_ss37_generalization_20261025/`。
- 会话：39104。
- 父PID：20000，创建时间2026-10-25 21:34:56（本地）。
- 八个子PID：18196、57868、25112、54576、52400、39380、60112、27444，创建时间21:34:58。
- 启动后核验8/180真实输出已写入，日志无Traceback；此时还未完成。
- 其他111台训练PID35688及其验证子进程保持运行，未干预。
- 命令只记录，**不得重复执行**：

```powershell
python -B -X utf8 -u tools/eval_raw_ss37_generalization.py run
```

## 检查与完成标准

- 新接口和既有Raw74评估相关测试18项通过（其中包含旧测试类的重复收集，不代表18项新测试）。
- 正常结束需会话exit0、日志`[raw37-complete]`且无Traceback、`completion.json`真实存在。
- 最终应180有序唯一记录，各档20；模型、训练config、评估config及输入哈希不变，失败列表与原始行一致。
- 能耗报J均值±样本SD；ESP@90为批内至少45/50按时的种子占比；Rel为批内所有工作流满足可靠性0.9的种子占比。
- `REPORT.md`每20条及完成时更新；CSV/JSONL每条更新。未完成不可当最终数据。
- 不自动中断、重选模型、换失败种子、重启或partial续跑。会话丢失时先合法核实原进程，不能只凭session unknown判崩溃。
- 未设置新的定时检查；本次用户要求是启动评估，不是新建自动任务。
