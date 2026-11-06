# Raw111完整九档评估

用户于2026-10-26提供111台训练完成run并要求“全部评估”。本次范围：SL/ML/LL × loose/medium/tight，固定原FCFS测试口径；不恢复已按用户要求停止的LM反馈消融，不启动训练，不改论文及原数据。

## 模型核验

- run：`results/raw_self_resource_models_20261025/sl_111/seed42_20261025_115207_654763_35688`。
- 训练manifest状态completed，Raw-Self、seed42、1200轮完整有序，43—1242工作流训练seed完整，60轮双验证完整。
- 原训练manifest中的10个来源文件SHA未漂移；checkpoint内config与manifest完全一致。
- 沿原`combine_checkpoint_metrics + metric_rank`验证规则选择ep1180，不按末轮或测试结果重选。
- 两块验证ESP@90及可靠性达标均1，合并验证能耗244367.24046513892 J（不是测试值）。
- checkpoint：该run的`best_d3qn_checkpoint.pt`。
- SHA256：`5a21e10a3f89bfabe5ef87b03daad2716ac1e4d6c6e10534188c2c51b0f65d10`。
- Raw反馈、下层Deadline budget0.08、84 edge+27 cloud、固定111槽位，SAC和上层Deadline Head均保留。

## 评估协议及启动

- 配置：`configs/v3_sl_111_new_eval_50wf.json`。
- 原`sec_job_fcfs_v1`、50wf/seed、0.03/s、可靠性要求0.9。
- small/medium/large × loose/medium/tight × 连续未筛选742000—742019，共180条，各档20条，所有失败保留。
- 8个worker，每进程Torch线程1；模型初始化seed742000，资源seed42。
- 复用已验证的Raw37 scale adapter、初始化入口及原credit evaluator的loader/输入构建/rollout/诊断；未修改这些旧函数。
- 复用Raw37冻结计划中的依赖文件清单，仅作来源核验，**没有复用任何Raw37结果或输入**；本次180个111台输入均重新生成并冻结身份指纹。
- 新入口及对应校验测试：`tools/eval_raw_sl111_generalization.py`、`tests/test_raw_sl111_generalization.py`。相关13项测试通过，包含旧测试类的重复收集，并非13项新测试。
- 启动前主机Python进程列表为空，未发现重复评估。
- 输出：`results/raw_sl111_generalization_20261026/`，会话54922。
- 父PID38260，创建时间2026-10-26 08:06:56（北京时间）；8个worker于08:06:57创建：61020、56884、28400、60172、27284、25056、47792、24532。
- 已通过主机进程列表确认父进程和8个worker存在。启动核验时正在计算首批，尚未声称完成。

已执行一次准备和一次运行，下面命令仅为记录，**不要重复执行**：

```powershell
python -B -X utf8 -u tools/eval_raw_sl111_generalization.py run
```

## 输出及完成标准

- `comparison_rows.csv/jsonl`每条更新；`workflow_details/`保存真实工作流级明细；`REPORT.md`每20条及最终完成时更新。
- 冻结来源、模型哈希、配置、各例输入物理/期限/工作量指纹见`frozen_plan.json`；模型选择见`validation_selection.json`。
- 最终需180有序唯一行、每档20条，所有来源hash保持一致，runtime输入与冻结指纹一致。
- 完成必须session退出0、日志`[raw111-complete]`且无Traceback、completion.json真实存在且失败列表与raw一致。
- 能耗J均值±样本SD；ESP@90为批内至少45/50按时的种子比例；Rel为批内所有工作流可靠性达到0.9的种子比例，不报平均DSR/rho代替达标率。
- 失败全部保留，不换种子、重选模型、自动重启或partial续跑；session unknown先合法核实进程身份，权限不足正常申请。
- 未新建定时检查，未动历史FCFS/HEFT实验及已停止的两组LM消融。
