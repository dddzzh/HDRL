# LM四版本Deadline反馈扩展

## 最新状态：用户要求停止，已执行

本聊天用户随后明确要求“停掉吧”。仅终止本次LM的两个父进程及8个worker，并再次通过主机进程列表确认已全部退出；111台Raw训练PID35688保持运行。

CF-Self与CF-Uniform均保留45/60条已完整写入结果（各loose20、medium20、tight5，最后seed742004），未完成，不可视作完整三档结论。没有completion.json，未删除或覆盖原始结果；未完成任务的进程内存不再保留。禁止自动重启或从partial盲目续跑，后续安排需用户重新决定。

用户讨论将原消融扩展到LS/LM/LL，并同意先跑现有74台模型支持的LM三档。执行前已明确：本次比较Full、CF-Self、CF-Uniform、Raw-Self，不是HEFT/去头结构消融；暂不训练37/111台消融模型。

## 固定协议

- 74服务器（56 edge+18 cloud）、Large科学工作流、每seed50wf、0.03/s、原共享FCFS期限。
- loose/medium/tight各连续未筛选742000—742019，共60条/版本。
- Full复用`results/legacy_train_new_eval_20261009/sm_74/comparison_rows.jsonl`的Large60条。
- Raw复用`results/raw_sm74_generalization_20261024/comparison_rows.jsonl`的Large60条。
- 两个引用目录和源模型保持不变，不重复评估Full/Raw。新目录中明确存为引用结果。
- CF-Self沿原双验证已选ep1200：SHA256 `6851c8a3a1461228aaac35a9acd2f58102706382c7dfde9b9a0009e7de7991ec`。
- CF-Uniform沿原双验证已选ep1180：SHA256 `7857daa9ad5a719e9d68d09d9932fe3c6c03950032cbc728ed2b58cc542032e9`。
- 通过既有`selected_model`重新核对验证排名、payload与原冻结记录完全一致，不按Large测试重选。
- 新增每版本60条；各4个worker，共8个；Torch线程1，模型初始化742000、资源seed42。
- 全部调用原loader、workflows构造、仿真rollout和工作流诊断；只新增独立评估适配器及测试，未改训练、网络、物理模型或论文。
- 原Small消融和Raw74冻结来源哈希均核对无漂移；Large输入指纹复用已核验Raw74冻结计划，运行时逐例重建核对。

## 已启动（2026-10-25本地时间）

|版本|session|父PID|创建时间|子PID|
|---|---:|---:|---|---|
|cf_self|8619|22536|22:43:32|30584,54924,40840,36280|
|cf_uniform|32933|59044|22:43:35|9872,54256,58264,37444|

核验时两父进程及8个worker均存在，正在计算首批，尚无新结果行；无failure.json。原111台Raw训练PID35688及其验证进程未干预。

输出：`results/lm_deadline_credit_20261025/`；汇总`REPORT.md`；各版本保留CSV/JSONL及新评估的工作流级明细。日志`cf_self.log`和`cf_uniform.log`。

已执行准备；以下仅记录启动命令，**不要重复运行**：

```powershell
python -B -X utf8 -u tools/eval_lm_deadline_credit.py run cf_self
python -B -X utf8 -u tools/eval_lm_deadline_credit.py run cf_uniform
```

相关14项测试通过（含导入旧测试类的重复收集）。本次没有创建定时检查。

## 完成与解释边界

- 两新增版本各60有序唯一真实行，各档20；来源SHA不变；runtime输入指纹与冻结计划一致。
- 各日志正常`[lm-complete]`且无Traceback、session退出0、completion.json真实存在，失败列表与原始行一致。
- 汇总可用`python -B -X utf8 tools/eval_lm_deadline_credit.py report`更新；读取的JSONL由原工具原子替换。
- 报J均值±样本SD、ESP@90、可靠性达标率，同时列实际迟到工作流数（每档1000个），不以平均DSR替代ESP。
- 所有失败保留，不重选模型/改参数/换种子/自动重启或partial续跑；session unknown先合法核实进程。
- Full为历史模型，不是同代码新重训的第四组，只有一个训练seed；该扩展在查看Small结果后开展，属于探索性分析，不能当成严格单变量因果证据或预注册独立确认。
- 此任务只覆盖LM，不代表LS/LL已完成，也不代表Raw版结构消融已完成。
