# SM74消融正式评估：2026-10-15

用户授权“把我们的消融实验跑起来”。仅评估已有模型，不启动新训练；保留全部原始结果与失败。

## 固定协议

- 74台=56 edge+18 cloud，科学Small工作流，50 workflows/seed，`sec_job_fcfs_v1`，到达率0.03/s，可靠性要求0.9；`configs/v3_sm_74_new_eval_50wf.json`。
- Loose→Medium→Tight，每档742000–742019，共60条/变体。资源seed42、模型初始化742000，四个新评估各2 worker、每进程Torch线程1，总计8个评估worker。
- 复用原始未筛选Full同键60条作为参考；不重跑Full、不使用条件筛选名单、不换失败种子。
- 输出独立 `results/ablation_sm74_new_eval_20261015`；原始场景主表暂不改写。

## 模型选择与四项新评估

1. HEFT+SAC：跨原训练和两次恢复的**真实完成**双验证块，复用 `combine_checkpoint_metrics` 和 `metric_rank`，先要求两块ESP@90=1且可靠性达标，再选最低验证能耗。原段42轮验证+最终恢复17轮=59轮；第860轮验证未完成，不补造。选定原第240轮best，双块平均验证能耗262758.5420106505J。恢复后最佳为累计1120轮457233.71536707086J，未超过原240轮。
2. HEFT rho=1：相同HEFT策略的满频诊断，不训练、不重新选checkpoint。
3. Full checkpoint rho=1：原Full第1200轮上层固定权重，禁用SAC并执行rho=1；这是冻结策略干预，不是独立重训的完整上层rho=1。
4. 去Deadline Head、rho=1：当前74台训练1200轮、60轮双验证均无可行模型；固定final_checkpoint第1200轮作**末轮失败诊断**，明确无可行验证，不按测试结果换成其他轮。

各模型三档Deadline共用一个固定checkpoint。全部来源hash、候选验证排名、loader配置、模型开关、样本物理量/DDL hash写入冻结计划及manifest。

## 口径与限制

- 无头模型加载其自身config，`deadline_head_enabled=false`、`use_low_agent=false`；同时校验checkpoint开关。不能复用通用loader中强制启用SAC的逻辑。
- 满频三项均校验实际rho_mean=1和无下层buffer，避免只改标签。
- 所有任务同一既定派发规则，改变的只有已声明的上层/频率策略。每个seed四项工作流与DDL指纹一致。
- 正式主消融指标：计算能耗J均值±样本SD、ESP@90、每批所有工作流可靠性达到0.9的比例；失败如实保留，不用低能耗掩盖违约。
- 当前缺少74台**独立训练完整上层rho=1**，因此无头最终模型与联合训练Full之间不是严格单因素Deadline Head因果对照。旧26台同类结果不混入本表。
- HEFT训练曾恢复两次，replay未保存会重新积累；完整记录此限制，不称逐位无缝续训。
- 单训练seed42、20个工作负载seed；不声称多训练重复。
- 各run目录创建一次，已有目录拒绝重复启动；异常不自动重启或partial续跑。结束核验60个有序唯一键、每档20、样本指纹配对、源hash不变、正常completion和日志。

## 入口

`python tools/eval_sm74_ablation.py prepare`：首次冻结计划；已存在则拒绝。

`python tools/eval_sm74_ablation.py run <variant>`：variant为heft_sac、heft_rho1、full_rho1、no_deadline_rho1；每项仅启动一次。

`python tools/eval_sm74_ablation.py report`：只刷新本实验报告，不开展推理或训练。
