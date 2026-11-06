# HEFT+SAC SM74 累计860轮恢复

2026-10-15用户明确选择“只恢复 HEFT+SAC”；不启动无DeadlineHead最终模型评估。

- 恢复源：`results/ablations/sm_74_legacy_train_20wf/heft_sac_resume_ep0854/20261015_114728/latest_checkpoint.pt`。
- payload本段episode6 + 保存config offset854 =累计860；SHA256 `74bc6eb89cf4a4adb52cc2e7704b8fc0b009803703bb0e5c97fa5a798657f062`。
- 上一段完成855–860轮后，在验证快照写入沙箱系统临时目录时权限异常退出；并非正常完成。所有原目录、日志、检查点均保留。
- 新配置 `configs/ablation_sm_74_heft_sac_resume_ep0860.json`：从原HEFT配置复制，仅改剩余轮数340、累计offset860、low_active_offset860、full checkpoint及输出路径；high_active_offset仍0。
- 保持20工作流旧训练协议、74台、训练seed42、SAC学习率/约束/队列校准、Torch线程1、验证workers2、总预算1200轮不变。不改训练器或算法代码。
- 下一轮累计861/seed903；末轮累计1200/seed1242。新checkpoint payload本段episode加860解释，不能再只加854。
- 860轮训练checkpoint有效，但860验证未完成；不伪造该轮验证，不自动写回历史。恢复后按既有global cadence验证880、900、…、1200；最终选模型只使用真实完成的双块验证，跨原段/恢复段共同按既定规则比较，原best全保留。
- 恢复模型、优化器、dual窗口/乘子、随机状态和beta_q=93.24580810040425。**Replay仍未保存，恢复后再次从空池积累，不是逐位无缝续训。** 不以此掩盖中断或宣称独立训练复现。
- 本次通过正常主机执行权限申请解决临时目录访问，不改TMP、不改worker、不绕过权限。
- 输出新根 `results/ablations/sm_74_legacy_train_20wf/heft_sac_resume_ep0860/<时间戳>/`，日志同父目录 `heft_sac_resume_ep0860_20261015.log`。不得重复启动。

## 实际启动与首次验证

正常权限审批后，2026-10-15 15:08:39.994423 +08:00启动一次，PID6076，工具session62104；实际输出 `results/ablations/sm_74_legacy_train_20wf/heft_sac_resume_ep0860/20261015_150842`。启动日志 full_checkpoint episode6/resume_full=True 对应累计860。已核对第一轮应为861/seed903。

CPU评估加速后的检查：累计937/1200（seed979），已真实完成880、900、920各两块验证，共6条验证历史，日志无Traceback；此前系统临时目录权限问题没有再次出现，训练继续。剩余263轮为此次检查时状态，后续以实际history为准。不启动无DeadlineHead的评估。
