# HEFT+SAC SM74 第854轮中断恢复（2026-10-15）

用户授权：电脑重启后继续未完成训练。不重新训练已完成的去 Deadline Head / rho=1 项，不自动启动评估或第三项训练。

## 冻结来源与恢复边界

- 原目录：`results/ablations/sm_74_legacy_train_20wf/heft_sac/20261014_155117`，全部文件保留，不原地覆盖。
- 恢复文件：该目录 `latest_checkpoint.pt`，payload episode=854；SHA256 `a6e2813ae73e535ddd3738acf781386390996119d3bce1ab1f90537b5a122162`。
- 原可行 best：`best_heft_checkpoint.pt`，episode=240，SHA256 `bc03579ba2e28c869cfa0876acdaf6eaa306ac0f48b616262225677173a8ef00`。原段双块验证合并能耗 262758.5420106505 J；这不是测试能耗。
- 直接使用生产入口的 `initial_full_checkpoint`，恢复网络、SAC 优化器、约束乘子及其窗口、随机状态、已校准队列权重（93.24580810040425）。不改策略、学习率、环境或资源。
- **旧配置未保存 replay buffer，检查点中 replay_state_dict=null。恢复后经验池从空重新积累，不是与未中断运行逐位一致的续训。** 后续实验报告需保留此恢复事实。

## 训练序列

- 配置：`configs/ablation_sm_74_heft_sac_resume_ep0854.json`，除恢复路径、输出路径、剩余预算及 episode offsets 外，与原配置相同。
- 总体目标1200轮；已完成854，续训346轮。下一轮 global_episode=855、episode_seed=897（42+855），最后 global_episode=1200、seed=1242。
- HEFT 不训练，high_active_episode_offset=0；SAC low_active_episode_offset=854，避免重置约束更新 warmup。
- SM=Small工作流/74台资源（56 edge+18 cloud），旧训练20工作流、0.1/s、Medium deadline不变。
- 验证保持累计轮次20的倍数：860、880、…、1200；两块542000/642000各20 seeds，workers=2不变。
- 生产入口 latest 每轮保存；编号 checkpoint 每本段20轮保存（首个 local20/global874），最后另存 final。未为恢复更改训练器保存逻辑。
- 本段 checkpoint 的 payload episode 是本段轮次，需加 config.training.episode_index_offset=854 解释；不要误称从零重训。
- 新输出根：`results/ablations/sm_74_legacy_train_20wf/heft_sac_resume_ep0854/<时间戳>/`。原历史不拼写覆盖。

## 完成后再做

核对原段854+续段346的累计1200轮、最终双验证和正常结束。生产入口本段 best 不继承原段 best 排名，必须按同一 combine_checkpoint_metrics/metric_rank 规则跨两段可行验证记录选最终模型，不能默认新段 best 优于原第240轮，不用测试重新选。

本次不自动启动最终消融评估。已完成的去 Deadline Head / rho=1 原训练1200轮无可行 best，其后续回退模型协议需另行确定，不能伪称通过约束。

## 启动状态

已于2026-10-15 11:47:25 +08:00启动一次；PID=39712，工具session=28445。启动前合法读取进程确认没有其他该训练实例，冻结 checkpoint hash 再次匹配。

- 命令：`D:/Miniconda/envs/py310/python.exe -u tools/train_coupled_hierarchical_d3qn_sac.py --config configs/ablation_sm_74_heft_sac_resume_ep0854.json`；不带 fixed-low-rho-one，不加载高层单独模型。
- 实际输出：`results/ablations/sm_74_legacy_train_20wf/heft_sac_resume_ep0854/20261015_114728`。
- 日志：`results/ablations/sm_74_legacy_train_20wf/heft_sac_resume_ep0854_20261015.log`。
- 启动日志确认 full_checkpoint episode=854、resume_full=True、high_algorithm=heft、low_algorithm=sac、low=cuda RTX5070、Torch线程1。
- 配置及已有恢复节奏回归：`python -m unittest tests.test_sm74_ablation_configs tests.test_sl_legacy_resume`，5项通过；没有用 smoke test 启动额外训练。
- 已实际完成累计855、856轮；首条记录 local=1/global=855/seed=897/high_active=0/low_active=855。SAC 每轮10次更新正常，首轮 replay 544条，次轮1107条；旧 latest SHA256保持不变。启动日志未见 Traceback。训练继续运行，尚未完成1200轮。

## 后续核查：累计860轮验证阶段权限异常

用户随后请求查看进度并先评估已完成项。实查本段只完成6条训练记录（global855–860），未产生验证记录或 final_checkpoint。日志在首次双进程验证的 TemporaryDirectory 快照写入时出现 `open file failed with error code: 5`，清理时同时出现 `PermissionError: [WinError 5]`，目标为沙箱 `AC/Temp/drlec_eval_*`。合法读取主机进程确认原PID39712及相关训练进程均不存在。

因此本段已异常退出，累计860/1200、尚差340轮；不能继续沿用“训练运行中”的早期状态。没有自动重启、改临时目录或改worker数以绕过权限。再次恢复需正常申请相应执行权限，并保留本段记录；新checkpoint payload local6需结合原offset854解释。

已完成的无 Deadline Head / rho=1 项保持1200轮、无可行best。尚未启动其测试；建议固定 final_checkpoint 作明确标注的末轮诊断评估，等待用户确认回退模型口径，不用测试挑选checkpoint。
