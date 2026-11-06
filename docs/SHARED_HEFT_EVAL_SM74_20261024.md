# Shared HEFT 训练模型的独立 HEFT 期限测试

## 授权与范围

用户要求保留原 FCFS，独立测试 shared_heft 模型的其他工作流规模。本次只覆盖该74台固定槽位模型的 SM/MM/LM × Loose/Medium/Tight × 20种子，共180条；不训练37/111台，不将其称作27档完整资源泛化。

## 固定模型

- Run：`results/deadline_reference_sm74_budget001/shared_heft/seed42_20261024_075301_286790_20940`。
- 训练1200轮完成，Full责任反馈不变，下层deadline预算0.01。
- 原始双验证542000/642000、各20seed的既有rank选出第1200轮；不是根据新测试选择。
- `best_d3qn_checkpoint.pt` SHA256：`c3167e8cf93e0ef075682f7c51c768ecdc1d6ebb8dfbe3a145397938ba2d2e04`。
- 验证参考仍为legacy isolated HEFT。新测试不宣称训练、验证、测试数据生成完全相同。

## 测试口径

1. 复用原 `configs/v3_sm_74_new_eval_50wf.json` 物理生成：56 edge +18 cloud，50工作流，0.03/s，连续未筛选742000—742019，资源seed42。
2. 同一个工作流集合先经原FCFS生成，保留其原2x/3x期限系数；不改DAG、计算量、通信量、到达时间或可靠性要求。
3. 使用训练所用的 `src/datasets/shared_heft_deadlines.py::assign_shared_heft_deadlines`，运行共享环境online HEFT、rho=1，得到各workflow的响应时间（不是整个环境单一makespan）。
4. 相对期限=原系数×该workflow的HEFT响应时间；绝对期限=到达时间+相对期限。原生成器profile仍标记物理来源sec_job_fcfs_v1，新结果另记deadline_reference=shared_environment_online_heft_rho1，不冒称原FCFS期限。
5. 真正评估执行仍是固定的D3QN+SAC模型，确定性推理，不训练或更新网络。加载使用checkpoint配置（budget0.01）及objective状态，初始化seed742000。
6. 原FCFS函数/配置/数据不改；只在独立工作进程中临时替换评估入口的样本构造调用，结束后恢复。旧的global训练与其他进程不干预。

## 执行与核验

- 新入口：`tools/eval_shared_heft_sm74.py`；命令 `python -u -B -X utf8 tools/eval_shared_heft_sm74.py start`。
- 两路并行，先生成180批参考期限、冻结物理输入及期限SHA；通过后自动评估180批。
- 输出：`results/shared_heft_eval_sm74_20261024/`；准备日志在同级`shared_heft_eval_sm74_20261024_launch.log`。
- `deadline_audit.json`保留每个workflow原FCFS参考/期限与新HEFT参考/期限；检查与此前Raw评估的FCFS物理输入一致。
- 保留逐工作流明细、原始CSV/JSONL、模型与来源SHA、失败名单。每档20唯一有序种子，全部完成后输出REPORT.md。
- ESP@90是每批至少45/50按时；Rel是每批所有workflow可靠性≥0.9。能耗为J均值±种子间样本SD。
- 15个不同测试通过（新评估5项、已有期限参考10项），包括真实50workflow三档构造、系数保留、参考独立于DDL、原FCFS不变及入口恢复；测试没有真实训练网络。
- 禁止重复启动、自动partial续跑、筛选失败seed或为追求100%调系数。

## 启动记录

- 2026-10-24已启动 `start`，执行session `74303`。启动前主机没有Python进程，目标输出目录和启动日志均不存在。
- 当前首先处于HEFT参考期限准备阶段；该阶段通过后同一命令自动进入模型评估，无须再次启动。
- 已再核对原Raw/FCFS冻结来源哈希未变。是否最终完成应以日志、180行校验、completion.json和进程正常退出共同确认，不能仅凭本启动记录判断。

## 解释限制

这是改变期限生成协议后的探索性实验，不是原FCFS条件下的算法提升。共享HEFT期限可能更紧或更松，必须由审计值判断，不能预设ESP100。新旧模型及期限均不同，不能把与旧Full/Raw的差异全部归因于训练参考改变。

## 用户授权的12路交接

- 用户明确批准结束本次评估3个父子进程，保留已完成结果，只将未完成MM/LM任务改12路并发。停止前核对PID/创建时间/命令：父2652，子27844、57876。没有停止其他程序。
- 切换时保存65条完整有序记录：SM60条，MM–Loose种子742000—742004五条；均通过ESP@90和Rel。其余115条重新派发，不重算已完成记录。
- 新适配入口 `tools/accelerate_shared_heft_sm74.py`，执行session `70344`。原评估脚本和冻结plan均不修改；plan里的workers=2描述初始运行，新`acceleration_12/handoff_manifest.json`明确记录实际workers=12、交接授权和来源SHA。
- MM/LM交错派发，原确定性D3QN+SAC推理函数不变，Torch每进程1线程；本模型没有DQN-Edge的跨seed attention EMA。每批环境独立重置，不训练，不更新模型或乘子。
- `acceleration_12/original_snapshot/`保存原CSV、JSONL、执行状态、日志和报告的副本及校验SHA。后续原JSONL保持标准遍历的连续前缀，所有已完成（含暂时跨过未完成位置）的真实结果另存`acceleration_12/comparison_rows.jsonl`及逐行JSON，进度看该目录`progress.json`/`REPORT.md`。
- 3项交接专项测试通过：不重复已完成case、前12任务MM/LM各6个、乱序完成合法校验及保留原连续前缀。完成后重新执行原180行全量校验和来源SHA检查，输出统一报告。

## 暂停至2026-10-25 06:00（北京时间）

- 用户要求“先结束掉，10.25早上六点开始评估”。已按暂停原进程、保留内存方式执行，不再新建评估或丢弃在途工作。
- `tools/control_shared_heft_pause.py pause`已对父48112及其12个子进程各执行一次NtSuspendProcess，全部返回0。两秒采样确认13个进程CPU时间增量均为0。
- 暂停时实际已完成80/180：SM60、MM14、LM6。所有结果仍在原目录和acceleration_12目录中，未删除。
- 进程完整身份与暂停记录在 `results/shared_heft_eval_sm74_20261024/acceleration_12/pause_state.json`。该文件同时防止重复暂停/恢复，恢复前校验全部PID、创建时间和命令。
- 已创建本聊天一次性恢复自动任务 `10-25-06-heft`，10月25日06:00（北京时间）执行。到时仅运行 `python -B -X utf8 tools/control_shared_heft_pause.py resume` 一次，再核对CPU时间或日志推进；不得重新运行start/run/accelerate，避免重复评估。
- 用户需保持电脑开机、Codex运行且不休眠/注销/重启；若进程缺失或身份变化，只报告，不盲目重启或拼接。若需额外权限按正常审批流程申请。
- 暂停身份校验3项专项测试通过；未改原评估策略、模型、期限、种子或FCFS来源文件。

## 2026-10-25 06:00定时恢复已执行

- 定时触发后核对无completion.json，pause_state为suspended；恢复脚本核对原13个进程完整身份全部一致。
- 仅运行一次 `control_shared_heft_pause.py resume`，原13个进程NtResumeProcess全部返回0，pause_state已记为resumed，禁止重复恢复。
- 三秒CPU采样显示12个工作进程各增长约2.9秒CPU时间；评估日志新增，实际进度82/180（SM60、MM15、LM7）。原父进程等待任务结果，未新建任何评估。
- 一次性恢复自动任务 `10-25-06-heft` 已删除；不每日重复，不新增完成监控任务。
