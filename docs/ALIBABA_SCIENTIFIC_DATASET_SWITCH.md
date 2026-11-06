# 科学工作流 / Alibaba 统一入口（2026-10-15）

本次只接通生成与环境、做协议预检；不训练、不自动启动四算法正式评估、不覆盖旧实验。Alibaba/111台资源不能直接解释为 Pegasus 的 Small 大小分组；这里的 SL 沿用用户所指111台资源设置，Alibaba不按策略表现筛选。

## 选择方式

配置：`configs/alibaba_sl_111_sec_job_100wf.json`。只改 `generator.source`：

- `alibaba_json`：从 `generator.alibaba_path=outputs/alibaba_test.json` 采样真实DAG，无放回、100个唯一job；相对计算量映射统计量固定拟合 `mapping_calibration_path=outputs/alibaba_train.json`，不按测试批次重新拟合。
- `pegasus`：从 `generator.path=data/sec_job_reference` 取科学工作流；`generator.dataset_scale=small` 可改为 `middle` / `large`。该字段不对Alibaba偷偷做大小过滤。

两者都经现有 `tools.hierarchical_common.build_workflows(config, resources, seed)` → `build_env(...)`，返回相同 `Workflow` / `Task` / `DependencyEdge` 对象。不修改状态编码器、动作空间、奖励、训练或checkpoint。

无需修改配置也能手动检查（已激活py310环境，在项目根目录运行）：

```powershell
python tools/inspect_workflows.py --config configs/alibaba_sl_111_sec_job_100wf.json --dataset alibaba_json --num-workflows 100 --seed 742000 --deadline-type loose
python tools/inspect_workflows.py --config configs/alibaba_sl_111_sec_job_100wf.json --dataset pegasus --num-workflows 100 --seed 742000 --deadline-type loose
```

把 `loose` 改成 `medium` / `tight` 即检查另外两档。配置固定resource_seed=42，检查seed不改变111台服务器配置。`inspect_workflows.py`原来把非pool来源硬编码成pegasus，此次已改为按实际选择路由。

## 新协议的固定定义

- 显式 `profile=sec_job_fcfs_v1` 才启用本次Alibaba新入口；旧Alibaba无profile或cai_2025_heft入口保留，不改变历史口径；非法profile报错，不悄悄退回HEFT。
- 资源111台=84 edge+27 cloud；100 workflows/episode；全系统一个到达流，`arrival_rate=0.03/s`，NumPy default_rng(seed)的指数间隔累加，首个到达也有指数等待。
- Alibaba DAG保留原job/task依赖；每个trace task的多个实例聚合成一个仿真任务，相对强度为 `duration * (plan_cpu/100) * max(1,instance_num)`，不是cycle实测值。
- 以独立校准池log1p(raw_weight)的p5/p95映射至0.5–20 Gcycles、加0.8–1.2抖动并截断到该范围；边通信量模拟为1–30 MB。模型不因数据来源改变。
- 同seed三档使用相同job、任务cycle、边MB、到达与参考响应时间；三档只变deadline倍率，随机流独立于任务映射，不因抽倍率导致后续任务物理量变化。
- 全部100个工作流、实际服务器和链路共同跑 **共享负载 FCFS/first-idle、rho=1** 参考；`T_ref = 参考完成时刻 - 该工作流到达时刻`。
- `D_relative = multiplier * T_ref`，`D_absolute = arrival + D_relative`；避免拿绝对完成时刻再乘倍率或重复加arrival。
- multiplier取2或3，取2的概率Loose=.2、Medium=.5、Tight=.8；相同逐工作流随机数保证Loose ≥ Medium ≥ Tight，允许部分工作流档间相等，实际100个样本比例不必恰好20/50/80。
- 可靠性要求显式0.9；旧配置的default_reliability=.95不应意外成为workflow约束。到达、deadline、通信、cycle和可靠性目标都是仿真输入，不能声称来自阿里原始实测。

## 预检复现与边界

```powershell
python tools/preflight_alibaba_sl111.py
```

默认独立输出 `results/alibaba_sl111_generation_preflight_20261015/`。若已有结果会拒绝覆盖；重做请显式用新的 `--output-dir`。导出三档原始workflow JSON、audit_summary.json、README。检查全部5万job的DAG与权重公式、train/test job不重叠、从batch_task.csv独立重建选中100个job、三档配对与可复现、参考任务执行/依赖通信/服务器无重叠、DDL公式及环境不重写DDL。另用无训练FCFS跑通三档Alibaba和100个科学工作流，**仅接线smoke，不是算法对比结果**。

现在可以接入环境不等于正式四算法结果已经完成。旧通用评估脚本的scale遍历对Alibaba没有Small/Medium/Large采样含义；正式评估应只跑一次Alibaba来源，并记录真实job名单和顺序，不能把同一池重复三次冒充三个规模。DQN-Edge仍须保留原有串行跨seed EMA规则。ready-task容量沿用模型既有48，不按数据源偷偷扩大网络；真实评估需报告截取行为，不能为适配而挑掉困难job。

## 已完成实测

固定seed742000/resource_seed42，三档共享100个唯一job、949个task、948条依赖。每job任务数5–34，中位7，未按结果或任务规模筛选。到达区间178.31658350606983–3154.5445286457493秒（首个到达也为指数间隔，不能强制移成0）。独立流式扫描原CSV共14295731行，选中100个job全部重建匹配；5万个池内job的DAG及raw_weight公式通过，校准/test job无重叠。

| 档位 | 2倍 / 3倍数量 | 平均相对DDL（秒） | 环境FCFS完成工作流 |
|---|---:|---:|---:|
| Loose | 15 / 85 | 57.534842 | 100 |
| Medium | 49 / 51 | 50.131171 | 100 |
| Tight | 75 / 25 | 45.381191 | 100 |

参考响应时间三档完全相同，均值20.202375秒；逐工作流DDL嵌套、物理量配对、同seed重现、环境保留DDL均通过。参考轨迹的ready峰值（含同刻释放）为12；这一条参考轨迹不能保证所有学习策略都不触及ready容量48。

仅切source为pegasus后，100个科学工作流也经同一环境FCFS完整运行。所有导出JSON已重新读入并构造Workflow，各100个；冻结输入与代码SHA256复核相同。

回归：`python -m unittest tests.test_alibaba_sec_job tests.test_sec_job_generation tests.test_arrival_process tests.test_legacy_generation_restore`，20项通过。无训练/模型权重加载。此100-job样本平均任务数仅9.49、到达率0.03/s、资源111台，并不自动代表高拥塞压力测试；不得据此预先宣称重负载泛化优势。
