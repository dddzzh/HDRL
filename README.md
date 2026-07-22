# DRLEC 动态边云工作流调度实验框架

这个项目用于做动态 edge-cloud workflow scheduling 实验。当前代码已经支持：

- 从 Alibaba workflow JSON 加载工作流 DAG。
- 给 workflow 预生成泊松到达时间。
- 按 arrival_time 动态释放 workflow。
- 使用 FCFS、random、fastest 三种调度策略运行仿真。
- 输出 summary、workflow 级结果、task 调度事件日志。
- 在运行时打印详细调度过程，方便观察 workflow 到达和 scheduler 决策。

## 目录结构

```text
configs/
  experiment_default.json              默认实验配置
  resources_edge_cloud_default.json    edge/cloud 服务器资源配置

dataset/
  原始 Alibaba trace 数据

outputs/
  alibaba_workflows.json               预处理后的 workflow DAG
  alibaba_workflows_with_arrivals.json 加入泊松到达后的 workflow DAG
  alibaba_workflow_arrival_summary.csv 到达时间摘要

src/
  datasets/                            数据集加载与标准化
  experiments/                         runner 和结果保存
  schedulers/                          调度策略
  simulator/                           仿真环境、资源、指标

tools/
  build_alibaba_workflows.py           从原始 trace 构造 workflow DAG
  add_poisson_arrivals.py              给 workflow 加泊松到达
  run_experiment.py                    跑实验的命令行入口

results/
  每次实验的输出结果
```

## Python 命令

当前这台机器上 `python` 和 `py` 没在 PATH 里，所以建议直接用完整解释器路径：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe
```

下面命令都以这个解释器为例。运行命令前请先进入项目根目录：

```powershell
cd D:\研究生期间\第二篇\DRLEC
```

## 一条命令跑随机策略并打印调度过程

最常用命令：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler random --trace-scheduling --trace-limit 100 --print-details --max-workflows 20 --max-events 50
```

这条命令会：

- 使用 `random` 随机调度策略。
- 打印前 100 行调度过程 trace。
- 实验结束后打印前 20 个 workflow 的到达结果。
- 实验结束后打印前 50 条 task 调度事件。
- 保存完整结果到 `results/alibaba_dynamic_workflow_test/random/seed_42/`。

如果只想看调度过程，不看结束后的详细表：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler random --trace-scheduling --trace-limit 500
```

如果想看更多 trace，把 `--trace-limit` 调大：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler random --trace-scheduling --trace-limit 2000
```

## 调度过程日志怎么看

打开 `--trace-scheduling` 后，会打印三类日志。

`ARRIVAL` 表示 workflow 到达并被释放：

```text
ARRIVAL time=144.252516 workflow=j_1047618 arrival_time=144.252516 tasks=11
```

字段含义：

- `time`：当前仿真时间。
- `workflow`：到达的 workflow id。
- `arrival_time`：该 workflow 数据中记录的到达时间。
- `tasks`：这个 workflow 中 task 数量。

`DECISION` 表示 scheduler 在当前 ready task 队列中做选择：

```text
DECISION time=144.252516 ready=4 [j_1047618:t0, j_1047618:t1, j_1047618:t2, j_1047618:t3] -> workflow=j_1047618 task=1 server=0 rho=0.600000
```

字段含义：

- `time`：当前仿真决策时间。
- `ready`：当前可调度 task 数量。
- `[...]`：ready task 队列预览，格式是 `workflow_id:t任务id`。
- `workflow/task/server/rho`：scheduler 最终选择的 workflow、task、服务器和 rho。

`EXECUTE` 表示该 task 实际执行后的结果：

```text
EXECUTE workflow=j_1047618 task=1 server=0 rho=0.600000 start=144.252516 finish=144.291405 exec=0.038889 energy=0.976889
```

字段含义：

- `start`：task 实际开始时间。
- `finish`：task 完成时间。
- `exec`：执行耗时。
- `energy`：该 task 消耗能量。

注意：`DECISION time` 和 `EXECUTE start` 不一定相同。原因是 task 开始时间还要满足父任务完成时间和服务器可用时间：

```text
start_time = max(current_time, workflow_arrival_time, parent_finish_time, server_available_time)
```

## 当前泊松到达是怎么实现的

泊松到达生成在：

```text
tools/add_poisson_arrivals.py
```

核心逻辑是：

```text
inter_arrival_time ~ Exponential(mean_interarrival)
arrival_time = cumulative sum of inter_arrival_time
```

这等价于齐次泊松过程的到达时刻生成方式。

默认输入：

```text
outputs/alibaba_workflows.json
```

默认输出：

```text
outputs/alibaba_workflows_with_arrivals.json
outputs/alibaba_workflow_arrival_summary.csv
```

重新生成泊松到达：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\add_poisson_arrivals.py --mean_interarrival 60 --seed 42
```

参数说明：

- `--mean_interarrival 60`：平均到达间隔为 60 秒。
- `--seed 42`：随机种子，保证可复现。
- `--start_time 0`：第一个 workflow 到达时间。
- `--shuffle`：生成到达前先打乱 workflow 顺序。

## Runner 执行流程

主入口是：

```text
tools/run_experiment.py
```

它会创建：

```text
src/experiments/runner.py::ExperimentRunner
```

runner 的流程：

1. 读取 `configs/experiment_default.json`。
2. 根据命令行覆盖 scheduler、dataset、seeds。
3. 加载 workflow dataset。
4. 加载 edge/cloud 资源配置。
5. 根据 scheduler 名称创建调度器。
6. 创建 `EdgeCloudWorkflowEnv` 仿真环境。
7. 调用 `env.run(scheduler)` 执行仿真。
8. 保存 summary、workflow_results、event_log。

## 默认实验配置

默认配置文件：

```text
configs/experiment_default.json
```

关键字段：

```json
{
  "dataset": {
    "type": "alibaba_json",
    "path": "outputs/alibaba_workflows_with_arrivals.json",
    "max_workflows": 100,
    "shuffle": false
  },
  "scheduler": {
    "name": "fcfs"
  },
  "runs": {
    "seeds": [42],
    "output_root": "results"
  }
}
```

说明：

- 默认 dataset 已经是带泊松到达时间的文件。
- `max_workflows` 控制本次实验最多加载多少个 workflow。
- 命令行 `--scheduler random` 会覆盖配置里的 scheduler。
- 命令行 `--seeds 1 2 3` 会覆盖配置里的 seeds。

## 支持的调度策略

调度器注册在：

```text
src/schedulers/registry.py
```

当前支持：

```text
fcfs     先选 arrival_time 最早的 ready task，再选 available_time 最早的 server
random   从 ready tasks、servers、rho_levels 中随机选择
fastest  选 arrival_time 最早的 ready task，再选 max_speed_mips 最高的 server
```

运行 FCFS：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler fcfs
```

运行最快服务器策略：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler fastest
```

运行随机策略：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler random
```

## 输出文件

每次运行后会输出到：

```text
results/<experiment_name>/<scheduler_name>/seed_<seed>/
```

例如随机策略 seed 42：

```text
results/alibaba_dynamic_workflow_test/random/seed_42/
```

目录内文件：

```text
config.json             本次运行实际使用的配置
summary.json            汇总指标
workflow_results.csv    workflow 级结果
event_log.csv           task 调度事件
```

上一级还会生成：

```text
results/alibaba_dynamic_workflow_test/random/aggregate_summary.csv
```

## 输出指标含义

`summary.json` 和控制台 summary 中常见字段：

- `total_energy`：所有 task 总能耗。
- `avg_workflow_makespan`：workflow 内部从第一个 task 开始到最后一个 task 完成的平均跨度。
- `avg_workflow_response_time`：workflow 从 arrival_time 到完成的平均响应时间。
- `avg_workflow_waiting_time`：workflow 从 arrival_time 到第一个 task 开始的平均等待时间。
- `total_finished_workflows`：完成的 workflow 数。
- `deadline_success_rate`：如果 workflow 有 deadline，则统计 deadline 满足率；当前默认是 `N/A`。
- `avg_task_execution_time`：平均 task 执行时间。
- `avg_server_utilization`：平均服务器利用率。

`workflow_results.csv` 字段：

- `workflow_id`：workflow id。
- `arrival_time`：workflow 到达时间。
- `finish_time`：workflow 完成时间。
- `response_time`：`finish_time - arrival_time`。
- `makespan`：workflow 内部 task 从首次开始到最后完成的跨度。
- `waiting_time`：第一个 task 开始时间减去 arrival_time。
- `total_energy`：该 workflow 下所有 task 总能耗。
- `deadline/deadline_met`：deadline 相关字段。

`event_log.csv` 字段：

- `workflow_id`：task 所属 workflow。
- `task_id`：task id。
- `server_id`：被调度到的服务器。
- `rho`：调度使用的 rho。
- `start_time`：实际开始时间。
- `finish_time`：完成时间。
- `exec_time`：执行耗时。
- `energy`：能耗。

## 常用命令合集

随机策略，打印调度过程：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler random --trace-scheduling --trace-limit 100
```

随机策略，打印过程和结果表：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler random --trace-scheduling --trace-limit 100 --print-details --max-workflows 20 --max-events 50
```

多 seed 运行：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler random --seeds 1 2 3
```

指定其他 dataset：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\run_experiment.py --scheduler random --dataset outputs/alibaba_workflows_with_arrivals.json
```

重新生成平均间隔 30 秒的泊松到达：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe .\tools\add_poisson_arrivals.py --mean_interarrival 30 --seed 42
```

## 代码阅读入口

建议按这个顺序看代码：

1. `tools/run_experiment.py`：命令行入口，负责解析参数和打印结果。
2. `src/experiments/runner.py`：实验编排，负责加载数据、资源、scheduler 和保存结果。
3. `src/datasets/alibaba_dataset.py`：读取 workflow JSON，校验字段，按 arrival_time 排序。
4. `src/simulator/edge_cloud_env.py`：核心仿真循环，包含 workflow 到达释放、ready task 生成、调度执行和 trace 打印。
5. `src/schedulers/random_scheduler.py`：随机策略。
6. `src/schedulers/fcfs_scheduler.py`：FCFS 策略。
7. `src/schedulers/fastest_server_scheduler.py`：最快服务器策略。
8. `src/simulator/metrics.py`：summary 指标计算。

## 当前实现边界

当前版本中：

- 泊松到达是在运行前预生成到 JSON 文件里，不是在 runner 内部在线采样。
- 仿真器根据 JSON 中的 `arrival_time` 动态释放 workflow。
- `communication_enabled=true` 还没有实现。
- backup scheduling 还没有实现。
- deadline 和 reliability 字段预留了配置，但默认实验没有启用。

