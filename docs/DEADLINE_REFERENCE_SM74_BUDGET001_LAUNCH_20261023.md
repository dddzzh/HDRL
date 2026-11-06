# SM74 两组 Full 训练：预算 0.01 与共享 HEFT 期限参考

日期：2026-10-23。只新增实验文件，不修改旧训练器、环境、策略、归因函数、配置或结果。本次由用户手动启动，不自动启动训练。

## 1. 两组的区别

| 组 | 训练 Deadline 参考 | 下层 deadline_budget | 其他学习配置 |
|---|---|---:|---|
| A / isolated_heft | 原单工作流 HEFT | 0.01 | 原 Full，保持不变 |
| B / shared_heft | 同批工作流在共享环境中的在线 HEFT，固定 rho=1 | 0.01 | 与 A 相同 |

两组都是完整 CAHDRL，上层 Deadline Head、SAC、反事实来源归因、可靠性反馈和 rho 能效下限均保留。不是 HEFT+SAC：HEFT 只产生训练任务的参考期限，真正学习和调度的仍是完整网络。

共同配置：SM（Small 工作流、56 edge + 18 cloud）、每轮 20 个工作流、旧生成器到达率 0.1/s、Medium、seed42、从头联合训练 1200 episode。工作流输入 seed 为 43—1242。除下层预算 0.08→0.01、输出路径和显式实验元数据外，新配置与原配置完全一致；入口会校验并拒绝意外差异。

## 2. VSCode 启动命令

在两个独立终端中运行；工作目录为 `D:/研究生期间/第二篇/DRLEC`，Python 环境使用 `D:/Miniconda/envs/py310/python.exe`。每条命令只运行一次。

A：原 Deadline 参考 + budget0.01：

```powershell
python -u tools/train_deadline_reference_experiment.py --config configs/diagnostic_sm74_isolated_heft_budget001.json
```

B：共享环境 HEFT 参考 + budget0.01：

```powershell
python -u tools/train_deadline_reference_experiment.py --config configs/diagnostic_sm74_shared_heft_budget001.json
```

可先在命令末尾加 `--check`：只校验配置，不创建训练目录、不启动网络训练。

**两组都必须使用上述新入口。不要用原训练入口读取 B 配置：原入口不会解释新增实验元数据，会按原单工作流 HEFT 生成期限。** 新入口不接受 resume、freeze 或更换算法参数；中断后的恢复需另行核对，不把只加载权重称为无缝恢复。

## 3. B 的期限生成定义

1. 先完整调用原生成器，包括原先隔离 HEFT 的期限计算和 2/3 倍率抽样。保留其随机数消耗，使 A、B 相同 seed 的工作流类型、任务、依赖、边数据、到达时间与倍率完全相同。
2. 深拷贝这些工作流，用现有 `EdgeCloudWorkflowEnv.run(HEFTScheduler(rho=1))` 做一次独立参考调度。复用真实环境的到达事件、依赖就绪、通信和服务器队列；不调用网络、不学习、不改变全局随机数状态。
3. HEFT 在已到达、依赖已完成的任务中选最高 upward rank，再按考虑通信与服务器排队的 EFT 放置。禁止备份，固定全频；参考策略不依据 Deadline 挑选动作。
4. 对每个工作流分别取 `reference = HEFT_finish - arrival`，令 `relative_deadline = 原倍率 × reference`，`absolute_deadline = arrival + relative_deadline`。
5. 仅将新期限字段交给正常训练环境。不是全环境共用一个截止时间，也不是让模型自己的完成时间决定 Deadline。

B 是“共享环境在线 HEFT 参考”，不是重新编写一个离线多 DAG 插空 HEFT；复用现有在线就绪规则。不能笼统声称其与隔离静态 HEFT 在任意 DAG 上除了排队外完全等价。它也不是当前评估用的共享 FCFS，因此只能研究共享环境参考的训练影响，不能声称已完全消除训练/测试分布差异。

## 4. 验证和后续测试不跟着放宽

为了 A/B 可比，**B 的新期限只用于训练 rollout**。两组训练内验证都保持原单工作流 HEFT、20 工作流、两块 20-seed 验证池（542000—542019 和 642000—642019）、每 20 轮验证，以及原可行性优先选模规则。若没有可行检查点，如实记录。

验证单进程时临时旁路训练生成 hook；Windows spawn 并行 worker 导入原生产模块，天然使用原生成器。默认每模型 Torch 线程 1、验证 worker 2，均沿用旧配置。后续 50 工作流、0.03/s、共享 FCFS 测试另外运行，本入口不启动它。

新 Full 与旧 budget0.08 Raw 的差异只能作探索。若最终采用新协议做正式归因消融，需要相应重训对照组；不保证 Full 胜过 Raw。

## 5. 输出和次日检查

每次启动都创建包含时间微秒和 PID 的独立 run 文件夹，不覆盖旧结果：

- A：`results/deadline_reference_sm74_budget001/isolated_heft/seed42_<timestamp>_<pid>/`
- B：`results/deadline_reference_sm74_budget001/shared_heft/seed42_<timestamp>_<pid>/`

每个 run 保存：

- 原训练器的 `train_history.jsonl`、`eval_history.jsonl` 和 checkpoints；选模型时看验证记录及 best checkpoint 的 episode，不默认选择最后一轮。
- `resolved_config.json`：此次实际配置。
- `deadline_reference_manifest.json`：模式、种子表、源码/配置 SHA256、状态和是否有验证可行模型；状态是 completed 才表示训练函数正常返回。
- `training_inputs.jsonl`：每轮实际输入 seed、物理输入哈希、期限哈希；B 还保存每个工作流的旧参考、新共享参考、倍率及新期限，便于与 A 对齐核验。

诊断同时看 `low_deadline_constraint_value`、`low_deadline_constraint_batch_mean`、更新前后 lambda 与实际 `low_lagrangian_penalty_mean`。最后一个是 Deadline 惩罚，不是总惩罚。不要把原日志的 `low_deadline_constraint_cost_mean` 当成工作流平均责任量：它会被 SAC 采样任务代价的同名诊断覆盖，旧逻辑此次未修改。

## 6. 实现与检查

- [独立训练入口](D:/研究生期间/第二篇/DRLEC/tools/train_deadline_reference_experiment.py)
- [共享 HEFT 期限适配](D:/研究生期间/第二篇/DRLEC/src/datasets/shared_heft_deadlines.py)
- [A 配置](D:/研究生期间/第二篇/DRLEC/configs/diagnostic_sm74_isolated_heft_budget001.json)
- [B 配置](D:/研究生期间/第二篇/DRLEC/configs/diagnostic_sm74_shared_heft_budget001.json)
- [新增测试](D:/研究生期间/第二篇/DRLEC/tests/test_deadline_reference_experiment.py)

新增测试覆盖配置差异、参考调度的单任务/共享争用/晚到达/DAG通信、旧期限不影响参考完成时间、输入及全局 RNG 不变、真实 XML 输入配对、验证旁路与异常后的 hook 恢复，以及模拟训练调用的 manifest 状态。模拟调用不会训练网络。

最终检查：10 项新增测试与相关 HEFT、旧/新生成器、归因消融、SAC、上层违约控制回归合计 99 项通过。测试临时目录使用工作区 tmp（系统沙箱临时目录曾阻止 PyTorch 写入测试检查点）；未修改旧测试。两条启动命令的 `--check` 均通过。创建文件前后的 241 份已有 Python/JSON 文件 SHA256 全部一致；正式实验输出根目录尚不存在，未启动正式训练。

真实 20 工作流生成预检已跑通 seed43 和44；seed44 的新旧参考比值平均1.0861，范围1.0051—1.3368。这只是输入生成检查，不是训练性能结果。seed44 的整批生成约12秒（包括资源构建）；B 每轮多一次参考仿真，预计比 A 慢，不承诺明早一定训练完成。
