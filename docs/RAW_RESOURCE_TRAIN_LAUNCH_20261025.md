# Raw资源规模补齐：只提供指令，未启动训练

用户决定沿用原来的FCFS测试协议，采用Raw-Self，不再以反事实归因为贡献。本次新增37/111台配置，其他生产代码及既有实验不改。

## 训练配置

- 基于已验证的 `configs/ablation_sm_74_raw_self_legacy_train.json`，仅改资源数、固定槽位数和输出/phase名称。
- `configs/raw_self_ss_37_legacy_train.json`：28 edge+9 cloud，max_servers=37。
- `configs/raw_self_sl_111_legacy_train.json`：84 edge+27 cloud，max_servers=111。
- 共用原Raw入口 `tools/train_deadline_credit_ablation.py`。不得替换为普通coupled入口，否则顶层Raw配置不会按该入口的hook执行。
- 1200轮、训练seed42、20个Small工作流/轮、0.1/s、Medium期限、legacy单工作流HEFT生成、低层budget0.08；上下层从头联合训练，Deadline Head和可靠性控制保留。
- 每20轮沿用542000/642000双验证，各20seed，先ESP与可靠性可行，再选最低验证能耗；不要按最终测试选择checkpoint。
- 配置中的`low_dual_attribution=workflow_counterfactual`保留旧采集接口名；本专用入口用`deadline_credit_ablation.variant=raw_self`替换实际训练成本，不能单凭旧接口字符串认定仍按Full训练。

VS Code终端工作目录为项目根，Python解释器选择 `D:/Miniconda/envs/py310/python.exe`。可在两个终端各运行一条，不要重复启动同一配置：

```powershell
python -u tools/train_deadline_credit_ablation.py --config configs/raw_self_ss_37_legacy_train.json --episodes 1200 --seed 42
```

```powershell
python -u tools/train_deadline_credit_ablation.py --config configs/raw_self_sl_111_legacy_train.json --episodes 1200 --seed 42
```

结果分别在 `results/raw_self_resource_models_20261025/ss_37/` 和 `sl_111/` 下新建带时间/PID的run目录，不覆盖74台或历史Full。训练后还需另做FCFS期限的九档测试；训练完成不等于泛化测试达标。

检查：`tests.test_raw_resource_configs`通过；两个命令均加`--check`执行通过、training_started=False。未运行真实训练。

## 旧FCFS训练记录的核实

2026-10-25读取实际`latest_checkpoint.pt`内config和相邻train_history.jsonl（不是只查看当前配置文件）：

|资源|旧Full run|记录轮数|checkpoint生成协议|
|---|---|---:|---|
|37|results/v3_ss_37_sec_job_fcfs/20261008_101232|1200|sec_job_fcfs_v1，50wf，0.03/s|
|74|results/v3_sm_74_sec_job_fcfs/20261008_101252|1200|sec_job_fcfs_v1，50wf，0.03/s|
|111|results/v3_sl_111_sec_job_fcfs/20261008_101305|1120|sec_job_fcfs_v1，50wf，0.03/s|

这些run确有FCFS期限训练，但没有Raw标记，不是这次需要补齐的Raw模型，也不是后来legacy_train_new_eval主表所选的legacy-HEFT训练模型。表中轮数仅指列明目录现存记录，不推断其他归档目录或恢复run是否有后续轮次。
