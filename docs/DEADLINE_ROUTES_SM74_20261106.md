# SM74：两条下层Deadline路线（2026-11-06）

这是独立训练试验，不覆盖旧配置、checkpoint、评估或论文。代码写好后只推到Git，由用户在服务器拉取并启动；本次不连接SSH、不自动训练。

## 两路定义

| 项目 | A：固定时间成本 | B：自适应Deadline惩罚 |
|---|---|---|
| 配置 | `configs/deadline_route_a_slack_sm74_sac_affine_20261106.json` | `configs/deadline_route_b_dual_sm74_sac_affine_20261106.json` |
| `agent.deadline_route` / `agent.low.deadline_route` | `a_slack` | `b_dual` |
| 下层 `slack_time_weight` | 1，保持原固定时间成本 | 0，不再使用固定时间成本 |
| 下层Deadline乘子 | 固定0 | 按滚动批次失败率自适应 |
| 下层Deadline更新 | 初值、I步长和P增益均0；控制器仍enabled | 初值1；I步长0.2；P增益3；上限8；warmup20轮 |
| 下层Deadline观察量 | 不用于改变乘子 | 每轮一个 `1{本轮按时工作流比例<0.90}` |
| 下层Deadline窗口/预算 | 保留旧配置但不更新数值 | 最近60轮的批次失败率均值；预算0.10 |
| 上层Deadline | 原PI控制器与Q_D成本不变 | 同左 |
| 上层可靠性 | 双向工作流不达标率控制 | 同左 |
| 下层可靠性 | 双向工作流不达标率控制＋稠密风险消耗reward成本 | 同左 |

A不设置 `lagrangian.enabled=false`：那会走旧固定Deadline惩罚回退，无法实现“只有slack_time、没有下层Deadline惩罚”。固定0时需要I和P同时为0。

B从连续增量迟到观察量切换到二元批次失败信号，因此不沿用原P增益25。失败率0.4、预算0.1时，P=25单独产生7.5的比例项，叠加积分项后容易碰到上限8。P=3是本次显式设置，不是已验证的最优值，也不保证乘子不饱和或性能改善。

这两路同时改变“固定时间成本”和“下层Deadline控制方式”，用于比较两种时间约束处理路线，**不是仅删除自适应机制的严格单因素消融**。若之后检验自适应本身，需要在选定的同一路线、同一基础reward和成本下，另训练匹配的固定乘子版。

## 两路共用的可靠性修改

- 上层 `reliability_dual_mode=two_sided_miss_rate`，每轮按最近100个工作流的二元可靠性不达标比例更新：步长0.5、预算0.005、初值1、上限100。
- 下层可靠性控制器保持启用，使用最终工作流二元可靠性不达标比例：窗口100个工作流、步长0.5、预算0.005、初值1、上限50。
- 上层Q_R仍学习终局正log缺口，动作组合方式不变。
- 下层 `reliability_cost_mode=dense_budget_share_v1`，回放成本使用已有的 `reliability_budget_share_spent`，而不是投影终局违反增量。
- `physical_low_reward=false`保持不变。只替换可靠性回放成本；原能耗/时间基础reward的z-score归一化不切换成PhysicalDVFS版本。

这里的稠密成本衡量“相对全频额外消耗的风险预算”，并非二元失败率。上下层的策略成本与乘子观察量仍使用不同代理，因此应称为自适应软惩罚控制，不宣称同一约束函数的标准原始—对偶算法、强对偶或逐工作流硬可行性保证。可靠性长期满足时乘子允许下降至0，不人为设置正下限。

## 保持不变的训练口径

- 74台：56 edge＋18 cloud；资源配置 `configs/resources_edge_cloud_dvfs_v3.json`。
- 原 `cai_2025_heft` 工作流生成与参考期限，Small、20工作流/轮、Medium、到达率0.1/s。
- **训练和原双块验证的工作流可靠性要求仍为0.90**；这不是0.95训练，`default_reliability=0.95`也不等于工作流约束门槛。之后若评估要求0.95，需真实写入评估环境并另行标注。
- 1200轮、seed42；双验证块542000和642000，每块20个种子，验证/保存间隔20轮，原选模规则不变。
- 上层三头、下层双critic，`server_affine` SAC动作与Jacobian修正保持不变。
- 原能效动作下界、故障率、物理模型、低层归因reward定义均不因路线选择而改动；B只更换下层Deadline乘子的观察量。

## 服务器启动指令

在服务器项目根目录、正确Python环境下，从头训练两份独立模型：

```bash
python -u tools/train_deadline_routes_d3qn_sac.py --config configs/deadline_route_a_slack_sm74_sac_affine_20261106.json --episodes 1200 --seed 42
```

```bash
python -u tools/train_deadline_routes_d3qn_sac.py --config configs/deadline_route_b_dual_sm74_sac_affine_20261106.json --episodes 1200 --seed 42
```

输出分别位于：

```text
results/deadline_routes_sm74_sac_affine_20261106/a_slack/<run>/
results/deadline_routes_sm74_sac_affine_20261106/b_dual/<run>/
```

不添加旧Full的 `--resume-checkpoint` 或 `--high-checkpoint`，不从旧模型暖启动。两路可以分开启动，但并发与GPU选择由用户在服务器安排；代码默认device=auto，不修改其他进程的GPU设置。

## 验收重点

1. A的下层Deadline乘子及实际Deadline惩罚始终为0，固定时间成本仍生效。
2. B的下层乘子对真实批次失败有响应；达标后允许下降，不能把“长期非零”当作验收要求。
3. 两路均检查实际 `乘子×回放成本`，不能只看乘子曲线；记录真实Deadline、可靠性、rho与能耗J。
4. 新旧模式、路线与可靠性成本状态不能静默混载。需要断点续跑时，只使用同一路线、相同配置的本次checkpoint并核验身份。
5. 两路均保留所有失败结果，不筛选测试种子，不承诺B一定优于A。

自检结果和Git提交身份以本次交付聊天为准；本文件不是训练完成声明。
