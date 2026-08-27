# A1/A3/A4 消融实验

## 实验定义

- **A1**：随机上层分配 + 固定已训练 SAC。验证学习型上层分配是否必要。
- **A3**：单头 DQN + SAC。验证双头结构相对于普通单头 DQN 的贡献。
- **A4**：双头 D3QN + SAC。完整方法。

三个变体必须使用同一组 20 个评估 seeds、同一资源配置、同一工作流配置和同一 deadline 生成方式。

## 指标

- `total_energy`：主要优化目标；
- `deadline_success_rate`：约束指标；
- `episode_makespan`：只作辅助分析，不作为主结论；
- `rho_mean`：分析下层 DVFS 行为。

## 启动方式

```powershell
& 'D:\Miniconda\envs\py310\python.exe' tools/run_ablation_a1_a3_a4.py `
  --a3-checkpoint <单头DQN checkpoint> `
  --a4-checkpoint results/hierarchical_dual_head_d3qn_sac_joint/20260825_113856/best_d3qn_checkpoint.pt
```

## 结论判定

- A1 → A3：学习型上层是否降低能耗并保持 deadline 约束；
- A3 → A4：双头结构是否进一步降低能耗或提高 deadline 成功率；
- A4 相对 A1：完整方法的总体收益。

只报告均值 ± 标准差，并使用配对 seeds 进行比较。
