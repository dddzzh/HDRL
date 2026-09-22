# old/ — 归档,不参与任何当前实验

这里的东西**没有任何现役代码 import**,删掉也不影响训练和评估。

| 目录 / 文件 | 是什么 | 为什么归档 |
|---|---|---|
| `legacy_D3QN_for_Fog_Computing_1.6/` | 上一篇论文的完整旧代码库(`Training_R.py` / `Training_S.py` / `evaluate_*.py` 等约 40 个脚本) | 已被 `src/` + `tools/` 重写取代。全仓库只有两处**注释**提到它(`src/datasets/pegasus_workflow_generator.py:22`、`src/simulator/metrics.py:81`),用来说明数据集采样规则和 makespan 定义的出处,不是代码依赖 |
| `tools/train_reliability_aware_high.py` | 单头可靠性 D3QN 的数值诊断脚本,自己造假 transition 验证 Q 头和拉格朗日行为 | 零引用。功能已并入 `train_coupled_hierarchical_d3qn_sac.py` 的真实训练链路 |

---

## `tools/` 里剩下的 4 个 `train_*hierarchical*` 分别是什么

**别再移它们了** —— 除入口外全是当前入口的依赖,移动会直接 ImportError。

```
train_coupled_hierarchical_d3qn_sac.py   ← 唯一入口。你跑的就是这个
        │
        ├── train_reliability_aware_hierarchical_d3qn.py   三头(能耗/时限/可靠性)agent 定义
        │           └── train_hierarchical_d3qn.py         训练主循环 + CLI 参数 + 对偶更新调用
        │                       └── hierarchical_common.py 环境/资源/工作流构建、日志打印
        └── (同时也直接 import 上面两个)
```

四个文件都同时是「库」和「可独立运行的入口」,所以看起来重复。判断标准很简单:
**README 和你的命令行里出现的只有 `train_coupled_hierarchical_d3qn_sac.py`,其余三个只当库用。**

## 对比算法(都在用,不是旧的)

`train_dqn_edge.py` / `train_dream.py` / `train_oo_actor_critic.py` / `train_hirl.py`
—— 四个都被 `tools/eval_all_baselines.py` import,README 里也都有训练命令。
