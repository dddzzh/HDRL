# Alibaba跨数据来源泛化：待执行实验卡

状态：方案与本地数据核查，尚未启动新训练或评估。2026-10-15。

## Research Question Card

- Question：在Pegasus上训练并选定的调度策略，冻结学习参数后，能否在Alibaba批处理DAG派生负载中保持能耗优势及截止期/可靠性约束满足能力？
- Type：确认性跨来源评估；接入和数值检查是独立工程预检，不用预检表现选择模型/种子。
- Hypothesis：Full在新的任务依赖拓扑和相对计算强度分布下仍有竞争力。当前没有Alibaba新协议结果，不能先声称泛化成功。
- Current evidence：本地原始batch_task及已划分JSON存在，已有转换、加载、工作量映射代码。具体证据见下。
- Missing evidence：模型训练数据来源溯源；train/test job去重；DAG合法性；任务数及ready队列截断检查；与当前sec_job deadline/arrival协议的适配；冻结映射参数与抽样清单；实际配对测试结果。
- What would support it：预先冻结的同一测试样本上，完整报告能耗与约束指标，Full仍保持优势/较好的约束权衡，不能只取成功样本。
- What would falsify/weaken it：约束满足率明显下降、能耗优势消失，或者结果依赖重新训练/按测试结果换模型或种子。均保留并限定结论。
- Minimal next action：先做只读数据审计和接口测试，冻结SM74四法的模型来源、测试清单及全部映射/约束参数，再经用户执行授权启动独立目录评估。
- Decision：建议做；暂不自动启动。

## 建议最小规模

固定74台（56 edge+18 cloud），使用现有SM74四算法训练模型，不在Alibaba重新训练或更新梯度，不根据阿里测试选checkpoint。每seed50个真实DAG，Loose/Medium/Tight各20个预先固定seed，四算法共享完全相同的DAG、任务物理量、到达与约束。总计4×3×20=240条模型-seed记录；20个测试seed不是20个独立训练seed。

初始化seed、遍历顺序、评估状态处理显式记录。DQN-Edge保留既有跨seed EMA行为，不并行独立加载拼行；因此“无梯度评估”不等同于其内部EMA完全静止，不能笼统宣称所有方法全状态冻结。其余方法也按各自原协议处理。

先用未按结果筛选的测试清单；保留全部失败，不把此前success-selected实验当本实验抽样依据。指标为总能耗J均值±样本SD、ESP@90、可靠性达标率；不以平均DSR替代ESP@90。若以后扩展37/111台，另作资源维度实验，不必一开始再建九宫格。

## Evidence Records

### ER-20261015-alibaba-local-01

- Source type：dataset / code artifact。
- Source：`dataset/cluster-trace-v2018/batch_task.csv`（802261444 bytes）；`outputs/alibaba_test.json`（10000 workflows）；`outputs/alibaba_train.json`（40000 workflows）。
- Observed：test任务数min/中位/max=5/7/131，train=5/7/133；JSON未携带到达时间或deadline，task含duration/plan_cpu/instance_num/raw_weight。仅做字段和数量核查，未验证完整来源哈希及去重。
- Code：`tools/build_alibaba_workflows.py`、`tools/split_alibaba_dataset.py`、`src/datasets/alibaba_dataset.py`、`src/datasets/workload_mapping.py`、`src/datasets/workflow_factory.py`。
- Limitations：当前mapper用trace派生相对强度映射cycles，通信量为合成值；不能称为阿里生产能耗实测。现有factory给自定义DAG生成的是HEFT参考deadline，不能直接冒充当前Pegasus的sec_job_fcfs_v1协议。mapper当前在加载/截取后的workflow上拟合分位数，正式测试应先冻结独立校准池的统计量，不能按每个测试批次重拟合。
- Claim strength：observed（具备接入基础，不代表完成接入或已有效果）。

### ER-20261015-alibaba-official-02

- Source type：dataset documentation。
- Source：[官方v2018说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)、[官方schema](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt)。
- Supports：v2018含生产批任务DAG信息，依赖由task_name解析；任务有多实例，依赖语义是前驱全部实例完成。聚合实例为一个仿真任务时必须说明近似。
- Does not support：本地转换绝对正确、真实周期数/链路流量/截止期/可靠性/能耗直接来自trace、云边真实部署已验证。

## 允许与禁止结论

允许（实际完成后按结果限定）：基于Alibaba真实工作流拓扑和相对负载特征构造的trace-driven仿真中的跨来源泛化。

禁止：尚未测试就称跨数据集泛化成功；称为阿里线上部署/实测节能；把合成的通信、deadline和可靠性要求称为trace原始测量。没有在Alibaba重训才可称无重训迁移；源模型若用过该测试池则不能称未见数据。
