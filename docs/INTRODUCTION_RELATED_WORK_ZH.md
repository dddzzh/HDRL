# 引言与相关工作（中文初稿）

> 写作说明：本文采用作者—年份式叙述和编号式引文占位，便于后续转换为 LaTeX。文中没有写入尚未稳定的实验提升百分比；待最终 20-seed 结果冻结后，应在引言末段补入一组核心结果。

## 1 引言

边缘计算将计算能力部署到数据源附近，可以缩短交互路径并缓解核心网络压力；云端则提供更充足且弹性的计算资源。对于由多个相互依赖任务构成的科学计算、工业分析和智能物联网应用，将工作流表示为有向无环图（directed acyclic graph, DAG），并在异构边缘—云资源之间联合决定任务执行位置与处理速度，是获得低时延和低能耗服务的关键。经典 HEFT 算法通过上行秩确定任务优先级，并依据最早完成时间选择异构处理器，为 DAG 调度建立了具有代表性的列表调度范式 [1]。然而，实际边缘—云平台并不是一次性接收一个静态工作流：多个工作流会随机到达，节点计算能力、通信开销和队列状态持续变化，当前任务的放置还会改变后续任务可见的拥塞状态。因此，静态排序或一次性搜索得到的计划难以直接适应动态多工作流场景。

在这一场景中，节能调度还受到两个相互耦合的服务质量约束。其一，降低处理器频率可以减少动态功耗，却会延长任务执行时间，压缩工作流剩余关键路径上的时延裕量，并通过共享服务器队列将延迟传播给其他工作流。其二，较低电压和频率可能提高瞬态故障率，使节能收益与执行可靠性形成冲突 [19–22]。调度器因而需要同时回答两个不同类型的问题：将就绪任务放置到哪一台边缘或云服务器，以及在该服务器上采用何种 DVFS 频率比例。前者是随资源规模增长的离散组合决策，后者是连续控制决策；二者共同影响能耗、Deadline 达标和端到端可靠性。若将服务器编号和频率直接展开成单一动作空间，不仅会放大搜索空间，还会掩盖“放置造成的排队延迟”和“降频造成的执行拉伸”之间不同的责任边界。

近年来，深度强化学习被用于动态工作流调度。已有方法能够从运行状态中学习任务分配策略，并围绕完成时间、资源利用率、成本或能耗构造奖励 [4–7,25]。其中，Jayanetti 等将边缘/云层级和具体节点选择组织为分层动作，并以集成时延与能耗的奖励训练策略 [4]；Chen 等使用 D3QN 处理异构多工作流的多目标协同调度 [5]；Pan 和 Wei 使用正则化 DQN 实现动态多工作流在线分配 [6]；Wen 等提出 DQN-Edge，以独立的时间和能耗价值网络及上下文注意力完成动态边缘工作流调度 [7]。对于联合服务器与资源等级选择，Wang 和 Sun 将服务器编号与虚拟机等级编码为二进制动作，并以序数优化从 actor 邻域生成候选、由 critic 粗排后精确选择，以降低大离散动作空间的搜索代价 [25]。这些研究说明 DRL 具有适应动态环境的潜力，但代表性方法大多采用固定权重或注意力加权的标量奖励，Deadline 更多体现为奖励项、命中率指标或尽力满足的目标。它们通常不同时建模工作流级可靠性约束与连续 DVFS 控制，也没有显式区分放置层和频率层对工作流超期的因果责任。

本文研究动态异构边缘—云环境中的 Deadline 与可靠性约束计算能耗最小化问题，并提出一种约束感知的分层强化学习方法。上层使用三头 Dueling Double DQN [9,10] 选择任务的主执行服务器，三个价值头分别估计能耗收益、Deadline 成本和可靠性成本；动作选择时，利用在线更新的拉格朗日乘子将三个价值量组合，而不是在训练前固定多目标权重。下层使用连续动作 Soft Actor-Critic（SAC）[11] 为已选服务器输出 DVFS 频率比例 \(\rho\)。这种分解使离散放置与连续频率分别由适合其动作结构的学习器处理，同时保留二者通过环境状态、回报和约束反馈形成的耦合。

动态工作流中的约束信用分配不能只依赖任务完成时的局部惩罚。上层放置会改变服务器队列，而下层降频造成的执行拉伸也会经由队列先后关系影响其他任务。为此，本文在上层引入队列拥塞信用和剩余关键路径安全状态，使策略能够感知当前放置对未来可行性的影响；在下层引入工作流级 DVFS 反事实责任归因：保持服务器放置和服务器内任务顺序不变，将来源工作流的频率恢复为 \(\rho=1\)，重新传播 DAG 依赖与服务器先后约束，据此估计降频对本工作流及受影响工作流 Deadline 成本的贡献。上层和下层分别维护 Deadline 拉格朗日变量，使两个决策层按各自的作用范围调节约束压力；下层还维护可靠性拉格朗日变量，以响应频率相关的可靠性风险。

本文的主要贡献如下：

1. **面向混合决策结构的约束分层策略。** 本文将服务器放置建模为三头 Dueling Double DQN，将 DVFS 建模为连续动作 SAC，并以拉格朗日组合值连接能耗目标、Deadline 成本和可靠性成本，从而避免把异构的离散—连续决策粗粒度地展开为单一动作。
2. **面向不同决策层的约束调节。** 本文为上下层设置独立的 Deadline 对偶变量，并在下层设置可靠性对偶变量，使放置与降频能够依据各自观察到的约束成本调节策略，而不依赖预先固定的惩罚系数。
3. **面向动态队列的长程状态与责任归因。** 本文结合队列拥塞信用、剩余关键路径安全状态和工作流级 DVFS 反事实回放，将终端超期信号分解为更符合调度因果链的训练反馈。
4. **约束优先的模型选择。** 本文使用 ESP@90 与可靠性达标率筛选 checkpoint，并仅在满足验证门槛的候选模型之间比较能耗，以减少仅按平均奖励或能耗选模造成的约束偏移。

需要强调的是，本文采用经验约束、拉格朗日对偶更新和验证集门控来提高策略的约束达标能力；这些机制不等同于对每个工作流给出形式化的硬实时或逐样本可靠性证明。最终稿应在此处补充冻结后的主要实验结论，例如：“在 Small 和 Medium 场景的 20 个独立种子上，本文方法在达到预设 ESP@90 与可靠性门槛时，相比最强可行基线降低了 XX% 和 XX% 的计算能耗。”

## 2 相关工作

### 2.1 动态异构环境中的工作流调度

工作流调度通常将任务依赖建模为 DAG。HEFT 以平均计算和通信代价构造上行秩，并把任务分配给具有最早完成时间的处理器 [1]；该方法计算开销较低，但面向给定工作流和已知资源状态生成静态计划。针对工作流随机到达，Arabnejad 等提出 Dynamic Workflow Scheduler，在商业云中为具有 Deadline 的多个非周期工作流动态配置资源并降低成本 [2]。Ismayilov 和 Topcuoglu 将资源故障与目标数量变化建模为动态多目标优化问题，并利用神经网络预测环境变化后的 Pareto 解集 [3]。这些方法扩展了动态性建模，但在线重规划或群体搜索的计算代价会随工作流与资源规模增长，并且其主要目标是成本、完成时间或多目标 Pareto 优化。

DRL 为在线调度提供了另一条路径。Jayanetti 等面向边缘—云工作流提出多 actor、共享 critic 的混合策略，先区分边缘/云层级，再选择节点，以集成奖励平衡能耗和执行时间 [4]。Chen 等采用 D3QN、复合奖励和自适应时间步处理异构多工作流的 makespan、成本、公平性和连续性 [5]。Pan 和 Wei 提出 R-DQN，为动态云中的多个实时工作流在线选择虚拟机，主要优化 makespan 与资源利用率 [6]。Wen 等提出 DQN-Edge，先以改进上行秩确定任务顺序，再用时间和能耗两个 Q 网络及上下文注意力选择边缘虚拟机 [7]。这些方法已经覆盖动态到达、任务依赖和资源异构等要素，但 Deadline 往往通过奖励塑形或成功率评估间接处理，可靠性和连续 DVFS 通常未进入同一约束决策模型。

Wang 和 Sun 面向动态无线边缘计算中的独立任务请求，使用 actor–critic 与序数优化联合选择服务器和虚拟机等级 [25]。该方法将联合动作编码为二进制串，从 actor 输出附近生成 100 个候选，以 critic 选出前 10 个候选，再依据时延—能耗目标作精确选择。它为高维联合动作提供了高效搜索机制，但原问题不是 DAG 工作流，任务 Deadline 主要通过奖励惩罚和动作可行性条件处理，也不包含工作流级可靠性约束。因此，本文将其视为“联合放置—资源等级搜索”基线，而不是与本文问题设定完全一致的工作流基线。

本文与上述工作的联系在于同样利用价值学习适应动态队列；区别在于，本文把“最小化能耗且满足 Deadline 与可靠性要求”写成显式的经验约束优化问题，并让能耗、Deadline 和可靠性分别拥有价值头或成本通道。本文还同时学习服务器放置和连续 DVFS，而不是固定频率下只学习任务—服务器映射。

### 2.2 约束强化学习与风险敏感学习

约束马尔可夫决策过程（constrained Markov decision process, CMDP）通过在累计回报之外引入一个或多个累计成本预算，为服务质量和安全要求提供了标准建模框架。CPO 在信赖域策略优化中直接处理成本约束，并给出接近逐次迭代约束满足的理论界 [12]；RCPO 使用多时间尺度拉格朗日更新，以学习到的惩罚系数引导策略趋向可行解 [13]。Stooke 等指出，普通积分式拉格朗日更新容易产生超调和振荡，并以 PID 形式改善约束响应 [14]。对于尾部风险，Chow 和 Ghavamzadeh 将条件风险价值（CVaR）引入 MDP 优化，使策略能够关注小概率高损失结果而不仅是平均成本 [15]。

工作流调度中的 Deadline 和可靠性不是普通的加权偏好：当达标率低于服务要求时，进一步降低能耗不应抵消约束失败。固定惩罚系数需要针对负载和资源规模反复调节，也可能因不同目标的数值尺度而改变实际权衡。本文借鉴 CMDP 的拉格朗日思想，由观测到的 Deadline 与可靠性成本自适应更新乘子；下层 Deadline 控制还使用尾部统计量，使最差一部分工作流的超期风险能够影响频率策略。与 CPO、Lyapunov 安全策略或逐状态硬约束方法不同，本文没有声称训练期间或每次调度决策都严格可行，而是通过滚动统计、对偶控制和验证门控实现经验约束满足。这一边界应在实验和结论中保持一致。

### 2.3 分层强化学习与离散—连续混合动作

分层强化学习通过在不同抽象层上组织决策来降低长时程信用分配难度。Options 框架把持续多个时间步的闭环策略纳入半马尔可夫决策过程 [16]；h-DQN [23] 和 FeUdal Networks [24] 则分别通过高层目标与低层动作、Manager 与 Worker 的分工学习跨时间尺度策略。另一类工作直接研究参数化动作空间，即先选择离散动作类型，再决定该动作对应的连续参数。Hausknecht 和 Stone 将深度强化学习扩展到离散动作带连续参数的场景 [17]，P-DQN 进一步以 DQN 与连续动作网络共同处理混合动作 [18]。

服务器放置与 DVFS 也构成离散—连续混合动作，但其层次具有明确的物理语义：服务器选择决定排队位置、通信路径和可用处理器，频率比例决定已选服务器上的执行时间、动态能耗和频率相关故障风险。Jayanetti 等的层级动作主要在边缘/云层级和具体节点之间分解离散选择 [4]；本文则在服务器放置和连续频率控制之间分层，并允许两层使用不同算法、状态和约束变量。该设计同时缩小动作搜索难度，并为上层放置成本与下层 DVFS 成本建立可解释的信用边界。

从动作结构看，序数优化 actor–critic 将服务器与虚拟机等级压缩为一个二进制编码的离散联合动作，并通过候选生成和排序缓解组合搜索 [25]。这种平坦联合动作仍把放置与资源等级交给同一策略处理。本文则保留两类决策的语义边界：上层在服务器集合上作离散选择，下层在已选服务器的可行区间内输出连续频率比例，并分别接收与其责任范围相对应的约束反馈。

### 2.4 DVFS 能耗优化

DVFS 利用处理器动态功耗与电压、频率之间的关系，在时延裕量允许时降低运行频率以节省能耗。对工作流而言，频率调整必须服从任务依赖和端到端完成时间，而不能只根据单个任务的局部 slack。Wu 等提出软错误感知的三阶段调度方法，以 HEFT 获得初始计划，再通过任务重映射和可靠性感知 DVFS 降低能耗 [20]。Hassan 等提出 SERAS，将工作流 Deadline 分配到任务，并结合 DVFS、checkpoint 与回滚恢复兼顾能耗、完成时间和可靠性 [21]。Wu 等进一步研究实时工作流的能耗—Deadline—可靠性三指标调度，通过任务副本与分层映射满足可靠性阈值，并为各副本选择频率 [22]。Esnaashari 等提出 DReaM，用 DQN 在多核 IoT 设备上为实时任务联合选择处理器核、七档电压—频率等级以及 P/B、TMR 或 NMR 容错方式，在满足任务时限和目标可靠性的同时降低功耗 [26]。

上述方法清楚揭示了 DVFS 的核心耦合：降频可以节能，但会同时消耗时间裕量并增加可靠性风险。不过，它们主要针对离线已知工作流、周期性实例或设备级独立实时任务，并以规则、启发式搜索或有限频率档位作决策。当多个 DAG 工作流随机到达并共享服务器队列时，局部 slack 不能完整反映降频的外溢影响。本文使用连续动作 SAC 在线选择 \(\rho\)，并以剩余关键路径安全状态和固定放置/顺序下的工作流级反事实传播估计 DVFS 责任，从而把频率控制置于动态队列上下文中。

### 2.5 可靠性感知工作流调度

可靠性感知调度常采用可靠服务器选择、任务复制、checkpoint/rollback 或可靠性预算分解。Zhang 等将能耗与系统可靠性作为冲突目标，使用带 DVFS 的双目标遗传算法搜索 Pareto 解 [19]。Wu 等显式建模低电压/频率导致的软错误，并在完成时间和可靠性约束下分配虚拟机与频率 [20]。SERAS 使用 checkpoint 与回滚恢复抵消降频带来的故障风险 [21]。面向实时工作流，Wu 等通过为任务配置多个副本达到可靠性阈值，再联合优化副本频率与映射 [22]。DReaM 则将 P/B、TMR 和 NMR 作为 DQN 联合动作的一部分，使容错方式与处理器核及频率档位共同决定任务可靠性和功耗 [26]。这些研究表明，可靠性不能被简单等同于节点可用率；频率、执行时长、复制数量和恢复机制都会改变应用成功概率。

本文正式配置采用 primary-only 动作，不启用备份副本。因而，本文的可靠性机制不依赖以额外计算和能耗换取成功概率的复制，而是在服务器放置与 DVFS 频率选择过程中显式估计并约束端到端可靠性成本。这一设定使能耗—可靠性权衡更直接，但也意味着策略无法通过冗余执行修复单次任务失败；其适用范围和与复制型方法的公平比较应在实验设置中明确说明。

### 2.6 本文方法的定位

| 研究方向/代表方法 | 动态多工作流 | 异构边—云放置 | 连续 DVFS | 显式 Deadline 约束 | 显式可靠性约束 | 主要信用机制 |
|---|---:|---:|---:|---:|---:|---|
| HEFT [1] | 否 | 可扩展 | 否 | 否 | 否 | 上行秩与最早完成时间 |
| DWS [2] | 是 | 以云为主 | 否 | 是（启发式） | 否 | Deadline 分解与动态重排 |
| Jayanetti 等 [4] | 是/仿真动态 | 是 | 否 | 奖励/尽力满足 | 否 | 集成时延—能耗奖励 |
| HCDRL [5] | 是 | 以云为主 | 否 | 非核心约束 | 非核心约束 | 复合多目标奖励 |
| R-DQN [6] | 是 | 以云为主 | 否 | 实时目标 | 否 | DQN 即时回报 |
| DQN-Edge [7] | 是 | 边缘 | 否 | 成功率/奖励 | 否 | 双 Q 网络与注意力融合 |
| OO actor–critic [25] | 独立任务动态到达 | 是 | 离散资源等级 | 奖励/可行性条件 | 否 | 二进制联合动作、候选粗排与精确选择 |
| DReaM [26] | 设备级实时任务 | 否（多核设备） | 七档离散 DVFS | 任务时限惩罚 | 是 | 核—频率—容错方式的平坦联合 DQN 动作 |
| 可靠性 DVFS 启发式 [20–22] | 通常否或周期实例 | 云/同构平台为主 | 离散档位 | 是 | 是 | slack、复制、checkpoint 或可靠性分解 |
| **本文方法** | **是** | **是** | **是** | **上下层独立拉格朗日约束** | **价值成本与下层拉格朗日约束** | **队列信用、关键路径安全状态与工作流级反事实归因** |

该对比表用于说明问题覆盖范围，不表示本文已经获得比所有类别方法更强的理论保证。本文的创新重点是：在动态异构边缘—云队列中，把离散放置、连续 DVFS、Deadline 与可靠性经验约束以及工作流级责任归因统一到同一个可训练框架中。

## 3 三个主要对比算法的落位建议（写作备注）

- **DQN-Edge [7]：引言、相关工作和实验设置都应出现。** 它与本文的问题最接近，均处理动态边缘环境中的 DAG 工作流、任务排序/放置和能耗—时延权衡。引言中用一到两句说明它是最接近的直接先行工作；相关工作 2.1 详细比较其双 Q 网络与注意力加权和本文显式约束三头价值分解；实验中说明保留的论文组件、统一模拟器后的改动以及公平性设置。
- **OO actor–critic [25]：主要放在相关工作 2.1/2.3 和实验设置。** 它对应“服务器—资源等级联合动作及高维动作搜索”，能够直接对比本文的分层动作设计。引言只需与其他动态 DRL 方法合并提及，无需展开算法流程。实验中应称为 **adapted OOAC**，明确原文处理独立任务请求和离散 VM 等级，而本实现把等级适配为离散 \(\rho\) 并接入共同的工作流状态、目标与约束评估。
- **DReaM [26]：主要放在相关工作 2.4/2.5 和实验设置，不宜把它写成工作流调度先行工作。** 原文面向单个多核 IoT 设备的实时任务，联合选择核心、七档 DVFS 和 P/B、TMR、NMR 容错方式；它适合支撑“DVFS—可靠性联合决策”的比较。实验中应称为 **adapted DReaM-DQN**，明确当前正式版本将核心扩展为边缘/云服务器，并关闭原文的容错执行分支，因此比较的是其平坦离散“服务器×频率”DQN 思路，而不是对原系统的严格复现。

三者的实现性改动不要塞入引言或相关工作正文。正式英文稿应在 `Experimental Setup/Baselines` 中为每个算法单列“original decision model / retained components / adaptations for the common simulator”，否则审稿人容易把跨问题适配误认为原论文设置。

## 参考文献（已核验元数据）

[1] H. Topcuoglu, S. Hariri, and M.-Y. Wu, “Performance-effective and low-complexity task scheduling for heterogeneous computing,” *IEEE Transactions on Parallel and Distributed Systems*, 13(3):260–274, 2002. DOI: [10.1109/71.993206](https://doi.org/10.1109/71.993206).

[2] V. Arabnejad, K. Bubendorfer, and B. C. K. Ng, “Dynamic multi-workflow scheduling: A deadline and cost-aware approach for commercial clouds,” *Future Generation Computer Systems*, 100:98–108, 2019. DOI: [10.1016/j.future.2019.04.029](https://doi.org/10.1016/j.future.2019.04.029).

[3] G. Ismayilov and H. R. Topcuoglu, “Neural network based multi-objective evolutionary algorithm for dynamic workflow scheduling in cloud computing,” *Future Generation Computer Systems*, 102:307–322, 2020. DOI: [10.1016/j.future.2019.08.012](https://doi.org/10.1016/j.future.2019.08.012).

[4] A. Jayanetti, S. Halgamuge, and R. Buyya, “Deep reinforcement learning for energy and time optimized scheduling of precedence-constrained tasks in edge–cloud computing environments,” *Future Generation Computer Systems*, 137:14–30, 2022. DOI: [10.1016/j.future.2022.06.012](https://doi.org/10.1016/j.future.2022.06.012).

[5] G. Chen, J. Qi, Y. Sun, X. Hu, Z. Dong, and Y. Sun, “A collaborative scheduling method for cloud computing heterogeneous workflows based on deep reinforcement learning,” *Future Generation Computer Systems*, 141:284–297, 2023. DOI: [10.1016/j.future.2022.11.032](https://doi.org/10.1016/j.future.2022.11.032).

[6] J. Pan and Y. Wei, “A deep reinforcement learning-based scheduling framework for real-time workflows in the cloud environment,” *Expert Systems with Applications*, 255:124845, 2024. DOI: [10.1016/j.eswa.2024.124845](https://doi.org/10.1016/j.eswa.2024.124845).

[7] M. Wen, X. Liu, X. Ning, C. Liu, X. Chen, J. Nian, and L. Cheng, “Deep reinforcement learning for energy-efficient workflow scheduling in edge computing,” *Computer Networks*, 274:111790, 2026. DOI: [10.1016/j.comnet.2025.111790](https://doi.org/10.1016/j.comnet.2025.111790).

[8] V. Mnih et al., “Human-level control through deep reinforcement learning,” *Nature*, 518:529–533, 2015. DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236).

[9] H. van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with Double Q-learning,” *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1), 2016. DOI: [10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295).

[10] Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, and N. de Freitas, “Dueling network architectures for deep reinforcement learning,” *Proceedings of the 33rd International Conference on Machine Learning*, PMLR 48:1995–2003, 2016. [PMLR](https://proceedings.mlr.press/v48/wangf16.html).

[11] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80:1861–1870, 2018. [PMLR](https://proceedings.mlr.press/v80/haarnoja18b.html).

[12] J. Achiam, D. Held, A. Tamar, and P. Abbeel, “Constrained policy optimization,” *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70:22–31, 2017. [PMLR](https://proceedings.mlr.press/v70/achiam17a.html).

[13] C. Tessler, D. J. Mankowitz, and S. Mannor, “Reward constrained policy optimization,” *International Conference on Learning Representations*, 2019. [OpenReview](https://openreview.net/forum?id=SkfrvsA9FX), [arXiv:1805.11074](https://arxiv.org/abs/1805.11074).

[14] A. Stooke, J. Achiam, and P. Abbeel, “Responsive safety in reinforcement learning by PID Lagrangian methods,” *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119:9133–9143, 2020. [PMLR](https://proceedings.mlr.press/v119/stooke20a.html).

[15] Y. Chow and M. Ghavamzadeh, “Algorithms for CVaR optimization in MDPs,” *Advances in Neural Information Processing Systems*, 27, 2014. [NeurIPS](https://proceedings.neurips.cc/paper/2014/hash/35f1050a4381d2d216bf56ad46b0277d-Abstract.html).

[16] R. S. Sutton, D. Precup, and S. Singh, “Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning,” *Artificial Intelligence*, 112(1–2):181–211, 1999. DOI: [10.1016/S0004-3702(99)00052-1](https://doi.org/10.1016/S0004-3702(99)00052-1).

[17] M. J. Hausknecht and P. Stone, “Deep reinforcement learning in parameterized action space,” *International Conference on Learning Representations*, 2016. [ICLR record](https://mlanthology.org/iclr/2016/hausknecht2016iclr-deep/), [arXiv:1511.04143](https://arxiv.org/abs/1511.04143).

[18] J. Xiong et al., “Parametrized deep Q-networks learning: Reinforcement learning with discrete-continuous hybrid action space,” arXiv:1810.06394, 2018. [arXiv](https://arxiv.org/abs/1810.06394).

[19] L. Zhang, K. Li, C. Li, and K. Li, “Bi-objective workflow scheduling of the energy consumption and reliability in heterogeneous computing systems,” *Information Sciences*, 379:241–256, 2017. DOI: [10.1016/j.ins.2016.08.003](https://doi.org/10.1016/j.ins.2016.08.003).

[20] T. Wu, H. Gu, J. Zhou, T. Wei, X. Liu, and M. Chen, “Soft error-aware energy-efficient task scheduling for workflow applications in DVFS-enabled cloud,” *Journal of Systems Architecture*, 84:12–27, 2018. DOI: [10.1016/j.sysarc.2018.03.001](https://doi.org/10.1016/j.sysarc.2018.03.001).

[21] H. A. Hassan, S. A. Salem, and E. M. Saad, “A smart energy and reliability aware scheduling algorithm for workflow execution in DVFS-enabled cloud environment,” *Future Generation Computer Systems*, 112:431–448, 2020. DOI: [10.1016/j.future.2020.05.040](https://doi.org/10.1016/j.future.2020.05.040).

[22] Z. Wu, L. Han, J. Liu, Y. Robert, and F. Vivien, “Energy-aware mapping and scheduling strategies for real-time workflows under reliability constraints,” *Journal of Parallel and Distributed Computing*, 176:1–16, 2023. DOI: [10.1016/j.jpdc.2023.02.004](https://doi.org/10.1016/j.jpdc.2023.02.004).

[23] T. D. Kulkarni, K. Narasimhan, A. Saeedi, and J. B. Tenenbaum, “Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation,” *Advances in Neural Information Processing Systems*, 29, 2016. [NeurIPS](https://proceedings.neurips.cc/paper/2016/hash/f442d33fa06832082290ad8544a8da27-Abstract.html).

[24] A. S. Vezhnevets, S. Osindero, T. Schaul, N. Heess, M. Jaderberg, D. Silver, and K. Kavukcuoglu, “FeUdal Networks for hierarchical reinforcement learning,” *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70:3540–3549, 2017. [PMLR](https://proceedings.mlr.press/v70/vezhnevets17a.html).

[25] Y. Wang and S. Sun, “Dynamic task scheduling in wireless edge computing using deep reinforcement learning with ordinal optimization,” *EURASIP Journal on Wireless Communications and Networking*, 2025:96, 2025. DOI: [10.1186/s13638-025-02534-0](https://doi.org/10.1186/s13638-025-02534-0).

[26] M. Esnaashari, M. Ansari, and A. Ejlali, “DReaM: Deep reinforcement learning for joint reliability and power management in multicore IoT devices,” *IEEE Internet of Things Journal*, 12(24):54509–54522, 2025. DOI: [10.1109/JIOT.2025.3619513](https://doi.org/10.1109/JIOT.2025.3619513).

## 定稿前待补项目

- 将引言末段的 `XX%` 替换为冻结后的 20-seed 配对评估结果，并同时报告可行率或门控条件，避免只报告能耗均值。
- 根据目标期刊格式把编号引用转换为 `\cite{}`，并从 DOI/会议官网导出 BibTeX；不要手工猜测卷期、页码或作者拼写。
- 若实验继续保持 `enable_backup_duplication=false`，在系统模型与实验设置中明确“可靠性通过放置与频率控制实现，不使用任务复制”。
- 将 ESP@90 首次出现处定义为“单个评估 episode 内 Deadline 达标工作流比例不低于 90% 的 episode 占比”或项目最终采用的精确定义。
- 方法章节应给出上层动作选择式、三个价值头各自的 Bellman 目标、上下层对偶更新式，以及工作流级反事实图的构造与成本守恒规则。
