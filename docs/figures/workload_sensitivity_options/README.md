# 工作负载敏感性：三版预览

仅绘图，不重新评估、不修改论文或原始数据。运行：`python plot/plot_workload_sensitivity_options.py`。

三版使用同样两组已完成的原始未筛选数据：数量实验400条、到达率实验400条；其中80条是明确复现的共同基准，不当作新增独立样本。每点20工作负载种子，固定训练模型，误差线为能耗的样本标准差。

## 版式

当前选定调整：`python plot/plot_workload_sensitivity_options.py --bars`，另存`energy_grouped_bars_J.png`/SVG和`provenance_bars.json`。两幅分组柱状图只展示能耗J均值±样本SD，无内嵌表格、无小插图、纵轴从0起。每组算法从左至右DQN-Edge、DReaM-DQN、OOAC、D3QN+SAC；共享图例在底部。横轴是等宽的离散实验档位，不表示连续到达率间距。ESP/Rel表单独放在`constraint_attainment_table.tex`，是需要booktabs的LaTeX片段，尚未插入论文。

柱状图显示名称现为CAHDRL (ours)，底层模型键仍为d3qn_sac，模型和原始数据不改。子图标签居中放在各横轴下方，与共享图例至少留12pt间距（脚本实测断言）。同一命令还生成`energy_grouped_bars_J_refined_colors.png`/SVG同色系加深预览：DQN #D9A182、DReaM #93B18A、OOAC #88B9BF、CAHDRL #5F86B5，供用户选择，并未覆盖框架原配色。main2.tex未改。

- A：上排能耗双图，下方一张紧凑表，完整概括当前实际ESP与Rel模式。只有在原始记录核对通过且达标率确为这组模式时才生成，不能套用到新数据。颜色强调的90%仍达到论文90%经验服务门槛，不代表不可行。
- B：上排能耗，下排ESP热力图；每格标明数值，90/100用两档颜色区分，不把颜色当作连续刻度。Rel在两张热力图上方注明全100%，不是缺失指标。
- C：两行对应数量/到达率，三列对应Energy/ESP/Rel。百分比轴为明确标注的84%至103%局部视图。相等曲线不人为错开；这版直观展示Rel及多数ESP曲线重叠的冗余。

框架配色保持：DQN-Edge #F5AA83，DReaM #BBDDB2，OOAC #BAE0E3，D3QN+SAC #A8C0E4。用不同线型及深色细轮廓辅助辨认，不增加样本点、异常点或虚假抖动。图例统一在底部，ours最后。输出PNG和可编辑SVG；选定后再制作最终PDF与论文图注。

## 必须随最终图注保留的实验口径

新增J单位版A预览：`python plot/plot_workload_sensitivity_options.py --joules`，另存`option_A_joules_with_unit_energy.png`/SVG和`provenance_joules.json`，不覆盖前三版。数量主轴J用×10^6，到达率主轴J用×10^5；数量图新增四法单位工作流能耗J/workflow小图，数值及SD均按批次数量归一化，并与已有归一化汇总逐项比对。不做拟合、平滑或人工波动，保留两能耗主面板与原达标表。

SM=Small科学工作流、74台（56edge+18cloud）、Tight。数量扫描50至250，到达率固定0.03/s，参考DDL按批次重算；到达率扫描0.01至0.20/s，固定50工作流与0.03/s基准相对DDL预算。因此是互补单因素实验，不是数量×到达率的联合实验。

ESP=ESP@90：每批按时比例至少90%的种子比例，不等于每个工作流都按时。Rel=每批全部工作流满足R>=0.9的种子比例，不等于R=1，也不是实测硬件无故障概率。所有比例为有限样本经验值。

数量五档四法均ESP/Rel100%；到达率0.20/s下DQN-Edge/DReaM的ESP90%，其余全100%；所有Rel100%。原始失败不删。

能源轴都用kJ，但数量面板和到达率面板范围不同，独立标注刻度，不通过强制共用上限压扁到达率变化。具体输入SHA和逐行核对记录见provenance.json。
