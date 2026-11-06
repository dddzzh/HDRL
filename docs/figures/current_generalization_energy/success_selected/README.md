# 新配对评估能耗图（条件筛选样本）

数据来自已完成核验的12模型配对评估，不是上午的未筛选原始块。2160条、108箱体、每箱20种子；逐种子CSV和来源hash在results/success_seed_screening_20261013/exports/。

沿用SS/SM/SL、MS/MM/ML、LS/LM/LL九面板、底部图例、纯箱线图及独立线性纵轴。现按用户要求与敏感性柱状图增强配色统一：DQN-Edge #D9A182、DReaM-DQN #93B18A、OOAC #88B9BF、CAHDRL #5F86B5。单位J、乘数10^6。箱体为中位数和IQR，须线1.5-IQR。用户要求隐藏点状标记，未删除观察。

按用户要求移除图底部说明文字；每组箱线及底部图例从左到右为DQN-Edge、DReaM-DQN、OOAC、CAHDRL，不附加ours。名称仅是显示别名，底层模型键d3qn_sac与数据不改，main2.tex不改。该图仍为按结果筛选的条件样本，不能把筛选结果解释为未筛选总体100%成功率；数据来源和筛选属性保留在本文档及refined_figure_provenance.json中。上午的原始图在上一级目录保留。

生成命令（只绘图，不训练或评估）：

    python plot/plot_current_generalization_energy.py --success-selected

PDF：output/pdf/energy_boxplots_success_selected.pdf。PNG/SVG在本目录。
