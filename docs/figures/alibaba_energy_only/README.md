# Alibaba能耗单图

仅保留原三联图的能耗箱线图；去掉ESP与Rel子图、底部实验说明小字和不再需要的(a)子图编号，保留坐标轴与共享图例。

来源：原始未筛选Alibaba实验，111台、100工作流、三档Deadline、四算法各20个固定种子，共240条记录；没有重新评估。全部能耗观察保留，隐藏离群点标记但不删除数据。

展示名CAHDRL，内部键仍为d3qn_sac。沿用参考图的框架浅色配色，本方法放在每组最右侧。

- PNG、SVG：本目录alibaba_energy_boxplot.png/.svg。
- 矢量PDF：output/pdf/alibaba_energy_boxplot.pdf。
- 原始三联图paper/manuscript/pic/alibaba_energy_constraints.pdf和main2.tex均不覆盖。
- 重绘：python plot/plot_alibaba_energy_only.py（只读取既有实验，不运行模型）。

若手动插入论文，原图注中(b)/(c)等说明需要相应删改；正文描述应使用“20个未筛选种子上各法三档ESP@90与Rel均100%”，不写所有工作流都按时或物理可靠性等于1。

## 淡色分隔线试排版

`python plot/plot_alibaba_energy_only.py --separators` 另存 `alibaba_energy_boxplot_separators.png/.svg` 和 `output/pdf/alibaba_energy_boxplot_separators.pdf`，不覆盖无分隔线版本。两条淡灰色竖线位于Loose/Medium/Tight组间，透明度0.28，高度为绘图区的10%至86%，不贯穿整幅图。
