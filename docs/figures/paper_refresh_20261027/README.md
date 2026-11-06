# 论文数据与蓝色配色更新（2026-10-27）

## 当前论文图名（用户确认后简化）

`paper/manuscript/pic`现在仅含4张实际引用的图：

- `scheduling_framework.pdf`：调度算法框架。
- `main_experiment.pdf`：主实验27档能耗箱线图。
- `alibaba_generalization.pdf`：阿里工作流泛化实验。
- `sensitivity_analysis.pdf`：工作流数量与到达率敏感性实验。

`main2.tex`仅同步4条includegraphics路径，正文、数据、布局、图像文件内容完全不变。改名前的论文源与PDF存于`paper/manuscript/backups/figure_names_20261027/`。下方原文件名保留为历史追溯记录；重绘输出仍位于`output/pdf`，不向论文pic自动添加旧名称副本。

## 最新配色修订：低饱和分类色（当前生效）

用户确认采用此配色后，已将主实验、阿里、敏感性目录及论文pic/output中的44个被替代的PNG/SVG/PDF移至Windows回收站；未永久删除，可从回收站恢复。当前论文引用的4张新版图保留，旧版备份目录保留，实验数据、模型、表格和来源记录未清理。历史说明里指向旧图的链接可能不再打开，这是清理预期结果。

为避免删除旧导出PDF后影响框架重绘，`tools/refresh_paper_seed019_20261027.py`已改为读取`paper/manuscript/backups/main_data_palette_20261027/pic/framework1.pdf`这一保留源。当前论文源文件仍未改变。

用户在首版之后认为深蓝同色系过重，要求仅调整颜色。当前四方法颜色为：DQN-Edge `#D99A5B`（柔和橙）、DReaM-DQN `#83A783`（灰绿）、OOAC `#A28AA7`（灰紫）、CAHDRL `#4C78A8`（稳重蓝）。框架同步使用浅色变体；箱线图中位数恢复深灰以保持对比。下文蓝色配色记录描述首版，保留追溯；文件名中的blue保留兼容已有论文引用。

本次4张图重新读取相同原CSV；120个CSV/JSONL的SHA一致，2160条主实验数据不变，4项回归检查通过。`main2.tex`逐字未改，SHA256为`a37cf30c254957c0f5444aa9ed46a9263962e6e7e5e3fca0d3c5189c26761a3f`。图的尺寸、布局、坐标轴、方法顺序、文字和数据均未改变；只替换颜色并重新编译论文PDF。

首版深蓝图与论文PDF另存于`paper/manuscript/backups/palette_darkblue_20261027/`。这些是设计选择，不声称IEEE/Transactions指定这一套色值；设计参考IEEE关于清晰区分、对比度和缩小后可读性的图形要求。未新增训练或评估，也没有做实体打印色差测试。

## 已核验的主实验

- CAHDRL：`results/legacy_seed_block_0_19_20261026/{ss_37,sm_74,sl_111}/comparison_rows.csv`，共540条。这里是原完整CAHDRL，不是Raw模型。
- 三个基线：`results/baselines_seed_block_0_19_20261026/{resource}_{mode}/comparison_rows.csv`，共1620条。
- 共12模型、2160条、108个统计单元；每档相同真实工作流种子0—19，各20条；数值直接读取原CSV，并逐行与JSONL交叉核验。
- 9个基线会话均已确认退出0；各180条，日志无Traceback，有完成标记。三个CAHDRL资源评估此前已核验完成。
- CAHDRL、DQN-Edge、OOAC的27档ESP@90、Rel均100%；DReaM-DQN有8档至少一项低于100%，全部真实结果保留。
- 该连续种子段是在查看CAHDRL结果后采用的条件性探索配对样本；论文已在主实验数据说明中明确，不作为结果无关选取样本的总体成功率估计。

主表统计导出：`main_cells_from_original_csv.csv`。这是汇总，不代替原始逐种子CSV。原始文件SHA见`data_audit.json`。

## 论文与图

现有源文件原位更新：`paper/manuscript/main2.tex`；编译结果：同目录`main2.pdf`。

论文引用的4张图均已统一蓝色：

1. `paper/manuscript/pic/framework1_blue.pdf`：仅替换矢量颜色操作。原文字、公式、几何、字体和连线不变；浅色背景使用所给蓝色的白混合色，以维持深色文字对比度。
2. `paper/manuscript/pic/energy_boxplots_seed_0_19_blue_separators.pdf`：新的0—19主实验CSV。
3. `paper/manuscript/pic/alibaba_energy_boxplot_separators_blue.pdf`：原阿里CSV，数据未换。
4. `paper/manuscript/pic/energy_grouped_bars_J_blue.pdf`：原工作流数量/到达率CSV，数据未换。

四方法统一颜色：DQN-Edge `#011C6B`；DReaM-DQN `#01258F`；OOAC `#0231BD`；CAHDRL `#023BE6`。框架背景另使用`#011142`的浅色变体。

能耗均以J表示；主表均值保留两位小数，不新增标准差列；消融保留原样本及均值±样本SD，仅重新从原CSV换算为J（表格取整J）。成功率维持原有居中样式，不加计数分数。

阿里、敏感性和消融没有重新评估，也未冒充0—19种子结果。方法、公式、训练和其他正文未改。仅同步主实验数值、相关摘要/结论、数据来源说明以及J单位；主表行高轻微收紧以消除浮动页溢出。图的布局、方法顺序及隐藏异常点标记的设定保留，所有样本仍参与统计。

## 备份与验证

- 原论文、原PDF、4张原图、3份原绘图脚本备份：`paper/manuscript/backups/main_data_palette_20261027/`。
- 未删除或覆盖旧实验数据；旧图文件保留，论文改用新图名。
- `python -B -X utf8 -m unittest tests.test_paper_csv_refresh -v`：4项通过，覆盖CSV类型、ESP/Rel计算、2160条配对、论文修改范围和可复现性。
- 编辑器内置编译和本地pdflatex均成功；13页PDF；相关图表页已渲染目视检查。无Overfull或未定义引用；浮动页仅含图表的提示保留。
- 原CSV/JSONL的更新前后SHA一致；最终核验结果见`final_verification.json`。

## 重绘（只读数据，不启动评估）

```powershell
python -B -X utf8 plot/plot_current_generalization_energy.py --seed-block-0-19 --separators
python -B -X utf8 plot/plot_alibaba_energy_only.py --separators --paper-blue
python -B -X utf8 plot/plot_workload_sensitivity_options.py --paper-blue
```

矢量框架换色使用带pypdf的文档Python运行`tools/refresh_paper_seed019_20261027.py --recolor`；该工具不会修改原框架图。其`--tex-patch`只针对备份前的旧源生成补丁，不可对已更新的论文重复执行。
