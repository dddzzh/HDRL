# Experimental section: source and audit notes

## 使用方法

以 `main.tex` 为论文入口；实验正文单独放在 `experiments.tex`，主文件已经通过 `\input{experiments}` 引入。请上传整套文件，不要只复制主文件。参考文献文件名是 `references.bib`。

本次只完善论文和绘图，不修改算法、训练配置、checkpoint 或原始结果。实验正文为英文，采用 IEEE 双栏格式。

## 内容与图表

- 实验设置：S26、S37、M37、M53，训练 Medium deadline，固定模型评估 Loose / Medium / Tight。
- 主实验：48 个“场景 × deadline × 方法”组合，960 条逐种子记录；每组 20 个测试种子。
- 结构消融：Full、Full 固定满频、独立训练上层、HEFT+SAC、HEFT 满频、无 Deadline 头，共六组。
- 三张图：主实验能耗箱线图及 ESP 矩阵、消融能耗及 ESP、训练期验证曲线。
- 五张实验表：场景、设备参数、主实验能耗、消融结果、Wilcoxon 检验。加上前文符号表，Wilcoxon 应为 Table VI；最终编号以 LaTeX 编译结果为准。
- 图为矢量 PDF；能耗表使用 J，保留两位小数，标准差使用样本标准差。

## 数据入口

路径均相对于项目根目录 `D:/研究生期间/第二篇/DRLEC`。

| 内容 | 数据文件 |
|---|---|
| S26、M37 逐种子主结果 | `docs/figures/formal_comparison_selected_columns.csv` |
| S26、M37 对照汇总 | `docs/figures/formal_comparison_figure_summary.csv` |
| S37 逐种子及汇总 | `results/scenario_matrix_evaluation/S37_final/comparison_rows.csv`、`comparison_summary.csv` |
| M53 逐种子及汇总 | `results/scenario_matrix_evaluation/M53_complete_4alg_20260927/comparison_rows.csv`、`comparison_summary.csv` |
| 六组消融汇总 | `results/ablation_results/small_main_ablation/ablation_summary.csv` |
| 显著性检验与配对数据 | `docs/figures/main_comparison_all_scenarios/` 下的 Wilcoxon CSV、报告及逐种子配对文件 |
| 训练曲线来源 | `experiment_assets/data_audit.json` 内的 `learning` 字段逐项列出原始 `eval_history.jsonl` |
| 图表再生成与静态检查 | `plot/build_experiment_section.py` |

主实验全部使用原始完整测试块 `242000–242019`，没有采用后来替换失败种子的结果。消融使用 `742000–742019`，因此其 Full 数值不应直接拿来替换主实验 Full 数值。

## 统计口径

ESP@90 指一组测试中，有多少比例的 workload realization 至少完成 18/20 个工作流的 deadline 要求；不是工作流成功率的平均值。

论文的经验可行门槛是 ESP@90 ≥90%，且每个测试 realization 的所有工作流可靠性要求均满足。训练中 checkpoint 的严格选择目标可以是 ESP=100%，两者已分开描述。M53 的 Medium/Tight 为 90%，不能写为全部 20 次通过，也不能写成概率约束的理论保证。

M37 DReaM 的 Medium ESP 在这套原始种子结果中为 85%，不是部分历史 Markdown 中的 100%。本次以逐种子 CSV 与交叉校验后的汇总为准。

所有 12 个主条件中，本文方法在经验可行方法里能耗最低；相对每档最低能耗可行 baseline 节能 13.45%–48.93%。不可行 baseline 的低能耗不作为可行节能优势。

Wilcoxon 先在每个测试种子内平均 12 个条件的相对能耗差，再对 20 个种子级差值检验。它只支持能耗差异，不支持 Deadline 更优或多训练种子的稳定性结论。

## 提交前不能忽略的边界

1. 现有实验是每种配置分别训练，泛化证据主要是改变 deadline 分布。不能宣称一个模型零样本泛化到任意规模；已做 Medium→Large 探索的退化也已在局限性中说明。
2. 每个训练变体只有一条训练轨迹，20 个测试种子不是 20 个训练种子，不能据此写训练稳定性保证。
3. 历史 baseline checkpoint 选择预算并非完全一致：部分为两块各 5 个种子、部分各 20 个；S26 DReaM 使用过单块平均 Deadline 选择；S37 DReaM 重新验证排序且使用回退。正文已如实披露，不能再写“完全统一选择协议”。要消除此项限制，需要统一重选，而不是修改文字隐藏。
4. 可靠性在目前参数下未形成区分，各方法均达标；不应声称本实验单独证明可靠性模块带来提升。
5. 设备 CPU、内存、GPU 与完整软件版本未由现有结果可靠确认，因此没有杜撰硬件或运行耗时。若目标期刊要求，应据实际训练机器补写。
6. 本次三张图已逐张检查；引用、标签、图文件和表行已静态检查。内置 LaTeX 编译器报 `Unable to find standard directories for platform`，因此**整篇 PDF 未完成编译排版验证**。请在 Overleaf 上传完整 ZIP，并以 `main.tex` 为主文件编译后检查跨栏浮动位置、表宽和最终编号。

这是一份完整实验章节草稿及可复现图表包，不意味着既有实验的设计限制已经消除，也不意味着可以保证录用。
