import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
#
# def draw_combined_plots(csv_paths):
#     # 创建两行三列的组合图
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#     plt.subplots_adjust(hspace=0.3, wspace=0.3)
#
#     # 定义颜色列表
#     box_colors = [
#         'orange', 'green', 'blue', 'purple', 'red', 'lightblue',
#         'darkgrey', 'lightgrey', 'teal', 'gold', 'pink', 'navy', 'olive'
#     ]
#     pie_colors = [
#         '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
#         '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896', '#98df8a', '#aec7e8'
#     ]
#     box_colors = [
#         '#e67c93',
#         '#d08349',
#         '#aa903d',
#         '#8c9563',
#         '#588e41',
#         '#439f7f',
#         '#449e97',
#         '#489ed9',
#         '#51a0d6',
#         '#aba7e7',
#         '#d97aad',
#         '#e47bbc'
#     ]
#     # 读取每个 CSV 文件并绘制箱线图和饼图
#     all_labels = []
#     for i, csv_path in enumerate(csv_paths):
#         # 读取数据
#         df = pd.read_csv(csv_path)
#
#         # 设置基准列
#         baseline = df['PT_FCFS']
#
#         # 计算 NP 值
#         np_values = {}
#         for col in df.columns:
#             if col != 'PT_FCFS':
#                 np_values[col] = (baseline - df[col]) / baseline
#         np_df = pd.DataFrame(np_values)
#
#         # 绘制箱线图（第一行）
#         ax_box = axes[0, i]
#         # current_palette = sns.color_palette()
#         # sns.palplot(current_palette)
#
#         sns.boxplot(data=np_df, ax=ax_box)
#         ax_box.set_ylim(-0.2, 1)
#         ax_box.set_title(f'NP Values Box Plot - CSV {i+1}')
#         ax_box.set_xlabel('Algorithms')
#         ax_box.set_ylabel('NP Values')
#         ax_box.tick_params(axis='x', rotation=45)
#         for patch in ax_box.patches:
#             print(patch.get_facecolor())
#         # 添加不同颜色的虚线标出每个箱体对应的平均值
#         for idx, col in enumerate(np_df.columns):
#             mean_val = np_df[col].mean()
#             ax_box.axhline(y=mean_val, color=box_colors[idx % len(box_colors)], linestyle='--', linewidth=1)
#
#         # 计算胜率
#         win_df = pd.DataFrame(0, index=df.index, columns=df.columns)  # 初始化为0
#         for idx, row in df.iterrows():
#             winner = row.idxmin()  # 获取最小值对应的列名（算法名称）
#             win_df.at[idx, winner] = 1  # 设置为1表示该算法在该行中获胜
#         win_rate = win_df.sum(axis=0) / len(df)  # 计算每列的1的比例，即胜率
#         all_labels = list(win_rate.index)
#
#         # 绘制饼图（第二行）
#         ax_pie = axes[1, i]
#         wedges, texts, autotexts = ax_pie.pie(win_rate, labels=None, autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
#                                               startangle=90, colors=pie_colors[:len(win_rate)])
#         ax_pie.set_title(f'Win Rate of Algorithms - CSV {i+1}')
#
#     # 创建饼图共用图例
#     fig_legend = plt.figure(figsize=(16, 1))
#     handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='None', markersize=10) for color in pie_colors[:len(all_labels)]]
#     fig_legend.legend(handles, all_labels, title="Algorithms", loc='center', ncol=6, frameon=True, fontsize=10)
#     plt.axis('off')  # 关闭坐标轴
#
#     plt.tight_layout(pad=0.5)
#     plt.show()
#
# if __name__ == '__main__':
#     csv_path_1 = "checkpoints/D3QN/RA1111_1700smallD2E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2129_RA270_SA90_integrated(new)/algorithm_comparison.csv"
#     csv_path_2 = "checkpoints/D3QN/RA1111_1707smallD3E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2135_RA220_SA60_integrated(new)/algorithm_comparison.csv"
#     csv_path_3 = "checkpoints/D3QN/RA1111_1715smallD4E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2140_RA210_SA50_integrated(new)/algorithm_comparison.csv"
#     csv_paths = [csv_path_1, csv_path_2, csv_path_3]
#     draw_combined_plots(csv_paths)
def draw_combined_plots_old(csv_paths):
    # 创建两行三列的组合图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # 定义颜色列表
    box_colors = [
        '#e67c93',
        '#d08349',
        '#aa903d',
        '#8c9563',
        '#588e41',
        '#439f7f',
        '#449e97',
        '#489ed9',
        '#51a0d6',
        '#aba7e7',
        '#d97aad',
        '#e47bbc'
    ]
    pie_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896', '#98df8a', '#aec7e8'
    ]

    # 读取每个 CSV 文件并绘制箱线图和饼图
    all_win_rates = []
    for i, csv_path in enumerate(csv_paths):
        # 读取数据
        df = pd.read_csv(csv_path)

        # 设置基准列
        baseline = df['PT_FCFS']

        # 计算 NP 值
        np_values = {}
        for col in df.columns:
            if col != 'PT_FCFS':
                np_values[col] = (baseline - df[col]) / baseline
        np_df = pd.DataFrame(np_values)

        # 绘制箱线图（第一行）
        ax_box = axes[0, i]
        sns.boxplot(data=np_df, ax=ax_box)
        ax_box.set_ylim(-0.2, 1)
        ax_box.set_title(f'NP Values - Large {i+1}', fontsize=14)
        ax_box.set_xlabel('Algorithms', fontsize=12)
        ax_box.set_ylabel('NP Values', fontsize=12)
        ax_box.tick_params(axis='x', rotation=45, labelsize=10)
        ax_box.tick_params(axis='y', labelsize=10)

        # 添加不同颜色的虚线标出每个箱体对应的平均值
        for idx, col in enumerate(np_df.columns):
            mean_val = np_df[col].mean()
            ax_box.axhline(y=mean_val, color=box_colors[idx % len(box_colors)], linestyle='--', linewidth=1)

        # 计算胜率
        win_df = pd.DataFrame(0, index=df.index, columns=df.columns)  # 初始化为0
        for idx, row in df.iterrows():
            winner = row.idxmin()  # 获取最小值对应的列名（算法名称）
            win_df.at[idx, winner] = 1  # 设置为1表示该算法在该行中获胜
        win_rate = win_df.sum(axis=0) / len(df)  # 计算每列的1的比例，即胜率

        # 绘制饼图
        ax_pie = axes[1, i]
        colors = [pie_colors[idx % len(pie_colors)] for idx in range(len(win_rate))]
        wedges, texts, autotexts = ax_pie.pie(win_rate, labels=[label if value > 0 else '' for label, value in zip(win_rate.index, win_rate)],
                                              autopct=lambda pct: func(pct, win_rate), startangle=90, colors=colors)
        ax_pie.set_title(f'Win Rate  - Large {i+1}', fontsize=14)

        # 设置标签文字的属性
        for text in texts:
            if text.get_text() == '':  # 隐藏没有标签的文字
                text.set_visible(False)
            else:
                text.set_fontsize(12)
        for autotext in autotexts:
            if autotext.get_text() == '':  # 隐藏没有显示的百分比
                autotext.set_visible(False)
            else:
                autotext.set_fontsize(12)

    # 调整布局，并保存图像
    plt.tight_layout(pad=0.5)
    # 保存图像到指定路径
    save_path = r"C:\Users\10927\Desktop\D3QN_for_Fog_Computing(1.6）\picture\Large.svg"
    plt.savefig(save_path, format='svg')  # 保存为SVG格式
    plt.show()

# 定义一个函数来隐藏0%的数值
def func(pct, allvals):
    absolute = pct / 100. * sum(allvals)
    return f'{pct:.1f}%' if pct > 0 else ''  # 如果为 0%，不显示
def draw_combined_plots(csv_paths, size):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 定义颜色列表
    colors = [
        '#FFC107',  # 明亮金黄 (Bright Amber)
        '#8BC34A',  # 青柠绿 (Lime Green)
        '#E57373',  # 浅红色 (Light Red)
        '#BA68C8',  # 明亮紫罗兰 (Vivid Violet)
        '#795548',  # 巧克力棕 (Chocolate Brown)
        '#2196F3',  # 明亮蓝色 (Bright Blue)
        '#F06292',  # 浅玫瑰色 (Light Rose)
        '#4CAF50',  # 浅绿色 (Fresh Green)
        '#FF9800',  # 明亮橙色 (Bright Orange)
        '#9E9E9E',  # 灰色 (Gray)
        '#3F51B5',  # 靛蓝色 (Indigo)
        '#4682B4',  # 钢蓝色
    ]

    # 创建图像布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, csv_path in enumerate(csv_paths):
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 替换 RA 和 SA 为 AMAD3QN
        df.rename(columns={'RA_SA': 'AMAD3QN'}, inplace=True)

        # 设置基准列
        baseline = df['PT_FCFS']

        # 计算 NP 值
        np_values = {}
        for col in df.columns:
            if col != 'PT_FCFS':
                np_values[col] = (baseline - df[col]) / baseline
        np_df = pd.DataFrame(np_values)

        # 将 'AMAD3QN' 放到最后
        if 'AMAD3QN' in np_df.columns:
            cols = [col for col in np_df.columns if col != 'AMAD3QN']
            np_df = np_df[cols + ['AMAD3QN']]

        # 颜色映射
        color_map = {col: colors[idx % len(colors)] for idx, col in enumerate(np_df.columns)}

        # 胜率统计
        win_count = {col: 0 for col in df.columns if col != 'PT_FCFS'}
        for _, row in df.iterrows():
            winner = row.idxmin()
            if winner != 'PT_FCFS':
                win_count[winner] += 1

        # 绘制箱线图（第一行）
        ax_box = axes[0, i]
        sns.boxplot(data=np_df, palette=[color_map[col] for col in np_df.columns], ax=ax_box)
        for col in np_df.columns:
            mean_val = np_df[col].mean()
            ax_box.axhline(y=mean_val, color=color_map[col], linestyle='--', linewidth=1)

        # 设置固定的纵轴范围
        ax_box.set_ylim(-0.2, 1)

        # 设置标题和标签
        ax_box.set_title(f'NP of Algorithms - {size} {i + 1}', fontsize=16)
        #ax_box.set_xlabel('Algorithms', fontsize=12)
        ax_box.set_ylabel('Normalized Performance', fontsize=12)
        ax_box.tick_params(axis='x', rotation=45, labelsize=10)
        ax_box.tick_params(axis='y', labelsize=12)

        # 算法名称加粗显示 'AMAD3QN'
        x_labels = np_df.columns
        x_labels = [f'$\mathbf{{{label}}}$' if label == 'AMAD3QN' else label for label in x_labels]
        ax_box.set_xticks(range(len(x_labels)))
        ax_box.set_xticklabels(x_labels, fontsize=12)

        # 绘制饼图（第二行）
        total_runs = len(df)
        labels = [
            f'$\mathbf{{{label}}}$ ({win_count[label] / total_runs * 100:.1f}%)'
            if label == 'AMAD3QN'
            else f'{label} ({win_count[label] / total_runs * 100:.1f}%)'
            for label in win_count if win_count[label] > 0
        ]
        sizes = [win_count[label] / total_runs for label in win_count if win_count[label] > 0]

        ax_pie = axes[1, i]
        wedges, texts, autotexts = ax_pie.pie(
            sizes,
            labels=labels,
            autopct='',
            startangle=140,
            colors=[color_map[label] for label in win_count if win_count[label] > 0],
        )
        ax_pie.set_title(f'WR of Algorithms - {size} {i + 1}', fontsize=16)

        # 设置标签字体大小
        for text in texts:
            text.set_fontsize(14)

    # 调整布局
    plt.tight_layout(pad=0.5)
    save_path = rf"C:\Users\10927\Desktop\D3QN_for_Fog_Computing(1.6）\picture\{size}.svg"
    plt.savefig(save_path, format='svg')  # 保存为SVG格式
    plt.show()

def draw_combined_plots_x(csv_paths):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    colors = [
        '#FFC107',  # 明亮金黄 (Bright Amber)
        '#8BC34A',  # 青柠绿 (Lime Green)
        '#E57373',  # 浅红色 (Light Red)
        '#BA68C8',  # 明亮紫罗兰 (Vivid Violet)
        '#795548',  # 巧克力棕 (Chocolate Brown)
        '#2196F3',  # 明亮蓝色 (Bright Blue)
        '#F06292',  # 浅玫瑰色 (Light Rose)
        '#4CAF50',  # 浅绿色 (Fresh Green)
        '#FF9800',  # 明亮橙色 (Bright Orange)
        '#9E9E9E',  # 灰色 (Gray)
        '#3F51B5',  # 靛蓝色 (Indigo)
        '#4682B4',  # 钢蓝色
    ]
    for i, csv_path in enumerate(csv_paths):
        # 读取数据
        df = pd.read_csv(csv_path)
        # 设置基准列
        baseline = df['PT_FCFS']
        # 计算 NP 值
        np_values = {}
        for col in df.columns:
            if col != 'PT_FCFS':
                np_values[col] = (baseline - df[col]) / baseline
        np_df = pd.DataFrame(np_values)
        color_map = {col: colors[idx % len(colors)] for idx, col in enumerate(np_df.columns)}


if __name__ == '__main__':
    # csv_path_1 = "checkpoints/D3QN/RA1111_1723smallD2E15C15W45L0.15_with_deadline/RoutingAgent/experiments1119_2145_RA360_SA90_integrated(new)/algorithm_comparison.csv"
    # csv_path_2 = "checkpoints/D3QN/RA1111_1735smallD3E15C15W45L0.15_with_deadline/RoutingAgent/experiments1119_2151_RA310_SA90_integrated(new)/algorithm_comparison.csv"
    # csv_path_3 = "checkpoints/D3QN/RA1111_1748smallD4E15C15W45L0.15_with_deadline/RoutingAgent/experiments1119_2228_RA330_SA120_integrated(new)/algorithm_comparison.csv"
    # csv_paths = [csv_path_1, csv_path_2, csv_path_3]
    # draw_combined_plots(csv_paths,size="Large")

    # csv_path_1 = "checkpoints/D3QN/RA1111_1700smallD2E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2129_RA270_SA90_integrated(new)/algorithm_comparison.csv"
    # csv_path_2 = "checkpoints/D3QN/RA1111_1707smallD3E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2135_RA220_SA60_integrated(new)/algorithm_comparison.csv"
    # csv_path_3 = "checkpoints/D3QN/RA1111_1715smallD4E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2140_RA210_SA50_integrated(new)/algorithm_comparison.csv"
    # csv_paths = [csv_path_1, csv_path_2, csv_path_3]
    # draw_combined_plots(csv_paths,size="Middle")


    csv_path_1 = "checkpoints/D3QN/RA1111_1644smallD2E6C6W20L0.15_with_deadline/RoutingAgent/experiments1119_2113_RA150_SA80_integrated(new)/algorithm_comparison.csv"
    csv_path_2 = "checkpoints/D3QN/RA1111_1649smallD3E6C6W20L0.15_with_deadline/RoutingAgent/experiments1119_2317_RA160_SA110_integrated(new)/algorithm_comparison.csv"
    csv_path_3 = "checkpoints/D3QN/RA1111_1654smallD4E6C6W20L0.15_with_deadline/RoutingAgent/experiments1120_1545_RA140_SA60_integrated(new)/algorithm_comparison.csv"
    csv_paths = [csv_path_1, csv_path_2, csv_path_3]
    draw_combined_plots(csv_paths,size="Small")