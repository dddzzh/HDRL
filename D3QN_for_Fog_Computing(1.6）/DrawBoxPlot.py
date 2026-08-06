import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def drawBoxPlot_SA(csv_path):
    # 导入csv文件
    df = pd.read_csv(csv_path)

    # 图片保存地址
    plot_dir = os.path.dirname(csv_path)
    plot_path_combined = os.path.join(plot_dir, "combined_plots.svg")

    # 以 FCFS 列作为 baseline
    baseline = df['FCFS']

    # 计算 NP 值
    np_values = {}
    win_count = {col: 0 for col in df.columns if col != 'FCFS'}  # 初始化胜率统计
    for col in df.columns:
        if col != 'FCFS':
            np_values[col] = (baseline - df[col]) / baseline
            # 统计每一列胜利次数（低于 baseline 即为胜利）
            win_count[col] = (df[col] < baseline).sum()

    # 将结果转换为 DataFrame
    np_df = pd.DataFrame(np_values)

    # 创建并列图的布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制箱线图 (左侧)
    sns.boxplot(data=np_df, ax=ax1)
    colors = ['orange', 'green', 'blue', 'purple', 'red']  # 定义颜色
    for idx, col in enumerate(np_df.columns):
        mean_val = np_df[col].mean()
        ax1.axhline(y=mean_val, color=colors[idx % len(colors)], linestyle='--', linewidth=1, label=f'{col} Mean')

    # 旋转X轴标签防止文字重叠
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # 添加图例
    ax1.legend()

    # 设置箱线图标题和标签
    ax1.set_title('NP Values Box Plot')
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('NP Values')

    # 绘制胜率饼图 (右侧)
    labels = win_count.keys()
    sizes = win_count.values()
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    ax2.set_title('Algorithms Win Rate')

    # 调整布局避免重叠
    plt.tight_layout()

    # 保存两张图的并列展示
    plt.savefig(plot_path_combined, format='svg')

    # 显示图像
    plt.show()


# Example of usage:
# drawBoxPlot_SA('path_to_your_csv_file.csv')


def drawBoxPlot_RA_old(csv_path):
    df = pd.read_csv(csv_path)
    plot_dir = os.path.dirname(csv_path)
    plot_path = os.path.join(plot_dir, "boxplot.svg")
    baseline = df['PT']

    np_values = {}
    for col in df.columns:
        if col != 'PT':
            np_values[col] = (baseline - df[col]) / baseline

    # 将结果转换为 DataFrame
    np_df = pd.DataFrame(np_values)

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=np_df)
    plt.ylim(-0.2, 1)  #将纵轴控制在-0.2到1之间
    # 添加不同颜色的虚线标出每个箱体对应的平均值
    colors = [
        'orange',  # 橙色
        'green',  # 绿色
        'blue',  # 蓝色
        'purple',  # 紫色
        'red',  # 红色
        'lightblue',  # 淡蓝色
        'darkgrey',  # 深灰色
        'lightgrey',  # 浅灰色
        'teal',  # 青色
        'gold',  # 金色
        'pink',  # 粉色
        'navy',  # 海军蓝
        'olive'  # 橄榄色
    ]
    for idx, col in enumerate(np_df.columns):
        mean_val = np_df[col].mean()
        plt.axhline(y=mean_val, color=colors[idx % len(colors)], linestyle='--', linewidth=1, label=f'{col} Mean')

    # 添加图例
    plt.legend()

    # 设置标题和标签
    plt.title('NP Values Box Plot')
    plt.xlabel('Algorithms')
    plt.ylabel('NP Values')

    plt.savefig(plot_path, format='svg')
    plt.show()

    # 计算胜率（最小值即为胜者）
    win_df = pd.DataFrame(0, index=df.index, columns=df.columns)  # 初始化为0

    # 遍历每一行，找出最小的值（即获胜的算法）
    for idx, row in df.iterrows():
        # 找到最小值所在的列
        winner = row.idxmin()  # 获取最小值对应的列名（算法名称）
        win_df.at[idx, winner] = 1  # 设置为1表示该算法在该行中获胜
    win_rate = win_df.sum(axis=0) / len(df)  # 计算每列的1的比例，即胜率
    custom_colors = ['#00008B', '#00FF00', '#FFA500', '#ADD8E6', '#A9A9A9', '#D3D3D3', '#008000']
    # 定义一个函数来隐藏0%的数值
    def func(pct, allvals):
        absolute = pct / 100. * sum(allvals)
        return f'{pct:.1f}%' if pct > 0 else ''  # 如果为 0%，不显示

    # 绘制饼图
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(win_rate, labels=None, autopct=lambda pct: func(pct, win_rate), startangle=90,
                                       colors=custom_colors)

    # 设置图例
    plt.legend(wedges, win_rate.index, title="Algorithms", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 设置标题
    plt.title('Win Rate of Algorithms')

    # 显示图形
    plt.show()

def drawBoxPlot_RA(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    df = pd.read_csv(csv_path)
    plot_dir = os.path.dirname(csv_path)
    plot_path = os.path.join(plot_dir, "boxplot.svg")
    baseline = df['PT']

    # 重新排列列顺序，将 'RA' 放到最后
    if 'RA' in df.columns:
        cols = [col for col in df.columns if col != 'RA']
        df = df[cols + ['RA']]

    # 计算 NP 值
    np_values = {}
    for col in df.columns:
        if col != 'PT':
            np_values[col] = (baseline - df[col]) / baseline

    # 将结果转换为 DataFrame
    np_df = pd.DataFrame(np_values)

    # 定义颜色列表
    colors = [
        '#FFAA33',  # 柔和橙色
        '#66CDAA',  # 中绿色
        '#FF6F61',  # 珊瑚红
        '#9370DB',  # 淡紫色
        '#8B4513',  # 棕褐色
        '#4682B4',  # 钢蓝色
    ]

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=np_df, palette=colors)
    plt.ylim(-0.2, 1)

    # 设置 x 轴和 y 轴刻度的字体大小，并旋转 x 轴标签
    x_labels = np_df.columns
    x_labels = [f'$\mathbf{{{label}}}$' if label == 'RA' else label for label in x_labels]
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, fontsize=12)
    plt.yticks(fontsize=12)

    # 添加不同颜色的虚线标出每个箱体对应的平均值
    for idx, col in enumerate(np_df.columns):
        mean_val = np_df[col].mean()
        plt.axhline(y=mean_val, color=colors[idx % len(colors)], linestyle='--', linewidth=1, label=f'{col} Mean')

    # 添加图例
    #plt.legend(fontsize=10)

    # 设置标题和标签
    plt.title('NP of Algorithms', fontsize=14)
    #plt.xlabel('Algorithms', fontsize=14)
    plt.ylabel('Normalized Performance', fontsize=14)

    # 保存图形
    plt.savefig(plot_path, format='svg')
    plt.show()

    # 计算胜率（最小值即为胜者）
    win_df = pd.DataFrame(0, index=df.index, columns=df.columns)  # 初始化为0

    # 遍历每一行，找出最小的值（即获胜的算法）
    for idx, row in df.iterrows():
        # 找到最小值所在的列
        winner = row.idxmin()  # 获取最小值对应的列名（算法名称）
        win_df.at[idx, winner] = 1  # 设置为1表示该算法在该行中获胜
    win_rate = win_df.sum(axis=0) / len(df)  # 计算每列的1的比例，即胜率

    # 绘制饼图
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        win_rate,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
        startangle=90,
        colors=colors
    )

    # 设置图例
    plt.legend(wedges, win_rate.index, title="Algorithms", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 设置标题
    plt.title('Win Rate of Algorithms', fontsize=14)

    # 显示图形
    plt.show()


def drawBoxPlot_Integrated(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path)
    plot_dir = os.path.dirname(csv_path)

    # 设置基准列
    baseline = df['PT_FCFS']

    # 计算 NP 值
    np_values = {}
    for col in df.columns:
        if col != 'PT_FCFS':
            np_values[col] = (baseline - df[col]) / baseline
    np_df = pd.DataFrame(np_values)

    # 绘制箱线图
    boxplot_path = os.path.join(plot_dir, "boxplot.svg")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=np_df)
    plt.ylim(-0.2, 1)  # 将纵轴控制在-0.2到1之间
    plt.title('NP Values Box Plot')
    plt.xlabel('Algorithms')
    plt.ylabel('NP Values')
    plt.xticks(rotation=45)  # 将横轴标签旋转45度
    # 添加不同颜色的虚线标出每个箱体对应的平均值
    colors = [
        'orange',  # 橙色
        'green',  # 绿色
        'blue',  # 蓝色
        'purple',  # 紫色
        'red',  # 红色
        'lightblue',  # 淡蓝色
        'darkgrey',  # 深灰色
        'lightgrey',  # 浅灰色
        'teal',  # 青色
        'gold',  # 金色
        'pink',  # 粉色
        'navy',  # 海军蓝
        'olive'  # 橄榄色
    ]
    for idx, col in enumerate(np_df.columns):
        mean_val = np_df[col].mean()
        plt.axhline(y=mean_val, color=colors[idx % len(colors)], linestyle='--', linewidth=1, label=f'{col} Mean')

    # 添加图例
    plt.legend()
    plt.savefig(boxplot_path, format='svg')
    plt.show()

#————————————饼图————————————
    piechart_path = os.path.join(plot_dir, "piechart.svg")
    # 计算胜率（最小值即为胜者）
    win_df = pd.DataFrame(0, index=df.index, columns=df.columns)  # 初始化为0

    # 遍历每一行，找出最小的值（即获胜的算法）
    for idx, row in df.iterrows():
        # 找到最小值所在的列
        winner = row.idxmin()  # 获取最小值对应的列名（算法名称）
        win_df.at[idx, winner] = 1  # 设置为1表示该算法在该行中获胜
    win_rate = win_df.sum(axis=0) / len(df)  # 计算每列的1的比例，即胜率
    custom_colors = [
    '#1f77b4',  # 蓝色 (Blue)
    '#ff7f0e',  # 橙色 (Orange)
    '#2ca02c',  # 绿色 (Green)
    '#d62728',  # 红色 (Red)
    '#9467bd',  # 紫色 (Purple)
    '#8c564b',  # 棕色 (Brown)
    '#e377c2',  # 粉色 (Pink)
    '#7f7f7f',  # 灰色 (Gray)
    '#bcbd22',  # 黄绿色 (Yellow-Green)
    '#17becf',  # 青色 (Teal)
    '#ff9896',  # 浅红色 (Light Red)
    '#98df8a',  # 浅绿色 (Light Green)
    '#aec7e8'   # 浅蓝色 (Light Blue)
]

    # 定义一个函数来隐藏0%的数值
    def func(pct, allvals):
        absolute = pct / 100. * sum(allvals)
        return f'{pct:.1f}%' if pct > 0 else ''  # 如果为 0%，不显示

    # 绘制饼图
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(win_rate, labels=None, autopct=lambda pct: func(pct, win_rate), startangle=90,
                                       colors=custom_colors)

    # 设置图例
    plt.legend(wedges, win_rate.index, title="Algorithms", loc='lower left', bbox_to_anchor=(-0.1, -0.1),
               ncol=1, frameon=True, fontsize=10)

    # 设置标题
    plt.title('Win Rate of Algorithms')

    plt.savefig(piechart_path, format='svg')
    plt.show()

#旧箱线图
# def drawBoxPlot_Integrated(csv_path):
#     df = pd.read_csv(csv_path)
#     plot_dir = os.path.dirname(csv_path)
#     plot_path_combined = os.path.join(plot_dir, "boxplot_and_piechart.svg")
#
#     # 设置基准列
#     baseline = df['PT_FCFS']
#
#     # 计算 NP 值
#     np_values = {}
#     for col in df.columns:
#         if col != 'PT_FCFS':
#             np_values[col] = (baseline - df[col]) / baseline
#     np_df = pd.DataFrame(np_values)
#
#     # 创建一个包含2个图的并列布局
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1行2列的布局
#
#     # 绘制箱线图
#     sns.boxplot(data=np_df, ax=axes[0])
#     axes[0].set_ylim(-0.2, 1)  # 将纵轴控制在-0.2到1之间
#     axes[0].set_title('NP Values Box Plot')
#     axes[0].set_xlabel('Algorithms')
#     axes[0].set_ylabel('NP Values')
#
#     # 添加不同颜色的虚线标出每个箱体对应的平均值
#     colors = [
#         'orange',  # 橙色
#         'green',  # 绿色
#         'blue',  # 蓝色
#         'purple',  # 紫色
#         'red',  # 红色
#         'lightblue',  # 淡蓝色
#         'darkgrey',  # 深灰色
#         'lightgrey',  # 浅灰色
#         'teal',  # 青色
#         'gold',  # 金色
#         'pink',  # 粉色
#         'navy',  # 海军蓝
#         'olive'  # 橄榄色
#     ]
#     for idx, col in enumerate(np_df.columns):
#         mean_val = np_df[col].mean()
#         axes[0].axhline(y=mean_val, color=colors[idx % len(colors)], linestyle='--', linewidth=1, label=f'{col} Mean')
#
#     axes[0].legend()
#
#     # 旧版的饼图
#     win_counts = df.apply(lambda row: row.idxmin(), axis=1).value_counts()
#
#     # 绘制饼图，表示胜率
#     axes[1].pie(win_counts, labels=win_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
#     axes[1].set_title('Win Rate by Algorithm')
#
#     # 保存并列图像为矢量图格式
#     plt.savefig(plot_path_combined, format='svg')
#
#     # 显示图像
#     plt.show()

if __name__ == '__main__':

    #drawBoxPlot_SA("checkpoints/D3QN/SA1112_1533smallD3E10C10W30L0.15_with_deadline/SequencingAgent/experiments1115_1426_RA220_SA60_SA_performance/SA_performance_data.csv")
    drawBoxPlot_RA("checkpoints/D3QN/RA1111_1707smallD3E10C10W30L0.15_with_deadline/RoutingAgent/experiments1115_1433_RA220_SA60/RA_utility_data.csv")
    #drawBoxPlot_Integrated('checkpoints/D3QN/RA1111_1748smallD4E15C15W45L0.15_with_deadline/RoutingAgent/experiments1119_2228_RA330_SA120_integrated(new)/algorithm_comparison.csv')