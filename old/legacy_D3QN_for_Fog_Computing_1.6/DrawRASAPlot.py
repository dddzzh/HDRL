import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

####  SA的在这个文件  RA的在原来的代码里
def draw_combined_boxplot_and_pie(csv_path):
    df = pd.read_csv(csv_path)
    plot_dir = os.path.dirname(csv_path)
    plot_path = os.path.join(plot_dir, "combined_boxplot_pie.svg")

    # 设置基准列
    baseline = df['PT']

    # 计算 NP 值
    np_values = {}
    for col in df.columns:
        if col != 'PT':
            np_values[col] = (baseline - df[col]) / baseline

    # 将 NP 值转为 DataFrame
    np_df = pd.DataFrame(np_values)

    # 绘制图形：箱线图和饼图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 绘制箱线图
    ax_box = axes[0]
    sns.boxplot(data=np_df, ax=ax_box)
    ax_box.set_ylim(-0.2, 1)
    ax_box.set_title('NP Values Box Plot', fontsize=28)  # Set title font size
    ax_box.set_xlabel('Algorithms', fontsize=24)  # Set x-axis label font size
    ax_box.set_ylabel('NP Values', fontsize=24)  # Set y-axis label font size
    ax_box.tick_params(axis='x', rotation=45, labelsize=18)  # Increase x-axis tick label size
    ax_box.tick_params(axis='y', labelsize=18)  # Increase y-axis tick label size

    # 添加每个箱体的平均值线
    box_colors = ['orange', 'green', 'blue', 'purple', 'red', 'lightblue', 'darkgrey', 'lightgrey', 'teal', 'gold',
                  'pink', 'navy', 'olive']
    for idx, col in enumerate(np_df.columns):
        mean_val = np_df[col].mean()
        ax_box.axhline(y=mean_val, color=box_colors[idx % len(box_colors)], linestyle='--', linewidth=2,
                       label=f'{col} Mean')

    # 计算胜率（最小值即为胜者）
    win_df = pd.DataFrame(0, index=df.index, columns=df.columns)
    for idx, row in df.iterrows():
        winner = row.idxmin()
        win_df.at[idx, winner] = 1
    win_rate = win_df.sum(axis=0) / len(df)

    # 定义一个函数来隐藏胜率为 0% 的算法的名称
    def func(pct, allvals):
        return f'{pct:.1f}%' if pct > 0 else ''  # 如果为 0%，不显示

    # 绘制饼图
    ax_pie = axes[1]
    custom_colors = ['#00008B', '#00FF00', '#FFA500', '#ADD8E6', '#A9A9A9', '#D3D3D3', '#008000']

    # 过滤掉胜率为0%的算法
    win_rate_filtered = win_rate[win_rate > 0]

    wedges, texts, autotexts = ax_pie.pie(win_rate_filtered, labels=win_rate_filtered.index,
                                          autopct=lambda pct: func(pct, win_rate_filtered), startangle=90,
                                          colors=custom_colors)

    # 设置标题
    ax_pie.set_title('Win Rate of Algorithms', fontsize=28)  # Set title font size

    # 设置图例
    ax_pie.legend(wedges, win_rate_filtered.index, title="Algorithms", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=20)

    # 设置饼图标签的字体大小
    for text in texts + autotexts:
        text.set_fontsize(20)

    plt.tight_layout(pad=1)
    plt.savefig(plot_path, format='svg')
    plt.show()


def drawBoxPlot_SA_old(csv_path):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 绘制箱线图 (左侧)
    sns.boxplot(data=np_df, ax=ax1)
    colors = ['orange', 'green', 'blue', 'purple', 'red', 'lightblue', 'darkgrey', 'lightgrey', 'teal', 'gold', 'pink',
              'navy', 'olive']
    for idx, col in enumerate(np_df.columns):
        mean_val = np_df[col].mean()
        ax1.axhline(y=mean_val, color=colors[idx % len(colors)], linestyle='--', linewidth=2, label=f'{col} Mean')

    # 旋转X轴标签防止文字重叠
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # 设置标题和标签字体大小
    ax1.set_title('NP Values Box Plot', fontsize=22)  # Set title font size
    ax1.set_xlabel('Algorithms', fontsize=20)  # Set x-axis label font size
    ax1.set_ylabel('NP Values', fontsize=20)  # Set y-axis label font size
    ax1.tick_params(axis='x', rotation=45, labelsize=12)  # Increase x-axis tick label size
    ax1.tick_params(axis='y', labelsize=16)  # Increase y-axis tick label size

    # 绘制胜率饼图 (右侧)
    labels = [label for label, win in win_count.items() if win > 0]  # 只保留胜率大于0的算法
    sizes = [win for win in win_count.values() if win > 0]  # 只保留胜率大于0的值

    # 定义一个函数来隐藏胜率为 0% 的算法的名称
    def func(pct, allvals):
        return f'{pct:.1f}%' if pct > 0 else ''  # 如果为 0%，不显示

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct=lambda pct: func(pct, sizes), startangle=140,
                                       colors=colors)
    ax2.set_title('Algorithms Win Rate', fontsize=22)  # Set title font size

    # 设置饼图标签的字体大小
    for text in texts + autotexts:
        text.set_fontsize(16)

    # 调整布局避免重叠
    plt.tight_layout(pad=1)

    # 保存两张图的并列展示
    plt.savefig(plot_path_combined, format='svg')

    # 显示图像
    plt.show()

def drawBoxPlot_SA(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 导入 csv 文件
    df = pd.read_csv(csv_path)

    # 图片保存地址
    plot_dir = os.path.dirname(csv_path)
    plot_path_combined = os.path.join(plot_dir, "combined_plots.svg")

    # 替换列名为缩写
    df.rename(columns={'UpwardRank': 'UR', 'DownwardRank': 'DR'}, inplace=True)

    # 以 FCFS 列作为 baseline
    baseline = df['FCFS']

    # 计算 NP 值
    np_values = {}
    for col in df.columns:
        if col != 'FCFS':
            np_values[col] = (baseline - df[col]) / baseline

    # 将结果转换为 DataFrame
    np_df = pd.DataFrame(np_values)

    # 将 'SA' 放到最后并加粗显示
    if 'SA' in np_df.columns:
        cols = [col for col in np_df.columns if col != 'SA']
        np_df = np_df[cols + ['SA']]

    # 胜率统计：每轮中选取最低值对应的算法
    win_count = {col: 0 for col in df.columns if col != 'FCFS'}  # 初始化胜率统计
    for _, row in df.iterrows():
        winner = row.idxmin()  # 每轮的最小值对应的算法
        if winner != 'FCFS':  # 忽略 FCFS
            win_count[winner] += 1

    # 创建并列图的布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

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

    # 构造颜色映射表，确保颜色一致性
    color_map = {col: colors[idx % len(colors)] for idx, col in enumerate(np_df.columns)}

    # 绘制箱线图 (左侧)
    sns.boxplot(data=np_df, palette=[color_map[col] for col in np_df.columns], ax=ax1)
    for idx, col in enumerate(np_df.columns):
        mean_val = np_df[col].mean()
        ax1.axhline(y=mean_val, color=color_map[col], linestyle='--', linewidth=2)

    # 设置标题和标签字体大小
    ax1.set_title('NP of Algorithms', fontsize=18)
    #ax1.set_xlabel('Algorithms', fontsize=20)
    ax1.set_ylabel('Normalized Performance', fontsize=16)
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    ax1.tick_params(axis='y', labelsize=16)

    # 设置 X 轴标签，SA 加粗
    x_labels = np_df.columns
    x_labels = [f'$\mathbf{{{label}}}$' if label == 'SA' else label for label in x_labels]
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, fontsize=16)

    # 绘制胜率饼图 (右侧)
    total_runs = len(df)
    labels = [
        f'$\mathbf{{{label}}}$ ({win_count[label] / total_runs * 100:.1f}%)'
        if label == 'SA'
        else f'{label} ({win_count[label] / total_runs * 100:.1f}%)'
        for label in win_count if win_count[label] > 0
    ]
    sizes = [win_count[label] / total_runs for label in win_count if win_count[label] > 0]

    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=labels,
        autopct='',
        startangle=140,
        colors=[color_map[label] for label in win_count if win_count[label] > 0]
    )
    ax2.set_title('WR of Algorithms', fontsize=22)

    # 设置标签字体大小
    for text in texts:
        text.set_fontsize(20)

    # 调整布局避免重叠
    plt.tight_layout(pad=1)

    # 保存图像
    plt.savefig(plot_path_combined, format='svg')

    # 显示图像
    plt.show()



if __name__ == '__main__':
    csv_path = "checkpoints/D3QN/SA1112_1533smallD3E10C10W30L0.15_with_deadline/SequencingAgent/experiments1115_1426_RA220_SA60_SA_performance/SA_performance_data.csv"  # Replace with your actual CSV file path
    drawBoxPlot_SA(csv_path)

# if __name__ == '__main__':
#     csv_path = "checkpoints/D3QN/SA1112_1533smallD3E10C10W30L0.15_with_deadline/SequencingAgent/experiments1115_1426_RA220_SA60_SA_performance/SA_performance_data.csv"  # Replace with your actual CSV file path
#     draw_combined_boxplot_and_pie(csv_path)
