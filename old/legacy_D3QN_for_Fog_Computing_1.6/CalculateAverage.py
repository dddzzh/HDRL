
import pandas as pd


def calculate_algorithm_means(csv_paths):
    # 初始化一个字典，用于存储每个算法的平均值
    algorithm_means = {}

    # 遍历每个 CSV 文件
    for csv_path in csv_paths:
        # 读取数据
        df = pd.read_csv(csv_path)

        # 计算每个算法（列）的平均值，排除 PT_FCFS 列
        for col in df.columns:
            if col != 'PT_FCFS':  # 排除基准列 PT_FCFS
                mean_val = df[col].mean()
                if col not in algorithm_means:
                    algorithm_means[col] = []
                algorithm_means[col].append(mean_val)

    # 打印结果
    for algorithm, means in algorithm_means.items():
        print(f"Algorithm: {algorithm}, Means: {means}")

    return algorithm_means


if __name__ == '__main__':
    # CSV 文件路径列表

    #small
    #csv_path_1 ="checkpoints/D3QN/RA1111_1644smallD2E6C6W20L0.15_with_deadline/RoutingAgent/experiments1119_2113_RA150_SA80_integrated(new)/algorithm_comparison.csv"
    #csv_path_2 ="checkpoints/D3QN/RA1111_1649smallD3E6C6W20L0.15_with_deadline/RoutingAgent/experiments1119_2317_RA160_SA110_integrated(new)/algorithm_comparison.csv"
    #csv_path_3 ="checkpoints/D3QN/RA1111_1654smallD4E6C6W20L0.15_with_deadline/RoutingAgent/experiments1120_1545_RA140_SA60_integrated(new)/algorithm_comparison.csv"
    #middle
    # csv_path_1 = "checkpoints/D3QN/RA1111_1700smallD2E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2129_RA270_SA90_integrated(new)/algorithm_comparison.csv"
    # csv_path_2 = "checkpoints/D3QN/RA1111_1707smallD3E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2135_RA220_SA60_integrated(new)/algorithm_comparison.csv"
    # csv_path_3 = "checkpoints/D3QN/RA1111_1715smallD4E10C10W30L0.15_with_deadline/RoutingAgent/experiments1119_2140_RA210_SA50_integrated(new)/algorithm_comparison.csv"


    #large
    csv_path_1 = "checkpoints/D3QN/RA1111_1723smallD2E15C15W45L0.15_with_deadline/RoutingAgent/experiments1119_2145_RA360_SA90_integrated(new)/algorithm_comparison.csv"
    csv_path_2 = "checkpoints/D3QN/RA1111_1735smallD3E15C15W45L0.15_with_deadline/RoutingAgent/experiments1119_2151_RA310_SA90_integrated(new)/algorithm_comparison.csv"
    csv_path_3 ="checkpoints/D3QN/RA1111_1748smallD4E15C15W45L0.15_with_deadline/RoutingAgent/experiments1119_2228_RA330_SA120_integrated(new)/algorithm_comparison.csv"
    # 将 CSV 路径放入列表中
    csv_paths = [csv_path_1, csv_path_2, csv_path_3]

    # 调用函数并打印每个算法的平均值
    algorithm_means = calculate_algorithm_means(csv_paths)