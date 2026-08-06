import argparse

parser = argparse.ArgumentParser()


# 环境参数
parser.add_argument('--numMobileDevices', type=int, default=3, help='移动设备数')
parser.add_argument('--numEdgeServers', type=int, default=20, help='边缘服务器数')
parser.add_argument('--numCenterServers', type=int, default=20, help='中心服务器数')
parser.add_argument('--numWorkflows', type=int, default=20, help='工作流数')
parser.add_argument('--lambdaRate', type=float, default=0.01, help='工作流生成频率')
parser.add_argument('--datasetScale', type=str, default="small", help='数据集尺寸')


#训练参数
parser.add_argument('--max_episodes', type=int, default=1000, help='轮次数')
parser.add_argument('--alpha', type=float, default=0.0003, help='学习率')
parser.add_argument('--gamma', type=float, default=0.99, help='折扣率')
parser.add_argument('--fc1_dim', type=int, default=256, help='全连接层1尺寸')
parser.add_argument('--fc2_dim', type=int, default=256, help='全连接层2尺寸')

#保存参数
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/D3QN/')
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')

configs = parser.parse_args()