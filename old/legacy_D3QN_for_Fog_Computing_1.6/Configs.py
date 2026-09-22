import os
import pickle
class InstanceConfigs:
    def __init__(self,
                 numMobileDevices=3,
                 numEdgeServers=20,
                 numCenterServers=20,
                 numWorkflows=20,
                 lambdaRate=0.01,
                 datasetScale="small",):

        # 环境参数
        self.numMobileDevices = numMobileDevices
        self.numEdgeServers = numEdgeServers
        self.numCenterServers = numCenterServers
        self.numWorkflows = numWorkflows
        self.lambdaRate = lambdaRate
        self.datasetScale = datasetScale

    def print_params(self):
        print("========训练实例参数========")
        print(f"(移动设备数): {self.numMobileDevices}")
        print(f"(边缘服务器数): {self.numEdgeServers}")
        print(f"(中心服务器数): {self.numCenterServers}")
        print(f"(工作流数): {self.numWorkflows}")
        print(f"(工作流生成频率): {self.lambdaRate}")
        print(f"(数据集尺寸): {self.datasetScale}")

    def save_configs(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'instance_configs.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)



class RoutingAgentConfigs:
    def __init__(self,
                 max_episodes=1000,
                 alpha=0.0003,
                 gamma=0.99,
                 tau = 0.005,
                 epsilon = 1.0,
                 eps_end = 0.1,
                 eps_dec = 2e-4,
                 max_size = 1000000,
                 batch_size = 256,
                 fc1_dim=256,
                 fc2_dim=256,
                 save_freq = 50, #保存频率
                 training_per_episode = 10, #每episode训练次数
                 ckpt_dir='./checkpoints/D3QN/',):
        # 训练参数
        self.max_episodes = max_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.max_size = max_size
        self.batch_size = batch_size

        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.save_freq = save_freq
        self.training_per_episode = training_per_episode
        # 保存地址参数
        self.ckpt_dir = ckpt_dir

    def print_params(self):
        print("========RoutingAgent参数========")
        print(f"(最大训练轮次): {self.max_episodes}")
        print(f"(学习率): {self.alpha}")
        print(f"(折扣率): {self.gamma}")
        print(f"(目标网络软更新率): {self.tau}")
        print(f"(初始探索率): {self.epsilon}")
        print(f"(最小探索率): {self.eps_end}")
        print(f"(探索率衰减): {self.eps_dec}")
        print(f"(经验回放缓冲区最大容量): {self.max_size}")
        print(f"(批处理大小): {self.batch_size}")
        print(f"(全连接层1的尺寸): {self.fc1_dim}")
        print(f"(全连接层2的尺寸): {self.fc2_dim}")
        print(f"(模型保存频率): {self.save_freq}")
        print(f"(每集训练次数): {self.training_per_episode}")
        print(f"(检查点保存目录): {self.ckpt_dir}")
    def save_configs(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'routing_agent_configs.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

class SequencingAgentConfigs:
    def __init__(self,
                 max_episodes=1000,
                 alpha=0.0003,
                 gamma=0.99,
                 tau = 0.005,
                 epsilon = 1.0,
                 eps_end = 0.1,
                 eps_dec = 2e-4,
                 max_size = 1000000,
                 batch_size = 256,
                 fc1_dim=256,
                 fc2_dim=256,
                 save_freq = 50, #保存频率
                 training_per_episode = 10, #每episode训练次数
                 ckpt_dir='./checkpoints/D3QN/',):
        # 训练参数
        self.max_episodes = max_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.max_size = max_size
        self.batch_size = batch_size

        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.save_freq = save_freq
        self.training_per_episode = training_per_episode
        # 保存地址参数
        self.ckpt_dir = ckpt_dir

    def print_params(self):
        print("========SequencingAgent参数========")
        print(f"(最大训练轮次): {self.max_episodes}")
        print(f"(学习率): {self.alpha}")
        print(f"(折扣率): {self.gamma}")
        print(f"(目标网络软更新率): {self.tau}")
        print(f"(初始探索率): {self.epsilon}")
        print(f"(最小探索率): {self.eps_end}")
        print(f"(探索率衰减): {self.eps_dec}")
        print(f"(经验回放缓冲区最大容量): {self.max_size}")
        print(f"(批处理大小): {self.batch_size}")
        print(f"(全连接层1的尺寸): {self.fc1_dim}")
        print(f"(全连接层2的尺寸): {self.fc2_dim}")
        print(f"(模型保存频率): {self.save_freq}")
        print(f"(每集训练次数): {self.training_per_episode}")
        print(f"(检查点保存目录): {self.ckpt_dir}")
    def save_configs(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'sequencing_agent_configs.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
# class Config:
#     def __init__(self,
#                  numMobileDevices=3,
#                  numEdgeServers=20,
#                  numCenterServers=20,
#                  numWorkflows=20,
#                  lambdaRate=0.01,
#                  datasetScale="middle",



