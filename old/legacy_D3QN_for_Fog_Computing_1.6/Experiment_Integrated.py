import time
import operator
import os
import pickle
from D3QN import  D3QN
from deap import gp
from datetime import datetime
from CreateDataset import create_random_dataset
from Simulator import simulation
import pandas as pd
#给GP用的

def Experiment_Integrate():
    RA_model_dir = './checkpoints/D3QN/RA1111_1654smallD4E6C6W20L0.15_with_deadline/RoutingAgent/'
    SA_model_dir = './checkpoints/D3QN/SA1111_2310smallD4E6C6W20L0.15_with_deadline/SequencingAgent/'
    RA_episode = 140 #150
    SA_episode = 60 #110
    GP_dir = 'GP_rules/exp_3_M4E6C6W20L015Ssmall/exp_20241111_0346/best_ind_gen_50.pkl'

    load_path = os.path.join(RA_model_dir, 'instance_configs.pkl')
    with open(load_path, 'rb') as f:
        InsConfigs = pickle.load(f)
    InsConfigs.print_params()

    load_path = os.path.join(RA_model_dir, 'routing_agent_configs.pkl')
    with open(load_path, 'rb') as f:
        RAConfigs = pickle.load(f)

    load_path = os.path.join(SA_model_dir, 'sequencing_agent_configs.pkl')
    with open(load_path, 'rb') as f:
        SAConfigs = pickle.load(f)

    #读取GP的文件  设置GP相关
    with open(GP_dir, 'rb') as file:
        GP_individual =  pickle.load(file)

    GP_func_RA = gp.compile(GP_individual[0], pset = pset)
    GP_func_SA = gp.compile(GP_individual[1], pset = pset)

    agent_R = D3QN(
        alpha=RAConfigs.alpha,  # 学习率
        state_dim=5 * (InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1),  # 状态空间
        action_dim=InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1,  # 动作空间
        fc1_dim=RAConfigs.fc1_dim, fc2_dim=RAConfigs.fc2_dim,  # 模型参数
        ckpt_dir=RA_model_dir,  # 保存地址
        gamma=RAConfigs.gamma, tau=RAConfigs.tau,  # 折扣系数
        epsilon=RAConfigs.epsilon, eps_end=RAConfigs.eps_end, eps_dec=RAConfigs.eps_dec,  # 探索率相关
        max_size=RAConfigs.max_size, batch_size=RAConfigs.batch_size)  #

    agent_R.load_models(RA_episode)

    agent_S = D3QN(
        alpha=SAConfigs.alpha,  # 学习率
        state_dim=16,  # 状态空间
        action_dim=4,  # 动作空间
        fc1_dim=SAConfigs.fc1_dim, fc2_dim=SAConfigs.fc2_dim,  # 模型参数
        ckpt_dir=SA_model_dir,  # 保存地址
        gamma=SAConfigs.gamma, tau=SAConfigs.tau,  # 折扣系数
        epsilon=SAConfigs.epsilon, eps_end=SAConfigs.eps_end, eps_dec=SAConfigs.eps_dec,  # 探索率相关
        max_size=SAConfigs.max_size, batch_size=SAConfigs.batch_size)

    agent_S.load_models(SA_episode)

    # 创建文件夹
    current_time = datetime.now().strftime('%m%d_%H%M')
    ckpt_dir = agent_R.checkpoint_dir
    experiments_dir = os.path.join(ckpt_dir, f'experiments{current_time}_RA{RA_episode}_SA{SA_episode}_integrated(new)')
    os.makedirs(experiments_dir, exist_ok=True)

    #生成并保存（或者导入）数据集 100个实例
    envs = []
    for _ in range(30):  # 非固定验证集合
        env = create_random_dataset(InsConfigs.numMobileDevices,
                                    InsConfigs.numEdgeServers,
                                    InsConfigs.numCenterServers,
                                    InsConfigs.numWorkflows,
                                    InsConfigs.lambdaRate,
                                    InsConfigs.datasetScale)
        envs.append(env)

    tardiness_RA_SA = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="RA", sequencing_mode="SA")
        tardiness_RA_SA.append(tard)

    tardiness_PT_FCFS = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="PT", sequencing_mode="FCFS")
        tardiness_PT_FCFS.append(tard)

    tardiness_EFT_HCR = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="HCR")
        tardiness_EFT_HCR.append(tard)

    tardiness_EFT_MS = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="MS")
        tardiness_EFT_MS.append(tard)

    tardiness_EFT_EASPT = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="EASPT")
        tardiness_EFT_EASPT.append(tard)

    tardiness_EFT_LRW = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="LRW")
        tardiness_EFT_LRW.append(tard)

    tardiness_EA_HCR = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EA", sequencing_mode="HCR")
        tardiness_EA_HCR.append(tard)

    tardiness_EA_MS = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EA", sequencing_mode="MS")
        tardiness_EA_MS.append(tard)

    tardiness_EA_EASPT = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EA", sequencing_mode="EASPT")
        tardiness_EA_EASPT.append(tard)

    tardiness_EA_LRW = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EA", sequencing_mode="LRW")
        tardiness_EA_LRW.append(tard)

    tardiness_RA_alone = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="RA", sequencing_mode="FCFS")
        tardiness_RA_alone.append(tard)

    tardiness_SA_alone = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="PT", sequencing_mode="SA")
        tardiness_SA_alone.append(tard)

    tardiness_MTGP = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="MTGP", sequencing_mode="MTGP", GP_func_RA=GP_func_RA, GP_func_SA=GP_func_SA)
        tardiness_MTGP.append(tard)

    data = {
        'RA_SA': tardiness_RA_SA,
        'PT_FCFS': tardiness_PT_FCFS,
        'EFT_HCR':tardiness_EFT_HCR,
        'EFT_MS': tardiness_EFT_MS,
        'EFT_EASPT': tardiness_EFT_EASPT,
        'EFT_LRW': tardiness_EFT_LRW,
        'EA_HCR': tardiness_EA_HCR,
        'EA_MS': tardiness_EA_MS,
        'EA_EASPT': tardiness_EA_EASPT,
        'EA_LRW': tardiness_EA_LRW,
        'RA_alone':tardiness_RA_alone,
        'SA_alone':tardiness_SA_alone,
        'MTGP': tardiness_MTGP
    }

    # 将字典转换为DataFrame
    df = pd.DataFrame(data)
    csv_path = os.path.join(experiments_dir, 'algorithm_comparison.csv')
    df.to_csv(csv_path, index=False)
class CustomPrimitiveSet(gp.PrimitiveSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_terminal_ratio = 0.1  # 自定义终端比例

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        total = float(self.terms_count + self.prims_count)
        if total == 0:
            return 0
        return self._custom_terminal_ratio

    @terminalRatio.setter
    def terminalRatio(self, value):
        """Set the custom terminal ratio."""
        self._custom_terminal_ratio = value

def protected_divide(x, y):
    return x / y if y != 0 else 1  # 选择适当的默认值或处理方式

pset = CustomPrimitiveSet("MAIN", 10)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_divide, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
pset.renameArguments(ARG0='NIQ', ARG1='WIQ', ARG2='MRT', ARG3='UT',
                     ARG4='DT', ARG5='PT', ARG6='TTIQ', ARG7='TIS',
                     ARG8='TWT', ARG9='NTR')

if __name__ == '__main__':
    start_time = time.time()
    Experiment_Integrate()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")