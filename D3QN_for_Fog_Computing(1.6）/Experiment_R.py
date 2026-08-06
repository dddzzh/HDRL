#对RA进行实验 结果保存在RA文件夹之下即可 保存一下实验使用的SA的地址
import time
from datetime import datetime
import os
import pickle
from D3QN import D3QN
from CreateDataset import create_random_dataset
import heapq
from Event import Event, workflowArrivalEvent, jobArrivalEvent, jobStartUploadEvent, jobStartProcessEvent, \
    jobFinishedEvent, jobStartDownloadEvent
from obs import  get_obs_R,get_obs_S_with_deadline
from SequencingAction_with_deadline import FCFS,SPT,HCR,MS
import matplotlib.pyplot as plt
import pandas as pd


#对SA进行实验 结果保存在SA文件夹之下即可
import time
from datetime import datetime
import os
import pickle
from D3QN import D3QN
from CreateDataset import create_random_dataset
import heapq
from Event import Event, workflowArrivalEvent, jobArrivalEvent, jobStartUploadEvent, jobStartProcessEvent, \
    jobFinishedEvent, jobStartDownloadEvent
from obs import  get_obs_R,get_obs_S_with_deadline
from SequencingAction_with_deadline import FCFS,SPT,HCR,MS
import matplotlib.pyplot as plt
import pandas as pd
from Simulator import simulation
def Experiment_R():

    RA_model_dir = './checkpoints/D3QN/RA1111_1748smallD4E15C15W45L0.15_with_deadline/RoutingAgent/'
    SA_model_dir = './checkpoints/D3QN/SA1112_1642smallD4E15C15W45L0.15_with_deadline/SequencingAgent/'
    RA_episode = 330
    SA_episode = 60
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


    agent_R = D3QN(
        alpha=RAConfigs.alpha,  # 学习率
        state_dim=5 * (InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1),  # 状态空间
        action_dim=InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1,  # 动作空间
        fc1_dim=RAConfigs.fc1_dim, fc2_dim=RAConfigs.fc2_dim,  # 模型参数
        ckpt_dir=RA_model_dir,    # 保存地址
        gamma=RAConfigs.gamma, tau=RAConfigs.tau,  # 折扣系数
        epsilon=RAConfigs.epsilon, eps_end=RAConfigs.eps_end, eps_dec=RAConfigs.eps_dec,  # 探索率相关
        max_size=RAConfigs.max_size, batch_size=RAConfigs.batch_size) #

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

    #创建文件夹
    current_time = datetime.now().strftime('%m%d_%H%M')
    ckpt_dir = agent_R.checkpoint_dir
    experiments_dir = os.path.join(ckpt_dir, f'experiments{current_time}_RA{RA_episode}_SA{SA_episode}')
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

    tardiness_RA = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode = "RA", sequencing_mode= "FCFS")
        tardiness_RA.append(tard)

    tardiness_EFT = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="FCFS")
        tardiness_EFT.append(tard)

    tardiness_PT = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="PT", sequencing_mode="FCFS")
        tardiness_PT.append(tard)

    tardiness_WT = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="WT", sequencing_mode="FCFS")
        tardiness_WT.append(tard)

    tardiness_UR = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="UR", sequencing_mode="FCFS")
        tardiness_UR.append(tard)

    tardiness_SQ = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="SQ", sequencing_mode="FCFS")
        tardiness_SQ.append(tard)

    tardiness_EA = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EA", sequencing_mode="FCFS")
        tardiness_EA.append(tard)

    # tardiness_GP = []
    # for env in envs:
    #     env.reset()
    #     tard = simulation(agent_R, agent_S, env, routing_mode="MTGP", sequencing_mode="SA")
    #     tardiness_GP.append(tard)

    # tardiness_Random = []
    # for env in envs:
    #     env.reset()
    #     tard = simulation(agent_R, agent_S, env, routing_mode="Random", sequencing_mode="SA")
    #     tardiness_Random.append(tard)


# 使用pandas保存数据 以excel或者csv格式保存  文件名 第几代模型
    data = {
        'RA': tardiness_RA,
        'EFT': tardiness_EFT,
        'PT': tardiness_PT,
        'WT': tardiness_WT,
        'UR': tardiness_UR,
        'SQ': tardiness_SQ,
        'EA': tardiness_EA,
        #'GP':tardiness_GP,
        #'Random':tardiness_Random
    }

    # 将字典转换为DataFrame
    df = pd.DataFrame(data)
    csv_path = os.path.join(experiments_dir, 'RA_utility_data.csv')
    # 保存 DataFrame 为 CSV 文件
    df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    start_time = time.time()
    Experiment_R()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")