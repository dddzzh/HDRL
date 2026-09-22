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
#先尝试使用其他调度规则和 SA结合进行对比

def Experiment_S():
    #导入模型
    RA_model_dir = './checkpoints/D3QN/RA1111_1707smallD3E10C10W30L0.15_with_deadline/RoutingAgent/'
    SA_model_dir = './checkpoints/D3QN/SA1112_1533smallD3E10C10W30L0.15_with_deadline/SequencingAgent/'
    RA_episode = 220
    SA_episode = 60

    load_path = os.path.join(SA_model_dir, 'instance_configs.pkl')
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
    ckpt_dir = agent_S.checkpoint_dir
    experiments_dir = os.path.join(ckpt_dir, f'experiments{current_time}_RA{RA_episode}_SA{SA_episode}_SA_performance')
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

    #SA
    tardiness_SA = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode = "EFT", sequencing_mode= "SA")
        tardiness_SA.append(tard)
    #先来先服务
    tardiness_FCFS = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="FCFS")
        tardiness_FCFS.append(tard)

    #最小处理时间
    tardiness_SPT = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="SPT")
        tardiness_SPT.append(tard)

    #最大完成率优先
    tardiness_HCR = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="HCR")
        tardiness_HCR.append(tard)
    #最小松弛优先
    tardiness_MS = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="MS")
        tardiness_MS.append(tard)

    tardiness_UpwardRank = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="UpwardRank")
        tardiness_UpwardRank.append(tard)

    tardiness_DownwardRank = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="DownwardRank")
        tardiness_DownwardRank.append(tard)


    tardiness_LPT = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="LPT")
        tardiness_LPT.append(tard)

    tardiness_RSR = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="RSR")
        tardiness_RSR.append(tard)

    tardiness_EASPT = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="RA", sequencing_mode="EASPT")
        tardiness_EASPT.append(tard)

    tardiness_LRW = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="LRW")
        tardiness_LRW.append(tard)

    tardiness_ATC = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="ATC")
        tardiness_ATC.append(tard)

    tardiness_Random = []
    for env in envs:
        env.reset()
        tard = simulation(agent_R, agent_S, env, routing_mode="EFT", sequencing_mode="Random")
        tardiness_Random.append(tard)

    # 使用pandas保存数据 以excel或者csv格式保存  文件名 第几代模型
    data = {
        'SA': tardiness_SA,
        'FCFS': tardiness_FCFS,
        'SPT': tardiness_SPT,
        'HCR': tardiness_HCR,
        'MS': tardiness_MS,
        'UpwardRank':tardiness_UpwardRank,
        'DownwardRank': tardiness_DownwardRank,
        'LPT':tardiness_LPT,
        'RSR':tardiness_RSR,
        'EASPT':tardiness_EASPT,
        'LRW': tardiness_LRW,
        'ATC': tardiness_ATC,
        'Random':tardiness_Random
    }

    # 将字典转换为DataFrame
    df = pd.DataFrame(data)
    csv_path = os.path.join(experiments_dir, 'SA_performance_data.csv')
    # 保存 DataFrame 为 CSV 文件
    df.to_csv(csv_path, index=False)










def drawBoxPlot(csv_path):
    #导入excel地址

    pass

    #绘制箱线图

if __name__ == '__main__':
    start_time = time.time()
    Experiment_S()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")