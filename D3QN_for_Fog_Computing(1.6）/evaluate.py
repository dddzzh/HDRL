from Params import configs
from D3QN import D3QN
from CreateDataset import create_random_dataset
import heapq
from Event import Event, workflowArrivalEvent, jobArrivalEvent, jobStartUploadEvent, jobStartProcessEvent, \
    jobFinishedEvent, jobStartDownloadEvent
from obs import  get_obs_R
import matplotlib.pyplot as plt
import os
import pickle
import time
def evaluate():

    model_dir = './checkpoints/D3QN/0822_0942smallD3E8C8W20L0.1/RoutingAgent/'
    #model_dir = './checkpoints/D3QN/0819_0956smallD3E8C8W20L0.01/RoutingAgent/'
    #导入实例和模型参数
    load_path = os.path.join(model_dir, 'instance_configs.pkl')
    with open(load_path, 'rb') as f:
        InsConfigs = pickle.load(f)
    InsConfigs.print_params()

    load_path = os.path.join(model_dir, 'routing_agent_configs.pkl')
    with open(load_path, 'rb') as f:
        RAConfigs = pickle.load(f)

    #导入模型
    agent_R = D3QN(
        alpha=RAConfigs.alpha,  # 学习率
        state_dim=5 * (InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1),  # 状态空间
        action_dim=InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1,  # 动作空间
        fc1_dim=RAConfigs.fc1_dim, fc2_dim=RAConfigs.fc2_dim,  # 模型参数
        ckpt_dir=model_dir,  # 保存地址
        gamma=RAConfigs.gamma, tau=RAConfigs.tau,  # 折扣系数
        epsilon=RAConfigs.epsilon, eps_end=RAConfigs.eps_end, eps_dec=RAConfigs.eps_dec,  # 探索率相关
        max_size=RAConfigs.max_size, batch_size=RAConfigs.batch_size)  # memory buffer大小 训练批次大小

    #数据集参数

    #生成环境集
    envs = []
    for _ in range(30):  #非固定验证集合
        env = create_random_dataset(InsConfigs.numMobileDevices,
                                     InsConfigs.numEdgeServers,
                                     InsConfigs.numCenterServers,
                                     InsConfigs.numWorkflows,
                                     InsConfigs.lambdaRate,
                                     InsConfigs.datasetScale)
        envs.append(env)

    #绘图 对所有的数据集的平均makespan进行绘图  横轴是迭代次数 纵轴是平均表现
    makespan_per_episode = []
    for episode in range(RAConfigs.save_freq, RAConfigs.max_episodes +1, RAConfigs.save_freq): #50 ,1001, 50
        agent_R.load_models(episode)
        makespan_per_env = []
        for env in envs:
            env.reset()
            makespan = simulation(agent_R, env)
            makespan_per_env.append(makespan)
        makespan_per_episode.append(makespan_per_env)

    makespan_EFT = []
    for env in envs:
        makespan = eval_EFT(env)
        makespan_EFT.append(makespan)
    makespan_FCFS = []
    # for env in envs:
    #     makespan = eval_FCFS(env)
    #     makespan_FCFS.append(makespan)




    average_makespan_per_episode = [sum(episode) / len(episode) for episode in makespan_per_episode]
    average_makespan_EFT = sum(makespan_EFT) / len(makespan_EFT)
    #average_makespan_FCFS = sum(makespan_FCFS) / len(makespan_FCFS)



    #x_values = list(range(50, 1050, 50))  #
    x_values = list(range(RAConfigs.save_freq, RAConfigs.max_episodes +RAConfigs.save_freq, RAConfigs.save_freq))
    ckpt_dir = agent_R.checkpoint_dir
    plot_dir = os.path.join(ckpt_dir, 'plots')

    # 创建文件夹
    os.makedirs(plot_dir, exist_ok=True)

    # 图表文件路径
    plot_path = os.path.join(plot_dir, 'average_makespan_per_episode.png')

    # 绘制折线图并保存
    plt.figure(figsize=(10, 5))
    plt.plot(x_values[:len(average_makespan_per_episode)], average_makespan_per_episode, marker='o',
             label='Average Makespan per Episode')
    plt.axhline(y=average_makespan_EFT, color='r', linestyle='--', label='Average Makespan EFT (Baseline)')
    #plt.axhline(y=average_makespan_FCFS, color='g', linestyle='--', label='Average Makespan FCFS (Baseline)')
    plt.legend()
    plt.title('Average Makespan per Episode with Baseline')
    plt.xlabel('Episodes (increment of 50)')
    plt.ylabel('Average Makespan')
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()


def simulation(agent_R, env):
    env.reset()
    select_job = None  # 用于记录当前被选中的任务
    agent_R.q_eval.eval()
    agent_R.q_target.eval()

    while True:
        # 1、完成当前时刻所有任务
        while env.event_list and env.event_list[0].time == env.current_time:
            event = heapq.heappop(env.event_list)
            event.trigger(env)

        # 2、处理当前机器
        for device in env.devices:
            if device.isBusy == False and device.wait_list != []:
                job = device.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, device,
                                                  env.current_time))

        # 3、处理当前时刻进行的 Sequencing Decision
        for server in env.servers:
            if server.isBusy == False and server.wait_list != []:  # 如果有存在机器不忙 且等待列表不为空的情况
                job = server.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue

        # 4、处理当前时刻待进行的 Routing Decision
        if env.ready_job_set:
            select_job = min(env.ready_job_set, key=lambda job: job.workload)
            obs_R = get_obs_R(env, select_job)
            action = agent_R.choose_action(obs_R, isTrain=False)

            if action == len(env.servers):  # 表示本地卸载
                env.addEvent(
                    jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job,
                                        env.devices[select_job.sourceDeviceId],
                                        env.current_time))
            else:
                env.addEvent(
                    jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action],
                                        env.current_time))

            select_job = None

            continue

        if env.event_list == []:
            check = True
            for wf in env.workflows:
                for job in wf.jobs:
                    if job.state != "已完成":
                        print(f"工作流{wf.id} 任务{job.id}未完成")
                        check = False
            print(check)
            break

            # 5、更新时间
        if env.current_time != env.event_list[0].time:
            env.current_time = env.event_list[0].time

    # 计算所有工作流的完成时长之和
    makespan = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        makespan += max_finish_time_job.finishTime - wf.arrivalTime


    agent_R.q_eval.train()
    agent_R.q_target.train()
    return makespan


def eval_EFT(env):
    env.reset()
    while True:
        # 1、完成当前时刻所有任务
        while env.event_list and env.event_list[0].time == env.current_time:
            event = heapq.heappop(env.event_list)
            event.trigger(env)
        # 2、处理当前机器
        for device in env.devices:
            if device.isBusy == False and device.wait_list != []:
                job = device.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, device,
                                                  env.current_time))
        # 4、处理当前时刻进行的 Sequencing Decision
        for server in env.servers:
            if server.isBusy == False and server.wait_list != []:  # 如果有存在机器不忙 且等待列表不为空的情况
                job = server.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue
        # 3、处理当前时刻待进行的 Routing Decision
        if env.ready_job_set:
            select_job = min(env.ready_job_set, key=lambda job: job.workload)

            EarliestFinishTime = []
            for server in env.servers:
                jobs = server.trans_list + server.wait_list
                total_workload = sum(job.workload for job in jobs)
                Allpt = total_workload / server.pa
                if server.isBusy == True:
                    EAT = server.processing_job.endProcessingTime - env.current_time
                else:
                    EAT = 0
                time_of_this_job = (select_job.data_up+select_job.data_down) / server.bandwidth + select_job.workload / server.pa

                EarliestFinishTime.append(Allpt + EAT+time_of_this_job)
            device = env.devices[select_job.sourceDeviceId]
            if device.isBusy == True:
                EAT = device.processing_job.endProcessingTime - env.current_time
            else:
                EAT = 0

            total_workload = sum(job.workload for job in device.wait_list)
            Allpt = total_workload / device.pa
            time_of_this_job = select_job.workload / device.pa
            EarliestFinishTime.append(Allpt + EAT + time_of_this_job)

            action = EarliestFinishTime.index(min(EarliestFinishTime))

            if action == len(env.servers):  # 表示本地卸载
                env.addEvent(
                    jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job,
                                        env.devices[select_job.sourceDeviceId],
                                        env.current_time))
            else:
                env.addEvent(
                    jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action],
                                        env.current_time))

            continue
        if env.event_list == []:
            check = True
            for wf in env.workflows:
                for job in wf.jobs:
                    if job.state != "已完成":
                        print(f"工作流{wf.id} 任务{job.id}未完成")
                        check = False
            print(check)
            break

            # 5、更新时间
        if env.current_time != env.event_list[0].time:
            env.current_time = env.event_list[0].time
    makespan = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        makespan += max_finish_time_job.finishTime - wf.arrivalTime
    return makespan
    #print('最早完成优先makespan:{}'.format(makespan))


def eval_FCFS(env):
    env.reset()
    while True:
        # 1、完成当前时刻所有任务
        while env.event_list and env.event_list[0].time == env.current_time:
            event = heapq.heappop(env.event_list)
            event.trigger(env)
        # 2、处理当前机器
        for device in env.devices:
            if device.isBusy == False and device.wait_list != []:
                job = device.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, device,
                                                  env.current_time))
        # 4、处理当前时刻进行的 Sequencing Decision
        for server in env.servers:
            if server.isBusy == False and server.wait_list != []:  # 如果有存在机器不忙 且等待列表不为空的情况
                job = server.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue
        # 3、处理当前时刻待进行的 Routing Decision
        if env.ready_job_set:
            select_job = min(env.ready_job_set, key=lambda job: job.workload)

            FirstComeFirstService = []
            for server in env.servers:
                if server.isBusy == True:
                    idle_time = server.processing_job.endProcessingTime - env.current_time

                else :
                    idle_time = 0
                FirstComeFirstService.append(idle_time)

            device = env.devices[select_job.sourceDeviceId]
            if device.isBusy == True:
                idle_time = device.processing_job.endProcessingTime - env.current_time
            else:
                idle_time = 0
            FirstComeFirstService.append(idle_time)
            action = FirstComeFirstService.index(min(FirstComeFirstService))

            if action == len(env.servers):  # 表示本地卸载
                env.addEvent(
                    jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job,
                                        env.devices[select_job.sourceDeviceId],
                                        env.current_time))
            else:
                env.addEvent(
                    jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action],
                                        env.current_time))

            continue
        if env.event_list == []:
            check = True
            for wf in env.workflows:
                for job in wf.jobs:
                    if job.state != "已完成":
                        print(f"工作流{wf.id} 任务{job.id}未完成")
                        check = False
            print(check)
            break

            # 5、更新时间
        if env.current_time != env.event_list[0].time:
            env.current_time = env.event_list[0].time
    makespan = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        makespan += max_finish_time_job.finishTime - wf.arrivalTime
    return makespan

if __name__ == '__main__':
    start_time = time.time()
    evaluate()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")