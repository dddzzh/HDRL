import time
import os
import pickle
from D3QN import D3QN
from CreateDataset import create_random_dataset
import heapq
from Event import Event, workflowArrivalEvent, jobArrivalEvent, jobStartUploadEvent, jobStartProcessEvent, \
    jobFinishedEvent, jobStartDownloadEvent
from obs import  get_obs_R,get_obs_S_with_deadline
from SequencingAction_with_deadline import FCFS, SPT,HCR, MS#,EDF
import matplotlib.pyplot as plt
from datetime import datetime
def eval_slack(env, agent_R,agent_S):
    env.reset()
    select_job = None  # 用于记录当前被选中的任务
    agent_R.q_eval.eval()
    agent_R.q_target.eval()
    agent_S.q_eval.eval()
    agent_S.q_target.eval()

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
            if server.isBusy == False and len(server.wait_list) == 1:  # 如果只有一个任务 直接选这个任务
                job = server.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue
            elif server.isBusy == False and len(server.wait_list) > 1:  # 如果有多个任务 使用sequencing decision
                job = MS(env, server)
                # 把任务开始处理发送上去
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))

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
        makespan += max(max_finish_time_job.finishTime - wf.deadline , 0)


    agent_R.q_eval.train()
    agent_R.q_target.train()
    return makespan
def eval_min_process(env, agent_R,agent_S):
    env.reset()
    select_job = None  # 用于记录当前被选中的任务
    agent_R.q_eval.eval()
    agent_R.q_target.eval()
    agent_S.q_eval.eval()
    agent_S.q_target.eval()

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
            if server.isBusy == False and len(server.wait_list) == 1:  # 如果只有一个任务 直接选这个任务
                job = server.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue
            elif server.isBusy == False and len(server.wait_list) > 1:  # 如果有多个任务 使用sequencing decision
                job = SPT(env, server)
                # 把任务开始处理发送上去
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))

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
    tardiness = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        tardiness += max(max_finish_time_job.finishTime - wf.deadline , 0)


    agent_R.q_eval.train()
    agent_R.q_target.train()
    return tardiness

def eval_FCFS(env, agent_R,agent_S):
    env.reset()
    select_job = None  # 用于记录当前被选中的任务
    agent_R.q_eval.eval()
    agent_R.q_target.eval()
    agent_S.q_eval.eval()
    agent_S.q_target.eval()

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
            if server.isBusy == False and len(server.wait_list) == 1:  # 如果只有一个任务 直接选这个任务
                job = server.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue
            elif server.isBusy == False and len(server.wait_list) > 1:  # 如果有多个任务 使用sequencing decision
                job = server.wait_list[0]
                # 把任务开始处理发送上去
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))

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
        makespan += max(max_finish_time_job.finishTime - wf.deadline , 0)


    agent_R.q_eval.train()
    agent_R.q_target.train()
    return makespan


def eval_CR(env, agent_R,agent_S):
    env.reset()
    select_job = None  # 用于记录当前被选中的任务
    agent_R.q_eval.eval()
    agent_R.q_target.eval()
    agent_S.q_eval.eval()
    agent_S.q_target.eval()

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
            if server.isBusy == False and len(server.wait_list) == 1:  # 如果只有一个任务 直接选这个任务
                job = server.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue
            elif server.isBusy == False and len(server.wait_list) > 1:  # 如果有多个任务 使用sequencing decision
                job = HCR(env, server)
                # 把任务开始处理发送上去
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))

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
        makespan += max(max_finish_time_job.finishTime - wf.deadline , 0)


    agent_R.q_eval.train()
    agent_R.q_target.train()
    return makespan
def evaluate():
    RA_model_dir = './checkpoints/D3QN/RA0911_0036smallD3E10C10W30L0.15_with_deadline/RoutingAgent/'
    SA_model_dir = 'checkpoints/D3QN/SA1017_2135smallD3E10C10W30L0.15_with_deadline_and_obs_change_性能很差/SequencingAgent/'
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
        max_size=RAConfigs.max_size, batch_size=RAConfigs.batch_size)

    agent_R.load_models(150)

    agent_S = D3QN(
        alpha=SAConfigs.alpha,  # 学习率
        state_dim=32,  # 状态空间
        action_dim=4,  # 动作空间
        fc1_dim=SAConfigs.fc1_dim, fc2_dim=SAConfigs.fc2_dim,  # 模型参数
        ckpt_dir=SA_model_dir,  # 保存地址
        gamma=SAConfigs.gamma, tau=SAConfigs.tau,  # 折扣系数
        epsilon=SAConfigs.epsilon, eps_end=SAConfigs.eps_end, eps_dec=SAConfigs.eps_dec,  # 探索率相关
        max_size=SAConfigs.max_size, batch_size=SAConfigs.batch_size)

    envs = []
    for _ in range(5):  # 非固定验证集合
        env = create_random_dataset(InsConfigs.numMobileDevices,
                                    InsConfigs.numEdgeServers,
                                    InsConfigs.numCenterServers,
                                    InsConfigs.numWorkflows,
                                    InsConfigs.lambdaRate,
                                    InsConfigs.datasetScale)
        envs.append(env)

    makespan_per_episode = []
    for episode in range(SAConfigs.save_freq, SAConfigs.max_episodes + 1, SAConfigs.save_freq):  # 50 ,1001, 50
        agent_S.load_models(episode)
        makespan_per_env = []
        for env in envs:
            env.reset()
            makespan = simulation(agent_R, agent_S, env)
            makespan_per_env.append(makespan)
        makespan_per_episode.append(makespan_per_env)

    makespan_FCFS = []
    for env in envs:
        makespan = eval_FCFS(env,agent_R,agent_S)
        makespan_FCFS.append(makespan)


    makespan_CR = []#完成率最高优先
    for env in envs:
        makespan = eval_CR(env,agent_R,agent_S)
        makespan_CR.append(makespan)

    makespan_min_process = []
    for env in envs:
        makespan = eval_min_process(env, agent_R, agent_S)
        makespan_min_process.append(makespan)


    makespan_slack = []  # 松弛
    for env in envs:
        makespan = eval_slack(env, agent_R, agent_S)
        makespan_slack.append(makespan)

    win_probability_per_episode = []
    for episode_idx, makespan_per_env in enumerate(makespan_per_episode):
        win_count = 0
        total_envs = len(envs)

        for env_idx in range(total_envs):
            if makespan_per_env[env_idx] < makespan_slack[env_idx] and makespan_per_env[env_idx] < makespan_CR[env_idx]:
                win_count += 1

        win_probability = win_count / total_envs
        win_probability_per_episode.append(win_probability)

    # 输出每个 episode 赢过 makespan_CR 的概率
    for episode, prob in enumerate(win_probability_per_episode, start=1):
        print(f"Episode {episode}: Win Probability = {prob * 100:.2f}%")

    average_makespan_per_episode = [sum(episode) / len(episode) for episode in makespan_per_episode]
    average_makespan_FCFS = sum(makespan_FCFS) / len(makespan_FCFS)
    average_makespan_CR = sum(makespan_CR) / len(makespan_CR)
    average_makespan_min_process = sum(makespan_min_process) / len(makespan_min_process)
    average_makespan_slack = sum(makespan_slack) / len(makespan_slack)

    x_values = list(range(SAConfigs.save_freq, SAConfigs.max_episodes + SAConfigs.save_freq, SAConfigs.save_freq))
    ckpt_dir = agent_S.checkpoint_dir
    plot_dir = os.path.join(ckpt_dir, 'plots')

    # 创建文件夹
    os.makedirs(plot_dir, exist_ok=True)

    # 图表文件路径
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    plot_filename = f'average_tardiness_per_episode_{timestamp}.png'
    plot_path = os.path.join(plot_dir, plot_filename)

    # 绘制折线图并保存
    plt.figure(figsize=(10, 5))
    plt.plot(x_values[:len(average_makespan_per_episode)], average_makespan_per_episode, marker='o',label='RA + SA')
    #plt.axhline(y=average_makespan_EFT, color='r', linestyle='--', label='Average Makespan EFT (Baseline)')
    plt.axhline(y=average_makespan_FCFS, color='g', linestyle='--', label='RA + FCFS')
    plt.axhline(y=average_makespan_CR, color='y', linestyle='--', label='RA + CR')
    plt.axhline(y=average_makespan_min_process, color='r', linestyle='--', label='RA + min process')
    plt.axhline(y=average_makespan_slack, color='b', linestyle='--', label='RA + min slack')
    plt.legend()
    plt.title('Average tardiness per Episode with Baseline')
    plt.xlabel('Episodes (increment of 50)')
    plt.ylabel('Average tardiness')
    plt.grid(True)

    plt.savefig(plot_path)
    plt.show()
    #plt.close()
def simulation(agent_R,agent_S, env):
    env.reset()
    select_job = None  # 用于记录当前被选中的任务
    agent_R.q_eval.eval()
    agent_R.q_target.eval()
    agent_S.q_eval.eval()
    agent_S.q_target.eval()
    a0 = 0
    a1 = 0
    a2 = 0
    a3 = 0

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
            if server.isBusy == False and len(server.wait_list) == 1:  # 如果只有一个任务 直接选这个任务
                job = server.wait_list[0]
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue
            elif server.isBusy == False and len(server.wait_list) > 1:  # 如果有多个任务 使用sequencing decision
                obs_S = get_obs_S_with_deadline(env, server)  # 观测值 只观测这一时刻的状态16
                last_s = server.last_state  # 机器保存的 上一时刻的状态 16
                input_obs = obs_S + [a - b for a, b in zip(obs_S, last_s)]  # 最终输入机器的状态 32
                action = agent_S.choose_action(input_obs)
                if action == 0:
                    job = FCFS(env, server)
                    a0+=1
                elif action == 1:
                    job = SPT(env, server)
                    a1 += 1
                elif action == 2:
                    a2 += 1
                    job = HCR(env, server)
                elif action == 3:
                    a3 += 1
                    job = MS(env, server)

                # 把任务开始处理发送上去
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))

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
    tardiness = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        tardiness += max(max_finish_time_job.finishTime - wf.deadline, 0)

    print(f'0:{a0} 1:{a1} 2:{a2} 3:{a3} ')
    agent_R.q_eval.train()
    agent_R.q_target.train()
    agent_S.q_eval.train()
    agent_S.q_target.train()
    return tardiness


if __name__ == '__main__':
    start_time = time.time()
    evaluate()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")