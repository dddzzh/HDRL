import os
import time
import heapq
from Event import Event, workflowArrivalEvent, jobArrivalEvent, jobStartUploadEvent, jobStartProcessEvent, \
    jobFinishedEvent, jobStartDownloadEvent
from Configs import InstanceConfigs, SequencingAgentConfigs, RoutingAgentConfigs
from CreateDataset import create_random_dataset
from datetime import datetime
from utils import create_directory
from D3QN import D3QN
from obs import get_obs_S_with_deadline,get_obs_R
from SequencingAction_with_deadline import FCFS,SPT,HCR,MS#,EDF
from reward import get_reward_S_with_deadline
import random
def eval_slack(agent_R, env):
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

        # 4、处理当前时刻进行的 Sequencing Decision
        for server in env.servers:
            if server.isBusy == False and server.wait_list != []:  # 如果有存在机器不忙 且等待列表不为空的情况
                job = MS(env, server)
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue

        # 3、处理当前时刻待进行的 Routing Decision
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
    print('slack:{}'.format(tardiness))

    agent_R.q_eval.train()
    agent_R.q_target.train()
def eval_Complete_Rate(agent_R, env):
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

        # 4、处理当前时刻进行的 Sequencing Decision
        for server in env.servers:
            if server.isBusy == False and server.wait_list != []:  # 如果有存在机器不忙 且等待列表不为空的情况
                job = HCR(env, server)
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue

        # 3、处理当前时刻待进行的 Routing Decision
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
    print('完成率最高优先:{}'.format(tardiness))

    agent_R.q_eval.train()
    agent_R.q_target.train()

def eval_agent_S(agent_R, agent_S, env):
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

        # 4、处理当前时刻进行的 Sequencing Decision
        for server in env.servers:
            if server.isBusy == False and server.wait_list != []:  # 如果有存在机器不忙 且等待列表不为空的情况
                obs = get_obs_S_with_deadline(env,server)
                action = agent_S.choose_action(obs, isTrain = False)
                if action == 0:
                    job = FCFS(env, server)
                    a0 += 1
                elif action == 1:
                    job = SPT(env, server)
                    a1 += 1
                elif action == 2:
                    job = HCR(env, server)
                    a2 += 1
                elif action == 3:
                    job = MS(env, server)
                    a3 += 1

                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue

        # 3、处理当前时刻待进行的 Routing Decision
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
    print('Agent_S_tardiness:{}'.format(tardiness))
    print(f'0:{a0} 1:{a1} 2:{a2} 3:{a3}')
    agent_R.q_eval.train()
    agent_R.q_target.train()
    agent_S.q_eval.train()
    agent_S.q_target.train()

def eval_fcfs(agent_R, env):
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
    print('fcfs_tardiness:{}'.format(tardiness))

    agent_R.q_eval.train()
    agent_R.q_target.train()
def collect_sequencing_experience(agent_S, env):
    for server in env.servers:
        new_memory = []  # 用于存储新的经验列表

        # 遍历每个server的memory
        for i in range(len(server.memory)):
            state, action, reward = server.memory[i]

            # 检查是否是最后一条经验
            if i < len(server.memory) - 1:
                next_state = server.memory[i + 1][0]  # 获取下一时刻的状态
                done = False
            else:
                next_state = state  #如果标记为完成 那么这条经验的下一时刻状态不会被用到 这里以当前时刻状态暂替
                done = True  # 标记为完成

            #保存经验
            agent_S.remember(state, action, reward, next_state, done)

def collect_sequencing_experience_changed(agent_S, env):
    for server in env.servers:
        new_memory = []  # 用于存储新的经验列表
        last_state = [0] * 16
        # 遍历每个server的memory
        for i in range(len(server.memory)):
            state, action, reward = server.memory[i]

            # 检查是否是最后一条经验
            if i < len(server.memory) - 1:
                next_state = server.memory[i + 1][0]  # 获取下一时刻的状态
                done = False
            else:
                next_state = state  #如果标记为完成 那么这条经验的下一时刻状态不会被用到 这里以当前时刻状态暂替
                done = True  # 标记为完成

            #diff_list = [a - b for a, b in zip(list1, list2)]
            real_state = state + [a - b for a, b in zip(state, last_state)]
            real_next_state = next_state + [a - b for a, b in zip(next_state, state)]

            #保存经验
            agent_S.remember(real_state, action, reward, real_next_state, done)
            last_state = state
def main(InsConfigs, RAConfigs, SAConfigs):
    eval_env = create_random_dataset(InsConfigs.numMobileDevices,
                                     InsConfigs.numEdgeServers,
                                     InsConfigs.numCenterServers,
                                     InsConfigs.numWorkflows,
                                     InsConfigs.lambdaRate,
                                     InsConfigs.datasetScale)

    ## 创建保存模型的文件夹 model_dir为最终地址
    current_time = datetime.now().strftime("%m%d_%H%M")
    folder_name = f"SA{current_time}{InsConfigs.datasetScale}D{InsConfigs.numMobileDevices}E{InsConfigs.numEdgeServers}C{InsConfigs.numCenterServers}W{InsConfigs.numWorkflows}L{InsConfigs.lambdaRate}_with_deadline/"
    model_dir = os.path.join(SAConfigs.ckpt_dir, folder_name)
    model_dir = os.path.join(model_dir, "SequencingAgent/")
    create_directory(model_dir, sub_dirs=['Q_eval', 'Q_target'])

    # 打印、保存参数文件
    InsConfigs.print_params()
    InsConfigs.save_configs(model_dir)

    SAConfigs.print_params()
    SAConfigs.save_configs(model_dir)

    # 创建模型
    agent_R = D3QN(
        alpha=RAConfigs.alpha,  # 学习率
        state_dim=5 * (InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1),  # 状态空间
        action_dim=InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1,  # 动作空间
        fc1_dim=RAConfigs.fc1_dim, fc2_dim=RAConfigs.fc2_dim,  # 模型参数
        ckpt_dir='./checkpoints/D3QN/RA1111_1748smallD4E15C15W45L0.15_with_deadline/RoutingAgent/',  # 保存地址
        gamma=RAConfigs.gamma, tau=RAConfigs.tau,  # 折扣系数
        epsilon=RAConfigs.epsilon, eps_end=RAConfigs.eps_end, eps_dec=RAConfigs.eps_dec,  # 探索率相关
        max_size=RAConfigs.max_size, batch_size=RAConfigs.batch_size)  # memory buffer大小 训练批次大小
    ##RA模型暂时使用预训练导入的模型

    agent_R.load_models(330)


    agent_S = D3QN(
        alpha=SAConfigs.alpha,  # 学习率
        state_dim=16 ,  # 状态空间  原本16
        action_dim= 4  ,  # 动作空间
        fc1_dim=SAConfigs.fc1_dim, fc2_dim=SAConfigs.fc2_dim,  # 模型参数
        ckpt_dir=model_dir,  # 保存地址
        gamma=SAConfigs.gamma, tau=SAConfigs.tau,  # 折扣系数
        epsilon=SAConfigs.epsilon, eps_end=SAConfigs.eps_end, eps_dec=SAConfigs.eps_dec,  # 探索率相关
        max_size=SAConfigs.max_size, batch_size=SAConfigs.batch_size)  # memory buffer大小 训练批次大小

    for episode in range(SAConfigs.max_episodes):
        env = create_random_dataset(InsConfigs.numMobileDevices,
                                    InsConfigs.numEdgeServers,
                                    InsConfigs.numCenterServers,
                                    InsConfigs.numWorkflows,
                                    InsConfigs.lambdaRate,
                                    InsConfigs.datasetScale)
        total_reward_SA = 0
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
                    obs_S = get_obs_S_with_deadline(env, server)
                    action = agent_S.choose_action(obs_S)
                    if action == 0:
                        job = FCFS(env, server)
                    elif action == 1:
                        job = SPT(env, server)
                    elif action == 2:
                        job = HCR(env, server)
                    elif action == 3:
                        job = MS(env, server)

                    reward = get_reward_S_with_deadline(env, job, server)
                    total_reward_SA += reward
                    # print(f"action: {action}  reward:{reward}")
                    # 收集经验
                    exp = []
                    exp.append(obs_S)
                    exp.append(action)
                    exp.append(reward)
                    server.memory.append(exp)

                    # 把任务开始处理发送上去
                    env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                      env.current_time))

                    continue
            # 4、处理当前时刻待进行的 Routing Decision
            if env.ready_job_set:
                select_job = min(env.ready_job_set, key=lambda job: job.workload)
                obs_R = get_obs_R(env, select_job)
                action = agent_R.choose_action(obs_R)

                if action == len(env.servers):  # 表示本地卸载
                    env.addEvent(
                        jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job,
                                            env.devices[select_job.sourceDeviceId],
                                            env.current_time))
                else:  # 卸载到对应服务器上
                    env.addEvent(
                        jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action],
                                            env.current_time))

                select_job = None
                continue
            # 5、检验是否完成
            if env.event_list == []:
                check = True
                for wf in env.workflows:
                    for job in wf.jobs:
                        if job.state != "已完成":
                            print(f"工作流{wf.id} 任务{job.id}未完成")
                            check = False
                break

            # 6、更新时间
            if env.current_time != env.event_list[0].time:
                env.current_time = env.event_list[0].time
        # 解析并保存经验
        collect_sequencing_experience(agent_S, env)
        #collect_sequencing_experience_changed(agent_S, env)
        # 训练模型
        for i in range(SAConfigs.training_per_episode):
            agent_S.learn()

        tardiness = 0
        for wf in env.workflows:
            max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
            tardiness += max(max_finish_time_job.finishTime - wf.deadline, 0)

        print('EP:{} Reward:{} Epsilon:{} tardiness:{}'.format(episode + 1, total_reward_SA, agent_S.epsilon, tardiness))


        #eval_fcfs(agent_R,eval_env )
        #eval_agent_S(agent_R, agent_S, eval_env)

        #eval_Complete_Rate(agent_R, eval_env)
        #eval_slack(agent_R,eval_env)

        if (episode + 1) % SAConfigs.save_freq == 0:
            agent_S.save_models(episode + 1)

if __name__ == '__main__':
    start_time = time.time()
    InsConfigs = InstanceConfigs(
        numMobileDevices=4,
        numEdgeServers=15,
        numCenterServers=15,
        numWorkflows=45,
        lambdaRate=0.15,
        datasetScale="small",
    )
    RAConfigs = RoutingAgentConfigs(
        max_episodes=1000,
        alpha=0.0003,
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        eps_end=0.1,
        eps_dec=5e-4,
        max_size=1000000,
        batch_size=256,
        fc1_dim=256,
        fc2_dim=256,
        save_freq=50,
        training_per_episode=10,
        ckpt_dir='./checkpoints/D3QN/',
    )
    SAConfigs = SequencingAgentConfigs(
        max_episodes=200,
        alpha=0.0003,
        gamma=0.99,
        tau=0.005,
        epsilon=1,
        eps_end=0.1,
        eps_dec=5e-3, #5e-3
        max_size=1000000,
        batch_size=256,
        fc1_dim=256,
        fc2_dim=256,
        save_freq=10,
        training_per_episode=1, #注意 这里改了 原为1
        ckpt_dir='./checkpoints/D3QN/',
    )

    main(InsConfigs, RAConfigs, SAConfigs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")
