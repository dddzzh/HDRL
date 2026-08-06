import random

from D3QN import D3QN
#from Params import configs
from CreateDataset import create_random_dataset
import heapq
from Event import Event, workflowArrivalEvent, jobArrivalEvent, jobStartUploadEvent, jobStartProcessEvent, \
    jobFinishedEvent, jobStartDownloadEvent
from utils import create_directory
from obs import get_obs_R
from datetime import datetime
import os
import time
from Configs import InstanceConfigs, RoutingAgentConfigs


def collect_experience(agent, env):
    for wf in env.workflows:
        for job in wf.jobs:
            agent.remember(job.S, job.A, job.R, job.next_S, job.terminal)


def eval_dataset(env):
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
                #jobs = server.trans_list + server.wait_list
                jobs = server.wait_list
                total_workload = sum(job.workload for job in jobs)
                Allpt = total_workload / server.pa
                if server.isBusy == True:
                    EAT = server.processing_job.endProcessingTime - env.current_time
                else:
                    EAT = 0
                time_of_this_job = (select_job.data_up + select_job.data_down) / server.bandwidth + select_job.workload / server.pa
                EarliestFinishTime.append(Allpt + EAT + time_of_this_job)
                #EarliestFinishTime.append(Allpt)
            device = env.devices[select_job.sourceDeviceId]
            if device.isBusy == True:
                EAT = device.processing_job.endProcessingTime - env.current_time
            else:
                EAT = 0

            total_workload = sum(job.workload for job in device.wait_list)
            Allpt = total_workload / device.pa
            time_of_this_job = select_job.workload / device.pa
            EarliestFinishTime.append(Allpt + EAT + time_of_this_job)
            #EarliestFinishTime.append(Allpt)
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
    tardiness = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        makespan += max_finish_time_job.finishTime - wf.arrivalTime
        tardiness += max(max_finish_time_job.finishTime - wf.deadline, 0)
    print(f"EFT_makespan为{makespan}, tardiness为{tardiness}")


def eval_agent(agent_R, env):
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

    makespan = 0
    tardiness = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        makespan += max_finish_time_job.finishTime - wf.arrivalTime
        tardiness += max(max_finish_time_job.finishTime - wf.deadline, 0)
    print(f"Agent_makespan为{makespan}, tardiness为{tardiness}")

    agent_R.q_eval.train()
    agent_R.q_target.train()


def main(InsConfigs, RAConfigs):

    # 用于验证的数据集
    eval_env = create_random_dataset(InsConfigs.numMobileDevices,
                                     InsConfigs.numEdgeServers,
                                     InsConfigs.numCenterServers,
                                     InsConfigs.numWorkflows,
                                     InsConfigs.lambdaRate,
                                     InsConfigs.datasetScale)
    ## 创建保存模型的文件夹 model_dir为最终地址
    current_time = datetime.now().strftime("%m%d_%H%M")
    folder_name = f"RA{current_time}{InsConfigs.datasetScale}D{InsConfigs.numMobileDevices}E{InsConfigs.numEdgeServers}C{InsConfigs.numCenterServers}W{InsConfigs.numWorkflows}L{InsConfigs.lambdaRate}_with_deadline/"
    model_dir = os.path.join(RAConfigs.ckpt_dir, folder_name)
    model_dir = os.path.join(model_dir, "RoutingAgent/")
    create_directory(model_dir, sub_dirs=['Q_eval', 'Q_target'])

    #打印、保存参数文件
    InsConfigs.print_params()
    InsConfigs.save_configs(model_dir)

    RAConfigs.print_params()
    RAConfigs.save_configs(model_dir)

    #创建模型
    agent_R = D3QN(
        alpha=RAConfigs.alpha,  # 学习率
        state_dim=5 * (InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1),  # 状态空间
        action_dim=InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1,  # 动作空间
        fc1_dim=RAConfigs.fc1_dim, fc2_dim=RAConfigs.fc2_dim,  # 模型参数
        ckpt_dir=model_dir,  # 保存地址
        gamma=RAConfigs.gamma, tau=RAConfigs.tau, #折扣系数
        epsilon=RAConfigs.epsilon, eps_end=RAConfigs.eps_end, eps_dec=RAConfigs.eps_dec, #探索率相关
        max_size=RAConfigs.max_size, batch_size=RAConfigs.batch_size) #memory buffer大小 训练批次大小

    for episode in range(RAConfigs.max_episodes):
        env = create_random_dataset(InsConfigs.numMobileDevices,
                                InsConfigs.numEdgeServers,
                                InsConfigs.numCenterServers,
                                InsConfigs.numWorkflows,
                                InsConfigs.lambdaRate,
                                InsConfigs.datasetScale)
        select_job = None  # 用于记录当前被选中的任务
        last_job = None  # 用于记录上一次被选中的任务 为其

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
                action = agent_R.choose_action(obs_R)

                if action == len(env.servers):  # 表示本地卸载
                    env.addEvent(
                        jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job,
                                            env.devices[select_job.sourceDeviceId],
                                            env.current_time))
                else:
                    env.addEvent(
                        jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action],
                                            env.current_time))

                select_job.S = obs_R
                select_job.A = action
                if last_job != None:
                    last_job.next_S = obs_R
                last_job = select_job
                select_job = None
                continue

            if env.event_list == []:
                check = True
                for wf in env.workflows:
                    for job in wf.jobs:
                        if job.state != "已完成":
                            print(f"工作流{wf.id} 任务{job.id}未完成")
                            check = False
                last_job.next_S = [0] * (len(env.servers) + 1) * 5
                last_job.terminal = True
                # print(check)
                break

                # 5、更新时间
            if env.current_time != env.event_list[0].time:
                env.current_time = env.event_list[0].time
        #收集经验
        collect_experience(agent_R, env)
        #训练
        for i in range(RAConfigs.training_per_episode):
            agent_R.learn()

        # 计算makespan 所有工作流的完成时长之和
        makespan = 0
        for wf in env.workflows:
            max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
            makespan += max_finish_time_job.finishTime - wf.arrivalTime

        total_reward = 0
        for wf in env.workflows:
            for job in wf.jobs:
                total_reward += job.R

        print('EP:{} Reward:{} Epsilon:{} makespan:{}'.format(episode + 1, total_reward, agent_R.epsilon, makespan))
        eval_dataset(eval_env)
        eval_agent(agent_R, eval_env)

        if (episode + 1) % RAConfigs.save_freq == 0:
            agent_R.save_models(episode + 1)


if __name__ == '__main__':
    start_time = time.time()
    # 实例参数
    InsConfigs = InstanceConfigs(
        numMobileDevices=3,
        numEdgeServers=10,  #20
        numCenterServers=10, #20
        numWorkflows=30,
        lambdaRate=0.1,
        datasetScale="small",
    )
    # 路由智能体参数
    RAConfigs = RoutingAgentConfigs(
        max_episodes=400,
        alpha=0.0003,
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        eps_end=0.1,
        eps_dec=2e-4,
        max_size=1000000,
        batch_size=256,
        fc1_dim=256,
        fc2_dim=256,
        save_freq=10,
        training_per_episode=10,
        ckpt_dir='./checkpoints/D3QN/',
    )
    for size in [6, 10 ,15]:
        for value in [2, 3, 4]:
            RAConfigs.max_episodes = 400
            if size == 6:
                InsConfigs.numWorkflows = 20
            elif size == 10:
                InsConfigs.numWorkflows = 30
            elif size == 15:
                InsConfigs.numWorkflows = 45
            InsConfigs.numEdgeServers = size
            InsConfigs.numCenterServers = size
            InsConfigs.lambdaRate = 0.15
            InsConfigs.numMobileDevices = value
            main(InsConfigs, RAConfigs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")
