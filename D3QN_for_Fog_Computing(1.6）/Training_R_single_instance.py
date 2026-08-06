import random

from D3QN import D3QN
from Params import configs
from CreateDataset import create_random_dataset
import heapq
from Event import Event, workflowArrivalEvent, jobArrivalEvent, jobStartUploadEvent, jobStartProcessEvent, \
    jobFinishedEvent, jobStartDownloadEvent
from obs import get_obs_R
import time

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
                jobs = server.trans_list + server.wait_list
                total_workload = sum(job.workload for job in jobs)
                Allpt = total_workload / server.pa
                if server.isBusy == True:
                    EAT = server.processing_job.endProcessingTime - env.current_time
                else:
                    EAT = 0

                EarliestFinishTime.append(Allpt + EAT)
            device = env.devices[select_job.sourceDeviceId]
            if device.isBusy == True:
                EAT = device.processing_job.endProcessingTime - env.current_time
            else:
                EAT = 0

            total_workload = sum(job.workload for job in device.wait_list)
            Allpt = total_workload / device.pa
            EarliestFinishTime.append(Allpt + EAT)

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
    print('最早完成优先makespan:{}'.format(makespan))


def main():
    env = create_random_dataset(configs.numMobileDevices ,
                                configs.numEdgeServers,
                                configs.numCenterServers,
                                configs.numWorkflows,
                                configs.lambdaRate,
                                configs.datasetScale)
    agent_R = D3QN(alpha=0.0003, state_dim=env.obs_space_R, action_dim=env.action_space_R,
                   fc1_dim=256, fc2_dim=256, ckpt_dir=configs.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                   eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=256)

    for episode in range(configs.max_episodes):
        env.reset()
        select_job = None  # 用于记录当前被选中的任务
        last_job = None  # 用于记录上一次被选中的任务
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
                print(check)
                break

                # 5、更新时间
            if env.current_time != env.event_list[0].time:
                env.current_time = env.event_list[0].time

        collect_experience(agent_R, env)
        for i in range(10):
            agent_R.learn()

        # 计算所有工作流的完成时长之和
        makespan = 0
        for wf in env.workflows:
            max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
            makespan += max_finish_time_job.finishTime - wf.arrivalTime

        total_reward = 0
        for wf in env.workflows:
            for job in wf.jobs:
                total_reward += job.R

        print('EP:{} Reward:{} Epsilon:{} makespan:{}'.format(episode + 1, total_reward, agent_R.epsilon, makespan))
        eval_dataset(env)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")