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
from SequencingAction_with_deadline import FCFS,SPT,HCR,MS,UpwardRank, DownwardRank, LPT, RSR, EASPT, Random_SA, LRW,ATC
from RoutingAction_with_deadline import EFT,PT,WT,UR,SQ,EA,Random_RA
from get_GP_Priority import get_GP_Priority_RA,get_GP_Priority_SA
import matplotlib.pyplot as plt
import pandas as pd
## 输入所使用的方法  输出该系统下的延迟时间
def simulation(agent_R, agent_S, env, routing_mode, sequencing_mode, GP_func_RA = None, GP_func_SA = None):
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

                if sequencing_mode == "SA":
                    obs_S = get_obs_S_with_deadline(env, server)
                    action = agent_S.choose_action(obs_S, isTrain=False)
                    if action == 0:
                        job = FCFS(env, server)
                    elif action == 1:
                        job = SPT(env, server)
                    elif action == 2:
                        job = HCR(env, server)
                    elif action == 3:
                        job = MS(env, server)
                elif sequencing_mode == "FCFS": #先来先服务
                    job = FCFS(env, server)
                elif sequencing_mode == "SPT": #最小处理优先
                    job = SPT(env, server)
                elif sequencing_mode == "HCR":  #最大完成率优先
                    job = HCR(env, server)
                elif sequencing_mode == "MS": #最小松弛时间
                    job = MS(env, server)
                elif sequencing_mode == "MTGP":
                    job = get_GP_Priority_SA(env, server, GP_func_SA)
                elif sequencing_mode == "UpwardRank":
                    job = UpwardRank(env, server)
                elif sequencing_mode == "DownwardRank":
                    job = DownwardRank(env, server)
                elif sequencing_mode == "LPT":
                    job = LPT(env, server)
                elif sequencing_mode == "RSR":
                    job = RSR(env, server)
                elif sequencing_mode =="EASPT":
                    job = EASPT(env, server)
                elif sequencing_mode =="LRW":
                    job = LRW(env, server)
                elif sequencing_mode =="ATC":
                    job = ATC(env, server)
                elif sequencing_mode == "Random":
                    job = Random_SA(env, server)

                # 把任务开始处理发送上去
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,env.current_time))

        # 4、处理当前时刻待进行的 Routing Decision
        if env.ready_job_set:
            select_job = min(env.ready_job_set, key=lambda job: job.workload)


            if routing_mode == "RA": #如果用Routing Agent
                obs_R = get_obs_R(env, select_job)
                action = agent_R.choose_action(obs_R, isTrain=False)
            elif routing_mode == "EFT":  #如果用 最早完成时间优先 EFT
                action = EFT(env,select_job)
            elif routing_mode == "PT":
                action = PT(env,select_job)
            elif routing_mode == "WT":
                action = WT(env, select_job)
            elif routing_mode == "UR":
                action = UR(env, select_job)
            elif routing_mode == "SQ":
                action = SQ(env, select_job)
            elif routing_mode == "EA":
                action = EA(env, select_job)
            elif routing_mode == "Random":
                action = Random_RA(env, select_job)
            elif routing_mode == "MTGP":
                action = get_GP_Priority_RA(env, select_job, GP_func_RA)


            if action == len(env.servers):  # 表示本地卸载
                env.addEvent(jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job,env.devices[select_job.sourceDeviceId],env.current_time))
            else: #表示卸载到云端或边缘端
                env.addEvent(jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action],env.current_time))
            select_job = None

            continue
        #退出条件
        if env.event_list == []:
            for wf in env.workflows:
                for job in wf.jobs:
                    if job.state != '已完成':
                        print("有任务未完成")
            break

            # 5、更新时间
        if env.current_time != env.event_list[0].time:
            env.current_time = env.event_list[0].time

    # 计算所有工作流的完成时长之和
    makespan = 0
    tardiness = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        makespan += max_finish_time_job.finishTime - wf.arrivalTime
        tardiness += max(max_finish_time_job.finishTime - wf.deadline, 0)

    agent_R.q_eval.train()
    agent_R.q_target.train()
    agent_S.q_eval.train()
    agent_S.q_target.train()
    return tardiness
