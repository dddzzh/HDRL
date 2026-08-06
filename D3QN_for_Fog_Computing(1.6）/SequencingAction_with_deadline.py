
#全部设置为 如果有延期的任务 优先选择延期动作
import random
import numpy as np
#先来先服务
def FCFS(env, server):

    return server.wait_list[0]


def SPT(env, server):
    #最小任务量优先 MINMIN
    min_workload_job = min(server.wait_list, key=lambda job: job.workload)
    return min_workload_job

def HCR(env, server):
    #Highest Completion Rate 完成率最高优先
    CR = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        completeRate = wf.getCompleteRate()
        CR.append(completeRate)

    max_index = CR.index(max(CR))
    highest_priority_job = server.wait_list[max_index]
    return highest_priority_job


def MS(env, server):
  #最短松弛时间优先 Min Slack time
    slack_time = []
    for job in server.wait_list:

        wf = env.workflows[job.workflow_id]

        ect = wf.getEstimatedCompletionTime(env)
        deadline = wf.deadline  ##都是绝对时间
        slack = deadline - ect
        slack_time.append(slack)
    min_slack_time = slack_time.index(min(slack_time))
    return server.wait_list[min_slack_time]


#向上等级排序 HEFT
def UpwardRank(env, server):
    UR = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]

        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_transmition = sum(server.bandwidth for server in env.servers) / len(env.servers)
        wf.calculate_upward_ranks(mean_pa, mean_transmition)
            #print(wf.jobs[job.id].UpwardRank)
        UR.append(wf.jobs[job.id].UpwardRank)
    max_UR = UR.index(max(UR))
    return server.wait_list[max_UR]


def DownwardRank(env, server):
    DR = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
        mean_transmition = sum(server.bandwidth for server in env.servers) / len(env.servers)
        wf.calculate_downward_ranks(mean_pa, mean_transmition)
        DR.append((wf.jobs[job.id].DownwardRank))
    max_DR = DR.index(min(DR))
    return server.wait_list[max_DR]

#最大处理时间 优先 MAXMIN
def LPT(env, server):
    max_workload_job = max(server.wait_list, key=lambda job: job.workload)
    return max_workload_job

#最早deadline
def RSR(env, server):
    mean_pa = sum(server.pa for server in env.servers) / len(env.servers)
    dl = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        work = sum(job.workload for job in wf.jobs if job.state!="已完成")
        work_time = work / mean_pa
        ttd = wf.deadline - env.current_time
        dl.append(work_time/ ttd)
    max_dl = dl.index(max(dl))
    return server.wait_list[max_dl]

#最早可用+最小工作流
def EASPT(env, server) :
    easpt = []
    for job in server.wait_list:
        easpt.append(job.arrivalTime + job.workload / server.pa)
    min_easpt = easpt.index(min(easpt))
    return server.wait_list[min_easpt]
#最小剩余任务量
def LRW(env, server):
    LRW = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        work = sum(job.workload for job in wf.jobs if job.state != "已完成")
        LRW.append(work)
    min_LRW = LRW.index(min(LRW))
    return server.wait_list[min_LRW]


#http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_23.pdf 表观延误成本
def ATC(env, server):
    atc = []
    average_pt = sum(job.workload for job in server.wait_list) / len(server.wait_list)
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        cost = max(wf.deadline - env.current_time - job.workload / server.pa, 0)
        priority = np.exp(- cost / (0.05 * average_pt)) / job.workload / server.pa
        atc.append(priority)
    max_atc = atc.index(max(atc))
    return server.wait_list[max_atc]
def Random_SA(env, server):
    return random.choice(server.wait_list)