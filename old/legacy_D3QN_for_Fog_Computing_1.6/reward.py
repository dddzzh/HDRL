import copy
import networkx as nx
import itertools
import numpy as np
def get_reward_R(env, job):
    total_bandwidth = 0
    total_pa = 0
    num_servers = len(env.servers)

    for server in env.servers:
        total_bandwidth += server.bandwidth
        total_pa += server.pa

    average_bandwidth = total_bandwidth / num_servers if num_servers > 0 else 0
    average_pa = total_pa / num_servers if num_servers > 0 else 0

    average_time = (job.data_up+job.data_down) / average_bandwidth + job.workload / average_pa


    #return -1  *  (job.endProcessingTime - job.readyTime - average_time) #发现个问题 这里不应该是结束处理时间吧
    #return -1  *  (job.endProcessingTime - job.readyTime )
    return -1  *  (job.finishTime - job.readyTime - average_time)

def get_reward_R1(env, job):
    #考虑延期惩罚
    total_bandwidth = 0
    total_pa = 0
    num_servers = len(env.servers)

    for server in env.servers:
        total_bandwidth += server.bandwidth
        total_pa += server.pa

    average_bandwidth = total_bandwidth / num_servers if num_servers > 0 else 0
    average_pa = total_pa / num_servers if num_servers > 0 else 0

    average_time = (job.data_up+job.data_down) / average_bandwidth + job.workload / average_pa

    wf = env.workflows[job.workflow_id]
    deadline = wf.deadline
    #TTD = deadline - job.finishTime-
    #return -1  *  (job.endProcessingTime - job.readyTime - average_time) #发现个问题 这里不应该是结束处理时间吧
    #return -1  *  (job.endProcessingTime - job.readyTime )

    return -1  *  (job.finishTime - job.readyTime - average_time) - 0.1*(max(job.finishTime - wf.deadline,0))
def get_reward_S(env, job, server):
    #前后预估完成时间之差
    #可以考虑是如果有改进就为1 无改进为0

    now = env.current_time
    selected_index = server.wait_list.index(job)
    processing_times = [j.workload / server.pa for j in server.wait_list]  #原版的处理时间

    old_p = list(itertools.accumulate(processing_times))
    #print(server.name)
    #print(old_p)
    #print(server.wait_list)
    new_p = [
        old_p[i] + processing_times[selected_index] if i < selected_index else
        processing_times[selected_index] if i == selected_index else
        old_p[i]
        for i in range(len(old_p))
    ]  #保持job顺序还是按照server.waitlist里的位置

    total_bandwidth = 0
    total_pa = 0
    num_servers = len(env.servers)
    for s in env.servers:
        total_bandwidth += s.bandwidth
        total_pa += s.pa
    average_bandwidth = total_bandwidth / num_servers if num_servers > 0 else 0
    average_pa = total_pa / num_servers if num_servers > 0 else 0

    workflow_set = set()
    #当前任务所处的工作流集合 再转化为列表
    for j in server.wait_list:
        workflow_set.add(env.workflows[j.workflow_id])
    workflow_list = list(workflow_set)


    old_makespan = 0
    for wf in workflow_list:
        eft = [None] * len(wf.jobs)  #预计结束时间
        topo_order = list(nx.topological_sort(wf.graph))  #拓扑排序 返回的是节点的id
        for node_id in topo_order: #按照拓扑排序遍历
            j = wf.jobs[node_id] #这个任务
            if j.finishTime != None:  #如果该任务的预计完成时间已知
                eft[node_id] = j.finishTime

            elif j in server.wait_list: # 如果未知  检查是否在等待序列中
                index = server.wait_list.index(j)
                #print(len(server.wait_list))
                #print(server.name)
                #print(server.wait_list)
                #print(old_p)
                process_time = old_p[index]
                download_time = job.data_down / server.bandwidth

                eft[node_id] = now + process_time + download_time
            else:
            # 否则根据前序任务估算  如果没有前序任务 说明是第一个 就根据自己的时间估算
                predecessors = list(wf.graph.predecessors(node_id))

                if predecessors: #如果有前序
                    start_time = None
                    # 遍历前序节点
                    for pre in predecessors:
                        # 获取对应前序节点在 eft 列表中的值
                        if start_time is None or (eft[pre] is not None and eft[pre] > start_time):
                            start_time = eft[pre]


                else:
                    start_time = wf.arrivalTime
                process_time = job.workload / average_pa
                trans_time = (job.data_down + job.data_up)/average_bandwidth

                eft[node_id] = start_time + process_time + trans_time

        t = max(eft) - wf.arrivalTime
        old_makespan += t

    new_makespan = 0
    for wf in workflow_list:
        eft = [None] * len(wf.jobs)  # 预计结束时间
        topo_order = list(nx.topological_sort(wf.graph))  # 拓扑排序 返回的是节点的id
        for node_id in topo_order:  # 按照拓扑排序遍历
            j = wf.jobs[node_id]  # 这个任务
            if j.finishTime != None:  # 如果该任务的预计完成时间已知
                eft[node_id] = j.finishTime

            elif j in server.wait_list:  # 如果未知  检查是否在等待序列中
                index = server.wait_list.index(j)
                process_time = new_p[index]
                download_time = job.data_down / server.bandwidth

                eft[node_id] = now + process_time + download_time
            else:
                # 否则根据前序任务估算  如果没有前序任务 说明是第一个 就根据自己的时间估算
                predecessors = list(wf.graph.predecessors(node_id))

                if predecessors:  # 如果有前序
                    start_time = None
                    # 遍历前序节点
                    for pre in predecessors:
                        # 获取对应前序节点在 eft 列表中的值
                        if start_time is None or (eft[pre] is not None and eft[pre] > start_time):
                            start_time = eft[pre]


                else:
                    start_time = wf.arrivalTime
                process_time = job.workload / average_pa
                trans_time = (job.data_down + job.data_up) / average_bandwidth

                eft[node_id] = start_time + process_time + trans_time

        t = max(eft) - wf.arrivalTime
        new_makespan += t
    # if old_makespan > new_makespan:
    #     return 1
    # elif old_makespan == new_makespan:
    #     return 0
    # else:
    #     return -1
    return  old_makespan - new_makespan  #是否比改变前好

def get_reward_S_2(env,job,server):

    return -1 * (env.current_time - job.arrivalTime)

def get_reward_S_3(env, job, server):
    now = env.current_time
    selected_index = server.wait_list.index(job)
    processing_times = [j.workload / server.pa for j in server.wait_list]  # 原版的处理时间

    old_p = list(itertools.accumulate(processing_times))

    new_p = [
        old_p[i] + processing_times[selected_index] if i < selected_index else
        processing_times[selected_index] if i == selected_index else
        old_p[i]
        for i in range(len(old_p))
    ]
    workflow_set = set()
    # 当前任务所处的工作流集合 再转化为列表
    for j in server.wait_list:
        workflow_set.add(env.workflows[j.workflow_id])
    workflow_list = list(workflow_set)

    #临时的改变一下finish_time
    for i, job in enumerate(server.wait_list):
        job.finishTime = old_p[i] + now + (job.data_down / server.bandwidth)

    old_makespan = 0
    for wf in workflow_list:
        old_makespan += wf.getEstimatedCompletionTime(env)


    for i, job in enumerate(server.wait_list):
        job.finishTime = new_p[i] + now + (job.data_down / server.bandwidth)

    new_makespan = 0
    for wf in workflow_list:
        new_makespan += wf.getEstimatedCompletionTime(env)

    for i, job in enumerate(server.wait_list):
        job.finishTime = None

    reward = (old_makespan - new_makespan) / len(workflow_list)
    #reward = np.clip(reward / 10 , -1, 1)


    return reward

def get_reward_S_4(env, job, server):
    #只关注当前工作流
    now = env.current_time
    selected_index = server.wait_list.index(job)
    processing_times = [j.workload / server.pa for j in server.wait_list]  # 原版的处理时间

    old_p = list(itertools.accumulate(processing_times))

    new_p = [
        old_p[i] + processing_times[selected_index] if i < selected_index else
        processing_times[selected_index] if i == selected_index else
        old_p[i]
        for i in range(len(old_p))
    ]

    wf = env.workflows[job.workflow_id]
    job.finishTime = now + old_p[selected_index] +job.data_down/server.bandwidth
    old_makespan = wf.getEstimatedCompletionTime(env)

    job.finishTime = now + new_p[selected_index] +job.data_down/server.bandwidth
    new_makespan = wf.getEstimatedCompletionTime(env)
    job.finishTime = None

    return old_makespan - new_makespan
def get_reward_S_with_deadline(env, job, server):
    #考虑makespan 和 延期时间 的变化

    now = env.current_time
    selected_index = server.wait_list.index(job)
    processing_times = [j.workload / server.pa for j in server.wait_list]  # 原版的处理时间

    old_p = list(itertools.accumulate(processing_times))
    new_p = [
        old_p[i] + processing_times[selected_index] if i < selected_index else
        processing_times[selected_index] if i == selected_index else
        old_p[i]
        for i in range(len(old_p))
    ]
    workflow_set = set()
    # 当前任务所处的工作流集合 再转化为列表
    for j in server.wait_list:
        workflow_set.add(env.workflows[j.workflow_id])
    workflow_list = list(workflow_set)
    #临时的改变一下finish_time
    for i, job in enumerate(server.wait_list):
        job.finishTime = old_p[i] + now + (job.data_down / server.bandwidth)

    old_makespan = 0
    old_tardiness = 0
    for wf in workflow_list:
        ect = wf.getEstimatedCompletionTime(env)
        arrival_time = wf.arrivalTime
        deadline = wf.deadline
        old_makespan += ect - arrival_time
        old_tardiness += max(ect - deadline, 0 )


    for i, job in enumerate(server.wait_list):
        job.finishTime = new_p[i] + now + (job.data_down / server.bandwidth)
    new_makespan = 0
    new_tardiness = 0
    for wf in workflow_list:
        ect = wf.getEstimatedCompletionTime(env)
        arrival_time = wf.arrivalTime
        deadline = wf.deadline
        new_makespan += ect - arrival_time
        new_tardiness += max(ect - deadline, 0 )


    for i, job in enumerate(server.wait_list):
        job.finishTime = None

    makespan_reward = (old_makespan - new_makespan)# / len(workflow_list)

    tardiness_reward = (old_tardiness - new_tardiness)# / len(workflow_list)
    #print(tardiness_reward)
    #return np.clip(tardiness_reward / 20, -1, 1)   #旧的reward 10.18
    return np.clip(tardiness_reward / len(server.wait_list), -3, 3)
    #return tardiness_reward/20
def get_reward_S_with_deadline_2(env, job, server): #
    #只考虑当前选中任务所在工作流的情况
    pass