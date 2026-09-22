import random

#输入环境和动作 输出传输到哪个服务器

#最早完成优先  考虑排队时间
def EFT(env, select_job): #Earliest Finish Time  最早完成时间 原理上应该是 只考虑最早开始时间 + 最早处理时间+ 返回时间  考虑在机器上排队的任务
    EarliestFinishTime = []
    for server in env.servers:
        #jobs = server.trans_list + server.wait_list
        #jobs = server.wait_list
        #传输队列中更快到达的任务
        trans_job = [job for job in server.trans_list if
                     job.arrivalTime < (env.current_time + select_job.data_up / server.bandwidth)]
        jobs =server.wait_list + trans_job
        total_processing_time = sum(job.workload for job in jobs) / server.pa  #队列中任务的处理时间
        available_time = server.processing_job.endProcessingTime - env.current_time if server.isBusy else 0  #机器下一次可用时间
        start_time = max(available_time + total_processing_time , select_job.data_up/ server.bandwidth)  #选中任务的开始时间
        time_of_this_job = select_job.data_down /  server.bandwidth + select_job.workload / server.pa  #任务的处理和返回时间

        EarliestFinishTime.append(start_time + time_of_this_job)


    device = env.devices[select_job.sourceDeviceId]
    available_time = device.processing_job.endProcessingTime - env.current_time if device.isBusy else 0
    total_processing_time = sum(job.workload for job in device.wait_list)/device.pa
    start_time = available_time + total_processing_time # 本地卸载不需要考虑起始时间
    time_of_this_job = select_job.workload / device.pa
    EarliestFinishTime.append(start_time + time_of_this_job)
    action = EarliestFinishTime.index(min(EarliestFinishTime))

    return action

#选择最早可用+处理时间最小的机器   Minimum Processing Time 考虑排队时间 不考虑传输时间
def PT(env, select_job):
    PT = []
    for server in env.servers:
        #jobs = server.trans_list + server.wait_list
        #jobs = server.wait_list
        trans_job = [job for job in server.trans_list if
                     job.arrivalTime < (env.current_time + select_job.data_up / server.bandwidth)]
        jobs = server.wait_list #+ trans_job
        total_processing_time = sum(job.workload for job in jobs) / server.pa
        available_time = server.processing_job.endProcessingTime - env.current_time if server.isBusy else 0  # 机器下一次可用时间

        start_time = max(available_time , select_job.data_up / server.bandwidth)  # 选中任务的开始时间
        time_of_this_job =  select_job.workload / server.pa  # 任务的处理时间
        PT.append(start_time +total_processing_time+ time_of_this_job )

    device = env.devices[select_job.sourceDeviceId]
    available_time = device.processing_job.endProcessingTime - env.current_time if device.isBusy else 0
    total_processing_time = sum(job.workload for job in device.wait_list) / device.pa
    start_time = available_time + total_processing_time
    time_of_this_job = select_job.workload / device.pa
    PT.append(start_time + time_of_this_job)
    action = PT.index(min(PT))
    return action

#选择 历史平均排队时间最短的机器
def WT(env, select_job):
    WT = []
    #
    # for server in env.servers:
    #     jobs = server.record
    #     workload = 0
    #     for job in jobs:
    #         workload += job.startProcessingTime - job.arrivalTime
    #     if len(server.record)!= 0:
    #         average_wait_time = workload / len(server.record)
    #     else:
    #         average_wait_time = 0
    #     WT.append(average_wait_time)
    # action = WT.index(min(WT))
    for server in env.servers:
        jobs =  server.wait_list #+ server.trans_list
        trans_job = [job for job in server.trans_list if
                     job.arrivalTime < (env.current_time + select_job.data_up / server.bandwidth)]
        jobs = server.wait_list + trans_job
        # trans_job = [job for job in server.trans_list if job.arrivalTime < env.current_time+ select_job.data_up / server.bandwidth]
        # jobs+=trans_job
        total_processing_time = sum(job.workload for job in jobs) / server.pa  # 队列中任务的处理时间
        #available_time = server.processing_job.endProcessingTime - env.current_time if server.isBusy else 0  # 机器下一次可用时间=
        #trans_time = select_job.data_up / server.bandwidth
        WT.append(total_processing_time )

    device = env.devices[select_job.sourceDeviceId]
    #available_time = device.processing_job.endProcessingTime - env.current_time if device.isBusy else 0
    total_processing_time = sum(job.workload for job in device.wait_list) / device.pa
    WT.append(total_processing_time)
    action = WT.index(min(WT))

    return action





#最小利用率
def UR(env, select_job):
    #直接比较机器的工作时间 计算为 在该机器上的总任务量 / 机器的处理时间

    UR = []
    for server in env.servers:
        jobs = server.record + server.trans_list + server.wait_list
        #jobs = server.record  + server.wait_list
        total_processing_time = sum(job.workload for job in jobs) / server.pa
        UR.append(total_processing_time)

    device = env.devices[select_job.sourceDeviceId]
    jobs = device.record + device.trans_list + device.wait_list
    total_processing_time = sum(job.workload for job in jobs) / device.pa
    UR.append(total_processing_time)
    action = UR.index(min(UR))
    return action
#最小队列尺寸
def SQ(env, select_job):
    SQ = []
    for server in env.servers:
        num = len( server.wait_list + server.trans_list)
        if server.isBusy:
            num +=1

        SQ.append(num)

    device = env.devices[select_job.sourceDeviceId]
    num = len(device.wait_list)
    if device.isBusy:
        num += 1
    SQ.append(num)
    action = SQ.index(min(SQ))
    return action

#最早可用
def EA(env, select_job):
    EA = []
    for server in env.servers:
        trans_job = [job for job in server.trans_list if
                     job.arrivalTime < (env.current_time + select_job.data_up / server.bandwidth)]
        jobs = server.wait_list + trans_job
        available_time = server.processing_job.endProcessingTime - env.current_time if server.isBusy else 0
        total_workload = sum(job.workload for job in jobs) / server.pa
        EA.append(available_time + total_workload)

    device = env.devices[select_job.sourceDeviceId]
    jobs =  device.wait_list
    total_processing_time = sum(job.workload for job in jobs) / device.pa
    EA.append(total_processing_time)
    action = EA.index(min(EA))
    return action

def Random_RA(env, select_job):
    random_index = random.randint(0, len(env.servers))
    return random_index