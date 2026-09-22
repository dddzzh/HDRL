

def Action_0(env, server):
    #先来先服务 FCFS
    return server.wait_list[0]

def Action_1(env, server):
    #最小任务量优先 MINMIN
    min_workload_job = min(server.wait_list, key=lambda job: job.workload)
    return min_workload_job

def Action_2(env, server):
    #最小处理+下载优先
    min_job = min(server.wait_list, key=lambda job: (job.workload / server.pa) + (job.data_down / server.bandwidth))
    return min_job

def Action_3(env, server):
    #完成率最高优先
    CR = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        completeRate = wf.getCompleteRate()
        CR.append(completeRate)

    max_index = CR.index(max(CR))
    highest_priority_job = server.wait_list[max_index]
    return highest_priority_job
    #return server.wait_list[0]

def Action_4(env, server):


    #最小剩余时间优先

    remaining_time = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        rt = wf.getEstimatedCompletionTime(env)
        rt = rt - env.current_time
        remaining_time.append(rt)
    min_index = remaining_time.index(min(remaining_time))
    min_job = server.wait_list[min_index]



    #min_job = min(server.wait_list, key=lambda job: job.arrivalTime + (job.workload / server.pa))
    return min_job



def Action_5(env, server):
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        UpwardRank = wf.getUpwardRank(job)





#最小任务量优先  MINMIN

#最大任务量优先  MAXMIN

#最大向上排序 HEFT


#最早释放时间 + 处理时间 CEAS


#传输时间最短 BWAWA