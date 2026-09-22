# NIQ The number of tasks in the waiting queue.
# WIQ The remaining work in the waiting queue.
# MRT The ready time of server/device.
# UT The upload time of the task.
# DT The download time of the task.
# PT The processing time of the task.
# TTIQ The total remaining time in the waiting queue.
# TIS The time in system: t - releaseTime.
# TWT The waiting time of the task.
# NTR The number of tasks remaining of the workflow.




def get_GP_Priority_RA( env ,selected_job, func_RA):
    #输入被选中的任务  和环境  返回机器下标
    score = []
    for server in env.servers:
        jobs_in_waiting_queue = server.wait_list + [job for job in server.trans_list if
                                                    job.arrivalTime < (
                                                                env.current_time + selected_job.data_up / server.bandwidth)]
        NIQ = len(jobs_in_waiting_queue)
        WIQ = sum(job.workload for job in jobs_in_waiting_queue)
        MRT =  server.processing_job.endProcessingTime - env.current_time if server.isBusy else 0
        UT = selected_job.data_up / server.bandwidth
        DT = selected_job.data_down / server.bandwidth
        PT = selected_job.workload /server.pa
        TTIQ = WIQ / server.pa
        TIS = env.current_time - env.workflows[selected_job.workflow_id].arrivalTime
        TWT = 0
        NTR = len([job for job in env.workflows[selected_job.workflow_id].jobs if job.state != "已完成"])
        result_RA = func_RA(NIQ, WIQ, MRT, UT, DT, PT, TTIQ, TIS, TWT, NTR)
        score.append(result_RA)

    device = env.devices[selected_job.sourceDeviceId]
    NIQ = len(device.wait_list)
    WIQ = sum(job.workload for job in device.wait_list)
    MRT = device.processing_job.endProcessingTime - env.current_time if device.isBusy else 0
    UT = 0
    DT = 0
    PT = selected_job.workload / device.pa
    TTIQ = WIQ / device.pa
    TIS = env.current_time - env.workflows[selected_job.workflow_id].arrivalTime
    TWT = 0
    NTR = len([job for job in env.workflows[selected_job.workflow_id].jobs if job.state != "已完成"])
    result_RA = func_RA(NIQ, WIQ, MRT, UT, DT, PT, TTIQ, TIS, TWT, NTR)
    score.append(result_RA)

    index = score.index(max(score))

    return index

def get_GP_Priority_SA(env,server, func_SA ):
    score = []
    #输入机器和env 直接返回被选中的任务
    for job in server.wait_list:
        NIQ = len(server.wait_list )
        WIQ = sum(job.workload for job in server.wait_list )
        MRT = server.processing_job.endProcessingTime - env.current_time if server.isBusy else 0
        UT = job.data_up / server.bandwidth
        DT = job.data_down / server.bandwidth
        PT = job.workload / server.pa
        TTIQ = WIQ / server.pa
        TIS = env.current_time - env.workflows[job.workflow_id].arrivalTime
        TWT = env.current_time -  job.arrivalTime
        NTR = len([job for job in env.workflows[job.workflow_id].jobs if job.state != "已完成"])

        result_SA = func_SA(NIQ, WIQ, MRT, UT, DT, PT, TTIQ, TIS, TWT, NTR)
        score.append(result_SA)

        index = score.index(max(score))
        return server.wait_list[index]