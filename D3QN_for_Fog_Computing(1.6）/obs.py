import numpy as np
def normalize_list(lst):
    min_value = min(lst)
    max_value = max(lst)

    if min_value == max_value:
        # 当列表中的所有元素相等时，归一化后的值都为 0
        return [0 for _ in lst]

    return [(x - min_value) / (max_value - min_value) for x in lst]

def standardize_list(data):
    """
    对列表进行标准化，使其具有零均值和单位标准差

    参数:
    data (list of float): 要标准化的列表

    返回:
    list of float: 标准化后的列表
    """
    mean = np.mean(data)
    std = np.std(data)
    standardized_data = [(x - mean) / std for x in data]
    return standardized_data

def get_obs_R(env, job):

    now = env.current_time #当前时间
    device = env.devices[job.sourceDeviceId]

    #最早可用时间
    EarliestAvailableTime = []
    for server in env.servers:
        if server.isBusy == True:
            EAT = server.processing_job.endProcessingTime - now
        else:
            EAT = 0
        EarliestAvailableTime.append(EAT)

    if device.isBusy == True:
        EAT = device.processing_job.endProcessingTime - now
    else:
        EAT = 0
    EarliestAvailableTime.append(EAT)

    #上传时间
    UploadTime = []
    for server in env.servers:
        ut = job.data_up / server.bandwidth
        UploadTime.append(ut)
    UploadTime.append(0)


    #处理时间
    ProcessTime = []
    for server in env.servers:
        pt = job.workload / server.pa
        ProcessTime.append(pt)
    ProcessTime.append(job.workload / device.pa)


    #下载时间
    DownloadTime = []
    for server in env.servers:
        dt = job.data_down / server.bandwidth
        DownloadTime.append(dt)
    DownloadTime.append(0)


    #平均等待时间 (可选）

    #总处理时间 （已分配该机器上的所有任务）
    SumOfProcessingTime = []
    for server in env.servers:
        jobs = server.trans_list + server.wait_list
        total_workload = sum(job.workload for job in jobs)
        Allpt = total_workload / server.pa
        SumOfProcessingTime.append(Allpt)


    total_workload = sum(job.workload for job in device.wait_list)
    Allpt = total_workload / device.pa
    SumOfProcessingTime.append(Allpt)
    #当前任务所在工作流的完成率

    obs = EarliestAvailableTime + UploadTime + ProcessTime + DownloadTime + SumOfProcessingTime
    #obs = normalize_list(obs)
    #obs = standardize_list(obs)
    #obs = [x / 10 for x in obs]
    return obs


def get_obs_S(env, server):
    obs = []
    wait_list = server.wait_list
    now = env.current_time

    #当前任务数量
    job_num = len(server.wait_list)
    #obs.append(job_num)


    #处理时间
    max_pt = 150000 * 1e6 /server.pa
    min_pt = 50000 *1e6 /server.pa
    mean_process_time = sum(job.workload / server.pa for job in wait_list) / job_num
    max_process_time = max(job.workload / server.pa for job in wait_list)
    min_process_time = min(job.workload / server.pa for job in wait_list)

    #处理时间归一化后的标准差（可选）
    normalized_mean_process_time = (mean_process_time - min_pt) / (max_pt - min_pt)
    normalized_max_process_time = (max_process_time - min_pt) / (max_pt - min_pt)
    normalized_min_process_time = (min_process_time - min_pt) / (max_pt - min_pt)

    # obs.append(mean_process_time)
    # obs.append(max_process_time)
    # obs.append(min_process_time)

    obs.append(normalized_mean_process_time)
    obs.append(normalized_max_process_time)
    obs.append(normalized_min_process_time)

    #等待时间
    mean_waiting_time = sum(now - job.arrivalTime for job in wait_list) / job_num
    max_waiting_time = max(now - job.arrivalTime for job in wait_list)
    min_waiting_time = min(now - job.arrivalTime for job in wait_list)

    #obs.append(mean_waiting_time)
    #obs.append(max_waiting_time)
    #obs.append(min_waiting_time)

    #下载时间
    max_d = 2880 * 1024 *1024 *8 / server.bandwidth
    min_d = 720* 1024 *1024 *8  /server.bandwidth
    mean_download_time = sum(job.data_down / server.bandwidth for job in wait_list) / job_num
    max_download_time = max(job.data_down / server.bandwidth for job in wait_list) / job_num
    min_download_time = min(job.data_down / server.bandwidth for job in wait_list) / job_num


    normalized_mean_download_time = (mean_download_time - min_d) / (max_d - min_d)
    normalized_max_download_time = (max_download_time - min_d) / (max_d - min_d)
    normalized_min_download_time = (min_download_time - min_d) / (max_d - min_d)

    # obs.append(mean_download_time)
    # obs.append(max_download_time)
    # obs.append(min_download_time)

    obs.append(normalized_mean_download_time)
    obs.append(normalized_max_download_time)
    obs.append(normalized_min_download_time)

    #完成率
    CR = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        completeRate = wf.getCompleteRate()
        CR.append(completeRate)

    mean_CR = sum(CR) / len(CR)
    max_CR = max(CR)
    min_CR = min(CR)

    obs.append(mean_CR)
    obs.append(max_CR)
    obs.append(min_CR)


    return obs

def get_obs_S_with_deadline(env, server):
    obs = []
    #当前列表中的任务数量
    job_num = len(server.wait_list)
    #处理时间的归一化均值 标准差 最小值 最大值
    max_pt = 150000 * 1e6 /server.pa
    min_pt = 50000 *1e6 /server.pa
    pt = [job.workload / server.pa for job in server.wait_list ]
    pt_normalized = [(x - min_pt) / (max_pt - min_pt) for x in pt]

    mean_pt = np.mean(pt_normalized)
    std_pt = np.std(pt_normalized)
    max_pt = np.max(pt_normalized)
    min_pt = np.min(pt_normalized)
    #处理时间+传输时间的归一化均值 标准差 最小值 最大值

    #完成率的均值 标准差 最小值 最大值
    CR = []
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        completeRate = wf.getCompleteRate()
        CR.append(completeRate)

    mean_CR = np.mean(CR)
    std_CR = np.std(CR)
    max_CR = np.max(CR)
    min_CR = np.min(CR)




    #松弛时间的归一化均值 标准差 最大值 最小值
    slack_time = []
    estimated_delayed_job_rate = []  #预计延迟作业比
    for job in server.wait_list:
        wf = env.workflows[job.workflow_id]
        ect,djr = wf.getEstimatedCompletionTimeAndDelayedJobRate(env)
        deadline = wf.deadline  ##都是绝对时间
        slack = deadline - ect
        slack_time.append(slack)
        estimated_delayed_job_rate.append(djr)

    min_slack = min(slack_time)
    max_slack = max(slack_time)
    if max_slack == min_slack:
        slack_time_normalized = [0] * len(slack_time)  # Or choose another handling method
    else:
        slack_time_normalized = [(x - min_slack) / (max_slack - min_slack) for x in slack_time]

    mean_slack_time = np.mean(slack_time_normalized)
    std_slack_time = np.std(slack_time_normalized)  #同一实例里进行归一化 其最小值最大值就是0 和 1 不存在意义



    #延期率的均值 标准差 最大值 最小值  延期率的定义:延期作业数占比 / 延期作业量占比 / max（预计完成时间 - deadline, 0）÷ 预计完成时间
    mean_EDJR = np.mean(estimated_delayed_job_rate)
    std_EDJR = np.std(estimated_delayed_job_rate)
    max_EDJR = np.max(estimated_delayed_job_rate)
    min_EDJR = np.min(estimated_delayed_job_rate)

    # 当前机器的 预计延期的工作流占比
    delayed_workflow_rate = sum(x > 0 for x in estimated_delayed_job_rate) / len(estimated_delayed_job_rate)


    obs.append(job_num/10)

    obs.append(mean_pt)
    obs.append(std_pt)
    obs.append(max_pt)
    obs.append(min_pt)

    obs.append(mean_CR)
    obs.append(std_CR)
    obs.append(max_CR)
    obs.append(min_CR)

    obs.append(mean_slack_time)
    obs.append(std_slack_time)

    obs.append(mean_EDJR)
    obs.append(std_EDJR)
    obs.append(max_EDJR)
    obs.append(min_EDJR)

    obs.append(delayed_workflow_rate)
    return obs





    #实际延期率的均值 标准差 最小值 最大值（可选


    #当前 系统 已到达的工作流的预计延期率

    # 当前 系统 已到达的工作流 的实际延期率

