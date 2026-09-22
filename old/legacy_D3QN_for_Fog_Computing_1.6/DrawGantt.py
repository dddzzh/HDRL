import os
import pickle
from D3QN import D3QN
from CreateDataset import create_random_dataset
from Event import Event, workflowArrivalEvent, jobArrivalEvent, jobStartUploadEvent, jobStartProcessEvent, \
    jobFinishedEvent, jobStartDownloadEvent
import heapq
from obs import  get_obs_R
import matplotlib.pyplot as plt
from SequencingAction_with_deadline import FCFS,MS

def draw_gantt(env):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.get_cmap('tab20').colors  # 颜色调色板
    color_idx = 0

    for server in env.servers:
        print(f"{server.name} :带宽{server.bandwidth} 计算速度{server.pa}")
    for device in env.devices:
        print(f"{device.name} 计算速度{device.pa}")

    # 绘制设备的任务
    for idx, device in enumerate(env.devices):
        for job in device.record:
            start = job.startProcessingTime
            end = job.endProcessingTime
            duration = end - start
            ax.broken_barh([(start, duration)], (idx - 0.4, 0.8),
                           facecolors=(colors[color_idx % len(colors)]))
            color_idx += 1

    # 绘制服务器的任务
    server_offset = len(env.devices)
    for idx, server in enumerate(env.servers):
        for job in server.record:
            start = job.startProcessingTime
            end = job.endProcessingTime
            duration = end - start
            ax.broken_barh([(start, duration)], (server_offset + idx - 0.4, 0.8),
                           facecolors=(colors[color_idx % len(colors)]))
            color_idx += 1

    # 设置轴标签和刻度
    ax.set_ylim(-1, len(env.servers) + len(env.devices))
    ax.set_xlim(left=0)
    ax.set_xlabel('Time')
    ax.set_ylabel('Devices and Servers')

    # 设置 y 轴刻度和标签
    y_ticks = list(range(len(env.servers) + len(env.devices)))
    y_labels = [device.name for device in env.devices] + [server.name for server in env.servers]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    ax.grid(True)
    plt.title('Gantt Chart for Devices and Servers')
    plt.show()




if __name__ == '__main__':
    model_dir = './checkpoints/D3QN/RA0911_0036smallD3E10C10W30L0.15_with_deadline/RoutingAgent/'
    # 导入实例和模型参数
    load_path = os.path.join(model_dir, 'instance_configs.pkl')
    with open(load_path, 'rb') as f:
        InsConfigs = pickle.load(f)
    InsConfigs.print_params()

    load_path = os.path.join(model_dir, 'routing_agent_configs.pkl')
    with open(load_path, 'rb') as f:
        RAConfigs = pickle.load(f)

    # 导入模型
    agent_R = D3QN(
        alpha=RAConfigs.alpha,  # 学习率
        state_dim=5 * (InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1),  # 状态空间
        action_dim=InsConfigs.numEdgeServers + InsConfigs.numCenterServers + 1,  # 动作空间
        fc1_dim=RAConfigs.fc1_dim, fc2_dim=RAConfigs.fc2_dim,  # 模型参数
        ckpt_dir=model_dir,  # 保存地址
        gamma=RAConfigs.gamma, tau=RAConfigs.tau,  # 折扣系数
        epsilon=RAConfigs.epsilon, eps_end=RAConfigs.eps_end, eps_dec=RAConfigs.eps_dec,  # 探索率相关
        max_size=RAConfigs.max_size, batch_size=RAConfigs.batch_size)  # memory buffer大小 训练批次大小
    agent_R.load_models(400)

    env = create_random_dataset(InsConfigs.numMobileDevices,
                                InsConfigs.numEdgeServers,
                                InsConfigs.numCenterServers,
                                InsConfigs.numWorkflows,
                                InsConfigs.lambdaRate,
                                InsConfigs.datasetScale)

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
                #job = server.wait_list[0]
                job = MS(env, server)
                env.addEvent(jobStartProcessEvent(env.workflows[job.workflow_id], job, server,
                                                  env.current_time))
                continue

        #3、处理当前时刻待进行的 Routing Decision
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
        # # 3、处理当前时刻待进行的 Routing Decision
        # if env.ready_job_set:
        #     select_job = min(env.ready_job_set, key=lambda job: job.workload)
        #
        #     EarliestFinishTime = []
        #     for server in env.servers:
        #         jobs = server.trans_list + server.wait_list
        #         total_workload = sum(job.workload for job in jobs)
        #         Allpt = total_workload / server.pa
        #         if server.isBusy == True:
        #             EAT = server.processing_job.endProcessingTime - env.current_time
        #         else:
        #             EAT = 0
        #         time_of_this_job = (select_job.data_up+select_job.data_down) / server.bandwidth + select_job.workload / server.pa
        #
        #         EarliestFinishTime.append(Allpt + EAT+time_of_this_job)
        #     device = env.devices[select_job.sourceDeviceId]
        #     if device.isBusy == True:
        #         EAT = device.processing_job.endProcessingTime - env.current_time
        #     else:
        #         EAT = 0
        #
        #     total_workload = sum(job.workload for job in device.wait_list)
        #     Allpt = total_workload / device.pa
        #     time_of_this_job = select_job.workload / device.pa
        #     EarliestFinishTime.append(Allpt + EAT + time_of_this_job)
        #
        #     action = EarliestFinishTime.index(min(EarliestFinishTime))
        #
        #     if action == len(env.servers):  # 表示本地卸载
        #         env.addEvent(
        #             jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job,
        #                                 env.devices[select_job.sourceDeviceId],
        #                                 env.current_time))
        #     else:
        #         env.addEvent(
        #             jobStartUploadEvent(env.workflows[select_job.workflow_id], select_job, env.servers[action],
        #                                 env.current_time))
        #
        #     continue

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
    makespan = 0
    tardiness = 0
    for wf in env.workflows:
        max_finish_time_job = max(wf.jobs, key=lambda job: job.finishTime)
        makespan += max_finish_time_job.finishTime - wf.arrivalTime
        tardiness += max(max_finish_time_job.finishTime - wf.deadline,0)
    print(f"makespan为{makespan}, tardiness为{tardiness}")
    draw_gantt(env)