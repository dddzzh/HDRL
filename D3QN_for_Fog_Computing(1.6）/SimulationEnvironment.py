import heapq
import networkx as nx
import copy
from Event import workflowArrivalEvent
from Params import configs


class SimEnv:
    def __init__(self, devices,workflows, servers):
        self.devices = devices
        self.servers = servers
        self.workflows = workflows
        self.event_list = []  # 存储事件对象
        self.current_time = 0  # 当前时间
        self.ready_job_set = set()  # 已经ready的任务列表 三元组（workflow_id, job_id, 到达时间 )

        self.obs_space_R = 5 * (len(self.servers) + 1)
        self.action_space_R = len(self.servers) + 1 #动作空间是所有服务器 + 本地卸载

        for wf in self.workflows:
            self.addEvent(workflowArrivalEvent(wf, None, None, wf.arrivalTime))
        #初始状态保存了工作流到达事件
        self.initial_state = copy.deepcopy({
            'devices': self.devices,
            'servers': self.servers,
            'workflows': self.workflows,
            'event_list': self.event_list,
            'current_time': self.current_time,
            'ready_job_set': self.ready_job_set
        })

    def reset(self):
        # 将初始状态深拷贝回来
        initial = copy.deepcopy(self.initial_state)
        self.devices = initial['devices']
        self.servers = initial['servers']
        self.workflows = initial['workflows']
        self.event_list = initial['event_list']
        self.current_time = initial['current_time']
        self.ready_job_set = initial['ready_job_set']

    def __str__(self):

        device_info = "\n设备信息:\n"
        for device in self.devices:
            device_info += f" {device.name} - 忙碌状态: {device.isBusy}, 等待任务数: {len(device.wait_list)}\n"

        server_info = "\n服务器信息:\n"
        for server in self.servers:
            server_info += f" {server.name} - 忙碌状态: {server.isBusy}, 等待任务数: {len(server.wait_list)}\n"

        workflow_info = "\n工作流信息:\n"
        for workflow in self.workflows:
            workflow_info += f"工作流 {workflow.id} - 到达状态: {workflow.isArrived}, 类型: {workflow.type}, 到达时间: {workflow.arrivalTime}, 来自设备: {workflow.sourceDeviceId}\n"
            for job in workflow.jobs:
                workflow_info += f"  任务 {job.id} - 完成状态: {job.state}, 上行数据量: {job.data_up}, 下行数据量: {job.data_down}, 工作量: {job.workload}\n"

        return f"仿真环境当前时间 {self.current_time}:\n{device_info}{server_info}{workflow_info}"

    def addEvent(self, event):
        heapq.heappush(self.event_list, event)  # 按照优先级插入事件
