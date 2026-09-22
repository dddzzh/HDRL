import networkx as nx
import random
from ReadXML import generate_workflow_job_from_xml_file
from matplotlib import patches, pyplot as plt
class Job:
    def __init__(self, job_id, workflow_id, data_up, data_down, workload, sourceDeviceId):
        self.id = job_id
        self.workflow_id = workflow_id
        self.state = "未准备"  # 已准备 上传中 排队中 处理中 下载中 已完成
        self.data_up = data_up
        self.data_down = data_down
        self.workload = workload

        self.readyTime = None
        self.arrivalTime = None
        self.startProcessingTime = None
        self.endProcessingTime = None
        self.finishTime = None
        self.sourceDeviceId = sourceDeviceId
        self.assignedServer = None  #分配到哪台服务器上 -1表示本地卸载

        self.S = None
        self.A = None
        self.R = None
        self.next_S = None
        self.terminal = False


        self.UpwardRank = None #向上等级排序
        self.DownwardRank = None


class Workflow:
    def __init__(self, wf_id, type, arrivalTime, G, jobs, sourceDeviceId):
        super().__init__()
        self.id = wf_id  # 工作流id
        self.type = type # 工作流类型
        self.graph = G  #networkx DAG图
        self.num = self.graph.number_of_nodes()  # 节点数量 不算起始节点和结束节点
        self.jobs = jobs  # jobs索引从0开始
        self.arrivalTime = arrivalTime  # 开始时间
        self.finishTime = None #结束时间
        self.sourceDeviceId = sourceDeviceId
        self.isArrived = False

        self.deadline = None
    def isReady(self, job):
        # 遍历子任务的所有先驱节点
        for predecessor in self.graph.predecessors(job.id):
            predecessor_job = self.jobs[predecessor]
            if predecessor_job.state != "已完成":
                return False
        return True
    def getALlReadyJob(self):
        for job in self.jobs:
            if job.state == "未准备":
                if self.isReady(job):
                    job.state = "已准备"

        ready_job_list = []
        for job in self.jobs:
            if job.state == "已准备":
                ready_job_list.append(job)
        return ready_job_list

    def getCompleteRate(self):
        total_workload = sum(job.workload for job in self.jobs)  # 总工作量
        incomplete_workload = sum(job.workload for job in self.jobs if job.state != "已完成")  # 未完成任务的总工作量
        if total_workload == 0:
            return 0  # 如果总工作量为0，返回0避免除零错误
        complete_rate = 1 - (incomplete_workload / total_workload)
        return complete_rate

    #计算向上等级排序
    def calculate_upward_ranks(self, avg_processing_speed, avg_transmission_speed):

        for job in self.jobs:
            job.UpwardRank = 0

        topological_order = list(nx.topological_sort(self.graph))

        for job_id in reversed(topological_order):
            current_job = self.jobs[job_id]

            upload_time = current_job.data_up / avg_transmission_speed
            download_time = current_job.data_down / avg_transmission_speed
            computation_time = current_job.workload / avg_processing_speed
            current_job_processing_time = upload_time + computation_time + download_time

            max_upward_rank = 0
            for successor_id in self.graph.successors(job_id):
                successor_job = self.jobs[successor_id]
                if successor_job.UpwardRank is not None:
                    max_upward_rank = max(max_upward_rank, successor_job.UpwardRank)

            current_job.UpwardRank = max_upward_rank + current_job_processing_time
    #计算向下等级排序
    def calculate_downward_ranks(self, avg_processing_speed, avg_transmission_speed):
        # 初始化每个任务的向下等级排序
        for job in self.jobs:
            job.DownwardRank = 0

        # 拓扑排序以确保正确的依赖顺序（从起始节点开始）
        topological_order = list(nx.topological_sort(self.graph))

        # 按拓扑顺序遍历节点，计算向下等级
        for job_id in topological_order:
            current_job = self.jobs[job_id]

            # 计算任务的上传、计算和下载时间
            upload_time = current_job.data_up / avg_transmission_speed
            download_time = current_job.data_down / avg_transmission_speed
            computation_time = current_job.workload / avg_processing_speed
            current_job_processing_time = upload_time + computation_time + download_time

            # 计算向下等级（从当前任务到结束节点的最长路径）
            max_downward_rank = 0
            for predecessor_id in self.graph.predecessors(job_id):
                predecessor_job = self.jobs[predecessor_id]
                if predecessor_job.DownwardRank is not None:
                    max_downward_rank = max(max_downward_rank, predecessor_job.DownwardRank)

            current_job.DownwardRank = max_downward_rank + current_job_processing_time

    def getEstimatedCompletionTime(self, env):
        mean_pa = sum(server.pa for server  in env.servers) / len(env.servers)
        mean_bandwidth = sum(server.bandwidth for server  in env.servers) / len(env.servers)
        topo_sorted_jobs = list(nx.topological_sort(self.graph))
        finish_times = {job_id: 0 for job_id in topo_sorted_jobs}  #记录的是绝对时间

        for job_id in topo_sorted_jobs:#根据拓扑排序遍历任务
            job = self.jobs[job_id]

            if job.finishTime != None:   #如果当前任务已经确定完成时间 （处理中、下载中、已完成的任务）
                finish_times[job_id] = job.finishTime
            else:
                #如果未完成  那有几种情况 （未准备 已准备 上传中 排队中） 如果是
                #未准备的情况 开始时间要根据前序任务的完成时间估算
                #已准备的情况 开始时间是确定的
                #上传中  到达时间是确定的 处理时间是按照上传到的机器来计算 是否要根据排队时间估算有待商榷
                #排队中 计算预计的处理时间
                if job.state == "未准备":
                    # 计算预计开始时间  如果有前序 那么其开始时间估计为前序的最大完成时间 否则为开始时间
                    ready_time = max(finish_times[pred] for pred in self.graph.predecessors(job_id)) if list(
                        self.graph.predecessors(job_id)) else self.arrivalTime
                    upload_time = job.data_up / mean_bandwidth
                    processing_time = job.workload / mean_pa
                    download_time = job.data_down / mean_bandwidth
                    finish_times[job_id] = ready_time + upload_time + processing_time + download_time

                elif job.state == "已准备":
                    ready_time = job.readyTime
                    upload_time = job.data_up / mean_bandwidth
                    processing_time = job.workload / mean_pa
                    download_time = job.data_down / mean_bandwidth
                    finish_times[job_id] = ready_time + upload_time + processing_time + download_time

                elif job.state == "上传中":
                    if job.assignedServer == -1: #表示本地卸载
                        pa = env.devices[job.sourceDeviceId].pa
                        finish_times[job_id] = job.workload / pa + job.readyTime
                    else: #表示卸载到某台服务器上
                        bandwidth = env.servers[job.assignedServer].bandwidth
                        pa = env.servers[job.assignedServer].pa
                        finish_times[job_id] = job.arrivalTime + job.workload / pa + job.data_down / bandwidth

                elif job.state == "排队中":
                    if job.assignedServer == -1: #表示本地卸载
                        pa = env.devices[job.sourceDeviceId].pa
                        if env.devices[job.sourceDeviceId].isBusy == True:
                            finish_times[job_id] = env.devices[job.sourceDeviceId].processing_job.finishTime + job.workload / pa
                        else :
                            finish_times[job_id] = env.current_time + job.workload / pa
                    else:
                        pa = env.servers[job.assignedServer].pa
                        bandwidth = env.servers[job.assignedServer].bandwidth
                        if env.servers[job.assignedServer].isBusy == True:

                            finish_times[job_id] = env.servers[job.assignedServer].processing_job.finishTime + job.workload / pa +job.data_down /bandwidth
                        else :
                            finish_times[job_id] = env.current_time + job.workload / pa + job.data_down / bandwidth
        for job_id, finish_time in finish_times.items():
            if finish_time is None:
                print(f"Job ID {job_id} has a finish time of None.")

        EstimatedCompletionTime = max(finish_times.values())
        return EstimatedCompletionTime

    def getEstimatedCompletionTimeAndDelayedJobRate(self,env): #预计延期作业占比
        mean_pa = sum(server.pa for server  in env.servers) / len(env.servers)
        mean_bandwidth = sum(server.bandwidth for server  in env.servers) / len(env.servers)
        topo_sorted_jobs = list(nx.topological_sort(self.graph))
        finish_times = {job_id: 0 for job_id in topo_sorted_jobs}  #记录的是绝对时间

        for job_id in topo_sorted_jobs:#根据拓扑排序遍历任务
            job = self.jobs[job_id]

            if job.finishTime != None:   #如果当前任务已经确定完成时间 （处理中、下载中、已完成的任务）
                finish_times[job_id] = job.finishTime
            else:
                #如果未完成  那有几种情况 （未准备 已准备 上传中 排队中） 如果是
                #未准备的情况 开始时间要根据前序任务的完成时间估算
                #已准备的情况 开始时间是确定的
                #上传中  到达时间是确定的 处理时间是按照上传到的机器来计算 是否要根据排队时间估算有待商榷
                #排队中 计算预计的处理时间
                if job.state == "未准备":
                    # 计算预计开始时间  如果有前序 那么其开始时间估计为前序的最大完成时间 否则为开始时间
                    ready_time = max(finish_times[pred] for pred in self.graph.predecessors(job_id)) if list(
                        self.graph.predecessors(job_id)) else self.arrivalTime
                    upload_time = job.data_up / mean_bandwidth
                    processing_time = job.workload / mean_pa
                    download_time = job.data_down / mean_bandwidth
                    finish_times[job_id] = ready_time + upload_time + processing_time + download_time

                elif job.state == "已准备":
                    ready_time = job.readyTime
                    upload_time = job.data_up / mean_bandwidth
                    processing_time = job.workload / mean_pa
                    download_time = job.data_down / mean_bandwidth
                    finish_times[job_id] = ready_time + upload_time + processing_time + download_time

                elif job.state == "上传中":
                    if job.assignedServer == -1: #表示本地卸载
                        pa = env.devices[job.sourceDeviceId].pa
                        finish_times[job_id] = job.workload / pa + job.readyTime
                    else: #表示卸载到某台服务器上
                        bandwidth = env.servers[job.assignedServer].bandwidth
                        pa = env.servers[job.assignedServer].pa
                        finish_times[job_id] = job.arrivalTime + job.workload / pa + job.data_down / bandwidth

                elif job.state == "排队中":
                    if job.assignedServer == -1: #表示本地卸载
                        pa = env.devices[job.sourceDeviceId].pa
                        if env.devices[job.sourceDeviceId].isBusy == True:
                            finish_times[job_id] = env.devices[job.sourceDeviceId].processing_job.finishTime + job.workload / pa
                        else :
                            finish_times[job_id] = env.current_time + job.workload / pa
                    else:
                        pa = env.servers[job.assignedServer].pa
                        bandwidth = env.servers[job.assignedServer].bandwidth
                        if env.servers[job.assignedServer].isBusy == True:

                            finish_times[job_id] = env.servers[job.assignedServer].processing_job.finishTime + job.workload / pa +job.data_down /bandwidth
                        else :
                            finish_times[job_id] = env.current_time + job.workload / pa + job.data_down / bandwidth


        EstimatedCompletionTime = max(finish_times.values())

        exceed_deadline_count = sum(1 for finish_time in finish_times.values() if finish_time > self.deadline)
        # 计算超过截止时间的任务数量占总数的比值
        exceed_ratio = exceed_deadline_count / len(finish_times)

        return EstimatedCompletionTime, exceed_ratio

    #def calUpwardRank(self, job, mean_bandwidth, mean_pa):
    # def calUpwardRank(self, mean_bandwidth, mean_pa):
    #     def compute_job_time(job):
    #         # 上传时间
    #         upload_time = job.data_up / mean_bandwidth
    #         # 处理时间
    #         processing_time = job.workload / mean_pa
    #         # 下载时间
    #         download_time = job.data_down / mean_bandwidth
    #         return upload_time, processing_time, download_time
    #
    #     def compute_upward_rank(job_id):
    #         job = self.jobs[job_id]
    #         # 获取上传时间、处理时间、下载时间
    #         upload_time, processing_time, download_time = compute_job_time(job)
    #
    #         if self.graph.out_degree(job_id) == 0:  # 没有子任务
    #             job.UpwardRank = processing_time + download_time
    #         else:
    #             max_child_rank = 0
    #             for successor in self.graph.successors(job_id):
    #                 if self.jobs[successor].UpwardRank is None:
    #                     compute_upward_rank(successor)
    #                 max_child_rank = max(max_child_rank, self.jobs[successor].UpwardRank)
    #
    #             job.UpwardRank = upload_time + processing_time + max_child_rank
    #
    #     # 从所有的入口节点开始计算
    #     for job_id in self.graph.nodes:
    #         if self.graph.in_degree(job_id) == 0:  # 入口节点
    #             compute_upward_rank(job_id)



    def draw(self):
        g = self.graph
        for layer, nodes in enumerate(nx.topological_generations(g)):
            for node in nodes:
                g.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(g, subset_key="layer")  # 使用 Fruchterman-Reingold 布局算法

        plt.figure(figsize=(10, 6))
        nx.draw(g, pos, with_labels=True, node_size=500, node_color="lightblue")
        plt.title(f"DAG for Workflow {self.id}")
        plt.show()