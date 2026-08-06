from reward import get_reward_R,get_reward_R1



class Event:
    def __init__(self, event_type, workflow, job, server, time, priority):
        self.event_type = event_type
        self.workflow = workflow
        self.job = job
        self.server = server
        self.time = time
        self.priority = priority

    def __lt__(self, other):
        # 先比较时间，时间相同时再比较优先级
        if self.time == other.time:
            return self.priority < other.priority
        else:
            return self.time < other.time


#工作流到达
class workflowArrivalEvent(Event):  # 工作流抵达事件
    def __init__(self, workflow, job, server, time):
        super().__init__('workflowArrivalEvent', workflow, job, server, time, priority=2)

    def trigger(self, env):
            # 修改工作流为已到达
            self.workflow.isArrived = True
            # 获取当前这个工作流已经准备完成的任务集合
            ready_job_set = set(self.workflow.getALlReadyJob())
            #记录准备完成时间
            for job in ready_job_set:
                job.readyTime = self.time
            #  将当前工作流所有准备完成的任务添加入集合
            env.ready_job_set.update(ready_job_set)
#任务开始上传
class jobStartUploadEvent(Event): #任务开始上传事件
    def __init__(self, workflow, job, server, time):
        super().__init__('jobStartUploadEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        self.job.state = "上传中"
        env.ready_job_set.remove(self.job)
        if self.server.server_type == "MobileDevice" and self.workflow.sourceDeviceId == self.server.id:
            self.job.assignedServer = -1 #表示本地卸载
            upload_time = 0
        else:
            self.job.assignedServer = self.server.id
            upload_time = self.job.data_up / self.server.bandwidth
            self.server.trans_list.append(self.job) #加入上传中列表

        self.job.arrivalTime = self.time + upload_time  #保存预计到达时间
        env.addEvent(jobArrivalEvent(self.workflow, self.job, self.server, self.time + upload_time))




class jobArrivalEvent(Event):  #任务抵达事件

    def __init__(self, workflow, job, server, time):
        super().__init__('jobArrivalEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        self.job.state = "排队中"
        if self.server.server_type != "MobileDevice":
            self.server.trans_list.remove(self.job) #任务到达 从传输队列中移除 本地卸载没有传输队列
        self.server.wait_list.append(self.job)  #加入等待队列 等待SA调度



class jobStartProcessEvent(Event): #任务开始处理事件
    def __init__(self, workflow, job, server, time):
        super().__init__('jobStartProcessEvent', workflow, job, server, time, priority=1)

    def trigger(self, env):
        self.job.state = "处理中"
        self.server.processing_job = self.job
        self.server.wait_list.remove(self.job)
        # 将服务器修改为busy
        self.server.isBusy = True
        process_time = self.job.workload / self.server.pa

        if self.server.server_type =="MobileDevice" and self.workflow.sourceDeviceId == self.server.id:
            download_time = 0
        else:
            download_time = self.job.data_down / self.server.bandwidth


        self.job.startProcessingTime = self.time
        self.job.endProcessingTime = self.time + process_time
        self.job.finishTime = self.time + process_time + download_time

        self.job.R = get_reward_R1(env, self.job)  #在这里修改奖励函数

        env.addEvent(jobStartDownloadEvent(self.workflow, self.job, self.server, self.time + process_time))
        #print(f"工作流 {self.workflow.id}的任务{self.job.id} 时刻 {self.time} 在服务器 {self.server.id}上开始处理")

class jobStartDownloadEvent(Event): #任务开始下载事件
    def __init__(self, workflow, job, server, time):
        super().__init__('jobStartDownloadEvent', workflow, job, server, time, priority=4)

    def trigger(self, env):
        self.job.state = "下载中"
        self.server.last_free_time = self.time
        self.server.processing_job = None
        self.server.isBusy = False
        self.server.record.append(self.job)
        env.addEvent(jobFinishedEvent(self.workflow, self.job, self.server, self.job.finishTime))

        #print(f"工作流 {self.workflow.id}的任务{self.job.id} 在时刻 {self.time} 在服务器 {self.server.id}上处理完成， 开始返回服务器{self.workflow.from_user}")

class jobFinishedEvent(Event):#任务完成事件
    def __init__(self, workflow, job, server, time):
        super().__init__('jobFinishedEvent', workflow, job, server, time, priority=2)

    def trigger(self, env):
        self.job.state = "已完成"

        #获取并更新当前工作流新的可用任务
        ready_job_set=set(self.workflow.getALlReadyJob())
        for job in ready_job_set :
            if job.readyTime ==None:
                job.readyTime = env.current_time
        env.ready_job_set.update(ready_job_set)
#任务到达

#任务开始处理

#任务完成处理开始下载

#任务完成