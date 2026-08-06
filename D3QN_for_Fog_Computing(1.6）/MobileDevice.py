class MobileDevice:
    def __init__(self, md_id, bandwidth, pa,  server_type, name):

        self.id = md_id  # id
        self.bandwidth = bandwidth  # 带宽
        self.pa = pa  # 处理能力
        self.isBusy = False
        self.wait_list = []  # 存储正在等待的任务 其中每个元素是job对象  排序直接按照先来后到的顺序
        self.server_type = server_type  # 服务器种类
        self.name = name
        self.record = []
        self.last_free_time = 0 #记录上一次空闲时间
        self.processing_job = None #记录正在处理的任务