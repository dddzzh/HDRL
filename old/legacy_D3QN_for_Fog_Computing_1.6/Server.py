
class Server:
    def __init__(self, es_id, bandwidth, pa,  server_type, name):
        self.id = es_id  # id
        self.bandwidth = bandwidth  # 带宽
        self.pa = pa  # 处理能力
        self.record = []  #用于记录在该设备上完成的任务  直接存储job对象
        self.isBusy = False
        self.wait_list = []  # 存储正在等待的任务 其中每个元素是job对象
        self.trans_list = [] #传输中的任务列表
        self.processing_job = None
        self.server_type = server_type  # 服务器种类
        self.name = name

        self.memory = [] #用来存储这个server上的[状态,动作,奖励]


        self.last_state = [0]*16