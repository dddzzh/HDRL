from Params import configs
import random
from ReadXML import generate_workflow_job_from_xml_file
from Workflow import Workflow, Job
from Server import Server
import numpy as np
from SimulationEnvironment import SimEnv
import copy
import math
import pickle
switcher = {
    # small
    0: "CyberShake_30.xml",
    1: "Epigenomics_24.xml",
    2: "Inspiral_30.xml",
    3: "Montage_25.xml",
    4: "Sipht_30.xml",
    # middle
    5: "CyberShake_50.xml",
    6: "Epigenomics_46.xml",
    7: "Inspiral_50.xml",
    8: "Montage_50.xml",
    9: "Sipht_60.xml",
    # large
    10: "CyberShake_100.xml",
    11: "Epigenomics_100.xml",
    12: "Inspiral_100.xml",
    13: "Montage_100.xml",
    14: "Sipht_100.xml",
    # huge
    15: "CyberShake_1000.xml",
    16: "Epigenomics_997.xml",
    17: "Inspiral_1000.xml",
    18: "Montage_1000.xml",
    19: "Sipht_1000.xml"
}
import networkx as nx

def transmission_rate(d):
    B = 1e8 #单位bps  相当于 100MHZ
    P = 1 #单位w
    w_0 = 1e-13 # -100dBm  参考BIJOR论文
    noise_from_other = 1e-11 # -80dBm

    g_i_cloud = d**-4  #参考BIJOR论文 他论文中有引用
    r_i_cloud = B * math.log2(1 + (P * g_i_cloud) / (w_0 + noise_from_other ))

    return r_i_cloud #单位bps
def get_dag_type_path(dag_type):
    base_path = "data/"
    dag_type_path = base_path + switcher.get(dag_type, "DAG path error!!!")
    return dag_type_path


def createTimeTable(numMobileDevices, numWorkflows, lambdaRate):
    timeTable = []
    for _ in range(numMobileDevices):
        workflow_list = [0]  # 第一个数为0
        for _ in range(1, numWorkflows):
            interval = random.expovariate(lambdaRate)
            workflow_list.append(workflow_list[-1] + interval)
        timeTable.append(workflow_list)

    # 截取前numWorkflows个时间
    all_times = [(time, i) for i, sublist in enumerate(timeTable) for time in sublist]
    sorted_times = sorted(all_times, key=lambda x: x[0])
    top_times = sorted_times[:numWorkflows]
    return top_times  # [(时间，设备id )，(时间，设备id ) ...]


def createWorkflows(timeTable, datasetScale):
    workflows = []
    wf_id = 0
    for releaseTime, deviceId in timeTable:
        if datasetScale == "small":
            dag_type = random.randint(0, 4)
        elif datasetScale == "middle":
            dag_type = random.randint(0, 9)
        elif datasetScale == "large":
            dag_type = random.randint(0, 14)

        dag_path = get_dag_type_path(dag_type)
        G = generate_workflow_job_from_xml_file(dag_path)
        numJobs = nx.number_of_nodes(G)
        data_up = [random.uniform(720, 2880) for _ in range(numJobs)]  # 单位MB
        data_up = [x * 1024 * 1024 * 8 for x in data_up]  # 转换成单位b方便计算
        data_down = [random.uniform(720, 2880) for _ in range(numJobs)]  # 单位MB
        data_down = [x * 1024 * 1024 * 8 for x in data_down]  # 转换成单位b方便统一计算
        workload = [random.randint(50000, 150000) for _ in range(numJobs)]  # 单位MegaCycles
        workload = [x * 1e6 for x in workload]  # 转换成单位 cycles
        jobs = []
        for job_id in range(0, numJobs):
            jobs.append(Job(job_id, wf_id, data_up[job_id], data_down[job_id], workload[job_id], deviceId))
        dagtype = switcher.get(dag_type, "Invalid")
        wf = Workflow(wf_id, dagtype, releaseTime, G, jobs, deviceId)
        workflows.append(wf)
        wf_id += 1


    return workflows


def createServers(numEdgeServers, numCenterServers):
    ess = []
    for ed_id in range(0, numEdgeServers):
        distance = random.uniform(50, 100)
        bandwidth = transmission_rate(distance)  # 写作bandwidth 其实是传输速率 单位b/s
        pa = random.uniform(2.5, 5)  # 单位 GHZ
        pa = pa * 1e9  # 单位HZ
        name = "EdgeServer_" + str(ed_id)
        es = Server(ed_id, bandwidth, pa, "EdgeServer", name)
        ess.append(es)
    css = []
    for cs_id in range(numEdgeServers, numEdgeServers + numCenterServers):
        rand = random.random()
        if rand < 0.2:
            # 20% 概率选择 200~300
            distance = random.uniform(200, 300)
        elif rand < 0.8:
            # 60% 概率选择 300~400
            distance = random.uniform(300, 400)
        else:
            # 20% 概率选择 400~500
            distance = random.uniform(400, 500)
        bandwidth = transmission_rate(distance)  # 写作bandwidth 其实是传输速率 单位b/s

        pa = random.uniform(5, 10)  # 单位 GHZ
        pa = pa * 1e9  # 单位HZ

        name = "CenterServer_" + str(cs_id - numEdgeServers)
        cs = Server(cs_id, bandwidth, pa, "CenterServer", name)
        css.append(cs)

    servers = ess + css
    return  servers

def createMobileDevice(numMobileDevices):
    mds = []
    for md_id in range(numMobileDevices):
        bandwidth = np.random.choice([128, 256, 512], p=[0.2, 0.6, 0.2])
        pa = random.uniform(1.2, 2.5)
        pa = pa * 1e9
        name = "MobileDevice_" + str(md_id)
        md = Server(md_id, bandwidth, pa, "MobileDevice", name)
        mds.append(md)
    return mds


def create_random_dataset(numMobileDevices, numEdgeServers, numCenterServers, numWorkflows, lambdaRate, datasetScale):

    # numMobileDevices = configs.numMobileDevices  # 默认3
    # numEdgeServers = configs.numEdgeServers  # 默认20
    # numCenterServers = configs.numCenterServers  # 默认20
    # numWorkflows = configs.numWorkflows  # 默认20
    # lambdaRate = configs.lambdaRate  # 默认0.01
    # datasetScale = configs.datasetScale  # 默认small

    timeTable = createTimeTable(numMobileDevices, numWorkflows, lambdaRate)
    wfs = createWorkflows(timeTable, datasetScale)
    mds = createMobileDevice(numMobileDevices)
    servers = createServers(numEdgeServers, numCenterServers)

    mean_pa = sum(server.pa for server in servers) / len(servers)
    mean_transmition = sum(server.bandwidth for server in servers) / len(servers)

    envState = SimEnv(mds, wfs, servers)
    for wf in envState.workflows:
        ect = wf.getEstimatedCompletionTime(envState)
        wf.deadline = wf.arrivalTime + (ect - wf.arrivalTime) * random.uniform(1.5, 2.5)
        wf.calculate_upward_ranks(mean_pa, mean_transmition)

    envState.initial_state['workflows'] = copy.deepcopy(envState.workflows)
    #print(envState.workflows[0].deadline)
    #print(envState)
    return envState


if __name__ == '__main__':

    #工作流
    env = create_random_dataset(configs.numMobileDevices ,
                                configs.numEdgeServers,
                                configs.numCenterServers,
                                configs.numWorkflows,
                                configs.lambdaRate,
                                configs.datasetScale)
    all = 0
    for wf in env.workflows:
        all += wf.getEstimatedCompletionTime(env) - wf.arrivalTime
    print(all)

