from xml.etree import ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt


class Task:
    def __init__(self, task_id):
        self.task_id = task_id
        self.parents = []
        self.children = []


def generate_workflow_job_from_xml_file(path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        mName2Task = {}
        job_id_starts_from = 0
        task_list = []

        # Define the namespace
        namespace = {'dax': 'http://pegasus.isi.edu/schema/DAX'}

        # Update element names with the namespace
        for node in root.findall('dax:job', namespace):
            node_id = node.get('id')
            node_name = node.get('name')

            task = Task(job_id_starts_from)
            job_id_starts_from += 1
            mName2Task[node_id] = task
            task_list.append(task)

        for node in root.findall('dax:child', namespace):
            child_name = node.get('ref')
            if child_name in mName2Task:
                child_task = mName2Task[child_name]

                for parent in node.findall('dax:parent', namespace):
                    parent_name = parent.get('ref')
                    if parent_name in mName2Task:
                        parent_task = mName2Task[parent_name]
                        parent_task.children.append(child_task)
                        child_task.parents.append(parent_task)

        # Clean up to save memory
        mName2Task.clear()
        G = nx.DiGraph()
        for task in task_list:
            G.add_node(task.task_id)
            for parent in task.parents:
                G.add_edge(parent.task_id, task.task_id)
        return G

    except ET.ParseError:
        print("XML Parsing Error; Please make sure your XML file is valid.")
        return []


def visualize_workflow(tasks):
    G = nx.DiGraph()

    for task in tasks:
        G.add_node(task.task_id)

        for parent in task.parents:
            G.add_edge(parent.task_id, task.task_id)

    # 选择不同的布局算法，这里使用了circular布局
    pos = nx.circular_layout(G)



    is_dag = nx.is_directed_acyclic_graph(G)
    if is_dag:
        print("G 是有向无环图（DAG）")
    else:
        print("G 不是有向无环图（DAG）")
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black",
            font_weight="bold", arrowsize=10)

    plt.title("Workflow Job Visualization")
    plt.show()

