import networkx as nx
from collections import defaultdict
import numpy as np
import torch

class Env(object):
    def __init__(self, examples, config, state_action_space=None):
        """Temporal Knowledge Graph Environment.
        examples: 四元组 (subject, relation, object, timestamps)；
        config: 配置字典；
        state_action_space: 预处理的动作空间；
        """
        self.config = config  # 保存环境的配置
        self.num_rel = config['num_rel']  # 获取关系数量
        # 根据传入的例子构建知识图谱和节点标签映射
        self.graph, self.label2nodes = self.build_graph(examples)
        # [0, num_rel) -> 正常关系；num_rel -> 不变操作，(num_rel, num_rel * 2] -> 反向关系
        self.NO_OP = self.num_rel  # 设置不变操作（NO_OP）
        self.ePAD = config['num_ent']  # 实体的填充符（Padding entity）
        self.rPAD = config['num_rel'] * 2 + 1  # 关系的填充符（Padding relation）
        self.tPAD = 0  # 时间的填充符（Padding time）
        self.state_action_space = state_action_space  # 预处理的动作空间
        if state_action_space:
            # 如果提供了预处理的动作空间，则获取其键（key）
            self.state_action_space_key = self.state_action_space.keys()

    def build_graph(self, examples):
        """图节点表示为 (实体, 时间)，边是有向并带标签的关系。
        返回：
            graph: nx.MultiDiGraph；构建的多重有向图。
            label2nodes: 一个字典，键是实体，值是该实体在图中的节点集合 (entity, time)。
        """
        graph = nx.MultiDiGraph()  # 创建一个空的多重有向图
        label2nodes = defaultdict(set)  # 创建一个默认值为set的字典，用于存储实体到节点的映射
        examples.sort(key=lambda x: x[3], reverse=True)  # 按照时间戳降序排序，确保按时间顺序处理
        for example in examples:
            src = example[0]  # 获取四元组中的源实体
            rel = example[1]  # 获取四元组中的关系
            dst = example[2]  # 获取四元组中的目标实体
            time = example[3]  # 获取四元组中的时间戳
            # 添加当前四元组的节点和边
            src_node = (src, time)  # 源实体节点，(实体, 时间)
            dst_node = (dst, time)  # 目标实体节点，(实体, 时间)
            # 如果源节点不存在，则添加源节点到图中
            if src_node not in label2nodes[src]:
                graph.add_node(src_node, label=src)
            # 如果目标节点不存在，则添加目标节点到图中
            if dst_node not in label2nodes[dst]:
                graph.add_node(dst_node, label=dst)
            # 添加从源节点到目标节点的边，关系为rel
            graph.add_edge(src_node, dst_node, relation=rel)
            # 添加从目标节点到源节点的反向边，关系为rel+self.num_rel+1
            graph.add_edge(dst_node, src_node, relation=rel + self.num_rel + 1)
            # 将源节点和目标节点添加到对应的实体标签节点集合中
            label2nodes[src].add(src_node)
            label2nodes[dst].add(dst_node)
        return graph, label2nodes  # 返回构建好的图和标签到节点的映射

    def get_state_actions_space_complete(self, entity, time, current_=False, max_action_num=None):
        """获取当前状态的动作空间。
        参数：
            entity: 当前状态的实体；
            time: 候选动作的最大时间戳；
            current_: 当前事件的时间是否可以被使用；
            max_action_num: 最大存储的事件数；
        返回：
            numpy 数组，形状为 [事件数量， 3]，(关系，目标实体，时间戳)
        """
        if self.state_action_space:  # 如果提供了预处理的动作空间
            # 如果当前状态的 (实体, 时间戳, 是否是当前事件) 存在于动作空间中，则直接返回该动作空间
            if (entity, time, current_) in self.state_action_space_key:
                return self.state_action_space[(entity, time, current_)]
        # 获取该实体在图中的所有节点
        nodes = self.label2nodes[entity].copy()  # 复制实体对应的节点集合
        if current_:
            # 如果当前事件可以使用，则过滤掉未来的事件，只保留小于等于当前时间戳的事件
            nodes = list(filter((lambda x: x[1] <= time), nodes))
        else:
            # 如果当前事件不能使用，则不考虑未来的事件，也不考虑当前事件，过滤掉大于等于当前时间戳的事件
            nodes = list(filter((lambda x: x[1] < time), nodes))
        # 按时间戳降序排序，确保最新的事件在前
        nodes.sort(key=lambda x: x[1], reverse=True)
        actions_space = []  # 存储所有可用的动作空间
        i = 0  # 计数器，用于控制最大动作数
        # 遍历所有节点
        for node in nodes:
            # 遍历当前节点的所有出边，获取源节点到目标节点的关系
            for src, dst, rel in self.graph.out_edges(node, data=True):
                actions_space.append((rel['relation'], dst[0], dst[1]))  # 将动作 (关系, 目标实体, 时间戳) 添加到动作空间中
                i += 1  # 增加计数器
                # 如果达到最大动作数，停止添加更多的动作
                if max_action_num and i >= max_action_num:
                    break
            # 如果已经达到最大动作数，跳出外层循环
            if max_action_num and i >= max_action_num:
                break
        # 返回动作空间数组，类型为 int32
        return np.array(list(actions_space), dtype=np.dtype('int32'))

    def next_actions(self, entites, times, query_times, max_action_num=200, first_step=False):
        """Get the current action space. There must be an action that stays at the current position in the action space.
        Args:
            entites: torch.tensor, shape: [batch_size], the entity where the agent is currently located;
            times: torch.tensor, shape: [batch_size], the timestamp of the current entity;
            query_times: torch.tensor, shape: [batch_size], the timestamp of query;
            max_action_num: The size of the action space;
            first_step: Is it the first step for the agent.
        Return: torch.tensor, shape: [batch_size, max_action_num, 3], (relation, entity, time)
        """
        if self.config['cuda']:
            entites = entites.cpu()
            times = times.cpu()
            query_times = times.cpu()

        entites = entites.numpy()
        times = times.numpy()
        query_times = query_times.numpy()

        actions = self.get_padd_actions(entites, times, query_times, max_action_num, first_step)

        if self.config['cuda']:
            actions = torch.tensor(actions, dtype=torch.long, device='cuda')
        else:
            actions = torch.tensor(actions, dtype=torch.long)
        return actions

    def get_padd_actions(self, entites, times, query_times, max_action_num=200, first_step=False):
        """Construct the model input array.
        If the optional actions are greater than the maximum number of actions, then sample,
        otherwise all are selected, and the insufficient part is pad.
        """
        actions = np.ones((entites.shape[0], max_action_num, 3), dtype=np.dtype('int32'))
        actions[:, :, 0] *= self.rPAD
        actions[:, :, 1] *= self.ePAD
        actions[:, :, 2] *= self.tPAD
        for i in range(entites.shape[0]):
            # NO OPERATION
            actions[i, 0, 0] = self.NO_OP
            actions[i, 0, 1] = entites[i]
            actions[i, 0, 2] = times[i]

            if times[i] == query_times[i]:
                action_array = self.get_state_actions_space_complete(entites[i], times[i], False)
            else:
                action_array = self.get_state_actions_space_complete(entites[i], times[i], True)

            if action_array.shape[0] == 0:
                continue

            # Whether to keep the action NO_OPERATION
            start_idx = 1
            if first_step:
                # The first step cannot stay in place
                start_idx = 0

            if action_array.shape[0] > (max_action_num - start_idx):
                # Sample. Take the latest events.
                actions[i, start_idx:, ] = action_array[:max_action_num-start_idx]
            else:
                actions[i, start_idx:action_array.shape[0]+start_idx, ] = action_array
        return actions