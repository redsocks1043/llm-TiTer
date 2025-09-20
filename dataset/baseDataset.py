from collections import defaultdict
from torch.utils.data import Dataset
import os
class baseDataset(object):
    def __init__(self, trainpath, testpath, statpath, validpath):
        """基础数据集类。读取数据文件并预处理。
            Args:
                trainpath: 训练数据文件路径
                testpath: 测试数据文件路径
                statpath: 实体数量和关系数量的统计文件路径
                validpath: 验证数据文件路径
        """
        # 加载训练、测试、验证集的四元组数据
        self.trainQuadruples = self.load_quadruples(trainpath)
        self.testQuadruples = self.load_quadruples(testpath)
        self.validQuadruples = self.load_quadruples(validpath)
        # 合并四元组数据
        self.allQuadruples = self.trainQuadruples + self.validQuadruples + self.testQuadruples
        # 获取实体总数和关系总数
        self.num_e, self.num_r = self.get_total_number(statpath)
        # 获取用于时间相关过滤的字典
        self.skip_dict = self.get_skipdict(self.allQuadruples)
        # 记录训练集中出现的所有实体
        self.train_entities = set()
        for query in self.trainQuadruples:
            self.train_entities.add(query[0])
            self.train_entities.add(query[2])
        # 获取关系-实体共现信息 ##TODO 这里是啥意思
        self.RelEntCooccurrence = self.getRelEntCooccurrence(self.trainQuadruples)  # -> dict

        def get_entity_id_map(statpath):
            entity_map = {}
            entity_path = os.path.join(os.path.dirname(statpath), 'entity2id.txt')
            try:
                with open(entity_path, 'r') as f:
                    # 跳过第一行
                    next(f)
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            entity_map[int(parts[1])] = parts[0]
            except FileNotFoundError:
                print(f"警告: {entity_path} 未找到。实体映射将不可用。")
            return entity_map

        self.id_to_entity = get_entity_id_map(statpath)

    def getRelEntCooccurrence(self, quadruples):
        """用于归纳式平均。获取训练集中的共现信息。
        返回:
            {'subject': 字典[键->关系, 值->共现的主体实体集合],
             'object': 字典[键->关系, 值->共现的客体实体集合]}
        """
        relation_entities_s = {}  # 关系-主体实体字典
        relation_entities_o = {}  # 关系-客体实体字典
        for ex in quadruples:
            s, r, o = ex[0], ex[1], ex[2]  # 分解四元组为主体、关系、客体
            reversed_r = r + self.num_r + 1  # 计算反向关系的ID
            # 存储正向关系的共现信息
            if r not in relation_entities_s.keys():
                relation_entities_s[r] = set()
            relation_entities_s[r].add(s)
            if r not in relation_entities_o.keys():
                relation_entities_o[r] = set()
            relation_entities_o[r].add(o)
            # 存储反向关系的共现信息
            if reversed_r not in relation_entities_s.keys():
                relation_entities_s[reversed_r] = set()
            relation_entities_s[reversed_r].add(o)
            if reversed_r not in relation_entities_o.keys():
                relation_entities_o[reversed_r] = set()
            relation_entities_o[reversed_r].add(s)
        return {'subject': relation_entities_s, 'object': relation_entities_o}

    def get_all_timestamps(self):
        """获取数据集中的所有时间戳
            返回：
                timestamps: 包含所有不重复时间戳的集合
        """
        timestamps = set()  # 创建一个空集合用于存储时间戳
        for ex in self.allQuadruples:  # 遍历所有四元组
            timestamps.add(ex[3])  # 将四元组中的时间戳（第4个元素）添加到集合中
        return timestamps  # 返回所有不重复的时间戳集合

    def get_skipdict(self, quadruples):
        """用于时间相关的过滤指标计算
        参数：
            quadruples: 四元组列表
        返回：
            filters: 一个默认字典，键为(实体,关系,时间戳)，值为对应的真实实体集合
        """
        filters = defaultdict(set)  # 创建一个默认字典，默认值为空集合
        for src, rel, dst, time in quadruples:  # 遍历所有四元组
            # 添加正向关系的过滤信息：(头实体,关系,时间) -> 尾实体集合
            filters[(src, rel, time)].add(dst)
            # 添加反向关系的过滤信息：(尾实体,反向关系,时间) -> 头实体集合
            filters[(dst, rel + self.num_r + 1, time)].add(src)
        return filters  # 返回构建好的过滤字典

    @staticmethod
    def load_quadruples(inpath):
        """读取训练/验证/测试数据文件
            参数：
                inpath: 数据文件路径，可以是train.txt、valid.txt或test.txt
            返回：
                quadrupleList: 包含所有四元组的列表，每个四元组格式为[头实体,关系,尾实体,时间戳]
            """
        with open(inpath, 'r') as f:  # 打开数据文件
            quadrupleList = []  # 创建空列表存储四元组
            for line in f:  # 逐行读取文件
                try:
                    line_split = line.split()  # 分割每行文本
                    head = int(line_split[0])  # 转换头实体为整数
                    rel = int(line_split[1])  # 转换关系为整数
                    tail = int(line_split[2])  # 转换尾实体为整数
                    time = int(line_split[3])  # 转换时间戳为整数
                    quadrupleList.append([head, rel, tail, time])  # 将四元组添加到列表中
                except:
                    print(line)  # 如果转换失败，打印出问题的行
            return quadrupleList  # 返回所有四元组列表

    @staticmethod
    def get_total_number(statpath):
        """读取统计信息文件
        参数：
            statpath: 统计文件路径
        返回：
            (实体数量, 关系数量)的元组
        """
        with open(statpath, 'r') as fr:  # 打开统计文件
            for line in fr:  # 读取文件第一行
                line_split = line.split()  # 分割行
                # 返回实体数量和关系数量
                return int(line_split[0]), int(line_split[1])


class QuadruplesDataset(Dataset):
    """PyTorch数据集类，用于加载四元组数据"""
    def __init__(self, examples, num_r):
        """
        examples: 四元组列表
        num_r: 关系的数量
        """
        self.quadruples = examples.copy()
        # 为每个四元组添加反向关系的四元组
        for ex in examples:
            self.quadruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3]])

    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, item):
        # 返回四元组中的各个元素：(主体，关系，客体，时间戳)
        return self.quadruples[item][0], \
               self.quadruples[item][1], \
               self.quadruples[item][2], \
               self.quadruples[item][3]
