import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict


class LLMPriorRewardGenerator:
    """使用大模型生成先验奖励的类"""

    def __init__(self, data_path: str, config: Dict[str, Any]):
        self.data_path = data_path
        self.config = config
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self.time2id = {}
        self.id2time = {}

        # 先验知识存储
        self.entity_types = {}  # 实体类型信息
        self.relation_semantics = {}  # 关系语义信息
        self.temporal_patterns = {}  # 时间模式

        # 奖励缓存
        self.reward_cache = {}

        self.logger = logging.getLogger(__name__)
        self._load_data()
        self._initialize_prior_knowledge()

    def _load_data(self):
        """加载数据文件"""
        try:
            # 加载实体、关系、时间映射
            with open(f"{self.data_path}/entity2id.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    entity, id_str = line.strip().split('\t')
                    ent_id = int(id_str)
                    self.entity2id[entity] = ent_id
                    self.id2entity[ent_id] = entity

            with open(f"{self.data_path}/relation2id.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    relation, id_str = line.strip().split('\t')
                    rel_id = int(id_str)
                    self.relation2id[relation] = rel_id
                    self.id2relation[rel_id] = relation

            # 如果有时间信息
            if hasattr(self.config, 'time_span'):
                for i in range(self.config.time_span):
                    self.time2id[i] = i
                    self.id2time[i] = i

            self.logger.info(f"加载了 {len(self.entity2id)} 个实体, {len(self.relation2id)} 个关系")

        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise

    def _initialize_prior_knowledge(self):
        """初始化先验知识"""
        # 1. 分析关系的语义类型
        self._analyze_relation_semantics()

        # 2. 分析实体类型（基于关系和上下文）
        self._analyze_entity_types()

        # 3. 分析时间模式
        self._analyze_temporal_patterns()

    def _analyze_relation_semantics(self):
        """分析关系的语义特征"""
        # 定义一些常见的关系语义类型
        semantic_patterns = {
            'location': ['located_in', 'capital_of', 'part_of', 'neighbor_of'],
            'social': ['friend_of', 'family_of', 'colleague_of', 'enemy_of'],
            'political': ['leader_of', 'member_of', 'ally_of', 'oppose'],
            'economic': ['trade_with', 'invest_in', 'own', 'work_for'],
            'temporal': ['before', 'after', 'during', 'simultaneous'],
            'causal': ['cause', 'result_in', 'influence', 'affect']
        }

        for rel_id, relation in self.id2relation.items():
            relation_lower = relation.lower().replace('_', ' ')

            # 基于关系名称推断语义类型
            semantic_type = 'general'  # 默认类型
            for sem_type, patterns in semantic_patterns.items():
                if any(pattern in relation_lower for pattern in patterns):
                    semantic_type = sem_type
                    break

            self.relation_semantics[rel_id] = {
                'type': semantic_type,
                'name': relation,
                'transitivity_score': self._estimate_transitivity(relation),
                'temporal_sensitivity': self._estimate_temporal_sensitivity(relation)
            }

    def _estimate_transitivity(self, relation: str) -> float:
        """估计关系的传递性强度"""
        transitive_keywords = ['part_of', 'located_in', 'member_of', 'subset_of']
        if any(keyword in relation.lower() for keyword in transitive_keywords):
            return 0.8
        return 0.3

    def _estimate_temporal_sensitivity(self, relation: str) -> float:
        """估计关系对时间的敏感性"""
        temporal_keywords = ['during', 'before', 'after', 'when', 'while']
        if any(keyword in relation.lower() for keyword in temporal_keywords):
            return 0.9
        return 0.4

    def _analyze_entity_types(self):
        """分析实体类型（基于关系使用模式）"""
        # 这里可以通过训练数据中的三元组来推断实体类型
        entity_relation_freq = defaultdict(lambda: defaultdict(int))

        # 读取训练数据来分析实体-关系共现模式
        try:
            with open(f"{self.data_path}/train.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        head = int(parts[0])
                        rel = int(parts[1])
                        entity_relation_freq[head][rel] += 1
        except:
            pass

        # 基于关系使用模式推断实体类型
        for ent_id in self.id2entity:
            if ent_id in entity_relation_freq:
                # 找到最常用的关系类型
                most_common_rel_types = defaultdict(int)
                for rel_id, freq in entity_relation_freq[ent_id].items():
                    if rel_id in self.relation_semantics:
                        rel_type = self.relation_semantics[rel_id]['type']
                        most_common_rel_types[rel_type] += freq

                if most_common_rel_types:
                    primary_type = max(most_common_rel_types.items(), key=lambda x: x[1])[0]
                    self.entity_types[ent_id] = {
                        'primary_type': primary_type,
                        'relation_profile': dict(most_common_rel_types)
                    }

    def _analyze_temporal_patterns(self):
        """分析时间模式"""
        # 这里可以分析不同关系在不同时间的活跃程度
        self.temporal_patterns = {
            'relation_time_activity': defaultdict(lambda: defaultdict(float)),
            'global_time_activity': defaultdict(float)
        }

    def compute_path_reward(self, path: List[Tuple[int, int, int]],
                            target_entity: int, current_time: int = None) -> float:
        """计算路径的先验奖励"""
        if not path:
            return 0.0

        # 缓存键
        cache_key = (tuple(path), target_entity, current_time)
        if cache_key in self.reward_cache:
            return self.reward_cache[cache_key]

        total_reward = 0.0
        path_length = len(path)

        # 1. 路径语义一致性奖励
        semantic_consistency = self._compute_semantic_consistency(path)
        total_reward += semantic_consistency * 0.3

        # 2. 路径长度惩罚（避免过长路径）
        length_penalty = max(0, 1.0 - (path_length - 2) * 0.2)
        total_reward += length_penalty * 0.2

        # 3. 目标导向性奖励
        target_orientation = self._compute_target_orientation(path, target_entity)
        total_reward += target_orientation * 0.4

        # 4. 时间一致性奖励
        if current_time is not None:
            temporal_consistency = self._compute_temporal_consistency(path, current_time)
            total_reward += temporal_consistency * 0.1

        # 归一化到 [0, 1] 区间
        total_reward = max(0.0, min(1.0, total_reward))

        # 缓存结果
        self.reward_cache[cache_key] = total_reward

        return total_reward

    def _compute_semantic_consistency(self, path: List[Tuple[int, int, int]]) -> float:
        """计算路径的语义一致性"""
        if len(path) < 2:
            return 1.0

        consistency_score = 0.0
        relation_types = []

        for _, rel_id, _ in path:
            if rel_id in self.relation_semantics:
                rel_type = self.relation_semantics[rel_id]['type']
                relation_types.append(rel_type)

        if not relation_types:
            return 0.5

        # 计算语义类型的一致性
        type_counts = defaultdict(int)
        for rel_type in relation_types:
            type_counts[rel_type] += 1

        # 如果主要使用同一语义类型的关系，给予较高奖励
        max_type_count = max(type_counts.values())
        consistency_score = max_type_count / len(relation_types)

        return consistency_score

    def _compute_target_orientation(self, path: List[Tuple[int, int, int]],
                                    target_entity: int) -> float:
        """计算路径的目标导向性"""
        if not path:
            return 0.0

        # 检查路径是否朝着目标实体的方向发展
        current_entity = path[0][0]  # 起始实体
        final_entity = path[-1][2]  # 最终实体

        # 如果直接到达目标，给最高奖励
        if final_entity == target_entity:
            return 1.0

        # 基于实体类型相似性计算方向性
        target_orientation_score = 0.0

        if target_entity in self.entity_types and final_entity in self.entity_types:
            target_type = self.entity_types[target_entity].get('primary_type', 'general')
            final_type = self.entity_types[final_entity].get('primary_type', 'general')

            if target_type == final_type:
                target_orientation_score += 0.7
            else:
                target_orientation_score += 0.3
        else:
            target_orientation_score = 0.4  # 默认中等分数

        # 考虑路径中关系的传递性
        transitivity_bonus = 0.0
        for _, rel_id, _ in path:
            if rel_id in self.relation_semantics:
                transitivity_score = self.relation_semantics[rel_id]['transitivity_score']
                transitivity_bonus += transitivity_score

        if path:
            transitivity_bonus /= len(path)

        target_orientation_score = (target_orientation_score + transitivity_bonus * 0.3) / 1.3

        return min(1.0, target_orientation_score)

    def _compute_temporal_consistency(self, path: List[Tuple[int, int, int]],
                                      current_time: int) -> float:
        """计算时间一致性"""
        if not path:
            return 1.0

        temporal_score = 0.0

        for _, rel_id, _ in path:
            if rel_id in self.relation_semantics:
                temporal_sensitivity = self.relation_semantics[rel_id]['temporal_sensitivity']
                # 根据关系的时间敏感性调整分数
                temporal_score += temporal_sensitivity

        if path:
            temporal_score /= len(path)

        return temporal_score

    def compute_action_reward(self, current_entity: int, action_relation: int,
                              next_entity: int, target_entity: int,
                              current_path_length: int = 0) -> float:
        """计算单步动作的先验奖励"""
        # 避免NO_OP操作获得高奖励
        if action_relation == 0 or next_entity == current_entity:
            # 如果路径长度已经很长，NO_OP可能是合理的
            if current_path_length >= self.config.get('path_length', 3):
                return 0.1
            else:
                return -0.2  # 惩罚早期的NO_OP

        # 如果直接到达目标
        if next_entity == target_entity:
            return 1.0

        reward = 0.0

        # 1. 关系质量奖励
        if action_relation in self.relation_semantics:
            rel_info = self.relation_semantics[action_relation]
            reward += 0.3  # 基础关系奖励

            # 根据关系类型调整
            if rel_info['type'] in ['location', 'social', 'political']:
                reward += 0.1  # 这些关系通常更有意义

        # 2. 实体类型匹配奖励
        if (target_entity in self.entity_types and
                next_entity in self.entity_types):

            target_type = self.entity_types[target_entity].get('primary_type', 'general')
            next_type = self.entity_types[next_entity].get('primary_type', 'general')

            if target_type == next_type:
                reward += 0.3
            else:
                reward += 0.1

        # 3. 路径长度考虑
        length_factor = max(0.1, 1.0 - current_path_length * 0.15)
        reward *= length_factor

        return min(1.0, max(-0.5, reward))

    def update_reward_based_on_success(self, path: List[Tuple[int, int, int]],
                                       success: bool, target_entity: int):
        """基于成功与否更新奖励（在线学习）"""
        cache_key = (tuple(path), target_entity, None)

        if cache_key in self.reward_cache:
            old_reward = self.reward_cache[cache_key]

            if success:
                # 成功的路径，增加奖励
                new_reward = min(1.0, old_reward + 0.1)
            else:
                # 失败的路径，减少奖励
                new_reward = max(0.0, old_reward - 0.05)

            self.reward_cache[cache_key] = new_reward

    def save_rewards(self, save_path: str):
        """保存奖励缓存"""
        reward_data = {
            'reward_cache': dict(self.reward_cache),
            'relation_semantics': self.relation_semantics,
            'entity_types': self.entity_types,
            'temporal_patterns': self.temporal_patterns
        }

        with open(save_path, 'wb') as f:
            pickle.dump(reward_data, f)

        self.logger.info(f"奖励数据保存到 {save_path}")

    def load_rewards(self, load_path: str):
        """加载奖励缓存"""
        try:
            with open(load_path, 'rb') as f:
                reward_data = pickle.load(f)

            self.reward_cache = reward_data.get('reward_cache', {})
            self.relation_semantics.update(reward_data.get('relation_semantics', {}))
            self.entity_types.update(reward_data.get('entity_types', {}))
            self.temporal_patterns.update(reward_data.get('temporal_patterns', {}))

            self.logger.info(f"从 {load_path} 加载了 {len(self.reward_cache)} 个奖励缓存")

        except Exception as e:
            self.logger.warning(f"无法加载奖励缓存: {e}")


# 使用示例
def create_prior_rewards(config, data_path: str, save_path: str):
    """创建并保存先验奖励"""
    generator = LLMPriorRewardGenerator(data_path, config)

    # 如果有现有的奖励文件，先加载
    try:
        generator.load_rewards(f"{data_path}/rewards.pkl")
    except:
        pass

    # 保存更新后的奖励
    generator.save_rewards(save_path)

    return generator


if __name__ == "__main__":
    # 测试代码
    class MockConfig:
        def __init__(self):
            self.path_length = 3
            self.time_span = 24


    config = MockConfig()
    generator = create_prior_rewards(config, "data/ICEWS14", "data/ICEWS14/rewards.pkl")

    # 测试奖励计算
    test_path = [(1, 5, 10), (10, 12, 25)]
    reward = generator.compute_path_reward(test_path, target_entity=25)
    print(f"测试路径奖励: {reward}")