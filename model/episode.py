import torch
import torch.nn as nn
import logging  # 添加日志模块


# 该文件实现了一个用于时间知识图谱推理的episode处理模块
# 主要包含前向推理和束搜索两种模式，用于处理动态关系路径的预测

class Episode(nn.Module):
    def __init__(self, env, agent, config):
        """初始化模块
        Args:
            env: 环境对象，提供动作空间信息
            agent: 智能体对象，包含策略网络和嵌入表示
            config: 配置参数字典
        """
        super(Episode, self).__init__()
        self.config = config  # 存储配置参数
        self.env = env  # 环境接口，用于获取可用动作
        self.agent = agent  # 智能体，包含策略网络和嵌入
        self.path_length = config['path_length']  # 推理路径的最大长度
        self.num_rel = config['num_rel']  # 关系类型的总数
        self.max_action_num = config['max_action_num']  # 每个步骤的最大候选动作数
        
        # 添加成功率统计相关属性
        self.total_queries = 0
        self.successful_queries = 0
        self.current_epoch = 0

    def calculate_success_rate(self, predicted_entities, target_entities):
        """计算推理成功率
        Args:
            predicted_entities: 预测的目标实体
            target_entities: 真实的目标实体
        Returns:
            success_rate: 成功率
            successful_count: 成功的查询数量
        """
        if predicted_entities.dim() > 1:
            # 对于beam search，取最佳预测
            predicted_entities = predicted_entities[:, 0]
        
        successful = (predicted_entities == target_entities).float()
        successful_count = successful.sum().item()
        success_rate = successful_count / len(target_entities) if len(target_entities) > 0 else 0.0
        
        return success_rate, successful_count

    def log_episode_info(self, success_rate, target_entities, predicted_entities, query_relations):
        """打印episode信息
        Args:
            success_rate: 成功率
            target_entities: 目标实体
            predicted_entities: 预测实体
            query_relations: 查询关系
        """
        batch_size = len(target_entities)
        
        # 更新全局统计
        self.total_queries += batch_size
        self.successful_queries += int(success_rate * batch_size)
        
        # 计算累积成功率
        cumulative_success_rate = self.successful_queries / self.total_queries if self.total_queries > 0 else 0.0
        
        # 只在每个epoch结束时打印一次详细信息，或者大幅减少打印频率
        if self.total_queries % 10000 == 0:  # 改为每10000个查询打印一次，大幅减少输出
            logging.info(f"Episode Info - Batch Success Rate: {success_rate:.4f}, "
                        f"Cumulative Success Rate: {cumulative_success_rate:.4f}, "
                        f"Target Entities: {target_entities[:3].tolist() if len(target_entities) > 3 else target_entities.tolist()}, "
                        f"Predicted Entities: {predicted_entities[:3].tolist() if len(predicted_entities) > 3 else predicted_entities.tolist()}")

    def forward(self, query_entities, query_timestamps, query_relations, target_entities=None):
        """前向传播处理完整推理路径（训练用）
        Args:
            query_entities: [batch_size] 查询的起始实体ID
            query_timestamps: [batch_size] 查询的时间戳
            query_relations: [batch_size] 目标关系的ID
            target_entities: [batch_size] 目标实体ID（用于计算成功率）
        Return:
            返回训练过程中各步骤的损失、logits、动作索引及最终状态
        """
        # 获取初始嵌入表示
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)

        # 初始化当前状态
        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # 初始化NO_OP操作

        # 存储各步骤信息
        all_loss = []
        all_logits = []
        all_actions_idx = []

        # 初始化LSTM隐藏状态
        self.agent.policy_step.set_hiddenx(query_relations.shape[0])

        # 逐步执行推理路径
        for t in range(self.path_length):
            first_step = (t == 0)  # 判断是否为第一步

            # 从环境获取可用动作
            action_space = self.env.next_actions(
                current_entites,
                current_timestamps,
                query_timestamps,
                self.max_action_num,
                first_step
            )

            # 通过智能体选择动作
            loss, logits, action_id = self.agent(
                prev_relations,
                current_entites,
                current_timestamps,
                query_relations_embeds,
                query_entities_embeds,
                query_timestamps,
                action_space,
            )

            # 提取选择的动作信息
            chosen_relation = torch.gather(action_space[:, :, 0], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity = torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=action_id).reshape(
                action_space.shape[0])

            # 保存当前步骤信息
            all_loss.append(loss)
            all_logits.append(logits)
            all_actions_idx.append(action_id)

            # 更新当前状态
            current_entites = chosen_entity
            current_timestamps = chosen_entity_timestamps
            prev_relations = chosen_relation

        # 计算并打印成功率信息
        if target_entities is not None:
            success_rate, successful_count = self.calculate_success_rate(current_entites, target_entities)
            self.log_episode_info(success_rate, target_entities, current_entites, query_relations)

        return all_loss, all_logits, all_actions_idx, current_entites, current_timestamps

    def beam_search(self, query_entities, query_timestamps, query_relations, target_entities=None):
        batch_size = query_entities.shape[0]
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)

        self.agent.policy_step.set_hiddenx(batch_size)

        # 第一步
        current_entities = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP
        action_space = self.env.next_actions(current_entities, current_timestamps,
                                             query_timestamps, self.max_action_num, True)
        loss, logits, action_id = self.agent(
            prev_relations,
            current_entities,
            current_timestamps,
            query_relations_embeds,
            query_entities_embeds,
            query_timestamps,
            action_space
        )

        action_space_size = action_space.shape[1]
        beam_size = min(self.config['beam_size'], action_space_size)
        beam_log_prob, top_k_action_id = torch.topk(logits, beam_size, dim=1)
        beam_log_prob = beam_log_prob.reshape(-1)

        current_entities = torch.gather(action_space[:, :, 1], dim=1, index=top_k_action_id).reshape(-1)
        current_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=top_k_action_id).reshape(-1)
        prev_relations = torch.gather(action_space[:, :, 0], dim=1, index=top_k_action_id).reshape(-1)

        # 扩展历史编码器以适配 beam_size
        self.agent.policy_step.expand_for_beam(beam_size)

        # 后续步骤
        for t in range(1, self.path_length):
            query_timestamps_roll = query_timestamps.repeat(beam_size).reshape(-1)
            query_entities_embeds_roll = query_entities_embeds.repeat(1, beam_size).reshape(batch_size * beam_size, -1)
            query_relations_embeds_roll = query_relations_embeds.repeat(1, beam_size).reshape(batch_size * beam_size, -1)

            action_space = self.env.next_actions(current_entities, current_timestamps,
                                                 query_timestamps_roll, self.max_action_num)

            loss, logits, action_id = self.agent(
                prev_relations,
                current_entities,
                current_timestamps,
                query_relations_embeds_roll,
                query_entities_embeds_roll,
                query_timestamps_roll,
                action_space
            )

            # 更新 beam 概率
            action_space_size = action_space.shape[1]
            beam_tmp = beam_log_prob.repeat(action_space_size, 1).T + logits

            # 之前不稳定的多样性惩罚已被移除
            beam_tmp = beam_tmp.reshape(batch_size, -1)
            
            new_beam_size = min(self.config['beam_size'], action_space_size * beam_size)
            top_k_log_prob, top_k_action_id = torch.topk(beam_tmp, new_beam_size, dim=1)
            offset = top_k_action_id // action_space_size  # beam 索引
        
            # # 更新隐藏状态
            # hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
            # offset_expanded = offset.unsqueeze(-1).repeat(1, 1, self.agent.policy_step.state_dim)
            # self.agent.policy_step.hx = torch.gather(hx_tmp, dim=1, index=offset_expanded).reshape(
            #     batch_size * new_beam_size, -1)
            # # 更新历史序列
            # history_tmp = self.agent.policy_step.history.reshape(batch_size, beam_size,
            #                                                      self.agent.policy_step.history_len, -1)
            # self.agent.policy_step.history = torch.gather(
            #     history_tmp, dim=1,
            #     index=offset_expanded.unsqueeze(2).repeat(1, 1, self.agent.policy_step.history_len, 1)
            # ).reshape(batch_size * new_beam_size, self.agent.policy_step.history_len, self.agent.policy_step.state_dim)
            # self.agent.policy_step.current_len = self.agent.policy_step.current_len.reshape(batch_size,
            #                                                                                 beam_size).gather(dim=1,
            #                                                                                                   index=offset).reshape(
            #     -1)
            # 使用在HistoryEncoder中新增的、更安全的方法来更新状态
            self.agent.policy_step.select_beams(batch_size, beam_size, new_beam_size, offset)
            # 更新当前状态
            current_entities = torch.gather(action_space[:, :, 1].reshape(batch_size, -1), dim=1,
                                            index=top_k_action_id).reshape(-1)
            current_timestamps = torch.gather(action_space[:, :, 2].reshape(batch_size, -1), dim=1,
                                              index=top_k_action_id).reshape(-1)
            prev_relations = torch.gather(action_space[:, :, 0].reshape(batch_size, -1), dim=1,
                                            index=top_k_action_id).reshape(-1)
            beam_log_prob = top_k_log_prob.reshape(-1)
            beam_size = new_beam_size

        final_entities = current_entities.reshape(batch_size, -1)
        final_probs = beam_log_prob.reshape(batch_size, -1)
        
        # 计算并打印成功率信息
        if target_entities is not None:
            success_rate, successful_count = self.calculate_success_rate(final_entities, target_entities)
            self.log_episode_info(success_rate, target_entities, final_entities, query_relations)

        return final_entities, final_probs

    def _calculate_diversity_penalty(self, entities, action_space_size):
        """计算实体多样性惩罚，鼓励选择不同的实体路径"""
        # 检查实体张量是否为空
        if entities.numel() == 0:
            return torch.zeros(0, dtype=torch.float)
        
        # 安全计算 batch_size，确保不会为 0
        beam_size = self.config['beam_size']
        total_size = entities.size(0)
        
        # 如果总大小小于 beam_size，则可能是最后一个批次或特殊情况
        if total_size < beam_size:
            # 直接返回零惩罚
            batch_size = 1
            actual_beam_size = total_size
        else:
            batch_size = total_size // beam_size
            actual_beam_size = beam_size
            
            # 确保能够整除
            if batch_size * actual_beam_size != total_size:
                # 如果不能整除，可能是最后一个批次，调整 beam_size
                actual_beam_size = total_size // batch_size
        
        # 重塑实体张量
        entities = entities[:batch_size * actual_beam_size].reshape(batch_size, actual_beam_size)
        
        # 计算每个batch中实体的重复程度
        penalty = torch.zeros_like(entities, dtype=torch.float)
        for i in range(batch_size):
            unique_entities, counts = torch.unique(entities[i], return_counts=True)
            for j, ent in enumerate(entities[i]):
                # 使用布尔索引
                idx = (unique_entities == ent).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:  # 确保找到了匹配的实体
                    penalty[i, j] = counts[idx[0]].float() - 1.0
        
        # 将惩罚扩展到与 beam_tmp 相同的形状
        # 每个实体的惩罚值需要重复 action_space_size 次
        expanded_penalty = penalty.unsqueeze(-1).repeat(1, 1, action_space_size)
        return expanded_penalty.reshape(batch_size, -1)