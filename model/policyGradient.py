import torch
import numpy as np
import math
from model.baseline import ReactiveBaseline

class PG(object):
    def __init__(self, config):
        # 初始化策略梯度（Policy Gradient）类
        self.config = config  # 配置文件
        self.positive_reward = 1.0  # 正奖励的值
        self.negative_reward = 0.0  # 负奖励的值
        self.baseline = ReactiveBaseline(config, config['lambda'])  # 基准值，用于奖励标准化
        self.now_epoch = 0  # 当前的训练轮次

    def get_reward(self, current_entites, answers):
        # 根据当前实体与答案是否一致来计算奖励
        positive = torch.ones_like(current_entites, dtype=torch.float32) * self.positive_reward  # 创建正奖励张量
        negative = torch.ones_like(current_entites, dtype=torch.float32) * self.negative_reward  # 创建负奖励张量
        reward = torch.where(current_entites == answers, positive, negative)  # 如果当前实体等于答案，则给正奖励，否则给负奖励
        return reward  # 返回奖励值

    def calc_cum_discounted_reward(self, rewards):
        # 计算累计折扣奖励
        running_add = torch.zeros([rewards.shape[0]])  # 初始化累计奖励为零
        cum_disc_reward = torch.zeros([rewards.shape[0], self.config['path_length']])  # 初始化累计折扣奖励矩阵

        if self.config['cuda']:  # 如果使用 GPU
            running_add = running_add.cuda()  # 将变量移动到 GPU
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.config['path_length'] - 1] = rewards  # 将最后一个时间步的奖励直接赋值给累计奖励
        for t in reversed(range(self.config['path_length'])):
            # 从后往前计算累计折扣奖励
            running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t]  # 累加奖励并乘以折扣因子
            cum_disc_reward[:, t] = running_add  # 更新当前时间步的累计折扣奖励
        return cum_disc_reward  # 返回累计折扣奖励

    def entropy_reg_loss(self, all_logits):
        # 计算熵正则化损失
        all_logits = torch.stack(all_logits, dim=2)  # 将所有的 logits 堆叠成一个张量
        entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))  # 计算熵正则化项
        return entropy_loss  # 返回熵损失

    def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward):
        # 计算 REINFORCE 的损失
        loss = torch.stack(all_loss, dim=1)  # 将每个时间步的损失堆叠成一个张量
        base_value = self.baseline.get_baseline_value()  # 获取基准值
        final_reward = cum_discounted_reward - base_value  # 计算最终奖励（当前奖励减去基准值）

        reward_mean = torch.mean(final_reward)  # 计算最终奖励的均值
        reward_std = torch.std(final_reward) + 1e-6  # 计算最终奖励的标准差，避免除零错误
        final_reward = torch.div(final_reward - reward_mean, reward_std)  # 标准化奖励

        loss = torch.mul(loss, final_reward)  # 将损失与标准化奖励相乘，进行加权
        # 熵正则化损失，动态调整权重，随着训练轮次逐渐衰减
        entropy_loss = self.config['ita'] * math.pow(self.config['zita'], self.now_epoch) * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss  # 总损失：加权损失的均值减去熵正则化损失
        return total_loss  # 返回最终的总损失

