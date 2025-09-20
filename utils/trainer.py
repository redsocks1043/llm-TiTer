import torch  # 导入PyTorch库
import json  # 导入JSON处理库
import os  # 导入操作系统功能库
import tqdm  # 导入进度条库

class Trainer(object):  # 定义训练器类
    def __init__(self, model, pg, optimizer, args, distribution=None):  # 初始化方法，接收模型、策略梯度、优化器、参数和分布
        self.model = model  # 保存模型实例
        self.pg = pg  # 保存策略梯度实例
        self.optimizer = optimizer  # 保存优化器实例
        self.args = args  # 保存参数
        self.distribution = distribution  # 保存分布实例，用于奖励整形
        self.reward_cache = {}  # 添加奖励缓存
        # 添加设备属性
        self.device = torch.device("cuda" if args.cuda else "cpu")

    def train_epoch(self, dataloader, ntriple):  # 训练一个epoch的方法
        self.model.train()  # 将模型设置为训练模式

        total_loss = 0.0  # 初始化总损失
        total_reward = 0.0  # 初始化总奖励
        total_success_rate = 0.0  # 添加总成功率统计
        counter = 0  # 初始化计数器
        with tqdm.tqdm(total=ntriple, unit='ex') as bar:  # 创建进度条，总数为ntriple，单位为'ex'
            bar.set_description('Train')  # 设置进度条描述为'Train'
            for src_batch, rel_batch, dst_batch, time_batch in dataloader:  # 遍历数据加载器中的批次数据
                if self.args.cuda:  # 如果启用了CUDA
                    src_batch = src_batch.cuda()  # 将源实体批次数据移至GPU
                    rel_batch = rel_batch.cuda()  # 将关系批次数据移至GPU
                    dst_batch = dst_batch.cuda()  # 将目标实体批次数据移至GPU
                    time_batch = time_batch.cuda()  # 将时间批次数据移至GPU

                # 传递目标实体给forward方法
                all_loss, all_logits, _, current_entities, current_time = self.model(src_batch, time_batch, rel_batch, dst_batch)

                reward = self.pg.get_reward(current_entities, dst_batch)  # 获取当前实体和目标实体之间的奖励
                
                # 计算成功率
                success_rate, _ = self.model.calculate_success_rate(current_entities, dst_batch)
                total_success_rate += success_rate
                
                if self.args.reward_shaping:  # 如果启用了奖励整形
                    # reward shaping  # 奖励整形
                    delta_time = time_batch - current_time  # 计算时间差

                    # 标准奖励计算
                    p_dt = []
                    for i in range(rel_batch.shape[0]):
                        rel = rel_batch[i].item()
                        dt = delta_time[i].item() // self.args.time_span
                        cache_key = (rel, dt)
                        if cache_key in self.reward_cache:
                            p_dt.append(self.reward_cache[cache_key])
                        else:
                            # 这里改为数组索引
                            try:
                                reward_val = self.distribution[rel, dt]
                            except IndexError:
                                reward_val = 0.0  # 或者其他默认值
                            self.reward_cache[cache_key] = reward_val
                            p_dt.append(reward_val)
                    p_dt = torch.tensor(p_dt, device=self.device)  # 直接在GPU上创建张量

                    beta = getattr(self.args, "reward_shaping_beta", 1.0)  # 可在args中添加reward_shaping_beta参数，默认1.0
                    # shaped_reward = reward + beta * p_dt
                    #p_dt_norm = (p_dt - p_dt.mean()) / (p_dt.std() + 1e-8)
                    shaped_reward = reward + beta * p_dt
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(shaped_reward)
                else:  # 如果未启用奖励整形
                    cum_discounted_reward = self.pg.calc_cum_discounted_reward(reward)  # 直接计算累积折扣奖励
                reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)  # 计算强化学习损失
                self.pg.baseline.update(torch.mean(cum_discounted_reward))  # 更新基线值
                self.pg.now_epoch += 1  # 当前epoch计数加1

                self.optimizer.zero_grad()  # 清空优化器梯度
                reinfore_loss.backward()  # 反向传播计算梯度
                if self.args.clip_gradient:  # 如果启用了梯度裁剪
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)  # 裁剪梯度
                self.optimizer.step()  # 优化器更新参数

                total_loss += reinfore_loss  # 累加损失
                total_reward += torch.mean(reward)  # 累加平均奖励
                counter += 1  # 计数器加1
                bar.update(self.args.batch_size)  # 更新进度条，步长为batch_size
                bar.set_postfix(loss='%.4f' % reinfore_loss, reward='%.4f' % torch.mean(reward).item(), success='%.4f' % success_rate)  # 在进度条后显示当前损失、奖励和成功率
        
        avg_success_rate = total_success_rate / counter
        return total_loss / counter, total_reward / counter, avg_success_rate  # 返回平均损失、平均奖励和平均成功率

    def save_model(self, checkpoint_path='checkpoint.pth'):  # 保存模型方法，默认保存路径为'checkpoint.pth'
        """Save the parameters of the model and the optimizer,"""  # 保存模型和优化器的参数
        argparse_dict = vars(self.args)  # 将参数转换为字典
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:  # 打开配置文件用于写入
            json.dump(argparse_dict, fjson)  # 将参数字典保存为JSON文件

        # 保存模型
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(save_dict, os.path.join(self.args.save_path, checkpoint_path))
