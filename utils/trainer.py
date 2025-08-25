import torch
import json
import os
import tqdm


class Trainer(object):
    def __init__(self, model, pg, optimizer, args, llm_rewards=None):
        self.model = model
        self.pg = pg
        self.optimizer = optimizer
        self.args = args
        self.llm_rewards = llm_rewards  # 新增：存储LLM奖励
        self.device = torch.device("cuda" if args.cuda else "cpu")

    def train_epoch(self, dataloader, ntriple):
        self.model.train()

        total_loss = 0.0
        total_reward = 0.0
        total_success_rate = 0.0
        counter = 0
        with tqdm.tqdm(total=ntriple, unit='ex') as bar:
            bar.set_description('Train')
            for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                if self.args.cuda:
                    src_batch = src_batch.cuda()
                    rel_batch = rel_batch.cuda()
                    dst_batch = dst_batch.cuda()
                    time_batch = time_batch.cuda()

                all_loss, all_logits, _, current_entities, current_time = self.model(src_batch, time_batch, rel_batch,
                                                                                     dst_batch)

                # 默认的RL环境奖励
                env_reward = self.pg.get_reward(current_entities, dst_batch)

                # 新增：结合LLM奖励
                final_reward = env_reward.clone()  # 先复制环境奖励
                if self.llm_rewards:
                    for i in range(len(src_batch)):
                        key = (src_batch[i].item(), rel_batch[i].item(), time_batch[i].item())
                        if key in self.llm_rewards:
                            # 结合奖励，例如：用LLM奖励覆盖，或者加权平均
                            # 这里我们简单地将LLM奖励加到原始奖励上
                            final_reward[i] += self.llm_rewards[key]

                success_rate, _ = self.model.calculate_success_rate(current_entities, dst_batch)
                total_success_rate += success_rate

                # 使用最终的组合奖励
                cum_discounted_reward = self.pg.calc_cum_discounted_reward(final_reward)

                reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward)
                self.pg.baseline.update(torch.mean(cum_discounted_reward))
                self.pg.now_epoch += 1

                self.optimizer.zero_grad()
                reinfore_loss.backward()
                if self.args.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
                self.optimizer.step()

                total_loss += reinfore_loss
                total_reward += torch.mean(final_reward)  # 使用final_reward
                counter += 1
                bar.update(self.args.batch_size)
                bar.set_postfix(loss='%.4f' % reinfore_loss, reward='%.4f' % torch.mean(final_reward).item(),
                                success='%.4f' % success_rate)

        avg_success_rate = total_success_rate / counter
        return total_loss / counter, total_reward / counter, avg_success_rate

    def save_model(self, checkpoint_path='checkpoint.pth'):
        argparse_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(save_dict, os.path.join(self.args.save_path, checkpoint_path))