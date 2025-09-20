import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalDynamicReward(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim, time_dim, hidden_dim):
        super().__init__()
        self.ent_embs = DynamicEmbedding(num_entities, entity_dim, time_dim)
        self.rel_embs = nn.Embedding(num_relations, relation_dim)
        self.proj = nn.Linear(entity_dim + relation_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, quadruples, query_times):
        subj = quadruples[:, 0]
        rel = quadruples[:, 1]
        obj = quadruples[:, 2]
        ts = quadruples[:, 3]
        dt = (query_times - ts).unsqueeze(1)

        subj_emb = self.ent_embs(subj, dt)
        obj_emb = self.ent_embs(obj, dt)
        rel_emb = self.rel_embs(rel)

        feat = torch.cat([subj_emb + obj_emb, rel_emb], dim=-1)
        proj_feat = self.proj(feat).unsqueeze(1)
        attn_output, _ = self.attn(proj_feat, proj_feat, proj_feat)
        reward = self.mlp(attn_output.squeeze(1))
        return reward.squeeze(-1)


class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t):
        super().__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent - dim_t)
        self.w = nn.Parameter(torch.from_numpy(1 / 10 ** torch.linspace(0, 9, dim_t).numpy()).float())
        self.b = nn.Parameter(torch.zeros(dim_t).float())

    def forward(self, entities, dt):
        dt = dt.view(-1, 1)
        t = torch.cos(self.w.view(1, -1) * dt + self.b.view(1, -1))
        e = self.ent_embs(entities)
        return torch.cat([e, t], dim=-1)


class RewardLearner:
    def __init__(self, quadruples, num_entities, num_relations, entity_dim, relation_dim, time_dim, hidden_dim,
                 device='cuda', lr=1e-3, batch_size=128, epochs=50, time_span=24):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.quadruples = torch.tensor(quadruples, dtype=torch.long).to(self.device)
        self.query_times = self.quadruples[:, 3]
        self.model = TemporalDynamicReward(num_entities, num_relations, entity_dim, relation_dim, time_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.time_span = time_span # 保存 time_span
        self.rewards = None
        self.num_relations = num_relations  # 添加这个属性

    def compute_targets(self, batch, alpha=0.5):
        rel_freq = torch.bincount(batch[:, 1], minlength=self.model.rel_embs.num_embeddings).float()
        subj_freq = torch.bincount(batch[:, 0], minlength=self.model.ent_embs.ent_embs.num_embeddings).float()
        obj_freq = torch.bincount(batch[:, 2], minlength=self.model.ent_embs.ent_embs.num_embeddings).float()

        rel_scores = rel_freq[batch[:, 1]] / (rel_freq.sum() + 1e-8)
        subj_scores = subj_freq[batch[:, 0]] / (subj_freq.sum() + 1e-8)
        obj_scores = obj_freq[batch[:, 2]] / (obj_freq.sum() + 1e-8)

        frequency_score = (rel_scores + subj_scores + obj_scores) / 3.0

        # 归一化时间差
        max_time = self.query_times.max()
        min_time = self.query_times.min()
        time_diffs = (max_time - batch[:, 3]).float()
        normalized_time_diffs = (time_diffs - time_diffs.min()) / (time_diffs.max() - time_diffs.min() + 1e-8)

        time_decay = torch.exp(-normalized_time_diffs)

        # 结合频率和时间衰减
        return alpha * frequency_score + (1 - alpha) * time_decay

    def train(self):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(self.quadruples)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch[0].to(self.device)
                target = self.compute_targets(batch).to(self.device)
                pred = self.model(batch, batch[:, 3])
                loss = self.criterion(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

        self.model.eval()
        with torch.no_grad():
            all_rewards = []
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
            for batch in loader:
                batch = batch[0].to(self.device)
                rewards = self.model(batch, batch[:, 3])
                all_rewards.append(rewards.cpu())
            self.rewards = torch.cat(all_rewards).numpy()
        return self.rewards

    def build_reward_matrix(self):
        """
        将训练得到的attention_rewards转换为(rel, dt)二维结构的reward_matrix
        
        Returns:
            reward_matrix: (num_rel, dt_size)的二维奖励矩阵
        """
        if self.rewards is None:
            raise ValueError("请先调用train()方法训练模型")
            
        quadruples = self.quadruples.cpu().numpy()
        attention_rewards = self.rewards
        
        # 统计所有 dt 的最大最小值，便于构建数组
        dt_list = []
        for i, quad in enumerate(quadruples):
            dt = (quad[3] - quad[3]) // self.time_span  # 这里dt=0，因为query_time=ts，实际用时需根据实际用法调整
            dt_list.append(dt)
        
        min_dt = min(dt_list)
        max_dt = max(dt_list)
        dt_offset = -min_dt  # 保证索引非负
        dt_size = max_dt - min_dt + 1
        
        # 初始化二维数组
        import numpy as np
        reward_matrix = np.zeros((self.num_relations, dt_size), dtype=np.float32)
        count_matrix = np.zeros((self.num_relations, dt_size), dtype=np.int32)
        
        # 聚合
        for i, quad in enumerate(quadruples):
            rel = quad[1]
            dt = (quad[3] - quad[3]) // self.time_span  # 这里dt=0，实际用时需根据实际用法调整
            idx_dt = dt + dt_offset
            reward_matrix[rel, idx_dt] += attention_rewards[i]
            count_matrix[rel, idx_dt] += 1
        
        # 求均值
        for rel in range(self.num_relations):
            for idx_dt in range(dt_size):
                if count_matrix[rel, idx_dt] > 0:
                    reward_matrix[rel, idx_dt] /= count_matrix[rel, idx_dt]
        
        return reward_matrix
    
    def train_and_build_matrix(self):
        """
        训练模型并直接返回reward matrix
        
        Returns:
            reward_matrix: (num_rel, dt_size)的二维奖励矩阵
        """
        self.train()
        return self.build_reward_matrix()
