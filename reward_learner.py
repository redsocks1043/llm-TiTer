import argparse
import os
import pickle
import time

import numpy as np
import torch

from dataset.baseDataset import baseDataset
from model.TemporalAttention import RewardLearner

def main():
    parser = argparse.ArgumentParser(description='Temporal Reward Learning')
    parser.add_argument('--data_dir', default='data/ICEWS14', type=str)
    parser.add_argument('--outfile', default='attention_rewards.pkl', type=str)
    parser.add_argument('--entity_dim', default=128, type=int, help='实体嵌入维度')
    parser.add_argument('--relation_dim', default=64, type=int, help='关系嵌入维度')
    parser.add_argument('--time_dim', default=16, type=int, help='时间嵌入维度')
    parser.add_argument('--hidden_dim', default=304, type=int, help='隐藏层维度')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--device', default='cuda', type=str, help='计算设备 (cuda/cpu)')
    parser.add_argument('--time_span', default=24, type=int, help='时间跨度，例如ICEWS14为24，YAGO为1') # 添加 time_span 参数
    args = parser.parse_args()

    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU进行计算")
        args.device = 'cpu'
    else:
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")

    # 构建数据文件路径
    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    if not os.path.exists(validF):
        validF = None

    # 创建数据集对象
    dataset = baseDataset(trainF, testF, statF, validF)

    print(f"Number of entities: {dataset.num_e}")
    print(f"Number of relations: {dataset.num_r}")
    print(f"Number of training quadruples: {len(dataset.trainQuadruples)}")

    print("Initializing reward learner...")
    start_time = time.time()

    # 创建奖励学习器
    reward_learner = RewardLearner(
        quadruples=dataset.trainQuadruples,
        num_entities=dataset.num_e,
        num_relations=dataset.num_r,
        entity_dim=args.entity_dim,
        relation_dim=args.relation_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        time_span=args.time_span  # 传递 time_span 参数
    )

    # 训练奖励模型
    rewards = reward_learner.train()

    training_time = time.time() - start_time
    print(f"\n训练完成，耗时: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")

    # 结果分析
    print("\nReward Distribution Analysis:")
    print(f"- Mean ± Std: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"- Median: {np.median(rewards):.4f}")
    print(f"- 25-75 Percentile: {np.percentile(rewards, 25):.4f} - {np.percentile(rewards, 75):.4f}")

    # 保存奖励文件
    out_path = os.path.join(args.data_dir, args.outfile)
    pickle.dump(rewards, open(out_path, 'wb'))
    print(f"Rewards saved to {out_path}")

if __name__ == '__main__':
    main()