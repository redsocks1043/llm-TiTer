import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from model.TemporalAttention import RewardLearner
from utils.logger import *
from utils.trainer import Trainer
from utils.tester import Tester
from dataset.baseDataset import baseDataset, QuadruplesDataset
from model.agent import Agent
from model.environment import Env
from model.episode import Episode
from model.policyGradient import PG
from model.lc import MultiAgentCoHPredictor
import os
import pickle

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Forecasting Models',
        usage='main2.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='whether to use GPU or not.')
    parser.add_argument('--data_path', type=str, default='data/ICEWS14', help='Path to data.')
    parser.add_argument('--do_train', action='store_true', help='whether to train.')
    parser.add_argument('--do_test', action='store_true', help='whether to test.')
    parser.add_argument('--save_path', default='logs', type=str, help='log and model save path.')
    parser.add_argument('--load_model_path', default='logs', type=str, help='trained model checkpoint path.')

    # Train Params
    parser.add_argument('--batch_size', default=512, type=int, help='training batch size.')
    parser.add_argument('--max_epochs', default=400, type=int, help='max training epochs.')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number used for dataloader.')
    parser.add_argument('--valid_epoch', default=30, type=int, help='validation frequency.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate.')
    parser.add_argument('--save_epoch', default=30, type=int, help='model saving frequency.')
    parser.add_argument('--clip_gradient', default=10.0, type=float, help='for gradient crop.')

    # Test Params
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='test batch size, it needs to be set to 1 when using IM module.')
    parser.add_argument('--beam_size', default=100, type=int, help='the beam number of the beam search.')
    parser.add_argument('--test_inductive', action='store_true', help='whether to verify inductive inference performance.')
    parser.add_argument('--IM', action='store_true', help='whether to use IM module.')
    parser.add_argument('--mu', default=0.1, type=float, help='the hyperparameter of IM module.')

    # Agent Params
    parser.add_argument('--ent_dim', default=100, type=int, help='Embedding dimension of the entities')
    parser.add_argument('--rel_dim', default=100, type=int, help='Embedding dimension of the relations')
    parser.add_argument('--state_dim', default=100, type=int, help='dimension of the LSTM hidden state')
    parser.add_argument('--hidden_dim', default=100, type=int, help='dimension of the MLP hidden layer')
    parser.add_argument('--time_dim', default=20, type=int, help='Embedding dimension of the timestamps')
    parser.add_argument('--entities_embeds_method', default='dynamic', type=str,
                        help='representation method of the entities, dynamic or static')

    # Environment Params
    parser.add_argument('--state_actions_path', default='state_actions_space.pkl', type=str,
                        help='the file stores preprocessed candidate action array.')

    # Episode Params
    parser.add_argument('--path_length', default=3, type=int, help='the agent search path length.')
    parser.add_argument('--max_action_num', default=50, type=int, help='the max candidate actions number.')

    # Policy Gradient Params
    parser.add_argument('--Lambda', default=0.99, type=float, help='update rate of baseline.')
    parser.add_argument('--Gamma', default=0.95, type=float, help='discount factor of Bellman Eq.')
    parser.add_argument('--Ita', default=0.01, type=float, help='regular proportionality constant.')
    parser.add_argument('--Zita', default=0.9, type=float, help='attenuation factor of entropy regular term.')

    # reward shaping params
    parser.add_argument('--reward_shaping', action='store_true', help='whether to use reward shaping.')
    parser.add_argument('--time_span', default=24, type=int, help='24 for ICEWS, 1 for WIKI and YAGO')
    # parser.add_argument('--alphas_pkl', default='dirchlet_alphas.pkl', type=str,
    #                     help='the file storing the alpha parameters of the Dirichlet distribution.')
    # parser.add_argument('--k', default=300, type=int, help='statistics recent K historical snapshots.')

    # 添加注意力模型参数
    parser.add_argument('--attention_rewards_file', default='attention_rewards.pkl', type=str,
                        help='Path to save/load attention rewards')
    parser.add_argument('--attention_hidden_dim', default=304, type=int,
                        help='Hidden dimension for attention model')
    parser.add_argument('--attention_epochs', default=50, type=int,
                        help='Number of epochs for training attention model')

    ## 添加大模型预测
    parser.add_argument('--use_llm_tester', action='store_true', help='whether to use LLM for testing.')
    return parser.parse_args(args)

def get_model_config(args, num_ent, num_rel):
    config = {
        'cuda': args.cuda,  # whether to use GPU or not.
        'batch_size': args.batch_size,  # training batch size.
        'num_ent': num_ent,  # number of entities
        'num_rel': num_rel,  # number of relations
        'ent_dim': args.ent_dim,  # Embedding dimension of the entities
        'rel_dim': args.rel_dim,  # Embedding dimension of the relations
        'time_dim': args.time_dim,  # Embedding dimension of the timestamps
        'state_dim': args.state_dim,  # dimension of the hidden state
        'action_dim': args.ent_dim + args.rel_dim,  # dimension of the actions
        'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_hidden_dim': args.hidden_dim,  # dimension of the MLP hidden layer
        'path_length': args.path_length,  # agent search path length
        'max_action_num': args.max_action_num,  # max candidate action number
        'lambda': args.Lambda,  # update rate of baseline
        'gamma': args.Gamma,  # discount factor of Bellman Eq.
        'ita': args.Ita,  # regular proportionality constant
        'zita': args.Zita,  # attenuation factor of entropy regular term
        'beam_size': args.beam_size,  # beam size for beam search
        'entities_embeds_method': args.entities_embeds_method,  # default: 'dynamic', otherwise static encoder will be used
    }
    return config

def main(args):
    #######################Set Logger#################################
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(args)

    #######################Create DataLoader#################################
    train_path = os.path.join(args.data_path, 'train.txt')
    test_path = os.path.join(args.data_path, 'test.txt')
    stat_path = os.path.join(args.data_path, 'stat.txt')
    valid_path = os.path.join(args.data_path, 'valid.txt')

    baseData = baseDataset(train_path, test_path, stat_path, valid_path)

    trainDataset  = QuadruplesDataset(baseData.trainQuadruples, baseData.num_r)
    train_dataloader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validDataset = QuadruplesDataset(baseData.validQuadruples, baseData.num_r)
    valid_dataloader = DataLoader(
        validDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    testDataset = QuadruplesDataset(baseData.testQuadruples, baseData.num_r)
    test_dataloader = DataLoader(
        testDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ######################Creat the agent and the environment###########################
    config = get_model_config(args, baseData.num_e, baseData.num_r)
    logging.info(config)
    logging.info(args)

    # creat the agent
    agent = Agent(config)

    # creat the environment
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)
    if not os.path.exists(state_actions_path):
        logging.info(f"状态动作空间文件 {state_actions_path} 不存在，将使用 None")
        state_action_space = None
    else:
        logging.info(f"尝试加载状态动作空间文件: {state_actions_path}")
        try:
            with open(state_actions_path, 'rb') as f:
                state_action_space = pickle.load(f)
            logging.info(f"成功加载状态动作空间文件: {state_actions_path}")
        except ModuleNotFoundError as e:
            logging.warning(f"由于 NumPy 兼容性问题，无法加载 {state_actions_path}: {str(e)}，使用 None")
            state_action_space = None
        except Exception as e:
            logging.error(f"加载 {state_actions_path} 失败: {str(e)}，使用 None")
            state_action_space = None
    env = Env(baseData.allQuadruples, config, state_action_space)

    # Create episode controller
    episode = Episode(env, agent, config)
    if args.cuda:
        episode = episode.cuda()
    pg = PG(config)  # Policy Gradient
    optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)

    # Load the model parameters
    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        episode.load_state_dict(params['model_state_dict'])
        # 只在训练模式下加载优化器状态
        if args.do_train and not args.do_test:
            try:
                optimizer.load_state_dict(params['optimizer_state_dict'])
                logging.info('成功加载优化器状态')
            except ValueError as e:
                logging.warning(f"无法加载优化器状态，错误信息：{str(e)}")
                logging.info("继续执行而不加载优化器状态...")
        else:
            logging.info('测试模式：跳过加载优化器状态')

        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    ######################Training and Testing###########################
    distributions = None
    if args.reward_shaping:
        reward_file = os.path.join(args.data_path, args.attention_rewards_file)
        
        # 首先创建RewardLearner对象（无论是否需要训练）
        reward_learner = RewardLearner(
            quadruples=baseData.trainQuadruples,
            num_entities=baseData.num_e,
            num_relations=baseData.num_r,
            entity_dim=args.ent_dim,
            relation_dim=args.rel_dim,
            time_dim=args.time_dim,
            hidden_dim=args.attention_hidden_dim,
            device='cuda' if args.cuda else 'cpu',
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.attention_epochs,
            time_span=args.time_span  # 添加time_span参数
        )
        
        if os.path.exists(reward_file):
            # 加载已计算的注意力奖励
            logging.info('Loading pre-computed attention rewards...')
            attention_rewards = pickle.load(open(reward_file, 'rb'))
            # 将加载的奖励设置到reward_learner中
            reward_learner.rewards = attention_rewards
        else:
            # 训练注意力奖励学习器
            logging.info('Training attention reward model...')
            # 获取计算的奖励
            attention_rewards = reward_learner.train()
        
            # 保存计算的奖励
            pickle.dump(attention_rewards, open(reward_file, 'wb'))
            logging.info(f'Saved attention rewards to {reward_file}')
        
            # 输出奖励统计信息
            logging.info(f"Reward statistics:")
            logging.info(f"Mean: {np.mean(attention_rewards):.4f}")
            logging.info(f"Std: {np.std(attention_rewards):.4f}")
            logging.info(f"Min: {np.min(attention_rewards):.4f}")
            logging.info(f"Max: {np.max(attention_rewards):.4f}")
        
        # 构建reward matrix
        distributions = reward_learner.build_reward_matrix()
    #     # === 统一处理 attention_rewards 为 (rel, dt) 二维结构 ===
    #     quadruples = baseData.trainQuadruples
    #     time_span = args.time_span
    #     num_rel = baseData.num_r
    #     # 统计所有 dt 的最大最小值，便于构建数组
    #     dt_list = []
    #     rel_list = []
    #     for i, quad in enumerate(quadruples):
    #         rel = quad[1]
    #         dt = (quad[3] - quad[3]) // time_span  # 这里dt=0，因为query_time=ts，实际用时需根据实际用法调整
    #         dt_list.append(dt)
    #         rel_list.append(rel)
    #     min_dt = min(dt_list)
    #     max_dt = max(dt_list)
    #     dt_offset = -min_dt  # 保证索引非负
    #     dt_size = max_dt - min_dt + 1
    #     # 初始化二维数组
    #     reward_matrix = np.zeros((num_rel, dt_size), dtype=np.float32)
    #     count_matrix = np.zeros((num_rel, dt_size), dtype=np.int32)
    #     # 聚合
    #     for i, quad in enumerate(quadruples):
    #         rel = quad[1]
    #         dt = (quad[3] - quad[3]) // time_span  # 这里dt=0，实际用时需根据实际用法调整
    #         idx_dt = dt + dt_offset
    #         reward_matrix[rel, idx_dt] += attention_rewards[i]
    #         count_matrix[rel, idx_dt] += 1
    #     # 求均值
    #     for rel in range(num_rel):
    #         for idx_dt in range(dt_size):
    #             if count_matrix[rel, idx_dt] > 0:
    #                 reward_matrix[rel, idx_dt] /= count_matrix[rel, idx_dt]
    #     # 用 reward_matrix 替换 distributions
    #     distributions = reward_matrix
    # # === 统一处理 attention_rewards 为 (rel, dt) 二维结构 ===

    # ===================初始化MutiAgentCoHPredictor开始==========================
    logging.info("Initializing Multi-Agent CoH Predictor...")
    coh_predictor = MultiAgentCoHPredictor(
        dataset_path=args.data_path,
        # 根据您的需求修改这里的模型名称
        execution_model='llama2:7b',  # 只保留执行模型
        device="cuda:0" if args.cuda else "cpu"  # 确保设备与参数同步
    )
    # ===================初始化MutiAgentCoHPredictor结束==========================
    logging.info("Multi-Agent CoH Predictor initialized.")
    trainer = Trainer(episode, pg, optimizer, args, distributions)
    tester = Tester(episode, args, baseData.train_entities, baseData.RelEntCooccurrence, distributions,coh_predictor)


    if args.do_train:
        logging.info('Start Training......')
        for i in range(args.max_epochs):
            loss, reward, success_rate = trainer.train_epoch(train_dataloader, trainDataset.__len__())
            logging.info('Epoch {}/{} Loss: {}, reward: {}, successful = {:.4f}'.format(i, args.max_epochs, loss, reward, success_rate))

            if i % args.save_epoch == 0 and i != 0:
                trainer.save_model('checkpoint_{}.pth'.format(i))
                logging.info('Save Model in {}'.format(args.save_path))

            if i % args.valid_epoch == 0 and i != 0:
                logging.info('Start Val......')
                metrics = tester.test(valid_dataloader,
                                      validDataset.__len__(),
                                      baseData.skip_dict,
                                      config['num_ent'])
                for mode in metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, metrics[mode]))

        trainer.save_model()
        logging.info('Save Model in {}'.format(args.save_path))

    if args.do_test:
        if args.use_llm_tester:
            # logging.info("Using LLM-enhanced Tester...")
            # metrics = tester.test_with_llm(test_dataloader,
            #                                testDataset.__len__(),
            #                                baseData.skip_dict,
            #                                config['num_ent'],
            #                                baseData.id_to_entity)  # 传递id->entity映射
            logging.info("Using two-phase LLM-enhanced Tester...")
            metrics = tester.test_with_llm_two_phase(test_dataloader,
                                       testDataset.__len__(),
                                       baseData.skip_dict,
                                       config['num_ent'],
                                       baseData.id_to_entity)
        else:
            logging.info("Using standard RL Tester...")
            metrics = tester.test(test_dataloader,
                                  testDataset.__len__(),
                                  baseData.skip_dict,
                                  config['num_ent'])

        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)

