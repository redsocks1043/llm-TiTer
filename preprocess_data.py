import pickle
import os
import argparse
from model.environment import Env
from dataset.baseDataset import baseDataset
from tqdm import tqdm

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='数据预处理', usage='preprocess_data.py [<args>] [-h | --help]')
    # 添加数据目录参数，默认为'data/ICEWS14'
    parser.add_argument('--data_dir', default='data/ICEWS14', type=str, help='数据路径')
    # 添加输出文件参数，用于保存预处理后的数据
    parser.add_argument('--outfile', default='state_actions_space.pkl', type=str,
                        help='预处理数据的保存文件名')
    # 添加存储动作数量参数，0表示存储所有
    parser.add_argument('--store_actions_num', default=0, type=int,
                        help='存储邻居的最大数量，0表示存储所有')
    args = parser.parse_args()

    # 构建训练、测试、统计和验证数据的文件路径
    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    # 如果验证集不存在，则设为None
    if not os.path.exists(validF):
        validF = None

    # 创建数据集对象
    dataset = baseDataset(trainF, testF, statF, validF)
    # 配置环境参数
    config = {
        'num_rel': dataset.num_r,  # 关系的数量
        'num_ent': dataset.num_e,  # 实体的数量
    }
    # 创建环境对象
    env = Env(dataset.allQuadruples, config)
    # 用于存储状态-动作空间的字典
    state_actions_space = {}
    # 获取所有时间戳
    timestamps = list(dataset.get_all_timestamps())
    print(args)

    # 使用tqdm显示处理进度
    with tqdm(total=len(dataset.allQuadruples)) as bar:
        # 遍历所有四元组(头实体,关系,尾实体,时间)
        for (head, rel, tail, t) in dataset.allQuadruples:
            # 如果头实体在该时间戳下的状态未处理
            if (head, t, True) not in state_actions_space.keys():
                # 获取头实体在该时间戳下的状态-动作空间（正向和反向）
                state_actions_space[(head, t, True)] = env.get_state_actions_space_complete(head, t, True,
                                                                                            args.store_actions_num)
                state_actions_space[(head, t, False)] = env.get_state_actions_space_complete(head, t, False,
                                                                                             args.store_actions_num)
            # 如果尾实体在该时间戳下的状态未处理
            if (tail, t, True) not in state_actions_space.keys():
                # 获取尾实体在该时间戳下的状态-动作空间（正向和反向）
                state_actions_space[(tail, t, True)] = env.get_state_actions_space_complete(tail, t, True,
                                                                                            args.store_actions_num)
                state_actions_space[(tail, t, False)] = env.get_state_actions_space_complete(tail, t, False,
                                                                                             args.store_actions_num)
            bar.update(1)

    # 将处理后的状态-动作空间保存到文件
    pickle.dump(state_actions_space, open(os.path.join(args.data_dir, args.outfile), 'wb'))
