# TITer

This is the data and coded for our EMNLP 2021 paper **TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting**

### Qucik Start

#### Data preprocessing

This is not necessary, but can greatly shorten the experiment time.

```
python preprocess_data.py --data_dir data/ICEWS14
```

#### Dirichlet parameter estimation

If you use the reward shaping module, you need to do this step.

```
python mle_dirichlet.py --data_dir data/ICEWS14 --time_span 24
```
```
python reward_learner.py --data_dir data/ICEWS14 --time_span 24
```
```
python reward_learner.py --data_dir data/YAGO --time_span 1
```

#### Train
you can run as following:
```
python main.py --data_path data/ICEWS14 --cuda --do_train --reward_shaping --time_span 24
```
```
python main2.py --data_path data/ICEWS14 --cuda --do_train --reward_shaping --time_span 24 --attention_rewards_file attention_rewards.pkl
```

#### Test
you can run as following:
```
python main.py --data_path data/ICEWS14 --cuda --do_test --IM --load_model_path logs/checkpoint.pth
```

```
python main2.py --data_path data/ICEWS14 --cuda --do_test --IM --load_model_path logs/checkpoint.pth
```
```
python main2.py --data_path data/ICEWS14 --cuda --do_test --load_model_path logs/checkpoint_30.pth --use_llm_tester

```

### Acknowledgments
model/dirichlet.py copy from https://github.com/ericsuh/dirichlet

### Cite

```
@inproceedings{Haohai2021TITer,
	title={TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting},
	author={Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.},
	booktitle={EMNLP},
	year={2021}
}
```