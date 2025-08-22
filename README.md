
### Qucik Start

#### Data preprocessing

This is not necessary, but can greatly shorten the experiment time.

```
python preprocess_data.py --data_dir data/ICEWS14
```

#### Train
you can run as following:

```
python main2.py --data_path data/ICEWS14 --cuda --do_train --reward_shaping --time_span 24 
```

#### Test
you can run as following:

```
python main2.py --data_path data/ICEWS14 --cuda --do_test --IM --load_model_path logs/checkpoint.pth
```


### Cite

```
@inproceedings{Haohai2021TITer,
	title={TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting},
	author={Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He.},
	booktitle={EMNLP},
	year={2021}
}
```