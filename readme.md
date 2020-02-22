### Usage
- 训练模型
```shell script
python main.py [-h] --pattern PATTERN --dataset DATASET --save_path SAVE_PATH [--device DEVICE] [--lr LR] [--momentum MOMENTUM] [--epochs EPOCHS] [--batch BATCH] [--num_worker NUM_WORKER] [--gamma GAMMA] [--phases PHASES] [--old OLD]
               [--attacker ATTACKER] [--epsilon EPSILON]

a cli that supports train/tl/attack

optional arguments:
  -h, --help            show this help message and exit
  --pattern PATTERN     use train/tl/attack to determine the pattern
  --dataset DATASET     choose the dataset, supporting cifar100 or cifar10
  --save_path SAVE_PATH
                        the saving path of model
  --device DEVICE       training device
  --lr LR               learning rate
  --momentum MOMENTUM   momentum
  --epochs EPOCHS       training epochs
  --batch BATCH         the size of each batch
  --num_worker NUM_WORKER
                        num_workers
  --gamma GAMMA         decrease rate when reach milestones
  --phases PHASES       epochs for warming model
  --old OLD             old model path
  --attacker ATTACKER   attackers, now support FGSMAttack and PGDAttack
  --epsilon EPSILON     epsilons

```

- 测试效果
```shell script
python test_attack.py [-h] --dataset DATASET --model_path MODEL_PATH [--save_path SAVE_PATH]

a cli that supports train/tl/attack

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     the dataset, supporting cifar10 and cifar100
  --model_path MODEL_PATH
                        the path of the model
  --save_path SAVE_PATH
                        the path to save result
```