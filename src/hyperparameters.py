import torch


# choose used device
DEFAULT_DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
# learning rate
DEFAULT_LR = 0.1
# TODO the effect of momentum?
DEFAULT_MOMENTUM = 0.9
# the size of each batch
DEFAULT_BATCH_SIZE = 128
# TODO https://www.cnblogs.com/hesse-summer/p/11343870.html
DEFAULT_NUM_WORKER = 4
# the number of iterations
DEFAULT_EPOCHS = 200
# use multiple GPUs or not
DEFAULT_PARALLELISM = False
# todo
# maybe overfit
# decrease learning rate every step
MILESTONES = [60, 120, 160]
# warm up training phases
WARM_PHASES = 1