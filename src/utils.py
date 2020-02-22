import numpy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch
from matplotlib import pyplot
import json
import os


DATA_DIR = "../data"

# default mean of cifar100
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# default std of cifar100
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# default mean of cifar10
CIFAR10_TRAIN_MEAN = (0.49139765, 0.48215759, 0.44653141)
# default std of cifar10
CIFAR10_TRAIN_STD = (0.24703199, 0.24348481, 0.26158789)


def get_training_dataloader(dataset, batch_size=128, num_workers=4, shuffle=True):

    if dataset == "cifar100":
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
        _data = torchvision.datasets.CIFAR100
    elif dataset == "cifar10":
        mean = CIFAR10_TRAIN_MEAN
        std = CIFAR10_TRAIN_STD
        _data = torchvision.datasets.CIFAR10
    else:
        raise ValueError(f'dataset "{dataset}" is not supported!')

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    training_dataset = _data(root=DATA_DIR, train=True, download=True,
                             transform=transform_train)

    training_loader = DataLoader(training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader


def get_test_dataloader(dataset, batch_size=128, num_workers=4, shuffle=False):

    if dataset == "cifar100":
        mean = CIFAR100_TRAIN_MEAN
        std = CIFAR100_TRAIN_STD
        _data = torchvision.datasets.CIFAR100
    elif dataset == "cifar10":
        mean = CIFAR10_TRAIN_MEAN
        std = CIFAR10_TRAIN_STD
        _data = torchvision.datasets.CIFAR10
    else:
        raise ValueError(f'dataset "{dataset}" is not supported!')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test = _data(root=DATA_DIR, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


# TODO use for warming up learning rate?
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def attack_loss(X, y, model: torch.nn.Module, attacker, eps=0.1, loss=torch.nn.CrossEntropyLoss()) -> torch.Tensor:
    y_hat = model(X)
    adv_y_hat = attacker(model, eps=eps).cal_perturbation(X, y)

    return loss(y_hat, y) + loss(adv_y_hat, y)


class DrawTools:
    """a tool to help draw"""

    def __init__(self, save_dir, epochs=200):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.epochs = epochs

    @property
    def plt(self):
        return pyplot

    def draw_loss(self, loss: list):
        self.plt.plot(range(self.epochs), loss)

        self.plt.xlabel("epochs")
        self.plt.ylabel("loss")

        self.plt.savefig(os.path.join(self.save_dir, "loss.svg"), format="svg")
        self.plt.clf()

    def draw_acc(self, train_acc: list, test_acc: list):
        self.plt.plot(range(self.epochs), train_acc, label="train")
        self.plt.plot(range(self.epochs), test_acc, label="test", linestyle="--")

        self.plt.xlabel("epochs")
        self.plt.ylabel("accuracy")
        self.plt.legend(loc="upper left")

        self.plt.savefig(os.path.join(self.save_dir, "accuracy.svg"), format="svg")
        self.plt.clf()

    def draw_all(self, loss: list, train_acc: list, test_acc: list):
        self.draw_acc(train_acc, test_acc)
        self.draw_loss(loss)
        self.save_result({
            "loss": loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        }, path=os.path.join(self.save_dir, "result.json"))

    def draw_adv(self, eps, acc):
        self.plt.plot(eps, acc)

        self.plt.xlabel("epsilons")
        self.plt.ylabel("incorrectness")

        self.plt.savefig(os.path.join(self.save_dir, "adv.svg"), format="svg")
        self.save_result({
            "eps": eps,
            "acc": acc
        }, path=os.path.join(self.save_dir, "adv.json"))
        self.plt.clf()

    @staticmethod
    def save_result(result, path):
        with open(path, "w", encoding="utf8") as f:
            json.dump(result, f)

    @staticmethod
    def load_result(path):
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
