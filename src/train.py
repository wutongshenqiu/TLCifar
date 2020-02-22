from dataclasses import dataclass
from typing import Dict
import os
import time
from utils import get_training_dataloader, get_test_dataloader
import resnet
from torch import optim
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.nn.modules import loss
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
from tl import TLResNet
from utils import WarmUpLR
import json
from hyperparameters import *


# # choose used device
# DEFAULT_DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
# # learning rate
# DEFAULT_LR = 0.1
# # TODO the effect of momentum?
# DEFAULT_MOMENTUM = 0.9
# # the size of each batch
# DEFAULT_BATCH_SIZE = 128
# # TODO https://www.cnblogs.com/hesse-summer/p/11343870.html
# DEFAULT_NUM_WORKER = 4
# # the number of iterations
# DEFAULT_EPOCHS = 200
# # use multiple GPUs or not
# DEFAULT_PARALLELISM = False
# # todo
# # maybe overfit
# # decrease learning rate every step
# MILESTONES = [60, 120, 160]
# # warm up training phases
# WARM_PHASES = 1

# if the batch_size and model structure is fixed, this may accelerate the training process
torch.backends.cudnn.benchmark = True


@dataclass
class HyperParameter:
    scheduler: optim.lr_scheduler
    optimizer: Optimizer
    # the loss function
    criterion: loss._Loss
    batch_size: int
    epochs: int
    device: str


class Trainer:

    def __init__(self, model: Module, train_loader: DataLoader, test_loader: DataLoader,
                 device=DEFAULT_DEVICE,  lr=DEFAULT_LR, momentum=DEFAULT_MOMENTUM,
                 epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                 parallelism=DEFAULT_PARALLELISM, milestones=MILESTONES,
                 gamma=0.2, warm_phases=WARM_PHASES, criterion=loss.CrossEntropyLoss()):
        print("initialize trainer")
        # parameter pre-processing
        self.test_loader = test_loader

        if torch.cuda.device_count() > 1 and parallelism:
            print(f"using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
        self.model.to(device)

        optimizer = optim.SGD(
            # choose whether train or not
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=5e-4
        )

        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        # warm phases
        self.warm_phases = warm_phases
        # warmup learning rate
        self.warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * self.warm_phases)

        self.hp = HyperParameter(
            scheduler=train_scheduler, optimizer=optimizer, criterion=criterion,
            batch_size=batch_size, epochs=epochs, device=device
        )

        self.train_loader = train_loader
        print("initialize finished")
        print(f"hyper parameter: {self.hp}")

    def train(self, save_path, attack=False, attacker=None, params: Dict = None):
        self._init_attacker(attack, attacker, params)

        batch_number = len(self.train_loader)
        # get current learning rate
        now_lr = self.hp.optimizer.state_dict().get("param_groups")[0].get("lr")
        # record best accuracy
        best_acc = 0

        for ep in range(1, self.hp.epochs+1):

            training_acc, running_loss = 0, .0
            start_time = time.process_time()

            for index, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.hp.device), data[1].to(self.hp.device)

                self.hp.optimizer.zero_grad()
                if attack:
                    # calculate this first, for this will zero the grad
                    adv_inputs = self.attacker.calc_perturbation(inputs, labels)
                    # zero the grad
                    self.hp.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    adv_outputs = self.model(adv_inputs)
                    _loss = self.hp.criterion(outputs, labels) + self.hp.criterion(adv_outputs, labels)
                else:
                    outputs = self.model(inputs)
                    _loss = self.hp.criterion(outputs, labels)

                _loss.backward()
                self.hp.optimizer.step()

                outputs: torch.Tensor
                training_acc += (outputs.argmax(dim=1) == labels).float().mean().item()

                # warm up learning rate
                if ep <= self.warm_phases:
                    self.warmup_scheduler.step()

                # detect learning rate change
                new_lr = self.hp.optimizer.state_dict().get("param_groups")[0].get("lr")
                if new_lr != now_lr:
                    now_lr = new_lr
                    print(f"learning rate changes to {now_lr:.6f}")

                running_loss += _loss.item()

                if index % batch_number == batch_number - 1:
                    end_time = time.process_time()

                    acc = self.test(self.model, test_loader=self.test_loader, device=self.hp.device)
                    print(f"epoch: {ep}   loss: {(running_loss / batch_number):.6f}   train accuracy: {training_acc / batch_number}   "
                          f"test accuracy: {acc}   time: {end_time - start_time:.2f}s")

                    if best_acc < acc:
                        best_acc = acc
                        self._save_best_model(save_path, ep, acc)

            # change learning rate by step
            self.hp.scheduler.step(ep)
        torch.save(self.model.state_dict(), f"{save_path}-latest")
        print("finished training")
        print(f"best accuracy on test set: {best_acc}")

    @staticmethod
    def test(model: Module, test_loader, device, debug=False):

        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                _, y_hats = model(inputs).max(1)
                match = (y_hats == labels)
                correct += len(match.nonzero())

        if debug:
            print(f"Testing: {len(test_loader.dataset)}")
            print(f"correct: {correct}")
            print(f"accuracy: {100*correct/len(test_loader.dataset):.3f}%")

        return correct / len(test_loader.dataset)

    def _init_attacker(self, attack, attacker, params):
        self.attack = attack
        if attack:
            print(f"robustness training with {attacker.__name__}")
            self.attacker = attacker(self.model, **params)
            self.attacker.print_params()
        else:
            print("normal training")

    def _save_best_model(self, save_path, current_epochs, accuracy):
        """save best model with current info"""
        info = {
            "current_epochs": current_epochs,
            "total_epochs": self.hp.epochs,
            "accuracy": accuracy
        }
        if self.attack:
            info.update({
                "attack": self.attack,
                "attacker": type(self.attacker).__name__,
                "epsilons": self.attacker.epsilon,
            })
        with open(os.path.join(os.path.dirname(save_path), "info.json"), "w", encoding="utf8") as f:
            json.dump(info, f)
        torch.save(self.model.state_dict(), f"{save_path}-best")

    @staticmethod
    def train_tl(origin_model_path, save_path, train_loader,
                 test_loader, device, choice="resnet50"):
        print(f"transform learning on model: {origin_model_path}")
        model = TLResNet.create_model(choice)
        model.load_model(origin_model_path)
        trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, device=device)
        trainer.train(save_path)


if __name__ == '__main__':
    train_loader = get_training_dataloader("cifar100")
    test_loader = get_test_dataloader("cifar100")
    model = resnet.resnet50()
    trainer = Trainer(model, train_loader, test_loader)
    trainer.train("1")




