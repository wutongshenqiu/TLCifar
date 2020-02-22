from __future__ import print_function
from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn.functional as functional


class FGSMAttack:

    def __init__(self, model: torch.nn.Module, clip_min=0, clip_max=1,
                 epsilon=0.3, p=-1, loss_function: Callable[[Any], Tensor] = None
                 ):
        self.model = model
        self.min = clip_min
        self.max = clip_max
        self.epsilon = epsilon
        self.p = p
        if loss_function is None:
            self.loss_function = functional.cross_entropy

    def calc_perturbation(self, x: Tensor, target: Tensor) -> Tensor:
        xt = x
        xt.requires_grad = True
        y_hat: Tensor = self.model(xt)
        # todo
        # this may interrupt other grad
        self.model.zero_grad()
        loss: Tensor = self.loss_function(y_hat, target)
        loss.backward()
        grad = xt.grad.data
        sign = grad.sign()
        return xt + self.epsilon * sign

    def print_params(self):
        print(f"epsilon: {self.epsilon}, loss: {self.loss_function.__name__}")


class PGDAttack:

    def __init__(self, model: torch.nn.Module, clip_min=0, clip_max=1,
                 random_init=True, epsilon=0.3, alpha=0.01, iter_num=5, p=-1,
                 loss_function: Callable[[Any], Tensor] = None
                 ):

        self.min = clip_min
        self.max = clip_max
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.p = p
        self.random_init = random_init
        self.iter_num = iter_num
        if loss_function is None:
            self.loss_function = functional.cross_entropy

    def random_delta(self, delta: Tensor) -> Tensor:
        if self.p == -1:
            delta.uniform_(-1, 1)
            delta = delta * self.epsilon
        else:
            pass
        return delta

    def calc_perturbation(self, x: Tensor, target: Tensor) -> Tensor:
        delta = torch.zeros_like(x)
        if self.random_init:
            delta = self.random_delta(delta)
        xt = x + delta
        xt.requires_grad = True

        for it in range(self.iter_num):
            y_hat = self.model(xt)  # type: Tensor
            loss = self.loss_function(y_hat, target)  # type: Tensor

            self.model.zero_grad()
            loss.backward()
            if self.p == -1:
                grad_sign = xt.grad.detach().sign()
                xt.data = xt.detach() + self.alpha * grad_sign
                xt.data = torch.clamp(xt - x, -self.epsilon, self.epsilon) + x
                xt.data = torch.clamp(xt.detach(), self.min, self.max)
            else:
                pass

            xt.grad.data.zero_()

        return xt

    def print_params(self):
        print(f"iter_num: {self.iter_num}, epsilon: {self.epsilon}, loss: {self.loss_function.__name__}")
