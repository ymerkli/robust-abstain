import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
import warnings
from typing import List
from abc import ABCMeta, abstractmethod


LR_SCHEDULERS = ['step_lr', 'cos_anneal', 'trades', 'trades_fixed', 'cosine', 'wrn']


class TradesLR(_LRScheduler):
    """LR schedule used in TRADES repo (https://github.com/yaodongyu/TRADES/blob/master/trades.py)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        n_epochs: int,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.n_epochs = n_epochs
        super(TradesLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        group_lrs = self.base_lrs
        if self.last_epoch >= 0.75 * self.n_epochs:
            group_lrs = [0.1 * lr for lr in group_lrs]
        return group_lrs

    def _get_closed_form_lr(self) -> List[float]:
        group_lrs = self.base_lrs
        if self.last_epoch >= 0.75 * self.n_epochs:
            group_lrs = [0.1 * lr for lr in group_lrs]
        return group_lrs


class TradesFixedLR(_LRScheduler):
    """LR schedule as in TRADES paper
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        n_epochs: int,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.n_epochs = n_epochs
        super(TradesFixedLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        group_lrs = self.base_lrs
        if self.last_epoch >= 0.75 * self.n_epochs:
            group_lrs = [0.1 * lr for lr in group_lrs]
        if self.last_epoch >= 0.9 * self.n_epochs:
            group_lrs = [0.01 * lr for lr in group_lrs]
        if self.last_epoch >= self.n_epochs:
            group_lrs = [0.001 * lr for lr in group_lrs]
        return group_lrs

    def _get_closed_form_lr(self) -> List[float]:
        group_lrs = self.base_lrs
        if self.last_epoch >= 0.75 * self.n_epochs:
            group_lrs = [0.1 * lr for lr in group_lrs]
        if self.last_epoch >= 0.9 * self.n_epochs:
            group_lrs = [0.01 * lr for lr in group_lrs]
        if self.last_epoch >= self.n_epochs:
            group_lrs = [0.001 * lr for lr in group_lrs]
        return group_lrs


class CosineLR(_LRScheduler):
    """Cosine LR schedule
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        n_epochs: int,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.n_epochs = n_epochs
        super(CosineLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [base_lr * 0.5 * (1 + np.cos((self.last_epoch - 1) / self.n_epochs * np.pi))
                for base_lr in self.base_lrs]

    def _get_closed_form_lr(self) -> List[float]:
        return [base_lr * 0.5 * (1 + np.cos((self.last_epoch - 1) / self.n_epochs * np.pi))
                for base_lr in self.base_lrs]


class WRNLR(_LRScheduler):
    """LR schedule as in WRN paper
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        n_epochs: int,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.n_epochs = n_epochs
        super(WRNLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        group_lrs = self.base_lrs
        if self.last_epoch >= 0.3 * self.n_epochs:
            group_lrs = [0.2 * lr for lr in group_lrs]
        if self.last_epoch >= 0.6 * self.n_epochs:
            group_lrs = [0.2 * 0.2 * lr for lr in group_lrs]
        if self.last_epoch >= 0.8 * self.n_epochs:
            group_lrs = [0.2 * 0.2 * 0.2 * lr for lr in group_lrs]
        return group_lrs

    def _get_closed_form_lr(self) -> List[float]:
        group_lrs = self.base_lrs
        if self.last_epoch >= 0.3 * self.n_epochs:
            group_lrs = [0.2 * lr for lr in group_lrs]
        if self.last_epoch >= 0.6 * self.n_epochs:
            group_lrs = [0.2 * 0.2 * lr for lr in group_lrs]
        if self.last_epoch >= 0.8 * self.n_epochs:
            group_lrs = [0.2 * 0.2 * 0.2 * lr for lr in group_lrs]
        return group_lrs


def get_lr_scheduler(
        args: object, lr_scheduler_type: str, opt: optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
    """Easily get a specified LR scheduler.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        lr_scheduler_type (str): Name of the scheduler.
        opt (optim.Optimizer): Optimizer.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: LR scheduler.
    """
    lr_scheduler = None
    if lr_scheduler_type == 'step_lr':
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    elif lr_scheduler_type == 'cos_anneal':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif lr_scheduler_type == 'trades':
        lr_scheduler = TradesLR(opt, args.epochs)
    elif lr_scheduler_type == 'trades_fixed':
        lr_scheduler = TradesFixedLR(opt, args.epochs)
    elif lr_scheduler_type == 'cosine':
        lr_scheduler = CosineLR(opt, args.epochs)
    elif lr_scheduler_type == 'wrn':
        lr_scheduler = WRNLR(opt, args.epochs)
    else:
        raise ValueError(f'Error: unkown LR schedule {lr_scheduler_type}')

    return lr_scheduler


class Scheduler(metaclass=ABCMeta):
    @abstractmethod
    def step(self, value: float, epoch: int) -> float:
        raise NotImplementedError


class StepScheduler(Scheduler):
    """Decay a learning parameter by gamma every step_size. Equivalent to
    torch.optim.lr_scheduler.StepLR, but can be applied to any parameter.

    Args:
        step_size (int): Period of parameter decay.
        gamma (float): Multiplicative factor of parameter decay. Defaults to 0.1.
    """

    def __init__(self, step_size: int, gamma: float = 0.1) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super(StepScheduler, self).__init__()

    def step(self, value: float, epoch: int) -> float:
        """Take a scheduler step and return the new value.

        Args:
            value (float): Learning parameter to decay.
            epoch (int): Current epoch.

        Returns:
            float: Decayed learning parameter.
        """
        if (epoch == 0) or (epoch % self.step_size != 0):
            return value

        return value * self.gamma
