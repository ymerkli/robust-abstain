import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import foolbox as fb
from typing import Optional

from robustabstain.attacker.base import Attacker


class PGD_Linf(Attacker):
    """ PGD Linf attack (Source: https://github.com/jeromerony/adversarial-library)

    Args:
        steps (int): Number of steps for the optimization.
        random_start (bool, optional): Whether to start with a random delta. Defaults to True.
        device (torch.device, optional): Device on which to perform the attack.
    """

    def __init__(
            self,
            steps: int,
            random_start: bool = True,
            device: torch.device = torch.device('cpu')
        ) -> None:
        super(PGD_Linf, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, eps: float, targeted: bool = False,
            restarts: int = 1, rel_step_size: float = 0.01 / 0.3, abs_step_size: Optional[float] = None
        ) -> torch.Tensor:

        batch_size = inputs.size(0)
        adv_inputs = inputs.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(restarts):
            adv_found_run, adv_inputs_run = self._attack(
                model, inputs=inputs[~adv_found], labels=labels[~adv_found], eps=eps, targeted=targeted,
                rel_step_size=rel_step_size, abs_step_size=abs_step_size
            )
            adv_inputs[~adv_found] = adv_inputs_run
            adv_found[~adv_found] = adv_found_run

            if adv_found.all():
                break

        return adv_inputs

    def _attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, eps: float, targeted: bool = False,
            rel_step_size: float = 0.01 / 0.3, abs_step_size: Optional[float] = None
        ) -> torch.Tensor:

        batch_size = inputs.size(0)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        clamp = lambda tensor: tensor.data.clamp_(min=-eps, max=eps).add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        if abs_step_size is not None:
            step_size = abs_step_size
        else:
            step_size = eps * rel_step_size

        if targeted:
            step_size *= -1

        delta = torch.zeros_like(inputs, requires_grad=True)
        delta_adv = torch.zeros_like(inputs)
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if self.random_start:
            delta.data.uniform_(-eps, eps)
            clamp(delta)
        else:
            delta.data.zero_()

        for i in range(self.steps):
            with torch.enable_grad():
                logits = model(inputs + delta)
                loss = F.cross_entropy(logits, labels, reduction='none')
                delta_grad = grad(loss.sum(), delta, only_inputs=True)[0]

            is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
            delta_adv = torch.where(batch_view(is_adv), delta.detach(), delta_adv)
            adv_found.logical_or_(is_adv)

            delta.data.add_(delta_grad.sign(), alpha=step_size)
            clamp(delta)

        return adv_found, inputs + delta_adv


class PGD_FB_Linf(Attacker):
    """Foolbox PGD Linf attack

    Args:
        steps (int): Number of steps for the optimization.
        random_start (bool, optional): Whether to start with a random delta. Defaults to True.
        device (torch.device, optional): Device on which to perform the attack.
    """
    def __init__(
            self,
            steps: int,
            random_start: bool = True,
            device: torch.device = torch.device('cpu')
        ) -> None:

        super(PGD_FB_Linf, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, eps: float, targeted: bool = False,
            restarts: int = 1, rel_step_size: float = 0.01 / 0.3, abs_step_size: Optional[float] = None
        ) -> torch.Tensor:

        fbmodel = fb.PyTorchModel(model, bounds=(0,1), device=self.device, preprocessing=None)
        step_size = rel_step_size
        if abs_step_size is not None:
            step_size = abs_step_size / eps

        attack = fb.attacks.LinfPGD(
            rel_stepsize=step_size, steps=self.steps, random_start=self.random_start
        )
        _, clipped_advs, _ = attack(fbmodel, inputs, labels, epsilons=eps)

        return clipped_advs


class PGD_FB_L2(Attacker):
    """Foolbox PGD L2 attack

    Args:
        steps (int): Number of steps for the optimization.
        random_start (bool, optionae): Whether to start with a random delta. Defaults to True.
        device (torch.device, optional): Device on which to perform the attack.
    """
    def __init__(
            self,
            steps: int,
            random_start: bool = True,
            device: torch.device = torch.device('cpu')
        ) -> None:

        super(PGD_FB_L2, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, eps: float, targeted: bool = False,
            restarts: int = 1, rel_step_size: float = 0.01 / 0.3, abs_step_size: Optional[float] = None
        ) -> torch.Tensor:

        fbmodel = fb.PyTorchModel(model, bounds=(0,1), device=self.device, preprocessing=None)
        step_size = rel_step_size
        if abs_step_size is not None:
            step_size = abs_step_size / eps

        attack = fb.attacks.L2PGD(
            rel_stepsize=step_size, steps=self.steps, random_start=self.random_start
        )
        _, clipped_advs, _ = attack(fbmodel, inputs, labels, epsilons=eps)

        return clipped_advs

