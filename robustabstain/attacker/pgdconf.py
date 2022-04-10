import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import foolbox as fb
from typing import Optional

from robustabstain.attacker.base import Attacker



class PGDCONF_Linf(Attacker):
    """ PGD CONF Linf attack. Maximizes the confidence of the resulting adversarial sample by
    running a targeted attack towards the top1 adversarial label.

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

        super(PGDCONF_Linf, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            eps: float, conf_threshold: float = 0.0, restarts: int = 1,
            rel_step_size: float = 0.01 / 0.3, abs_step_size: Optional[float] = None
        ) -> torch.Tensor:

        batch_size = inputs.size(0)
        adv_inputs = inputs.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(restarts):
            adv_found_run, adv_inputs_run = self._attack(
                model, inputs=inputs[~adv_found], labels=labels[~adv_found], eps=eps,
                conf_threshold=conf_threshold, rel_step_size=rel_step_size, abs_step_size=abs_step_size
            )
            adv_inputs[~adv_found] = adv_inputs_run
            adv_found[~adv_found] = adv_found_run

            if adv_found.all():
                break

        return adv_inputs

    def _attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            eps: float, conf_threshold: float = 0.0, rel_step_size: float = 0.01 / 0.3,
            abs_step_size: Optional[float] = None
        ) -> torch.Tensor:

        batch_size = inputs.size(0)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        clamp = lambda tensor: tensor.data.clamp_(min=-eps, max=eps).add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        if abs_step_size is not None:
            step_size = abs_step_size
        else:
            step_size = eps * rel_step_size

        # attack is targeted to the worst case adversarial label
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
                probs = F.softmax(logits, dim=1)

                # find top adv labels that are NOT equal to the base prediction
                top2_adv_labels = torch.argsort(logits, dim=1, descending=True)[:, :2] # top2 predicted labels
                adv_targets = torch.where(top2_adv_labels[:, 0] == labels, top2_adv_labels[:, 1], top2_adv_labels[:, 0])

                loss = F.cross_entropy(logits, adv_targets, reduction='none')
                delta_grad = grad(loss.sum(), delta, only_inputs=True)[0]

            adv_prob, adv_pred = probs.max(1)
            is_adv = (adv_pred != labels) & (adv_prob >= conf_threshold)
            delta_adv = torch.where(batch_view(is_adv), delta.detach(), delta_adv)
            adv_found.logical_or_(is_adv)

            delta.data.add_(delta_grad.sign(), alpha=step_size)
            clamp(delta)

        # use last delta for non-adversarial example to return best-effort adversarial samples
        delta_adv = torch.where(batch_view(~adv_found), delta.detach(), delta_adv)

        return adv_found, inputs + delta_adv


class PGDCONF_L2(Attacker):
    """ PGD CONF L2 attack. Maximizes the confidence of the resulting adversarial sample by
    running a targeted attack towards the top1 adversarial label.

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

        super(PGDCONF_L2, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.device = device

    def attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            eps: float, conf_threshold: float = 0.0, restarts: int = 1,
            rel_step_size: float = 0.01 / 0.3, abs_step_size: Optional[float] = None
        ) -> torch.Tensor:

        batch_size = inputs.size(0)
        adv_inputs = inputs.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(restarts):
            adv_found_run, adv_inputs_run = self._attack(
                model, inputs=inputs[~adv_found], labels=labels[~adv_found], eps=eps,
                conf_threshold=conf_threshold, rel_step_size=rel_step_size, abs_step_size=abs_step_size
            )
            adv_inputs[~adv_found] = adv_inputs_run
            adv_found[~adv_found] = adv_found_run

            if adv_found.all():
                break

        return adv_inputs

    def _attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            eps: float, conf_threshold: float = 0.0, rel_step_size: float = 0.01 / 0.3,
            abs_step_size: Optional[float] = None
        ) -> torch.Tensor:

        batch_size = inputs.size(0)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))

        if abs_step_size is not None:
            step_size = abs_step_size
        else:
            step_size = eps * rel_step_size

        # attack is targeted to the worst case adversarial label
        step_size *= -1

        delta = torch.zeros_like(inputs, requires_grad=True)
        delta_adv = torch.zeros_like(inputs)
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if self.random_start:
            delta.data.normal_()
            delta_flat = delta.view(batch_size, -1)
            n = batch_view(delta_flat.norm(p=2, dim=1))
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= eps * r / n
        else:
            delta.data.zero_()

        for i in range(self.steps):
            with torch.enable_grad():
                logits = model(inputs + delta)
                probs = F.softmax(logits, dim=1)

                # find top adv labels that are NOT equal to the base prediction
                top2_adv_labels = torch.argsort(logits, dim=1, descending=True)[:, :2] # top2 predicted labels
                adv_targets = torch.where(top2_adv_labels[:, 0] == labels, top2_adv_labels[:, 1], top2_adv_labels[:, 0])

                loss = F.cross_entropy(logits, adv_targets, reduction='none')
                delta_grad = grad(loss.sum(), delta, only_inputs=True)[0]
                delta_grad_norms = torch.norm(delta_grad.view(batch_size, -1), p=2, dim=1) + 1e-10
                delta_grad = delta_grad / batch_view(delta_grad_norms)

            adv_prob, adv_pred = probs.max(1)
            is_adv = (adv_pred != labels) & (adv_prob >= conf_threshold)
            delta_adv = torch.where(batch_view(is_adv), delta.detach(), delta_adv)
            adv_found.logical_or_(is_adv)

            delta.data.add_(delta_grad, alpha=step_size)

            # clamp delta
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, *[1] * (inputs.ndim - 1))

        # use last delta for non-adversarial example to return best-effort adversarial samples
        delta_adv = torch.where(batch_view(~adv_found), delta.detach(), delta_adv)

        return adv_found, inputs + delta_adv

