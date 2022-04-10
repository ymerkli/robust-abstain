import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from typing import Tuple

from robustabstain.utils.metrics import accuracy, adv_accuracy


def run_trades_loss(
        model: nn.Module, x_natural: torch.tensor, y: torch.tensor, step_size: float = 0.003,
        epsilon: float = 0.031, perturb_steps: int = 10, beta: int = 1.0, distance: str = 'Linf',
        adversarial: bool = True, mode: str = 'train', weight: torch.tensor = None
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, np.ndarray, np.ndarray]:
    """TRADES loss runner function (following Zhang et al.) that calculates both
    adversarial samples and the final TRADES loss.
    Modified from:
        - TRADES: https://github.com/yaodongyu/TRADES/blob/master/trades.py

    Args:
        model (nn.Module): Model to calculate loss for.
        x_natural (torch.tensor): Natural (unperturbed) inputs.
        y (torch.tensor): Ground truth labels.
        optimizer (optim.Optimizer): Optimizer.
        step_size (float, optional): Attack step size. Defaults to 0.003.
        epsilon (float, optional): Perturbation region size. Defaults to 0.031.
        perturb_steps (int, optional): Number of attack steps. Defaults to 10.
        beta (int, optional): TRADES beta. Defaults to 1.0.
        distance (str, optional): Adversarial norm. Defaults to 'Linf'.
        adversarial (bool, optional): If set, perturbations are adversarial. Defaults to True.
        mode (str, optional): 'train' or 'test' mode. Defaults to 'train'.
        weight (torch.tensor, optional): Per label weights for weighted loss. Defaults to None.

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, np.ndarray, np.ndarray]:
            TRADES loss, nat loss, adv loss, nat accuracy, adv accuracy, is_acc indicator, is_rob indicator
    """
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if adversarial:
        if distance == 'Linf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))

                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif distance == 'L2':
            delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                               F.softmax(model(x_natural), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        if distance == 'L2':
            x_adv = x_adv + epsilon * torch.randn_like(x_adv)
        else:
            raise ValueError(f'Error: no support for distance {distance} in stability training.')

    if mode == 'train':
        model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    nat_logits = model(x_natural)
    adv_logits = model(x_adv)

    # get robustness indicator
    nat_pred = nat_logits.argmax(1)
    adv_pred = adv_logits.argmax(1)
    is_acc = nat_pred.eq(y).int().cpu().numpy()
    is_rob = nat_pred.eq(adv_pred).int().cpu().numpy()

    loss_trades, loss_nat, loss_rob, nat_acc, adv_acc = trades_loss(
        nat_logits, adv_logits, y, beta, weight, reduction='mean'
    )

    return loss_trades, loss_nat, loss_rob, nat_acc, adv_acc, is_acc, is_rob


def trades_loss(
        nat_logits: torch.tensor, adv_logits: torch.tensor, y: torch.tensor,
        beta: float = 1.0, weight: torch.tensor = None, reduction: str = 'mean'
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """TRADES loss following Zhang et al. Modified from https://github.com/yaodongyu/TRADES.

    Args:
        nat_logits (torch.tensor): Model logits from natural samples.
        adv_logits (torch.tensor): Model logits from adversarial samples.
        y (torch.tensor): Labels.
        beta (float, optional): TRADES beta factor. Defaults to 1.0.
        weight (torch.tensor, optional): Per label weights for weighted loss. Defaults to None.
        reduction (str, optional): Loss reduction method. Defaults to 'mean'.

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
            TRADES loss, nat loss, adv loss, nat accuracy, adv accuracy
    """
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    batch_size = nat_logits.size(0)

    # calculate loss
    loss_natural = F.cross_entropy(nat_logits, y, reduction='none')
    loss_robust = criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(nat_logits, dim=1))

    # apply weight
    if weight is not None:
        for label_id, label_weight in enumerate(weight):
            loss_natural[y == label_id] = label_weight * loss_natural[y == label_id]
            loss_robust[y == label_id, :] = label_weight * loss_robust[y == label_id, :]

    if reduction == 'none':
        loss_robust = loss_robust.mean(dim=1) # reduce loss_robust to 1D tensor
    elif reduction == 'sum':
        loss_natural = loss_natural.sum()
        loss_robust = loss_robust.sum()
    elif reduction == 'mean' or reduction == 'batchmean':
        loss_natural = loss_natural.mean()
        loss_robust = (1.0 / batch_size) * loss_robust.sum()

    # combine losses to TRADES loss
    loss = loss_natural + beta * loss_robust

    # calculate accuracies
    nat_acc, _ = accuracy(nat_logits, y, topk=(1,))
    adv_acc, _ = adv_accuracy(adv_logits, nat_logits, y)

    return loss, loss_natural, loss_robust, nat_acc[0], adv_acc
