import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from robustabstain.utils.metrics import accuracy, adv_accuracy


def noise_loss(
        model: nn.Module, inputs: torch.tensor, targets: torch.tensor,
        noise_sd: float = 0.25, criterion: nn.modules.loss._Loss = None,
        clamp_x: bool = True, reduction: str ='mean'
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Augmenting the input with random noise as in Cohen et al.

    Args:
        model (nn.Module): Model to calculate loss for.
        inputs (torch.tensor): Input data.
        targets (torch.tensor): Labels.
        noise_sd (float, optional): Std deviation of noise. Defaults to 0.25.
        criterion (nn.modules.loss._Loss, optional): Criterion. Defaults to None.
        clamp_x (bool, optional): If set, perturbed inputs are clamped. Defaults to True.
        reduction (str, optional): Loss reduction method. Defaults to 'mean'.

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]: loss, nat_acc, adv_acc
    """
    inputs_noise = inputs + noise_sd * torch.randn_like(inputs)
    if clamp_x:
        inputs_noise = inputs_noise.clamp(0.0, 1.0)
    logits_noise = model(inputs_noise)
    logits_nat = model(inputs)
    loss = None
    if criterion is not None:
        loss = criterion(logits_noise, targets)
    else:
        loss = F.cross_entropy(logits_noise, targets, ignore_index=-1, reduction=reduction)

    nat_acc, _ = accuracy(logits_nat, targets, topk=(1,5))
    adv_acc, _ = adv_accuracy(logits_noise, logits_nat, targets)

    return loss, nat_acc[0], adv_acc
