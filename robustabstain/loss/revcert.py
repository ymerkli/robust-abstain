import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from typing import Tuple


REVCERT_LOSSES = ['revcertrad', 'revcertnoise', 'smoothmrevadv', 'smoothgrevadv']


def revcert_radius_loss(
        model: torch.nn.Module, inputs: torch.tensor, targets: torch.tensor,
        device: str, num_classes: int, num_noise_vec: int, noise_sd: float,
        revadv_beta: float, temp: float = 4.0, clamp_x: bool = True,
        reduction: str = 'mean'
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Abstain loss for certified robustness. Robustness on inaccurate samples
    is removed by directly minimizing the certified radius. The formulation of
    the differentiable certified radius follows Zhai et. al. (MACER).

    Args:
        model (torch.nn.Module): Model to calculate loss for.
        inputs (torch.tensor): Inputs to calculate loss for.
        targets (torch.tensor): Labels of the inputs.
        device (str): device.
        num_classes (int): Number of classes in the dataset.
        num_noise_vec (int): Number of noisy samples to sample per input.
        noise_sd (float): Standard deviation of the noise when sampling.
        revadv_beta (float): Hyperpararmeter controling strength of robust accuracy regularization.
        temp (float, optional): Temperature parameter in the cert radius. Defaults to 4.0.
        clamp_x (bool, optional): If set, noisy samples are clamped. Defaults to True.
        reduction (str, optional): Loss reduction method. Defaults to 'mean'.

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]:
            Combined loss, classification loss, certified radius loss
    """
    normal = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

    input_size = inputs.size(0)
    new_shape = [input_size * num_noise_vec]
    new_shape.extend(inputs[0].shape)
    inputs_rep = inputs.repeat((1, num_noise_vec, 1, 1)).view(new_shape)
    targets_rep = targets.unsqueeze(1).repeat(1, num_noise_vec).reshape(-1, 1).squeeze()
    noise = torch.randn_like(inputs_rep, device=device) * noise_sd
    noisy_inputs = inputs_rep + noise

    outputs = model(noisy_inputs)
    outputs = outputs.reshape((input_size, num_noise_vec, num_classes))
    outputs_softmax = F.softmax(outputs, dim=2).mean(1)
    p_inacc = 1 - torch.gather(outputs_softmax, dim=1, index=targets.unsqueeze(1)).squeeze()

    # classification loss
    loss_classification = F.cross_entropy(outputs.mean(1), targets, reduction='none')

    # cert radius
    beta_outputs = outputs * temp  # only apply temparature to the cert radius loss
    beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
    top2 = torch.topk(beta_outputs_softmax, 2)
    top2_probs, _ = top2
    out0, out1 = top2_probs[:, 0], top2_probs[:, 1]
    out0 -= 1e-6 # avoid nan (normal.icdf(1.0) = inf)
    out1 += 1e-6 # avoid nan (normal.icdf(0.0) = -inf)
    cert_radius = noise_sd/2 * (normal.icdf(out0) - normal.icdf(out1))

    # Final objective
    loss = p_inacc * cert_radius + revadv_beta * loss_classification

    if reduction == 'mean':
        loss = loss.mean()
        loss_classification = loss_classification.mean()
        cert_radius = cert_radius.mean()
    elif reduction == 'sum':
        loss = loss.sum()
        loss_classification = loss_classification.sum()
        cert_radius = cert_radius.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Error: invalid reduction method {reduction}')

    return loss, loss_classification, cert_radius


def revcert_noise_loss(
        model: torch.nn.Module, inputs: torch.tensor, targets: torch.tensor, device: str,
        num_classes: int, num_noise_vec: int, noise_sd: float, revadv_beta: float,
        topk_noise_vec: int = None, version: str = 'CE', reduction: str = 'mean'
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Abstain loss for certified robustness. Robustness on inaccurate samples
    is removed by training towards the most likely non-robust label on the
    topk_noise_vec noisy samples.

    Args:
        model (torch.nn.Module): Model to calculate loss for.
        inputs (torch.tensor): Inputs to calculate loss for.
        targets (torch.tensor): Labels of the inputs.
        device (str): device.
        num_classes (int): Number of classes in the dataset.
        num_noise_vec (int): Number of noisy samples to sample per input.
        noise_sd (float): Standard deviation of the noise when sampling.
        revadv_beta (float): Hyperpararmeter controling strength of robust accuracy regularization.
        topk_noise_vec (int, optional): Number of worst-case noise_vec to train towards
            most likely non-robust label. Defaults to None.
        version (str, optional): Version of revcert loss to use. Defaults to 'CE'.
        reduction (str, optional): Loss reduction method. Defaults to 'mean'.

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]:
            Combined loss, classification loss, revcert loss
    """

    if topk_noise_vec is not None:
        assert 0 < topk_noise_vec and topk_noise_vec <= num_noise_vec, \
            f'Error: topk_noise_samples (={topk_noise_vec} must be <= num_noise_vec (={num_noise_vec}))'
    else:
        topk_noise_vec = num_noise_vec

    input_size = inputs.size(0)
    new_shape = [input_size * num_noise_vec]
    new_shape.extend(inputs[0].shape)
    inputs_rep = inputs.repeat((1, num_noise_vec, 1, 1)).view(new_shape)
    targets_rep = targets.unsqueeze(1).repeat(1, topk_noise_vec).reshape(-1, 1).squeeze()
    noise = torch.randn_like(inputs_rep, device=device) * noise_sd
    noisy_inputs = inputs_rep + noise

    outputs = model(noisy_inputs)
    nat_pred_mean = outputs.reshape((input_size, num_noise_vec, num_classes)).mean(1).argmax(1)
    nat_pred_mean = nat_pred_mean.unsqueeze(1).repeat(1, num_noise_vec).reshape(-1, 1).squeeze()

    top2_labels = torch.argsort(outputs, dim=1, descending=True)[:, :2] # top2 predicted labels
    rev_labels = torch.where(top2_labels[:, 0] == nat_pred_mean, top2_labels[:, 1], top2_labels[:, 0])

    # select topk most likely adversarial noisy samples among all num_noise_vec noisy samples
    rev_logits = torch.gather(outputs, dim=1, index=rev_labels.unsqueeze(1)).view(input_size, num_noise_vec) # logits of the rev_labels
    topk_idx = torch.topk(rev_logits, k=topk_noise_vec, dim=1)[1]
    rev_labels = torch.gather(rev_labels.reshape((input_size, num_noise_vec)), dim=1, index=topk_idx)
    rev_labels = rev_labels.view(-1)

    # gather the outputs of the selected samples
    topk_idx = topk_idx.view(-1, 1).repeat(1, num_classes).view(input_size, topk_noise_vec, num_classes)
    outputs = torch.gather(outputs.reshape((input_size, num_noise_vec, num_classes)), dim=1, index=topk_idx)
    outputs = outputs.view(topk_noise_vec * input_size, num_classes)
    outputs_logsoftmax = F.log_softmax(outputs, dim=1)

    # classification loss (train for robust accurate prediction)
    loss_classification = F.cross_entropy(outputs, targets_rep, reduction='none')

    # revcert loss (train for noisy space to get low/0 certifiable radius)
    if version.lower() == 'kl':
        criterion_kl = nn.KLDivLoss(reduction='none')
        uniform_distr = 1/num_classes * torch.ones_like(outputs_logsoftmax)
        loss_revcert = criterion_kl(outputs_logsoftmax, uniform_distr).mean(1)
    else:
        loss_revcert = F.cross_entropy(outputs, rev_labels, reduction='none')

    outputs_softmax = F.softmax(outputs, dim=1)
    p_inacc = 1 - torch.gather(outputs_softmax, dim=1, index=targets_rep.unsqueeze(1)).squeeze()

    # Final objective
    loss = p_inacc * loss_revcert + revadv_beta * loss_classification

    if reduction == 'mean':
        loss = loss.mean()
        loss_classification = loss_classification.mean()
        loss_revcert = loss_revcert.mean()
    elif reduction == 'sum':
        loss = loss.sum()
        loss_classification = loss_classification.sum()
        loss_revcert = loss_revcert.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Error: invalid reduction method {reduction}')

    return loss, loss_classification, loss_revcert

