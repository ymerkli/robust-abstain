import torch
import torch.nn as nn
import torch.nn.functional as F


REVADV_LOSSES = ['mrevadv', 'grevadv', 'mrevadv_conf']


def revadv_loss(
        nat_logits: torch.tensor, adv_logits: torch.tensor, targets: torch.tensor,
        soft: bool = True, reduction: str = 'mean'
    ) -> torch.tensor:
    """Find the top predicted label of adversarial samples that is NOT equal to the base predcition,
    return CE loss of the adv output toward that label.

    Args:
        nat_logits (torch.tensor): Model logits from natural samples.
        adv_logits (torch.tensor): Model logits from adversarial samples.
        targets (torch.tensor): Ground truth labels.
        soft (bool, optional): If True, soft multiplier is used. Defaults to True.
        reduction (str, optional): Loss reduction method. Defaults to 'mean'.

    Returns:
        torch.tensor: revadv loss
    """
    nat_pred = nat_logits.argmax(1)
    nat_probs = F.softmax(nat_logits, dim=1)
    is_acc_batch = nat_pred.eq(targets).int()
    p_inacc = 1 - torch.gather(nat_probs, dim=1, index=targets.unsqueeze(1)).squeeze()
    inacc_multiplier = p_inacc if soft else 1 - is_acc_batch

    # find top adv labels that are NOT equal to the base prediction
    top2_adv_labels = torch.argsort(adv_logits, dim=1, descending=True)[:, :2] # top2 predicted labels
    adv_targets = torch.where(top2_adv_labels[:, 0] == nat_pred, top2_adv_labels[:, 1], top2_adv_labels[:, 0])

    revadv_loss = inacc_multiplier * F.cross_entropy(adv_logits, adv_targets, reduction='none')
    if reduction == 'mean':
        revadv_loss = revadv_loss.mean()
    elif reduction == 'sum':
        revadv_loss = revadv_loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Error: invalid reduction method {reduction}')

    return revadv_loss


def revadv_gambler_loss(
        nat_logits: torch.tensor, adv_logits: torch.tensor, targets: torch.tensor,
        conf: float = 1.0, reduction: str = 'mean'
    ) -> torch.tensor:
    """Gambler's loss combined with reversing adversarial robustness. Abastention is
    modeled as non-robustness towards the top adversarial target.

    Args:
        nat_logits (torch.tensor): Model logits from natural samples
        adv_logits (torch.tensor): Model logits from adversarial samples
        targets (torch.tensor): Ground truth labels
        conf (float, optional): Confidence, larger conf encourages prediction
            over non-robustness. Defaults to 1.0.
        reduction (str, optional): Loss reduction method. Defaults to 'mean'.

    Returns:
        torch.tensor: revadv gambler's loss
    """
    nat_pred = nat_logits.argmax(1)

    # find top adv labels that are NOT equal to the base prediction
    top2_adv_labels = torch.argsort(adv_logits, dim=1, descending=True)[:, :2] # top2 predicted labels
    adv_targets = torch.where(top2_adv_labels[:, 0] == nat_pred, top2_adv_labels[:, 1], top2_adv_labels[:, 0])
    abstain_logits = torch.gather(adv_logits, dim=1, index=adv_targets.unsqueeze(1))

    # combine nat_logits and abstain_logits and normalize to new distribution
    logits = torch.cat([nat_logits, abstain_logits], dim=1)
    probs = F.softmax(logits, dim=1)
    p_abstain = probs[:, -1] # last col contains the abstain probabilities
    p_y = torch.gather(probs, dim=1, index=targets.unsqueeze(1)).squeeze()

    # revadv gambler loss
    gambler_loss = -1 * torch.log(conf * p_y + p_abstain)
    if reduction == 'mean':
        gambler_loss = gambler_loss.mean()
    elif reduction == 'sum':
        gambler_loss = gambler_loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Error: invalid reduction method {reduction}')

    return gambler_loss


def revadv_conf_loss(
        nat_logits: torch.tensor, adv_logits: torch.tensor,
        targets: torch.tensor, soft: bool = True, reduction: str = 'mean'
    ) -> torch.tensor:
    """Revadv loss for confidence thresholding abstain, intending to maximize
    the commit precision by training a model to have low confidence on samples
    that are non-robust or inaccurate.

    Args:
        nat_logits (torch.tensor): Model logits from natural samples.
        adv_logits (torch.tensor): Model logits from adversarial samples.
        targets (torch.tensor): Ground truth labels.
        soft (bool, optional): If True, soft multiplier is used. Defaults to True.
        reduction (str, optional): Loss reduction method. Defaults to 'mean'.

    Returns:
        torch.tensor: revadv_conf loss
    """
    num_classes = nat_logits.size(1)
    uniform_distr = 1/num_classes * torch.ones_like(nat_logits)
    criterion_kl = nn.KLDivLoss(reduction='none')

    # predictions
    nat_pred = nat_logits.argmax(1)
    adv_pred = adv_logits.argmax(1)
    is_acc_batch = nat_pred.eq(targets).int()
    is_rob_batch = adv_pred.eq(nat_pred).int()

    # find top adv labels that are NOT equal to the base prediction
    top2_adv_labels = torch.argsort(adv_logits, dim=1, descending=True)[:, :2] # top2 predicted labels
    adv_targets = torch.where(top2_adv_labels[:, 0] == nat_pred, top2_adv_labels[:, 1], top2_adv_labels[:, 0])
    top_adv_logits = torch.gather(adv_logits, dim=1, index=adv_targets.unsqueeze(1))

    # combine nat_logits and top_adv_logits and normalize to new distribution
    logits = torch.cat([nat_logits, top_adv_logits], dim=1)
    probs = F.softmax(logits, dim=1)
    p_adv = probs[:, -1] # last col contains the adversarial probabilities
    p_y = torch.gather(probs, dim=1, index=targets.unsqueeze(1)).squeeze()

    # multiplier penalizing samples that are inaccurate OR non-robust
    multiplier = ((1 - p_y) + p_adv) if soft else (1-is_acc_batch)|(1-is_rob_batch)

    # revadv_conf loss
    loss = multiplier * criterion_kl(F.log_softmax(nat_logits, dim=1), uniform_distr).mean(1)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f'Error: invalid reduction method {reduction}.')

    return loss