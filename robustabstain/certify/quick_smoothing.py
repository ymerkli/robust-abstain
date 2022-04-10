"""Source: https://github.com/yaircarmon/semisup-adv
"""

import torch
import torch.nn.functional as F
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


def quick_smoothing(
        model, x, y, sigma=1.0, eps=1.0,
        num_smooth=100, batch_size=1000,
        softmax_temperature=100.0,
        detailed_output=False
    ):
    """Quick and dirty randomized smoothing 'certification', without proper
     confidence bounds. We use it only to monitor training.
    """
    x_noise = x.view(1, *x.shape) + sigma * torch.randn(num_smooth, *x.shape).cuda()
    x_noise = x_noise.view(-1, *x.shape[1:])

    # by setting a high softmax temperature, we are effectively using the
    # randomized smoothing approach as originally defined
    # it will be interesting to see if lower temperatures help
    preds = torch.cat([
        F.softmax(softmax_temperature * model(batch), dim=-1)
        for batch in torch.split(x_noise, batch_size)
    ])
    preds = preds.view(num_smooth, x.shape[0], -1).mean(dim=0)
    p_max, y_pred = preds.max(dim=-1)

    correct = (y_pred == y).cpu().numpy().astype('int64')
    radii = (sigma + 1e-16) * norm.ppf(p_max.detach().cpu().numpy())

    nat_acc = 100. * correct.sum() / len(correct)
    rob_acc = 100. * (correct * (radii >= eps)).sum() / len(correct)
    err = (1 - correct).sum()
    robust_err = (1 - correct * (radii >= eps)).sum()

    if not detailed_output:
        return nat_acc, rob_acc
    else:
        return correct, radii
