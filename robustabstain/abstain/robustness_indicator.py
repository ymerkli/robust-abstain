import torch
import torch.nn as nn

import numpy as np

from robustabstain.utils.loaders import get_rel_sample_indices
from robustabstain.eval.adv import empirical_robustness_eval


def robustness_indicator(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        eps: str, is_rob: np.ndarray = None
    ) -> np.ndarray:
    """Get a robustness indicator for abstaining.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): Trunk model to evaluate robustness indicator.
        device (str): device
        test_loader (torch.utils.data.DataLoader): Dataloader with test data.
        eps (str): Stringified perturbation region epsilon.
        is_rob (np.ndarray, optional): Robustness indicator. If given,
            this indicator is used. Defaults to None.

    Returns:
        np.ndarray: Binary indicator array indicating which samples are robust.
    """
    if is_rob is None:
        _, _, _, is_rob, _ = empirical_robustness_eval(
            args, model, device, test_loader, args.adv_attack, args.adv_norm, eps
        )

    return is_rob


