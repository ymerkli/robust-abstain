import torch
import torch.nn as nn

import numpy as np
from typing import Union, Tuple, Dict, List

from robustabstain.abstain.confidence_threshold import confidence_threshold
from robustabstain.abstain.robustness_indicator import robustness_indicator


ABSTAIN_METHODS = ['rob', 'conf']


def abstain_selector(
        args: object, model: nn.Module, model_dir: str, model_name: str,
        device: str, test_loader: torch.utils.data.DataLoader,
        method: str, eps: str = '', is_rob: np.ndarray = None, conf_threshold: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Selector wrapper function that selects which samples to abstain on in a compositional
    architecture. Returns binary arrays indicating which natural/ adversarial samples are sent
    to branch model (=1) and which samples are abstained on and sent to the trunk model (=0).

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): Trunk model to evaluate robustness indicator.
        model_dir (str): Directory in which the model is stored (and associated eval logs).
        model_name (str): Name of the model to evaluate.
        device (str): device
        test_loader (torch.utils.data.DataLoader): Dataloader with test data.
        method (str): Selection method. Must be in ['rob', 'conf'].
        eps (str, optional): Stringified perturbation region epsilon. Must be specified
            when method='rob'.  Defaults to ''.
        is_rob (np.ndarray, optional): Robustness indicator. If given,
            this indicator is used in 'rob' selector. Defaults to None.
        conf_threshold (float, optional): Confidence threshold to compare
            prediction confideces against. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: natural samples selector, adversarial samples selector
    """
    selector, adv_selector = None, None
    if method == 'rob':
        assert eps is not None, 'Error: specify eps for robustness selector.'
        selector = robustness_indicator(args, model, device, test_loader, eps, is_rob)
        adv_selector = selector
    elif method == 'conf':
        assert conf_threshold is not None, 'Error: specify conf_threshold for confidence selector.'
        selector, adv_selector, _ = confidence_threshold(
            args, model, model_dir, model_name, device, test_loader, eps, conf_threshold
        )
    else:
        raise ValueError(f'Error: invalid selector method {method}.')

    return selector, adv_selector