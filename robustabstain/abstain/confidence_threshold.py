import torch
import torch.nn as nn

import logging
import numpy as np
from typing import Union, Tuple, Dict, List

from robustabstain.eval.adv import empirical_robustness_eval, get_acc_rob_indicator
from robustabstain.eval.nat import natural_eval


def get_confidences(
        args: object, model: nn.Module, model_dir: str, model_name: str,
        device: str, test_loader: torch.utils.data.DataLoader, test_eps: str,
        adv_attack: str = 'apgdconf'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get confidences of the prediction of natural and adversarial samples.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): Model to evaluate.
        model_dir (str): Directory in which the model is stored (and associated eval logs).
        model_name (str): Name of the model to evaluate.
        device (str): device.
        test_loader (torch.utils.data.DataLoader): Dataloader with test data.
        test_eps (str): Size of adversarial perturbation region (stringified).
        adv_attack (str, optional): Type of adversarial attack to use. Defaults to 'apgdconf'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: nat confidences, adv confidences,
            sample indices, indicator array for accuracy, indicator array for robustness
    """
    # natural eval to get confidence on natural samples
    _, _, nat_conf, _, is_acc, nat_indices = natural_eval(args, model, device, test_loader)

    # robustness eval to get confidences on adversarial samples
    _, _, _, _, is_rob, _, _, adv_conf, adv_indices = get_acc_rob_indicator(
        args, model, model_dir, model_name, device, test_loader, args.eval_set,
        args.adv_norm, test_eps, adv_attack, use_existing=True, write_log=True
    )

    assert (nat_indices == adv_indices).all(), 'Error: sample indices orders do not match.'

    return nat_conf, adv_conf, nat_indices, is_acc, is_rob


def confidence_threshold(
        args: object, model: nn.Module, model_dir: str, model_name: str,
        device: str, test_loader: torch.utils.data.DataLoader,
        test_eps: str, conf_threshold: float, adv_attack: str = 'apgdconf'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check for each sample whether the natural and adversarial prediction confidence
    exceed a given confidence threshold.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): Model to evaluate.
        model_dir (str): Directory in which the model is stored (and associated eval logs).
        model_name (str): Name of the model to evaluate.
        device (str): device.
        test_loader (torch.utils.data.DataLoader): Dataloader with test data.
        test_eps (str): Size of adversarial perturbation region (stringified).
        conf_threshold (float): Confidence threshold to compare prediction confideces against.
        adv_attack (str, optional): Type of adversarial attack to use. Defaults to 'pgdconf'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Binary array indicating which natural samples are confident,
            binary array indicating which adversarial samples are confident, sample indices
    """
    logging.info(f'Evaluating abstain indicators by confidence thresholding with threshold {conf_threshold}.')

    nat_conf, adv_conf, indices, _, _ = get_confidences(
        args, model, model_dir, model_name, device, test_loader, test_eps, adv_attack
    )
    nat_is_confident = (nat_conf >= conf_threshold).astype(np.int64)
    adv_is_confident = (adv_conf >= conf_threshold).astype(np.int64)

    return nat_is_confident, adv_is_confident, indices