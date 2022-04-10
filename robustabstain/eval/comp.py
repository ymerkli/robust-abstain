import setGPU
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from typing import Tuple, List

from robustabstain.abstain.selector import abstain_selector
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.utils.checkpointing import load_checkpoint


def get_comp_model_paths(comp_dir: str) -> Tuple[str, str]:
    """Get branch and trunk model filepaths from compositional experiment directory.

    Args:
        comp_dir (str): Compositional experiment directory.
            Assumed to have two subdirectories 'branch/' and 'trunk/'.

    Returns:
        Tuple[str, str]: branch model path, trunk model path.
    """
    # match trunk, rob model checkpoints from comp_dir
    comp_dir = Path(comp_dir)
    match = re.match(r'^r(?:a|c)((?:\d+(?:_|\/)\d+)|(?:\d+(?:\.\d+)?))', comp_dir.name)
    if not match:
        raise ValueError("Error: comp_dir needs to follow naming convention 'r(a|c){eps}__'.")
    train_eps = match.group(1)

    branch_dir = comp_dir / 'branch'
    trunk_dir = comp_dir / 'trunk'
    branch_model_path, trunk_model_path = None, None
    for file in branch_dir.iterdir():
        branch_match = re.match(rf'^.+?r(?:a|c){train_eps}\.pt$', file.name)
        if branch_match:
            branch_model_path = file
            break

    for file in trunk_dir.iterdir():
        trunk_match = re.match(rf'^.+?_std_.*?r(?:a|c){train_eps}\.pt$', file.name)
        if trunk_match:
            trunk_model_path = file
            break

    if not branch_model_path or not trunk_model_path:
        raise ValueError(f'Error: branch/trunk model not found in {comp_dir}.')

    return str(branch_model_path), str(trunk_model_path)


def get_comp_key(branch_model_path: str, trunk_model_paths: List[str]) -> str:
    """Generate a model identifier for a compositional model architecture.

    Args:
        branch_model_name (str): Path to branch model.
        trunk_model_names (List[str]): Path(s) to trunk model(s).

    Returns:
        str: Compositional model identifier.
    """
    branch_model_name = os.path.splitext(os.path.basename(branch_model_path))[0]
    comp_key = f'comp_{branch_model_name}'
    for trunk_model_path in trunk_model_paths:
        trunk_model_name = os.path.splitext(os.path.basename(trunk_model_path))[0]
        comp_key += f'__{trunk_model_name}'

    return comp_key


def apply_comp_selector(
        branch_is_acc: np.ndarray, branch_is_rob: np.ndarray,
        trunk_is_acc: np.ndarray, trunk_is_rob: np.ndarray,
        selector: np.ndarray, adv_selector: np.ndarray = None,
        trunk_is_acc_adv: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply selector to compositional architecture to get compositional indicators.
    The adversary model is chosen such that the adversary attacks the model as well as
    the selection mechanism when evaluating robustness.

    Args:
        branch_is_acc (np.ndarray): Binary array indicating where branch model is accurate.
        branch_is_rob (np.ndarray): Binary array indicating where branch model is robust.
        trunk_is_acc (np.ndarray): Binary array indicating where trunk model is accurate.
        trunk_is_rob (np.ndarray): Binary array indicating where trunk model is robust.
        selector (np.ndarray): Binary array containing selectors for nat samples.
        adv_selector (np.ndarray, optional): Binary array containing adversarial selectors.
            If no adv_selector is given, selector is used. Defaults to None.
        trunk_is_acc_adv (np.ndarray, optional): Binary array indicating where trunk model
            is accurate when the selector is attacked (meaning the trunk model itself is
            a compositional model). Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Binary indicators for compositional model
            indicating accuracy on nat evaluated selector, accuracy on adversarial
            evaluated selector, robustness on adversarial evaluated selector.
    """
    if adv_selector is None:
        adv_selector = selector
    if trunk_is_acc_adv is None:
        trunk_is_acc_adv = trunk_is_acc

    # evaluate natural selector
    comp_is_acc_nat = selector * branch_is_acc + (1 - selector) * trunk_is_acc

    # evaluate adversarial selector
    comp_is_acc_adv = adv_selector * branch_is_acc + (1 - adv_selector) * trunk_is_acc_adv
    comp_is_rob_adv = adv_selector * branch_is_rob + (1 - adv_selector) * trunk_is_rob

    assert np.all((comp_is_acc_nat == 0)|(comp_is_acc_nat==1)), 'Error: indicator array comp_is_acc_nat can only contain 0s or 1s'
    assert np.all((comp_is_acc_adv == 0)|(comp_is_acc_adv ==1)), 'Error: indicator array comp_is_acc_adv can only contain 0s or 1s'
    assert np.all((comp_is_rob_adv == 0)|(comp_is_rob_adv ==1)), 'Error: indicator array comp_is_rob_adv can only contain 0s or 1s'

    return comp_is_acc_nat, comp_is_acc_adv, comp_is_rob_adv


def get_comp_indicator(
        args: object, branch_model_path: str, trunk_model_paths: List[str],
        device: str, dataloader: torch.utils.data.DataLoader, eval_set: str,
        eps_str: str, abstain_method: str, conf_threshold: float = None,
        use_existing: bool = False
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Get is_acc, is_rob indicators for a compositional architecture.
    The trunk model itself can itself be a compositional model again: if multiple
    trunk_models are given, trunk model 0 abstain to trunk model 1, trunk model 1
    abstain to trunk model 2, ..., trunk model n-2 abstain to trunk model n-1.
    Trunk model n-1 (last trunk) does not abstain.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        branch_model_path (str): Path to base branch model
        trunk_model_paths (List[str]): Paths to trunk models.
        device (str): device
        dataloader (torch.utils.data.DataLoader): Dataloader to evaluate.
        eval_set (str): Dataset split that is evaluated ('train', 'val', 'test').
        eps_str (str): Perturbation region size.
        abstain_method (str): Abstain selection method. Must be in ['rob', 'conf'].
        conf_threshold (float): Confidence threshold to compare prediction confideces against.
            Required when abstain_method = 'conf'. Defaults to None.
        use_existing (bool, optional): If set, an existing model eval log will be used (if such a log exists).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Comp nat accuracy, comp adv accuracy, binary indicators
            for compositional model indicating accuracy on natural evaluated selector,
            accuracy on adversarial evaluated selector, robustness on adversarial evaluated selector.
    """
    # get indices order of samples in dataloader
    dataset_indices = np.arange(len(dataloader.dataset))
    if type(dataloader.dataset) == torch.utils.data.dataset.Subset:
        # Subset dataset have random index order
        dataset_indices = dataloader.dataset.indices

    # load branch model
    branch_model, _, _ = load_checkpoint(
        branch_model_path, net=None, arch=None, dataset=args.dataset, device=device,
        normalize=not args.no_normalize, optimizer=None, parallel=True
    )
    branch_model_dir = os.path.dirname(branch_model_path)
    branch_model_name = os.path.splitext(os.path.basename(branch_model_path))[0]

    # get accuracy and robustness indicators on branch model
    branch_nat_acc, branch_adv_acc, _, branch_is_acc, branch_is_rob, _, _, _, _ = get_acc_rob_indicator(
        args, branch_model, branch_model_dir, branch_model_name, device, dataloader,
        eval_set, args.adv_norm, eps_str, args.test_adv_attack,
        use_existing=True, write_log=True, write_report=True
    )

    if len(trunk_model_paths) == 0:
        # if no trunk models are present, just return branch model evaluations in single model evaluation
        return branch_nat_acc, branch_adv_acc, branch_is_acc, branch_is_acc, branch_is_rob

    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob = None, None, None
    if len(trunk_model_paths) > 1:
        """
        If multiple trunk models are present, recursively evaluate the trunk model
        in compositional architecture. The first trunk model becomes the branch model
        in the new compositional architecture.
        """
        _, _, trunk_is_acc, trunk_is_acc_adv, trunk_is_rob = get_comp_indicator(
            args, branch_model_path=trunk_model_paths[0],
            trunk_model_paths=trunk_model_paths[1:], device=device,
            dataloader=dataloader, eval_set=eval_set, eps_str=eps_str,
            abstain_method=abstain_method, use_existing=True
        )
    else:
        """
        If a single trunk model is present, just use standard accuracy/ robustness evaluation
        """
        # load trunk model
        trunk_model, _, _ = load_checkpoint(
            trunk_model_paths[0], net=None, arch=None, dataset=args.dataset, device=device,
            normalize=not args.no_normalize, optimizer=None, parallel=True
        )
        trunk_model_dir = os.path.dirname(trunk_model_paths[0])
        trunk_model_name = os.path.splitext(os.path.basename(trunk_model_paths[0]))[0]

        # get accuracy and robustness indicators on trunk model
        _, _, _, trunk_is_acc, trunk_is_rob, _, _, _, _ = get_acc_rob_indicator(
            args, trunk_model, trunk_model_dir, trunk_model_name, device, dataloader,
            eval_set, args.adv_norm, eps_str, args.test_adv_attack,
            use_existing=True, write_log=True, write_report=True
        )
        trunk_is_acc_adv = trunk_is_acc # single model doesn't use a selector

    # get abstain selector
    selector, adv_selector = abstain_selector(
        args, branch_model, branch_model_dir, branch_model_name,
        device, dataloader, abstain_method, eps_str,
        is_rob=branch_is_rob, conf_threshold=conf_threshold
    )

    # get indicators for compositional model
    comp_is_acc, comp_is_acc_adv, comp_is_rob_adv = apply_comp_selector(
        branch_is_acc, branch_is_rob, trunk_is_acc, trunk_is_rob,
        selector, adv_selector, trunk_is_acc_adv
    )

    # get accuracies for compositional model
    comp_nat_acc, comp_adv_acc = compositional_accuracy(
        branch_is_acc, branch_is_rob, trunk_is_acc, trunk_is_rob,
        selector, adv_selector, trunk_is_acc_adv
    )

    return comp_nat_acc, comp_adv_acc, comp_is_acc, comp_is_acc_adv, comp_is_rob_adv


def compositional_accuracy(
        branch_is_acc: np.ndarray, branch_is_rob: np.ndarray,
        trunk_is_acc: np.ndarray, trunk_is_rob: np.ndarray,
        selector: np.ndarray, adv_selector: np.ndarray = None,
        trunk_is_acc_adv: np.ndarray = None
    ) -> Tuple[float, float]:
    """Natural and robust accuracy of compositional architecture:
       f_comp(x) = selector(x) * f_branch(x) + (1 - selector(x)) * f_trunk(x)

    Accuracy of a compositional model architecture consisting of a branch and a trunk model.
    The selector indicates whether a sample commits to the branch model or abstains and switches
    to the trunk model. '1' indicates committing to the branch model, '0' indicates abstaining
    from the branch model and switching to the trunk model instead.

    Args:
        branch_is_acc (np.ndarray): Binary array indicating where branch model is accurate.
        branch_is_rob (np.ndarray): Binary array indicating where branch model is robust.
        trunk_is_acc (np.ndarray): Binary array indicating where trunk model is accurate.
        trunk_is_rob (np.ndarray): Binary array indicating where trunk model is robust.
        selector (np.ndarray): Binary array containing selectors for nat samples.
        adv_selector (np.ndarray, optional): Binary array containing adversarial selectors.
            If no adv_selector is given, selector is used. Defaults to None.
        trunk_is_acc_adv (np.ndarray, optional): Binary array indicating where trunk model
            is accurate when the selector is attacked (meaning the trunk model itself is
            a compositional model). Defaults to None.

    Returns:
        Tuple[float, float]: compositional nat accuracy, compositional adv accuracy.
    """
    comp_is_acc_nat, comp_is_acc_adv, comp_is_rob_adv = apply_comp_selector(
        branch_is_acc, branch_is_rob, trunk_is_acc, trunk_is_rob,
        selector, adv_selector, trunk_is_acc_adv
    )

    comp_nat_acc = round(100.0 * np.average(comp_is_acc_nat), 2)
    comp_adv_acc = round(100.0 * np.average(comp_is_acc_adv * comp_is_rob_adv), 2)

    return comp_nat_acc, comp_adv_acc
