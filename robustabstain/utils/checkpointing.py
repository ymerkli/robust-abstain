import torch
import torch.nn as nn
import torchvision.models as torchmodels
import torch.backends.cudnn as cudnn
import torch.optim as optim

import os
import logging
from typing import Tuple, Dict
from efficientnet_pytorch import EfficientNet
import warnings

import robustabstain.archs as archs
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.loaders import get_robustbench_models
from robustabstain.utils.transforms import get_normalize_layer


def get_net(
        arch: str, dataset: str, num_classes: int, device: str, normalize: bool, parallel: bool = True
    ) -> nn.Module:
    """ Build net from given architecture.

    Args:
        arch (str): architecture of net.
        dataset (str): dataset net is trained on.
        num_classes (int): number of classes.
        device (str): device.
        normalize (bool): If set, normalization layer is prepended.
        parallel (bool, optional): If set, use DataParallel. Defaults to True.

    Returns:
        nn.Module: the model
    """

    logging.info(f'==> Building net {arch}')
    net = None
    in_dim, in_channels, num_classes = get_dataset_stats(dataset)

    # collect robustbench models from https://github.com/RobustBench/robustbench
    robustbench_models = get_robustbench_models()

    if arch.startswith('efficientnet'):
        tokens = arch.split('_')
        pretrained = 'pre' in tokens
        adv = 'adv' in tokens
        if pretrained:
            net = EfficientNet.from_pretrained(tokens[0], in_channels=in_channels, num_classes=num_classes, advprop=adv)
        else:
            net = EfficientNet.from_name(tokens[0], in_channels=in_channels, num_classes=num_classes)
    elif arch in archs.__dict__:
        if arch.startswith('aa_pyramidnet'):
            net = archs.__dict__[arch](dataset=dataset, num_classes=num_classes)
        elif arch == 'mininet':
            net = archs.__dict__[arch](in_dim=in_dim, num_classes=num_classes)
        else:
            net = archs.__dict__[arch](num_classes=num_classes)
    elif arch in torchmodels.__dict__:
        net = torchmodels.__dict__[arch](num_classes=num_classes)
    elif arch in robustbench_models:
        net = robustbench_models[arch][dataset]['model_dict']['model']()
    else:
        raise ValueError(f'Error: unknown architecture {arch}')

    # (most) robustbench models do not expect normalized inputs
    if normalize and (arch not in robustbench_models or arch in ['Hendrycks2019Using']):
        normalization_layer = get_normalize_layer(device, dataset)
        net = nn.Sequential(normalization_layer, net)

    net.to(device)
    if device == 'cuda' and parallel:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net


def load_checkpoint(
        filepath: str, net: nn.Module, arch: str, dataset: str, device: str,
        normalize: bool, optimizer: optim.Optimizer = None, parallel: bool = True,
        num_classes: int = None
    ) -> Tuple[nn.Module, optim.Optimizer, dict]:
    """Loads checkpoint from filepath, verify checkpoint to provided arguments. Then builds the
    model (if not provided) and loads model and optimizer state dict from checkpoint.
    Optionally, a normalization layer is prepended to the model.

    Args:
        filepath (str): Path to checkpoint.
        net (nn.Module): Model to save.
        arch (str): Architecture of net.
        dataset (str): Dataset net is trained on.
        device (str): device.
        normalize (bool): If set, model expects normalized data.
        optimizer (torch.optimizer, optional): optimizer. Defaults to None.
        parallel (bool, optional): If set, use DataParallel. Defaults to True.
        num_classes (int, optional): Number of classes. Defaults to None.

    Returns:
        Tuple[nn.Module, optim.Optimizer, dict]: loaded model, loaded optimizer, loaded checkpoint dict.
    """
    logging.info(f'==> Loading model from {filepath}')
    checkpoint = torch.load(filepath)

    if 'dataset' in checkpoint and checkpoint['dataset'] != dataset:
        warnings.warn(f"Warning: checkpoint references dataset {checkpoint['dataset']} but dataset {dataset} was provided.")

    if 'arch' in checkpoint:
        if arch:
            assert arch == checkpoint['arch'], f"Error: checkpoint references arch {checkpoint['arch']} but arch {arch} was provided."
        else:
            arch = checkpoint['arch']

    if not net:
        if not num_classes:
            _, _, num_classes = get_dataset_stats(dataset)
        net = get_net(arch, dataset, num_classes, device, normalize, parallel)

    # rename state_dict keys in case a torch.nn.DataParallel model was saved
    state_dict = None
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'net' in checkpoint:
        state_dict = checkpoint['net']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError(f'Error: no state dict found in {filepath}')

    state_dict = {k.replace('module.', ''):v for k,v in state_dict.items()}

    if type(net) == torch.nn.DataParallel:
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint


def save_checkpoint(
        filepath: str, model: nn.Module, arch: str, dataset: str,
        epoch: int, optimizer: optim.Optimizer, add_state: dict = {}
    ) -> None:
    """Checkpoints model state_dict and additional information.

    Args:
        filepath (str): Path to which the model should be saved to.
        model (nn.Module): Model to save.
        arch (str): Architecture of net.
        dataset (str): Dataset net is trained on.
        epoch (int): Current epoch.
        optimizer (torch.optimizer): Optimizer.
        add_state (dict): Additional state dict with additional parameters to add.
    """
    model_state_dict, opt_state_dict = None, None
    base_dir = os.path.dirname(filepath)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    # Don't store DataParallel models, rather store the underlying module
    if type(model) == torch.nn.DataParallel:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    try:
        opt_state_dict = optimizer.state_dict()
    except:
        pass

    logging.info(f'==> Saving model to {filepath}')
    state = {
        'model': model_state_dict,
        'optimizer': opt_state_dict,
        'epoch': epoch,
        'arch': arch,
        'dataset': dataset,
    }
    for key, value in add_state.items():
        if key not in state:
            state[key] = value

    torch.save(state, filepath)
