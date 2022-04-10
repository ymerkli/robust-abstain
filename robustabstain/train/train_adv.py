"""
Adversarial training script
"""

import setGPU
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import json
import os
import re
import logging
from tqdm import tqdm
from typing import Tuple, Dict, List
from robustness import model_utils, train, defaults
from robustness.datasets import DATASETS
from cox.utils import Parameters
import cox.store

import robustabstain.utils.args_factory as args_factory
from robustabstain.attacker.wrapper import AttackerWrapper
from robustabstain.attacker.smoothadv import SmoothAdv_PGD_L2, SmoothAdv_DDN
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.cert import get_acc_cert_indicator
from robustabstain.loss.trades import run_trades_loss
from robustabstain.utils.checkpointing import save_checkpoint, load_checkpoint, get_net
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.loaders import (
    get_dataloader, get_robustbench_models, default_data_dir,
    get_indicator_subsample, get_label_weights, get_rel_sample_indices)
from robustabstain.utils.log import write_config, default_serialization, logging_setup, init_logging, log
from robustabstain.utils.metrics import accuracy, adv_accuracy, AverageMeter, rob_inacc_perc
from robustabstain.utils.robustness import CIFAR, MTSD
from robustabstain.utils.schedulers import get_lr_scheduler
from robustabstain.utils.transforms import get_normalize_layer
from robustabstain.utils.model_utils import requires_grad_, finetune


def get_args():
    parser = args_factory.get_parser(
        description='Adversarial training for common architectures on cifar10, MTSD, etc.',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS,
            args_factory.ATTACK_ARGS, args_factory.SMOOTHING_ARGS
        ],
        required_args=['dataset', 'arch', 'epochs', 'train-eps', 'adv-norm']
    )
    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)
    if args.defense in ['stdadv', 'trades']:
        assert args.train_eps is not None, 'Specify --train-eps for adversarial training'
    elif args.defense == 'smoothadv':
        assert args.adv_norm == 'L2', 'Smooth-Adv training only for L2 norm supported'
        assert args.train_eps is not None, 'Specify --train-eps for adversarial training'
        assert args.noise_sd is not None, 'Specify --noise-sd for gaussian augmentation training'

    if not args.test_eps:
        args.test_eps = [args.train_eps]

    return args


def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]


def test_adv(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        epoch: int, selector: np.ndarray = None, writer: SummaryWriter = None, eval_set: str = ''
    ) -> Tuple[float, float, float]:
    """Test a model for adversarial robustness.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module
        device (PyTorch device): device
        test_loader (PyTorch dataloader): PyTorch loader with test data
        epoch (int): current epoch
        selector (np.ndarray, optional): Binary indicator for which samples to select. Defaults to None.
        writer (SummaryWriter, optional): SummaryWriter. Defaults to None.
        eval_set (str, optional): Dataset split being evaluated. Defaults to ''.

    Returns:
        Tuple[float, float, float, float]: adv loss, nat accuracy, adv accuracy, rob inacc percentage,
    """
    if selector is not None:
        assert len(selector) == len(test_loader.dataset), \
            "Number of training samples does not match number of sample weights."

    train_eps_float = convert_floatstr(args.train_eps) if args.train_eps is not None else None

    # switch to eval mode
    model.eval()

    # model metrics
    test_loss =AverageMeter()
    nat_acc1 = AverageMeter()
    adv_acc1 = AverageMeter()
    nonrob = AverageMeter() # fraction of non-robust samples
    rob_inacc = AverageMeter() # fraction of inaccurate robust samples


    attacker = AttackerWrapper(
        args.adv_attack, args.adv_norm, train_eps_float, args.train_att_n_steps,
        rel_step_size=args.train_att_step_size, version=args.autoattack_version, device=device
    )

    pbar = tqdm(test_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = get_indicator_subsample(test_loader, inputs, targets, sample_indices, selector)
        if inputs.size(0) == 0:
            continue

        # get natural predictions
        model.eval()
        nat_out = model(inputs)
        nat_pred = nat_out.argmax(1)

        # run attack
        adv_inputs = attacker.attack(model, inputs, nat_pred)
        adv_out = model(adv_inputs)
        adv_pred = adv_out.argmax(1)
        loss = F.cross_entropy(adv_out, targets)

        # accuracy and robustness indicators
        is_acc_batch = nat_pred.eq(targets).int().cpu().numpy()
        is_rob_batch = adv_pred.eq(nat_pred).int().cpu().numpy()

        # measure accuracy
        nat_accs, _ = accuracy(nat_out, targets, topk=(1,5))
        adv_acc, _ = adv_accuracy(adv_out, nat_out, targets)
        test_loss.update(loss.item(), inputs.size(0))
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))
        adv_acc1.update(adv_acc.item(), inputs.size(0))
        nonrob.update(100.0 * np.average(1-is_rob_batch), inputs.size(0))
        rob_inacc.update(100.0 * np.average(is_rob_batch & (1-is_acc_batch)), inputs.size(0))

        pbar.set_description(
            '[V] ADV epoch={}, adv_loss={:.4f}, nat_acc1={:.4f}, '
            'adv_acc1={:.4f} (eps={:.4f}), rob_inacc={:.4f}'.format(
                epoch, test_loss.avg, nat_acc1.avg, adv_acc1.avg, args.train_eps, rob_inacc.avg)
        )

    if writer:
        writer.add_scalar(f'{eval_set}/adv_loss', test_loss.avg, epoch)
        writer.add_scalar(f'{eval_set}/nat_acc', nat_acc1.avg, epoch)
        writer.add_scalar(f'{eval_set}/adv_acc{args.train_eps}', adv_acc1.avg, epoch)
        writer.add_scalar(f'{eval_set}/rob_inacc', rob_inacc.avg, epoch)
        writer.add_scalar(f'{eval_set}/nonrob', nonrob.avg, epoch)

    return (test_loss.avg, nat_acc1.avg, adv_acc1.avg, rob_inacc.avg)


def train_trades(
        args: object, model: nn.Module, device: str, train_loader: torch.utils.data.DataLoader,
        opt: optim.Optimizer, eps: float, epoch: int, selector: np.ndarray = None,
        weight: torch.tensor = None, writer: SummaryWriter = None
    ) -> Tuple[float, float, float]:
    """Single training iteration for TRADES adversarial training (https://arxiv.org/pdf/1901.08573.pdf).

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module
        device (PyTorch device): device
        train_loader (torch.utils.data.DataLoader): PyTorch loader with train data
        opt (torch.optimizer): optimizer
        eps (float): Perturbation region epsilon for adversarial training
        epoch (int): current epoch
        selector (np.ndarray, optional): Binary indicator for which samples to select. Defaults to None.
        weight (torch.tensor, optional): Per label weights for weighted loss. Defaults to None.
        writer (SummaryWriter, optional): Summarywriter. Defaults to None.

    Returns:
        Tuple[float, float, float]: trades train loss, nat accuracy, adv accuracy
    """
    if selector is not None:
        assert len(selector) == len(train_loader.dataset), \
            "Number of training samples does not match number of sample weights."

    model.train()
    nat_train_loss = AverageMeter()
    trades_train_loss = AverageMeter()
    nat_acc1 =  AverageMeter()
    adv_acc1 = AverageMeter()
    rob_inacc = AverageMeter()

    pbar = tqdm(train_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = get_indicator_subsample(train_loader, inputs, targets, sample_indices, selector)
        if inputs.size(0) == 0:
            continue

        # calculate robust loss
        step_size = eps * args.train_att_step_size
        loss, nat_loss, _, nat_acc, adv_acc, is_acc, is_rob = run_trades_loss(
            model=model, x_natural=inputs, y=targets, step_size=step_size,
            epsilon=eps, perturb_steps=args.train_att_n_steps, beta=args.trades_beta,
            distance=args.adv_norm, adversarial=True, mode='train', weight=weight
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        # update metrics
        trades_train_loss.update(loss.item(), inputs.size(0))
        nat_train_loss.update(nat_loss.item(), inputs.size(0))
        nat_acc1.update(nat_acc.item(), inputs.size(0))
        adv_acc1.update(adv_acc.item(), inputs.size(0))
        rob_inacc.update(rob_inacc_perc(is_acc, is_rob), inputs.size(0))

        pbar.set_description(
            '[T] TRADES epoch={}, trades_loss={:.4f}, nat_loss={:.4f}, nat_acc1={:.4f}, ' \
            'adv_acc1={:.4f} (eps={:.4f}), rob_inacc={:.4f}'.format(
                epoch, trades_train_loss.avg, nat_train_loss.avg,
                nat_acc1.avg, adv_acc1.avg, eps, rob_inacc.avg)
        )

    if writer:
        writer.add_scalar('train/trades_loss', trades_train_loss.avg, epoch)
        writer.add_scalar('train/nat_acc', nat_acc1.avg, epoch)
        writer.add_scalar(f'train/adv_acc{args.train_eps}', adv_acc1.avg, epoch)
        writer.add_scalar('train/rob_inacc', rob_inacc.avg, epoch)

    return (trades_train_loss.avg, nat_acc1.avg, adv_acc1.avg)


def test_trades(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        eps: float, epoch: int, selector: np.ndarray = None, eval_set: str = '',
        weight: torch.tensor = None, writer: SummaryWriter = None,
    ) -> Tuple[float, float, float]:
    """Evaluate natural and robust accuracies as well as TRADES loss on test set.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module
        device (str): device
        test_loader (PyTorch dataloader): PyTorch loader with test data
        eps (float): Perturbation region epsilon for adversarial training
        epoch (int): current epoch
        selector (np.ndarray, optional): Binary indicator for which samples to select. Defaults to None.
        eval_set (str, optional): Dataset split being evaluated ('val', 'test'). Defaults to ''.
        weight (torch.tensor, optional): Per label weights for weighted loss. Defaults to None.
        writer (SummaryWriter, optional): Summarywriter. Defaults to None.

    Returns:
        Tuple[float, float, float]: trades loss, nat accuracy, adv accuracy
    """
    if selector is not None:
        assert len(selector) == len(test_loader.dataset), \
            "Number of training samples does not match number of sample weights."

    model.eval()
    nat_test_loss = AverageMeter()
    trades_test_loss = AverageMeter()
    nat_acc1 =  AverageMeter()
    adv_acc1 = AverageMeter()
    rob_inacc = AverageMeter()

    pbar = tqdm(test_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = get_indicator_subsample(test_loader, inputs, targets, sample_indices, selector)
        if inputs.size(0) == 0:
            continue


        # run attacks to get adv accuracy and trades loss
        if batch_idx < args.eval_att_batches or args.eval_att_batches < 0:
            step_size = eps * args.test_att_step_size
            loss, nat_loss, _, nat_acc, adv_acc, is_acc, is_rob = run_trades_loss(
                model=model, x_natural=inputs, y=targets, step_size=step_size,
                epsilon=eps, perturb_steps=args.test_att_n_steps, beta=args.trades_beta,
                distance=args.adv_norm, adversarial=True, mode='test', weight=weight
            )

            # update metrics
            trades_test_loss.update(loss.item(), inputs.size(0))
            nat_test_loss.update(nat_loss.item(), inputs.size(0))
            nat_acc1.update(nat_acc.item(), inputs.size(0))
            adv_acc1.update(adv_acc.item(), inputs.size(0))
            rob_inacc.update(rob_inacc_perc(is_acc, is_rob), inputs.size(0))

        pbar.set_description(
            '[V] TRADES epoch={}, trades_loss={:.4f}, nat_loss={:.4f}, nat_acc1={:.4f}, ' \
            'adv_acc1={:.4f} (eps={:.4f}), rob_inacc={:.4f}'.format(
                epoch, trades_test_loss.avg, nat_test_loss.avg,
                nat_acc1.avg, adv_acc1.avg, eps, rob_inacc.avg)
        )

    if writer:
        writer.add_scalar(f'{eval_set}/trades_loss', trades_test_loss.avg, epoch)
        writer.add_scalar(f'{eval_set}/nat_acc', nat_acc1.avg, epoch)
        writer.add_scalar(f'{eval_set}/adv_acc{args.test_eps[0]}', adv_acc1.avg, epoch)
        writer.add_scalar(f'{eval_set}/rob_inacc', rob_inacc.avg, epoch)

    return (trades_test_loss.avg, nat_acc1.avg, adv_acc1.avg)


def train_smoothadv(
        args: object, model: nn.Module, device: str, train_loader: torch.utils.data.DataLoader,
        opt: torch.optim.Optimizer, epoch: int, writer: SummaryWriter
    ) -> Tuple[float, float, float, float]:
    """Train a model using 'Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers'
    (https://arxiv.org/abs/1906.04584). Source: https://github.com/Hadisalman/smoothing-adversarial

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module
        device (str): device
        train_loader (torch.utils.data.DataLoader): PyTorch loader with train data
        opt (torch.optimizer): optimizer
        epoch (int): current epoc
        writer (SummaryWriter, optional): SummaryWriter. Defaults to None.

    Returns:
        Tuple[float, float, float, float]:
            train loss, nat accuracy, adv accuracy, fraction robust inaccurate
    """
    # store for each train sample whether it is robust and/or accurate on the rob model
    is_acc = np.zeros(len(train_loader.dataset), dtype=np.int64)
    is_rob = np.zeros(len(train_loader.dataset), dtype=np.int64)

    # model metrics
    train_loss = AverageMeter()
    nat_acc1 = AverageMeter()
    adv_acc1 = AverageMeter()
    nonrob = AverageMeter() # fraction of non-robust samples
    rob_inacc = AverageMeter() # fraction of robust inaccurate samples

    train_eps_float = convert_floatstr(args.train_eps)
    noise_sd_float = convert_floatstr(args.noise_sd)
    _, _, num_classes = get_dataset_stats(args.dataset)
    attacker = AttackerWrapper(
        'smoothpgd', args.adv_norm, train_eps_float, args.train_att_n_steps,
        rel_step_size=args.train_att_step_size, version=args.autoattack_version, device=device
    )
    pbar = tqdm(train_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        rel_sample_indices = get_rel_sample_indices(train_loader, sample_indices)
        inputs, targets = inputs.to(device), targets.to(device)

        # sizing
        input_size = inputs.size(0)
        new_shape = [input_size * args.num_noise_vec]
        new_shape.extend(inputs[0].shape)

        # repeat inputs by number of noise samples per base sample
        inputs_rep = inputs.repeat((1, args.num_noise_vec, 1, 1)).view(new_shape)
        targets_rep = targets.unsqueeze(1).repeat(1, args.num_noise_vec).reshape(-1, 1).squeeze()
        noise = torch.randn_like(inputs_rep, device=device) * noise_sd_float

        model.eval()
        # natural prediction over noisy inputs and averaged prediction
        nat_out_rep = model(inputs_rep + noise)
        nat_out = nat_out_rep.reshape(input_size, args.num_noise_vec, num_classes).mean(1)
        nat_probs = F.softmax(nat_out_rep.reshape(input_size, args.num_noise_vec, num_classes), dim=2).mean(1)
        nat_pred = nat_probs.argmax(1)

        # smoothadv attack
        adv_inputs = attacker.attack(
            model, inputs_rep, nat_pred, noise=noise, num_noise_vectors=args.num_noise_vec
        )
        adv_out_rep = model(adv_inputs + noise)
        adv_out = adv_out_rep.reshape(input_size, args.num_noise_vec, num_classes).mean(1)
        adv_probs = F.softmax(adv_out, dim=1)
        adv_pred = adv_probs.argmax(1)

        # accuracy and robustness indicators
        is_acc_batch = nat_pred.eq(targets).int().cpu().numpy()
        is_rob_batch = adv_pred.eq(nat_pred).int().cpu().numpy()
        is_acc[rel_sample_indices] = is_acc_batch
        is_rob[rel_sample_indices] = is_rob_batch

        """
        train model
        """
        model.train()
        loss = F.cross_entropy(adv_out_rep, targets_rep, reduction='mean')

        opt.zero_grad()
        loss.backward()
        opt.step()

        # update metrics
        train_loss.update(loss.item(), inputs.size(0))
        nat_accs, _ = accuracy(nat_probs, targets, topk=(1,5))
        adv_acc, _ = adv_accuracy(adv_probs, nat_probs, targets)
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))
        adv_acc1.update(adv_acc.item(), inputs.size(0))
        nonrob.update(100.0 * np.average(1-is_rob_batch), inputs.size(0))
        rob_inacc.update(100.0 * np.average(is_rob_batch & (1-is_acc_batch)), inputs.size(0))

        pbar.set_description(
            '[T] SMOOTHADV epoch={}, loss={:.4f}, nat_acc1={:.4f}, '\
            'adv_acc1={:.4f} (eps={}), nonrob={:.4f}, rob_inacc={:.4f}'.format(
                epoch, train_loss.avg, nat_acc1.avg, adv_acc1.avg,
                args.train_eps, nonrob.avg, rob_inacc.avg)
        )

    if writer:
        writer.add_scalar('train/revadv_loss', train_loss.avg, epoch)
        writer.add_scalar('train/nat_acc', nat_acc1.avg, epoch)
        writer.add_scalar(f'train/adv_acc{args.train_eps}', adv_acc1.avg, epoch)
        writer.add_scalar('train/nonrob', nonrob.avg, epoch)
        writer.add_scalar('train/rob_inacc', rob_inacc.avg, epoch)

    return (train_loss.avg, nat_acc1.avg, adv_acc1.avg, rob_inacc.avg)


def test_smoothadv(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        epoch: int, writer: SummaryWriter = None
    ) -> Tuple[float, float, float, float]:
    """Test a model using 'Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers'
    (https://arxiv.org/abs/1906.04584). Source: https://github.com/Hadisalman/smoothing-adversarial

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch model to test.
        device (PyTorch device): device
        test_loader (PyTorch dataloader): PyTorch loader with test data.
        epoch (int): current epoch
        writer (torch.SummaryWriter): SummaryWriter.

    Returns:
        Tuple[float, float, float, float]:
            test loss, nat accuracy, test adv accuracy, fraction robust inaccurate
    """
    # store for each train sample whether it is robust and/or accurate
    is_acc = np.zeros(len(test_loader.dataset), dtype=np.int64)
    is_rob = np.zeros(len(test_loader.dataset), dtype=np.int64)

    # rob model metrics
    test_loss = AverageMeter()
    nat_acc1 = AverageMeter()
    adv_acc1 = AverageMeter()
    nonrob = AverageMeter() # fraction of non-robust samples
    rob_inacc = AverageMeter() # fraction of inaccurate robust samples

    model.eval()

    test_eps_float = convert_floatstr(args.test_eps[0])
    noise_sd_float = convert_floatstr(args.noise_sd)
    _, _, num_classes = get_dataset_stats(args.dataset)
    attacker = AttackerWrapper(
        'smoothpgd', args.adv_norm, test_eps_float, args.test_att_n_steps,
        rel_step_size=args.test_att_step_size, version=args.autoattack_version, device=device
    )
    pbar = tqdm(test_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        rel_sample_indices = get_rel_sample_indices(test_loader, sample_indices)
        inputs, targets = inputs.to(device), targets.to(device)

        # sizing
        input_size = inputs.size(0)
        new_shape = [input_size * args.num_noise_vec]
        new_shape.extend(inputs[0].shape)

        # repeat inputs and targets by number of noise samples per base sample
        inputs_rep = inputs.repeat((1, args.num_noise_vec, 1, 1)).view(new_shape)
        targets_rep = targets.unsqueeze(1).repeat(1, args.num_noise_vec).reshape(-1, 1).squeeze()
        noise = torch.randn_like(inputs_rep, device=device) * noise_sd_float

        model.eval()
        # natural prediction over noisy inputs and averaged prediction
        nat_out_rep = model(inputs_rep + noise)
        nat_out = nat_out_rep.reshape(input_size, args.num_noise_vec, num_classes).mean(1)
        nat_probs = F.softmax(nat_out_rep.reshape(input_size, args.num_noise_vec, num_classes), dim=2).mean(1)
        nat_pred = nat_probs.argmax(1)

        # smoothadv attack
        adv_inputs = attacker.attack(
            model, inputs_rep, nat_pred, noise=noise, num_noise_vectors=args.num_noise_vec
        )
        adv_out_rep = model(adv_inputs + noise)
        adv_out = adv_out_rep.reshape(input_size, args.num_noise_vec, num_classes).mean(1)
        adv_probs = F.softmax(adv_out, dim=1)
        adv_pred = adv_probs.argmax(1)

        # accuracy and robustness indicators
        is_acc_batch = nat_pred.eq(targets).int().cpu().numpy()
        is_rob_batch = adv_pred.eq(nat_pred).int().cpu().numpy()
        is_acc[rel_sample_indices] = is_acc_batch
        is_rob[rel_sample_indices] = is_rob_batch

        """
        test model
        """
        loss = F.cross_entropy(adv_out_rep, targets_rep, reduction='mean')

        # update metrics
        test_loss.update(loss.item(), inputs.size(0))
        nat_accs, _ = accuracy(nat_probs, targets, topk=(1,5))
        adv_acc, _ = adv_accuracy(adv_probs, nat_probs, targets)
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))
        adv_acc1.update(adv_acc.item(), inputs.size(0))
        nonrob.update(100.0 * np.average(1-is_rob_batch), inputs.size(0))
        rob_inacc.update(100.0 * np.average(is_rob_batch & (1-is_acc_batch)), inputs.size(0))

        pbar.set_description(
            '[V] SMOOTHADV epoch={}, loss={:.4f}, nat_acc1={:.4f}, '\
            'adv_acc1={:.4f} (eps={}), nonrob={:.4f}, rob_inacc={:.4f}'.format(
                epoch, test_loss.avg, nat_acc1.avg, adv_acc1.avg,
                args.train_eps, nonrob.avg, rob_inacc.avg)
        )

    if writer:
        writer.add_scalar('val/revadv_loss', test_loss.avg, epoch)
        writer.add_scalar('val/nat_acc', nat_acc1.avg, epoch)
        writer.add_scalar(f'val/adv_acc{args.train_eps}', adv_acc1.avg, epoch)
        writer.add_scalar('val/nonrob', nonrob.avg, epoch)
        writer.add_scalar('val/rob_inacc', rob_inacc.avg, epoch)

    return (test_loss.avg, nat_acc1.avg, adv_acc1.avg, rob_inacc.avg)


def train_stdadv(
        args: object, model: nn.Module, device: str, train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader, out_dir: str, exp_dir: str, exp_id: str
    ) -> None:
    """Train a model using standard adversarial training using the robustness library (https://github.com/MadryLab/robustness)

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module
        device (str): device
        train_loader (torch.utils.data.DataLoader): PyTorch loader with train data
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data
        out_dir (str): Parent directory of exp_dir
        exp_dir (str): Path to experiment directory
        exp_id (str): Experiment identifier
    """
    train_eps_float = convert_floatstr(args.train_eps)
    parsed_adv_norm = re.match(r'L?(\S+)', args.adv_norm)
    if not parsed_adv_norm:
        raise ValueError(f'Error: unknown advervsarial norm {args.adv_norm}.')
    train_kwargs = {
        # training parameters
        'out_dir': exp_dir,
        'epochs': args.epochs,
        'lr': args.lr,
        'step_lr': args.lr_step,
        'step_lr_gamma': args.lr_gamma,
        'adv_train': 1,
        'adv_eval': 1,
        'weight_decay': args.weight_decay,

        # adversarial training parameters
        'constraint': parsed_adv_norm.group(1),
        'eps': train_eps_float,
        'attack_lr': args.train_att_step_size * train_eps_float, # relative stepsize
        'attack_steps': args.train_att_n_steps,
        'use_best': 1,
        'random_start': 1,
        'defense': args.defense
    }
    train_args = Parameters(train_kwargs)

    # Fill whatever parameters are missing from the defaults
    ds_class = DATASETS[args.dataset]
    train_args = defaults.check_and_fill_args(train_args, defaults.TRAINING_ARGS, ds_class)
    train_args = defaults.check_and_fill_args(train_args, defaults.PGD_ARGS, ds_class)
    out_store = cox.store.Store(storage_folder=out_dir, exp_id=exp_id)
    write_config(train_args, exp_dir)

    # Train a model
    print(f'==> Starting training with config: ', \
        json.dumps(vars(train_args)['params'], default=default_serialization, indent=2)
    )
    train.train_model(args=train_args, model=model, loaders=(train_loader, test_loader), store=out_store)


def main_stdadv(
        args: object, device: str, out_dir: str, exp_id: str, exp_dir: str, chkpt_path: str
    ) -> None:
    """Main function for standard adversarial training using https://github.com/MadryLab/robustness as general
    adversarial training library.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        device (PyTorch device): device
        out_dir (str): Path to train_mode/dataset dir
        exp_id (str): Experiment ID
        exp_dir (str): Path to experiment directory (out_dir/exp_id)
        chkpt_path (str): Path to checkpoint file.
    """
    # add entries to robustness DATASETS dict
    DATASETS['mtsd'] = MTSD
    DATASETS['cifar'] = CIFAR
    DATASETS['cifar10'] = DATASETS['cifar']

    # build dataset
    dataset = DATASETS[args.dataset](default_data_dir(args.dataset))
    train_loader, test_loader = dataset.make_loaders(
        workers=args.num_workers, batch_size=args.train_batch, val_batch_size=args.test_batch,
        data_aug=True, shuffle_train=True, shuffle_val=False
    )

    # build model and resume from checkpoint if given
    model, _ = model_utils.make_and_restore_model(
        arch=args.arch, dataset=dataset, resume_path=args.resume, parallel=False
    )

    train_stdadv(args, model, device, train_loader, test_loader, out_dir, exp_dir, exp_id)

    """robustness library checkpoints models in robustness.attacker.AttackerModel format.
    Extract the underlying classifier (robustness.attacker.AttackerModel.model) and re-save.
    """
    best_chkpt_path = os.path.join(exp_dir, 'checkpoint.pt.best')
    best_model, best_chkpt = model_utils.make_and_restore_model(
        arch=args.arch, dataset=dataset, resume_path=best_chkpt_path, parallel=False
    )
    model = best_model.model

    # add a normalization layer in front of the model
    if not args.no_normalize and args.arch not in get_robustbench_models():
        normalization_layer = get_normalize_layer(device, dataset)
        model = nn.Sequential(normalization_layer, model)

    add_state = {'train_eps': args.train_eps, 'adv_norm': args.adv_norm}
    save_checkpoint(
        chkpt_path, model, args.arch, args.dataset,
        best_chkpt['epoch'], best_chkpt['optimizer'], add_state
    )


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()
    noise_sd_float = convert_floatstr(args.noise_sd) if args.noise_sd is not None else None
    train_eps_float = convert_floatstr(args.train_eps) if args.train_eps is not None else None

    # log directories
    train_mode = 'smoothadv' if 'smoothadv' in args.defense else 'adv'
    out_dir, exp_dir, exp_id, chkpt_path = logging_setup(args, train_mode=train_mode)
    model_name = os.path.splitext(os.path.basename(chkpt_path))[0]
    running_chkpt_dir = os.path.join(exp_dir, 'running_chkpt')

    if args.defense == 'stdadv':
        # stdadv training using https://github.com/MadryLab/robustness has its own main function
        main_stdadv(args, device, out_dir, exp_id, exp_dir, chkpt_path)
        return

    # setup logging
    log_filename = os.path.join(exp_dir, 'train_log.csv')
    log_list = init_logging(args, exp_dir)
    writer = SummaryWriter(log_dir=exp_dir)

    # build dataset
    train_loader, val_loader, test_loader, _, _, num_classes = get_dataloader(
        args, args.dataset, normalize=False, indexed=True, val_set_source='test', val_split=0.2
    )
    label_weights = None
    if args.weighted_loss:
        label_weights = get_label_weights(train_loader.dataset)
        label_weights = torch.from_numpy(label_weights).to(device).float()

    # build model
    model = get_net(args.arch, args.dataset, num_classes, device, normalize=not args.no_normalize, parallel=True)

    # loss, optimizer, lr scheduling
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = get_lr_scheduler(args, args.lr_sched, opt)

    # resume from checkpoint if given
    start_epoch = 0
    if args.resume:
        print(f'==> Resuming from checkpoint {args.resume}.')
        if args.load_opt:
            # load optimizer from checkpoint
            model, opt, checkpoint = load_checkpoint(
                args.resume, model, args.arch, args.dataset, normalize=not args.no_normalize,
                device=device, optimizer=opt, parallel=True
            )
        else:
            model, _, checkpoint = load_checkpoint(
                args.resume, model, args.arch, args.dataset, normalize=not args.no_normalize,
                device=device, optimizer=None, parallel=True
            )
        start_epoch = 0 if args.reset_epochs else checkpoint['epoch']

    if args.feature_extractor:
        finetune(model, device, feature_extract=True)

    # write args config
    train_args = vars(args)
    train_args['exp_dir'] = exp_dir
    train_args['chkpt_path'] = chkpt_path
    train_args['logfile'] = log_filename
    write_config(train_args, exp_dir)
    print(f'==> Starting training with config: ', \
        json.dumps(train_args, default=default_serialization, indent=2)
    )

    # setup attacker for smooth-adv training
    attacker = None
    if args.defense == 'smoothadv':
        assert args.adv_norm == 'L2', 'Only L2 norm supported for Smooth-Adv defense.'
        if args.adv_attack == 'pgd':
            attacker = SmoothAdv_PGD_L2(
                steps=args.train_att_n_steps, random_start=args.random_start,
                max_norm=train_eps_float, device=device
            )
        elif args.adv_attack == 'ddn':
            attacker = SmoothAdv_DDN(
                steps=args.train_att_n_steps, gamma=args.gamma_ddn, max_norm=train_eps_float,
                init_norm=args.init_norm_ddn, device=device
            )
        else:
            raise ValueError(f'Error: unknown attack {args.adv_attack}')


    best_loss, best_acc = float('inf'), -1
    for epoch in range(start_epoch, start_epoch+args.epochs):
        """
        Train
        """
        train_loss, train_nat_acc, train_adv_acc = float('inf'), -1, -1
        if args.defense == 'trades':
            # TRADES adversarial training
            train_loss, train_nat_acc, train_adv_acc = train_trades(
                args, model, device, train_loader, opt, train_eps_float, epoch,
                weight=label_weights, writer=writer
            )
        elif args.defense == 'smoothadv':
            # Smoothing-Adv training
            #attacker.max_norm = np.min([train_eps_float, (epoch + 1) * train_eps_float/args.warmup])
            #attacker.init_norm = np.min([train_eps_float, (epoch + 1) * train_eps_float/args.warmup])

            train_loss, train_nat_acc, train_adv_acc, train_rob_inacc = train_smoothadv(
                args, model, device, train_loader, opt, epoch, writer
            )
        else:
            raise ValueError(f'Error: unknown defense {args.defense}')

        """
        Validation
        """
        should_log = (epoch % args.val_freq == 0) or (epoch == args.epochs)
        if should_log:
            val_loss, val_acc, val_nat_acc, val_adv_acc = float('inf'), -1, -1, -1
            if args.defense == 'trades':
                # TRADES val
                val_loss, val_nat_acc, val_adv_acc = test_trades(
                    args, model, device, val_loader, train_eps_float, epoch, eval_set='val',
                    weight=label_weights, writer=writer
                )
                val_acc = val_adv_acc
            elif args.defense == 'smoothadv':
                # Smoothing-Adv val
                val_loss, val_nat_acc, val_adv_acc, val_rob_inacc = test_smoothadv(
                    args, model, device, val_loader, epoch, writer
                )
                val_acc = val_adv_acc
            else:
                raise ValueError(f'Error: unknown defense {args.defense}')

            # write to logfile
            log_list = log(log_filename, log_list, log_dict=dict(
                epoch=epoch, lr=lr_scheduler.get_last_lr()[0], train_loss=train_loss,
                train_nat_acc=train_nat_acc, train_adv_acc=train_adv_acc,
                val_loss=val_loss, val_nat_acc=val_nat_acc, val_adv_acc=val_adv_acc
            ))

            # checkpointing
            should_checkpoint = val_acc > best_acc
            if should_checkpoint:
                add_state = {'nat_prec1': val_nat_acc, 'adv_prec1': val_adv_acc}
                save_checkpoint(chkpt_path, model, args.arch, args.dataset, epoch, opt, add_state)
                best_loss, best_acc = val_loss, val_acc

        """
        Eval model on testset every args.eval_freq epochs and save running checkpoints
        """
        if args.test_freq > 0 and epoch % args.test_freq == 0:
            tmp_chkpt_dir = os.path.join(running_chkpt_dir, str(epoch))
            tmp_chkpt_path = os.path.join(tmp_chkpt_dir, f'{model_name}.pt')

            # evaluated model on test set and log it
            write_evals = args.running_checkpoint # only write logs when saving running checkpoints
            test_nat_acc1, test_adv_acc1, rob_inacc, is_acc_test, is_rob_test, _, _, _, _ = get_acc_rob_indicator(
                    args, model, tmp_chkpt_dir, model_name, device, test_loader,
                    'test', args.adv_norm, args.test_eps[0], args.test_adv_attack,
                    use_existing=False, write_log=write_evals, write_report=write_evals
            )
            writer.add_scalar('test/nat_acc', test_nat_acc1, epoch)
            writer.add_scalar(f'test/adv_acc{args.test_eps[0]}', test_adv_acc1, epoch)
            writer.add_scalar('test/rob_inacc', rob_inacc, epoch)

            if args.running_checkpoint:
                add_state = {'nat_prec1': test_nat_acc1, 'adv_prec1': test_adv_acc1}
                save_checkpoint(tmp_chkpt_path, model, args.arch, args.dataset, epoch, opt, add_state)

        lr_scheduler.step()

    # save final model
    split = os.path.splitext(chkpt_path)
    chkpt_path = f'{split[0]}_last{split[1]}'
    add_state = {'nat_prec1': val_nat_acc, 'adv_prec1': val_adv_acc}
    save_checkpoint(chkpt_path, model, args.arch, args.dataset, epoch, opt, add_state)

    # load best checkpoint model
    model, _, _ = load_checkpoint(
        chkpt_path, net=model, arch=args.arch, dataset=args.dataset,
        device=device, normalize=not args.no_normalize, optimizer=None, parallel=True
    )

    if 'smoothadv'in args.defense:
        logging.info('Evaluating cert accuracy of best checkpoint model on test set.')
        get_acc_cert_indicator(
            args, model, exp_dir, model_name, device, test_loader, eval_set='test',
            eps_str=args.test_eps[0], smooth=True, n_smooth_samples=500,
            use_existing=False, write_log=True, write_report=True
        )

    logging.info('Evaluating nat/adv accuracy of best checkpoint model on test set.')
    get_acc_rob_indicator(
        args, model, exp_dir, model_name, device, test_loader,
        'test', args.adv_norm, args.test_eps[0], args.test_adv_attack,
        use_existing=False, write_log=True, write_report=True
    )
    logging.info(120 * '=')

    writer.close()


if __name__ == '__main__':
    main()