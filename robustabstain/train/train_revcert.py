import setGPU
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import datetime
import numpy as np
import pandas as pd
import json
import os
import re
import logging
from tqdm import tqdm
from typing import Dict, Tuple, List

import robustabstain.utils.args_factory as args_factory
from robustabstain.attacker.wrapper import AttackerWrapper
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.cert import get_acc_cert_indicator
from robustabstain.eval.comp import compositional_accuracy
from robustabstain.eval.log import write_eval_report, write_smoothing_log
from robustabstain.loss.revadv import revadv_loss, revadv_gambler_loss
from robustabstain.loss.revcert import revcert_radius_loss, revcert_noise_loss
from robustabstain.utils.checkpointing import save_checkpoint, load_checkpoint
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.loaders import get_dataloader, get_rel_sample_indices
from robustabstain.utils.log import write_config, default_serialization, logging_setup, init_logging, log
from robustabstain.utils.metrics import accuracy, adv_accuracy, AverageMeter
from robustabstain.utils.schedulers import get_lr_scheduler, StepScheduler
from robustabstain.utils.model_utils import requires_grad_, finetune


def get_args() -> object:
    parser = args_factory.get_parser(
        description='Adversarial training for common architectures on cifar10, MTSD, etc.',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS,
            args_factory.ATTACK_ARGS, args_factory.SMOOTHING_ARGS, args_factory.COMP_ARGS
        ],
        required_args=[
            'dataset', 'epochs', 'train-eps', 'noise-sd', 'adv-norm',
            'branch-model', 'revadv-beta', 'smoothing-sigma'
        ]
    )
    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    args.test_eps = args.test_eps if args.test_eps else [args.train_eps]
    args.smoothing_sigma = args.smoothing_sigma if args.smoothing_sigma else args.noise_sd

    return args


def train_smoothrevadv(
        args: object, model: nn.Module, device: str, train_loader: torch.utils.data.DataLoader,
        opt: optim.Optimizer, epoch: int, beta: float = 1.0, soft: bool = True, variant: str = 'mrevadv',
        writer: SummaryWriter = None
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Reverse adversarial training - train a model to discourage robustness.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): PyTorch model to train.
        device (PyTorch device): device.
        train_loader (torch.utils.data.DataLoader): PyTorch loader with train data.
        opt (optim.Optimizer): optimizer.
        epoch (int): current epoch.
        beta (float, optional): Weighting of losses. Defaults to 0.1.
        soft (bool, optional): If set, soft revadv loss is used. Defaults to True.
        variant (str, optional): Revadv loss variant to use. Defaults to 'mrevadv'.
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
        rel_step_size=args.train_att_step_size, version=args.autoattack_version,
        gamma_ddn=args.gamma_ddn, init_norm_ddn=args.init_norm_ddn, device=device
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
        if variant == 'smoothmrevadv':
            revadv_train_loss = revadv_loss(nat_out_rep, adv_out_rep, targets_rep, soft=soft, reduction='none')
        elif variant == 'smoothgrevadv':
            revadv_train_loss = revadv_gambler_loss(
                nat_out_rep, adv_out_rep, targets_rep, conf=args.revadv_conf, reduction='none'
            )
        else:
            raise ValueError(f'Error: invalid loss variant {variant}.')

        noise_train_loss = F.cross_entropy(adv_out_rep, targets_rep, reduction='none')
        loss = (revadv_train_loss + beta * noise_train_loss).mean()

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
            '[T] SMOOTH-REVADV epoch={}, loss={:.4f}, nat_acc1={:.4f}, '\
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


def test_smoothrevadv(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        epoch: int, beta: float = 1.0, soft: bool = True,  variant: str = 'mrevadv',
        writer: SummaryWriter =None
    ) -> Tuple[float, float, float, float]:
    """Test a model that is trained using train_revadv - test robust accuracy of robust and
    accurate samples and test robustness of all other samples. The goal is to achieve high
    robust accuracy for robust and accurate samples and low robustness on all other samples.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): PyTorch model to test.
        device (str): device.
        test_loader (torch.utils.data.DataLoader): PyTorch loader with train data.
        epoch (int): current epoch.
        beta (float, optional): Weighting of losses. Defaults to 1.0.
        soft (bool, optional): If set, soft revadv loss is used. Defaults to True.
        variant (str, optional): Revadv loss variant to use. Defaults to 'mrevadv'.
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
        rel_step_size=args.test_att_step_size, version=args.autoattack_version,
        gamma_ddn=args.gamma_ddn, init_norm_ddn=args.init_norm_ddn, device=device
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
        if variant == 'smoothmrevadv':
            revadv_test_loss = revadv_loss(nat_out_rep, adv_out_rep, targets_rep, soft=soft, reduction='none')
        elif variant == 'smoothgrevadv':
            revadv_test_loss = revadv_gambler_loss(
                nat_out_rep, adv_out_rep, targets_rep, conf=args.revadv_conf, reduction='none'
            )
        else:
            raise ValueError(f'Error: invalid loss variant {variant}.')

        noise_test_loss = F.cross_entropy(adv_out_rep, targets_rep, reduction='none')
        loss = (revadv_test_loss + beta * noise_test_loss).mean()

        # update metrics
        test_loss.update(loss.item(), inputs.size(0))
        nat_accs, _ = accuracy(nat_probs, targets, topk=(1,5))
        adv_acc, _ = adv_accuracy(adv_probs, nat_probs, targets)
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))
        adv_acc1.update(adv_acc.item(), inputs.size(0))
        nonrob.update(100.0 * np.average(1-is_rob_batch), inputs.size(0))
        rob_inacc.update(100.0 * np.average(is_rob_batch & (1-is_acc_batch)), inputs.size(0))

        pbar.set_description(
            '[V] SMOOTH-REVADV epoch={}, loss={:.4f}, nat_acc1={:.4f}, '\
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


def train_revcert(
        args: object, model: nn.Module, device: str, train_loader: torch.utils.data.DataLoader,
        opt: optim.Optimizer, epoch: int, beta: float = 1.0, variant: str = 'revcertrad',
        writer: SummaryWriter = None
    ) -> Tuple[float, float]:
    """Reverse adversarial training - train a model to discourage robustness.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): PyTorch model to train.
        device (PyTorch device): device.
        train_loader (PyTorch dataloader): PyTorch loader with train data.
        opt (torch.optimizer): optimizer.
        epoch (int): current epoch.
        beta (float, optional): Weighting of losses. Defaults to 1.0.
        variant (str, optional): Revcert loss variant to use. Defaults to 'revcertrad'.
        writer (SummaryWriter, optional): SummaryWriter. Defaults to None.

    Returns:
        Tuple[float, float]: train loss, nat accuracy
    """
    train_eps_float = convert_floatstr(args.train_eps)
    noise_sd_float = convert_floatstr(args.noise_sd) if args.noise_sd is not None else None
    _, _, num_classes = get_dataset_stats(args.dataset)

    # model metrics
    train_loss = AverageMeter()
    train_loss_classif = AverageMeter()
    train_loss_revcert = AverageMeter()
    nat_acc1 = AverageMeter()

    pbar = tqdm(train_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        rel_sample_indices = get_rel_sample_indices(train_loader, sample_indices)
        inputs, targets = inputs.to(device), targets.to(device)
        model.eval()
        nat_out = model(inputs)

        """
        train model
        """
        model.train()
        if variant == 'revcertrad':
            loss, loss_classification, revcert_loss = revcert_radius_loss(
                model, inputs, targets, device, num_classes, args.num_noise_vec,
                noise_sd_float, beta, temp=args.macer_temp, reduction='mean'
            )
        elif variant == 'revcertnoise':
            loss, loss_classification, revcert_loss = revcert_noise_loss(
                model, inputs, targets, device, num_classes, args.num_noise_vec,
                noise_sd_float, beta, args.topk_noise_vec, version='CE', reduction='mean'
            )
        else:
            raise ValueError(f'Error: invalid loss variant {variant}.')

        opt.zero_grad()
        loss.backward()
        opt.step()

        # update metrics
        train_loss.update(loss.item(), inputs.size(0))
        train_loss_classif.update(loss_classification.item(), inputs.size(0))
        train_loss_revcert.update(revcert_loss.item(), inputs.size(0))
        nat_accs, _ = accuracy(nat_out, targets, topk=(1,5))
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))


        pbar.set_description(
            '[T] {} epoch={}, loss={:.4f}, loss_classification={:.4f}, ' \
            'revcert_loss={:.4f}, nat_acc1={:.4f}, '.format(
                variant.upper(), epoch, train_loss.avg, train_loss_classif.avg,
                train_loss_revcert.avg, nat_acc1.avg)
        )

    if writer:
        writer.add_scalar('train/loss', train_loss.avg, epoch)
        writer.add_scalar(f'train/loss_classification', train_loss_classif.avg, epoch)
        writer.add_scalar('train/loss_revcert', train_loss_revcert.avg, epoch)
        writer.add_scalar('train/nat_acc', nat_acc1.avg, epoch)

    return (train_loss.avg, nat_acc1.avg)


def test_revcert(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        epoch: int, beta: float = 1.0, variant: str = 'revcertrad', writer: SummaryWriter = None
    ) -> Tuple[float, float]:
    """Test a model that is trained using train_revadv - test robust accuracy of robust and
    accurate samples and test robustness of all other samples. The goal is to achieve high
    robust accuracy for robust and accurate samples and low robustness on all other samples.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): PyTorch model to test.
        device (PyTorch device): device.
        test_loader (PyTorch dataloader): PyTorch loader with train data.
        epoch (int): current epoch.
        beta (float, optional): Revcert beta parameter. Defaults to 1.0
        variant (str, optional): Revcert loss variant to use. Defaults to 'revcertrad'.
        writer (SummaryWriter, optional): SummaryWriter. Defaults to None.

    Returns:
        Tuple[float, float]: test loss, nat accuracy
    """
    train_eps_float = convert_floatstr(args.train_eps)
    noise_sd_float = convert_floatstr(args.noise_sd) if args.noise_sd is not None else None
    _, _, num_classes = get_dataset_stats(args.dataset)

    # model metrics
    test_loss = AverageMeter()
    test_loss_classif = AverageMeter()
    test_loss_revcert = AverageMeter()
    nat_acc1 = AverageMeter()

    model.eval()
    pbar = tqdm(test_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        rel_sample_indices = get_rel_sample_indices(test_loader, sample_indices)
        inputs, targets = inputs.to(device), targets.to(device)

        """
        test model
        """
        nat_out = model(inputs)
        if variant == 'revcertrad':
            loss, loss_classification, revcert_loss = revcert_radius_loss(
                model, inputs, targets, device, num_classes, args.num_noise_vec,
                noise_sd_float, beta, temp=args.macer_temp, reduction='mean'
            )
        elif variant == 'revcertnoise':
            loss, loss_classification, revcert_loss = revcert_noise_loss(
                model, inputs, targets, device, num_classes, args.num_noise_vec,
                noise_sd_float, beta, args.topk_noise_vec, version='CE', reduction='mean'
            )
        else:
            raise ValueError(f'Error: invalid loss variant {variant}.')

        # update metrics
        test_loss.update(loss.item(), inputs.size(0))
        test_loss_classif.update(loss_classification.item(), inputs.size(0))
        test_loss_revcert.update(revcert_loss.item(), inputs.size(0))
        nat_accs, _ = accuracy(nat_out, targets, topk=(1,5))
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))

        pbar.set_description(
            '[V] {} epoch={}, loss={:.4f}, loss_classification={:.4f}, ' \
            'revcert_loss={:.4f}, nat_acc1={:.4f}, '.format(
                variant.upper(), epoch, test_loss.avg, test_loss_classif.avg,
                test_loss_revcert.avg, nat_acc1.avg)
        )

    if writer:
        writer.add_scalar('val/loss', test_loss.avg, epoch)
        writer.add_scalar(f'val/loss_classification', test_loss_classif.avg, epoch)
        writer.add_scalar('val/loss_revcert', test_loss_revcert.avg, epoch)
        writer.add_scalar('val/nat_acc', nat_acc1.avg, epoch)

    return (test_loss.avg, nat_acc1.avg)


def smoothrevadv_iter(
        args: object, model: nn.Module, model_name: str, model_arch: str,
        model_chkpt_path_best: str, running_chkpt_dir: str, device: str,
        train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader, opt: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, revadv_beta: float,
        best_loss: float, best_acc: float,
        log_list: List[Dict], log_filename: str, writer: SummaryWriter = None
    ) -> Tuple[float, float, List[Dict]]:
    """Complete epoch iteration for smoothrevadv abstain training, including training,
    validation and testing.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): Model to train.
        model_name (str): Model identifier.
        model_arch (str): Architecture of the model.
        model_chkpt_path_best (str): Filepath to checkpointed model.
        running_chkpt_dir (str): Path to directory containing running checkpoints.
        device (str): device.
        train_loader (torch.utils.data.DataLoader): Dataloader containing training data.
        val_loader (torch.utils.data.DataLoader): Dataloader containing validation data.
        test_loader (torch.utils.data.DataLoader): Dataloader containing test data.
        opt (torch.optim.Optimizer): optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epoch (int): Epoch of the current iteration.
        revadv_beta (float): revadv_beta parameter in abstain training.
        best_loss (float): Best validation loss encountered so far.
        best_acc (float): Best validation accuracy encountered so far.
        log_list (List[Dict]): List of dicts containing epoch logs.
        log_filename (str): Filename to which log_list is logged.
        writer (SummaryWriter, optional): Summarywriter. Defaults to None.

    Returns:
        Tuple[float, float, List[Dict]]: Updated best_loss, best_acc, log_list
    """

    """
    Train branch model
    """
    train_loss, train_nat_acc, train_adv_acc, branch_rob_inacc_train, = train_smoothrevadv(
        args, model, device, train_loader, opt, epoch,
        revadv_beta, variant=args.revcert_loss, writer=writer
    )

    """
    Validate branch model
    """
    should_log = (epoch % args.val_freq == 0) or (epoch == args.epochs)
    if should_log:
        val_loss, val_nat_acc, val_adv_acc, branch_rob_inacc_val, = test_smoothrevadv(
            args, model, device, val_loader, epoch,
            revadv_beta, variant=args.revcert_loss, writer=writer
        )
        val_acc = val_adv_acc

        # write to log file
        log_list = log(log_filename, log_list,
            log_dict=dict(
                epoch=epoch, lr=lr_scheduler.get_last_lr()[0], train_loss=train_loss,
                train_nat_acc=train_nat_acc, train_adv_acc=train_adv_acc,
                train_rob_inacc=branch_rob_inacc_train, val_loss=val_loss,
                val_nat_acc=val_nat_acc, val_adv_acc=val_adv_acc,
                val_rob_inacc=branch_rob_inacc_val
        ))

        # checkpointing
        should_checkpoint = True
        if should_checkpoint:
            add_state = {'nat_prec1': val_nat_acc, 'adv_prec1': val_adv_acc}
            save_checkpoint(model_chkpt_path_best, model, model_arch, args.dataset, epoch, opt, add_state)
            best_loss, best_acc, best_rob_inacc = val_loss, val_acc, branch_rob_inacc_val

    """
    Eval model on testset every args.eval_freq epochs and save running checkpoints
    """
    if args.test_freq > 0 and epoch % args.test_freq == 0:
        tmp_chkpt_dir = os.path.join(running_chkpt_dir, str(epoch))
        tmp_chkpt_path = os.path.join(tmp_chkpt_dir, f'{model_name}.pt')

        # evaluate branch model on test set and log it
        write_evals = args.running_checkpoint # only write logs when saving running checkpoints
        (
            branch_nat_acc1, branch_adv_acc1, branch_rob_inacc,
            branch_is_acc_test, branch_is_rob_test, _, _, _, _
        ) = get_acc_rob_indicator(
                args, model, tmp_chkpt_dir, model_name, device,
                test_loader, 'test', args.adv_norm, args.test_eps[0],
                args.test_adv_attack, use_existing=False,
                write_log=write_evals, write_report=write_evals
        )
        writer.add_scalar('test/nat_acc', branch_nat_acc1, epoch)
        writer.add_scalar(f'test/adv_acc{args.test_eps[0]}', branch_adv_acc1, epoch)
        writer.add_scalar('test/rob_inacc', branch_rob_inacc, epoch)

        if args.running_checkpoint:
            add_state = {'nat_prec1': branch_nat_acc1, 'adv_prec1': branch_adv_acc1}
            save_checkpoint(tmp_chkpt_path, model, model_arch, args.dataset, epoch, opt, add_state)

    return best_loss, best_acc, log_list


def revcert_iter(
        args: object, model: nn.Module, model_name: str, model_arch: str,
        model_chkpt_path_best: str, running_chkpt_dir: str, device: str,
        train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader, opt: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int, revadv_beta: float,
        best_loss: float, best_acc: float,
        log_list: List[Dict], log_filename: str, writer: SummaryWriter = None
    ) -> Tuple[float, float, List[Dict]]:
    """Complete epoch iteration for revcert abstain training, including training,
    validation and testing.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): Model to train.
        model_name (str): Model identifier.
        model_arch (str): Architecture of the model.
        model_chkpt_path_best (str): Filepath to checkpointed model.
        running_chkpt_dir (str): Path to directory containing running checkpoints.
        device (str): device.
        train_loader (torch.utils.data.DataLoader): Dataloader containing training data.
        val_loader (torch.utils.data.DataLoader): Dataloader containing validation data.
        test_loader (torch.utils.data.DataLoader): Dataloader containing test data.
        opt (torch.optim.Optimizer): optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epoch (int): Epoch of the current iteration.
        revadv_beta (float): revadv_beta parameter in abstain training.
        best_loss (float): Best validation loss encountered so far.
        best_acc (float): Best validation accuracy encountered so far.
        log_list (List[Dict]): List of dicts containing epoch logs.
        log_filename (str): Filename to which log_list is logged.
        writer (SummaryWriter, optional): Summarywriter. Defaults to None.

    Returns:
        Tuple[float, float, List[Dict]]: Updated best_loss, best_acc, log_list
    """

    """
    Train model
    """
    train_loss, train_nat_acc = train_revcert(
            args, model, device, train_loader, opt,
            epoch, revadv_beta, variant=args.revcert_loss, writer=writer
    )

    """
    Validate model
    """
    should_log = (epoch % args.val_freq == 0) or (epoch == args.epochs)
    if should_log:
        val_loss, val_nat_acc = test_revcert(
            args, model, device, val_loader, epoch,
            revadv_beta, variant=args.revcert_loss, writer=writer
        )
        val_acc = val_nat_acc

        # write to log file
        log_list = log(log_filename, log_list,
            log_dict=dict(
                epoch=epoch, lr=lr_scheduler.get_last_lr()[0], train_loss=train_loss,
                train_nat_acc=train_nat_acc, val_loss=val_loss, val_nat_acc=val_nat_acc
        ))

        # checkpointing
        should_checkpoint = val_acc > best_acc
        should_checkpoint = True
        if should_checkpoint:
            add_state = {'nat_prec1': val_nat_acc}
            save_checkpoint(model_chkpt_path_best, model, model_arch, args.dataset, epoch, opt, add_state)
            best_loss, best_acc = val_loss, val_acc

    """
    Eval model on testset every args.eval_freq epochs and save running checkpoints
    """
    if args.test_freq > 0 and epoch % args.test_freq == 0:
        tmp_chkpt_dir = os.path.join(running_chkpt_dir, str(epoch))
        tmp_chkpt_path = os.path.join(tmp_chkpt_dir, f'{model_name}.pt')

        # evaluate branch model on test set and log it
        write_evals = args.running_checkpoint # only write logs when saving running checkpoints
        (
            branch_nat_acc1, branch_cert_acc1, branch_cert_inacc,
            branch_is_acc_test, branch_is_cert_test, _, _
        ) = get_acc_cert_indicator(
                args, model, tmp_chkpt_dir, model_name, device,
                test_loader, 'test', args.test_eps[0], smooth=True, n_smooth_samples=100,
                use_existing=False, write_log=write_evals, write_report=write_evals
        )

        if args.running_checkpoint:
            add_state = {'nat_prec1': branch_nat_acc1}
            save_checkpoint(tmp_chkpt_path, model, model_arch, args.dataset, epoch, opt, add_state)

    return best_acc, best_loss, log_list


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()
    train_eps_float = convert_floatstr(args.train_eps) if args.train_eps is not None else None
    noise_sd_float = convert_floatstr(args.noise_sd) if args.noise_sd is not None else None
    n_smooth_samples = 500

    # log directories
    out_dir, exp_dir, exp_id, branch_chkpt_path_best = logging_setup(args, train_mode=args.revcert_loss)
    branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
    branch_source_dir = os.path.dirname(args.branch_model)
    running_chkpt_dir = os.path.join(exp_dir, 'running_chkpt')

    # setup logging
    log_filename = os.path.join(exp_dir, 'train_log.csv')
    log_list = init_logging(args, exp_dir)
    writer = SummaryWriter(log_dir=exp_dir)

    # build dataset
    train_loader, val_loader, test_loader, _, _, num_classes = get_dataloader(
        args, args.dataset, normalize=False, indexed=True, val_set_source='test', val_split=0.2
    )

    # load branch model
    branch_model, _, branch_chkpt = load_checkpoint(
        args.branch_model, net=None, arch=None, dataset=args.dataset, device=device,
        normalize=not args.no_normalize, optimizer=None, parallel=True
    )
    branch_arch = branch_chkpt['arch']

    trunk_model = None
    if args.trunk_models:
        # if a trunk model is given, evaluate compositional accuracies
        trunk_model, _, trunk_chkpt = load_checkpoint(
            args.trunk_models[0], net=None, arch=None, dataset=args.dataset, device=device,
            normalize=not args.no_normalize, optimizer=None, parallel=True
        )
        trunk_arch = trunk_chkpt['arch']

    # setup models for finetuning
    if args.feature_extractor:
        finetune(branch_model, device, feature_extract=True)

    # loss, optimizer, lr scheduling
    opt = optim.SGD(branch_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = get_lr_scheduler(args, args.lr_sched, opt)

    # load optimizer state
    if args.load_opt and 'optimizer' in branch_chkpt:
        opt = opt.load_state_dict(branch_chkpt['optimizer'])

    # write args config
    args.test_eps = args.test_eps if args.test_eps else [args.train_eps]
    train_args = vars(args)
    train_args['exp_dir'] = exp_dir
    train_args['chkpt_path'] = branch_chkpt_path_best
    train_args['logfile'] = log_filename
    write_config(train_args, exp_dir)
    logging.info('==> Starting training with config: {}'.format(
        json.dumps(train_args, default=default_serialization, indent=2)
    ))

    # get accuracy and robustness indicators on testset
    branch_nat_acc1, branch_cert_acc1, branch_cert_inacc, \
        branch_is_acc_test, branch_is_cert_test, _, _ = get_acc_cert_indicator(
            args, branch_model, branch_source_dir, branch_model_name, device,
            test_loader, 'test', args.test_eps[0], smooth=True, n_smooth_samples=n_smooth_samples,
            use_existing=True, write_log=True, write_report=True
    )
    logging.info('Initial branch model test accuracy ({} eps={}): nat_acc={:.4f}, ' \
        'cert_acc={:.4f}, cert_inacc={:.4f} (smooth{})'.format(
            args.adv_norm, args.test_eps[0], branch_nat_acc1,
            branch_cert_acc1, branch_cert_inacc, args.smoothing_sigma
    ))

    if trunk_model:
        # if a trunk model is given, evaluate compositional accuracies
        trunk_model_dir = os.path.dirname(args.trunk_models[0])
        trunk_model_name = os.path.splitext(os.path.basename(args.trunk_models[0]))[0]

        #NOTE: trunk model is assumed non-certifiable
        trunk_nat_acc1, trunk_cert_acc1, trunk_cert_inacc, \
            trunk_is_acc_test, trunk_is_cert_test, _, _ = get_acc_cert_indicator(
                args, trunk_model, trunk_model_dir, trunk_model_name, device,
                test_loader, 'test', args.test_eps[0], smooth=False, n_smooth_samples=n_smooth_samples,
                use_existing=True, write_log=True, write_report=True
        )

        # initial compositional accuracy
        comp_nat_acc, comp_cert_acc = compositional_accuracy(
            branch_is_acc=branch_is_acc_test, branch_is_rob=branch_is_cert_test,
            trunk_is_acc=trunk_is_acc_test, trunk_is_rob=trunk_is_cert_test,
            selector=branch_is_cert_test
        )

        logging.info('Initial trunk model test accuracy ({} eps={}): nat_acc={:.4f}, ' \
            'cert_acc={:.4f}, cert_inacc={:.4f} (smooth{})'.format(
                args.adv_norm, args.test_eps[0], trunk_nat_acc1,
                trunk_cert_acc1, trunk_cert_inacc, args.smoothing_sigma
        ))
        logging.info('Initial compositional test accuracy ({} eps={}): ' \
            'nat_acc={:.4f}, cert_acc={:.4f} (smooth{})'.format(
                args.adv_norm, args.test_eps[0], comp_nat_acc, comp_cert_acc, args.smoothing_sigma
        ))
    logging.info(120 * '=')

    # update model name
    branch_model_name = os.path.splitext(os.path.basename(branch_chkpt_path_best))[0]

    # train the model
    best_loss, best_acc, best_cert_inacc = float('inf'), -1, 100
    revadv_beta = args.revadv_beta
    revadv_beta_scheduler = StepScheduler(args.revadv_beta_step, args.revadv_beta_gamma)
    for epoch in range(0, args.epochs):
        if 'smooth' in args.revcert_loss:
            best_loss, best_acc, log_list = smoothrevadv_iter(
                args, branch_model, branch_model_name, branch_arch, branch_chkpt_path_best,
                running_chkpt_dir, device, train_loader, val_loader, test_loader,
                opt, lr_scheduler, epoch, revadv_beta, best_loss, best_acc,
                log_list, log_filename, writer
            )
        else:
            best_loss, best_acc, log_list = revcert_iter(
                args, branch_model, branch_model_name, branch_arch, branch_chkpt_path_best,
                running_chkpt_dir, device, train_loader, val_loader, test_loader,
                opt, lr_scheduler, epoch, revadv_beta, best_loss, best_acc,
                log_list, log_filename, writer
            )

        writer.add_scalar('revadv_beta', revadv_beta, epoch)
        revadv_beta = revadv_beta_scheduler.step(revadv_beta, epoch)

        lr_scheduler.step()

    # save final models
    split = os.path.splitext(branch_chkpt_path_best)
    save_checkpoint(f'{split[0]}_last{split[1]}', branch_model, branch_arch, args.dataset, epoch, opt)
    logging.info(120 * '=')

    # load best checkpoint model
    branch_model, _, _ = load_checkpoint(
        branch_chkpt_path_best, net=branch_model, arch=branch_arch, dataset=args.dataset, device=device,
        normalize=not args.no_normalize, optimizer=None, parallel=True
    )

    # get accuracy and robustness indicators on testset
    logging.info('Evaluating compositional accuracy of best checkpoint models on test set.')
    (
        branch_nat_acc1, branch_cert_acc1, branch_cert_inacc,
        branch_is_acc_test, branch_is_cert_test, _, _
    ) = get_acc_cert_indicator(
            args, branch_model, exp_dir, branch_model_name, device, test_loader,
            eval_set='test', eps_str=args.test_eps[0], smooth=True, n_smooth_samples=n_smooth_samples,
            use_existing=True, write_log=True, write_report=True
    )
    logging.info('Branch model test accuracy ({} eps={}): nat_acc={:.4f}, ' \
        'cert_acc={:.4f}, cert_inacc={:.4f}'.format(
            args.adv_norm, args.test_eps[0], branch_nat_acc1,
            branch_cert_acc1, branch_cert_inacc
    ))

    if trunk_model:
        # if a trunk model is given, evaluate compositional accuracies
        comp_nat_acc, comp_cert_acc = compositional_accuracy(
            branch_is_acc=branch_is_acc_test, branch_is_rob=branch_is_cert_test,
            trunk_is_acc=trunk_is_acc_test, trunk_is_rob=trunk_is_cert_test,
            selector=branch_is_cert_test
        )

        logging.info('Trunk model test accuracy ({} eps={}): nat_acc={:.4f}, ' \
            'cert_acc={:.4f}, cert_inacc={:.4f}'.format(
                args.adv_norm, args.test_eps[0], trunk_nat_acc1,
                trunk_cert_acc1, trunk_cert_inacc
        ))
        logging.info('Compositional test accuracy ({} eps={}): nat_acc={:.4f}, cert_acc={:.4f}'.format(
            args.adv_norm, args.test_eps[0], comp_nat_acc, comp_cert_acc
        ))

        # write eval report
        eps_str = args.test_eps[0]
        comp_key = f'comp_{branch_model_name}__{trunk_model_name}'
        selector_key = 'cert'
        comp_accs = {
            eps_str: {
                comp_key: {
                    selector_key: {
                        'comp_nat_acc': comp_nat_acc,
                        'comp_cert_acc': comp_cert_acc,
                        'branch_model': branch_chkpt_path_best,
                        'trunk_model': args.trunk_models[0]
        }}}}
        write_eval_report(args, out_dir=exp_dir, comp_accs=comp_accs)

    logging.info(120 * '=')
    writer.close()


if __name__ == '__main__':
    main()