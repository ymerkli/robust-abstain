"""
Reverse adversarial training script
"""

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
from typing import Dict, Tuple

import robustabstain.utils.args_factory as args_factory
from robustabstain.attacker.wrapper import AttackerWrapper
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.comp import compositional_accuracy
from robustabstain.eval.log import write_eval_report
from robustabstain.loss.revadv import revadv_loss, revadv_gambler_loss, revadv_conf_loss
from robustabstain.loss.trades import trades_loss
from robustabstain.utils.checkpointing import save_checkpoint, load_checkpoint, get_net
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.loaders import get_dataloader, get_rel_sample_indices
from robustabstain.utils.log import write_config, default_serialization, logging_setup, init_logging, log
from robustabstain.utils.metrics import accuracy, adv_accuracy, AverageMeter
from robustabstain.utils.schedulers import get_lr_scheduler, StepScheduler
from robustabstain.utils.model_utils import requires_grad_, finetune


def get_args() -> object:
    parser = args_factory.get_parser(
        description='Abstain training',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS,
            args_factory.ATTACK_ARGS, args_factory.SMOOTHING_ARGS, args_factory.COMP_ARGS
        ],
        required_args=[
            'dataset', 'epochs', 'train-eps', 'adv-norm', 'revadv-beta'
        ]
    )
    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)
    assert len(args.trunk_models) <= 1, 'Error: specify at most one trunk model'
    args.test_eps = args.test_eps if args.test_eps else [args.train_eps]

    return args


def train_revadv(
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
        Tuple[float, float, float, float, np.ndarray, np.ndarray]:
            train loss, nat accuracy, adv accuracy, fraction robust inaccurate,
            is_acc indicator, is_rob indicator
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

    train_eps_float = convert_floatstr(args.train_eps) if args.train_eps is not None else None
    attacker = AttackerWrapper(
        args.adv_attack, args.adv_norm, train_eps_float, args.train_att_n_steps,
        rel_step_size=args.train_att_step_size, version=args.autoattack_version, device=device
    )
    pbar = tqdm(train_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        rel_sample_indices = get_rel_sample_indices(train_loader, sample_indices)
        inputs, targets = inputs.to(device), targets.to(device)

        model.eval()
        # get natural and adversarial predictions on the rob model
        nat_out = model(inputs)
        nat_pred = nat_out.argmax(1)
        nat_probs = F.softmax(nat_out, dim=1)
        adv_inputs = attacker.attack(model, inputs, nat_pred)
        adv_out = model(adv_inputs)
        adv_pred = adv_out.argmax(1)
        adv_probs = F.softmax(adv_out, dim=1)

        # accuracy and robustness indicators
        is_acc_batch = nat_pred.eq(targets).int().cpu().numpy()
        is_rob_batch = adv_pred.eq(nat_pred).int().cpu().numpy()
        is_acc[rel_sample_indices] = is_acc_batch
        is_rob[rel_sample_indices] = is_rob_batch

        model.train()

        """
        train model
        """
        if variant == 'mrevadv':
            revadv_train_loss = revadv_loss(nat_out, adv_out, targets, soft, reduction='none')
        elif variant == 'grevadv':
            revadv_train_loss = revadv_gambler_loss(
                nat_out, adv_out, targets, conf=args.revadv_conf, reduction='none'
            )
        elif variant == 'mrevadv_conf':
            revadv_train_loss = revadv_conf_loss(nat_out, adv_out, targets, soft, reduction='none')
        else:
            raise ValueError(f'Error: invalid loss variant {variant}.')

        trades_train_loss, _, _, _, _ = trades_loss(
            nat_out, adv_out, targets, beta=args.trades_beta, reduction='none'
        )
        loss = (revadv_train_loss + beta * trades_train_loss).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        # update metrics
        train_loss.update(loss.item(), inputs.size(0))
        nat_accs, _ = accuracy(nat_out, targets, topk=(1,5))
        adv_acc, _ = adv_accuracy(adv_out, nat_out, targets)
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))
        adv_acc1.update(adv_acc.item(), inputs.size(0))
        nonrob.update(100.0 * np.average(1-is_rob_batch), inputs.size(0))
        rob_inacc.update(100.0 * np.average(is_rob_batch & (1-is_acc_batch)), inputs.size(0))

        pbar.set_description(
            '[T] REVADV epoch={}, loss={:.4f}, nat_acc1={:.4f}, '\
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

    return (train_loss.avg, nat_acc1.avg, adv_acc1.avg, rob_inacc.avg, is_acc, is_rob)


def test_revadv(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        epoch: int, beta: float = 1.0, soft: bool = True,  variant: str = 'mrevadv',
        writer: SummaryWriter =None
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
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
        Tuple[float, float, float, float, np.ndarray, np.ndarray]:
            branch_model train loss, branch_model nat accuracy, branch_model adv accuracy,
            branch_model fraction branchust inaccurate, is_acc indicator, is_rob indicator
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
    attacker = AttackerWrapper(
        args.adv_attack, args.adv_norm, test_eps_float, args.test_att_n_steps,
        rel_step_size=args.test_att_step_size, version=args.autoattack_version, device=device
    )
    pbar = tqdm(test_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        rel_sample_indices = get_rel_sample_indices(test_loader, sample_indices)
        inputs, targets = inputs.to(device), targets.to(device)

        # get natural and adversarial predictions
        nat_out = model(inputs)
        nat_pred = nat_out.argmax(1)
        nat_probs = F.softmax(nat_out, dim=1)
        adv_inputs = attacker.attack(model, inputs, nat_pred)
        adv_out = model(adv_inputs)
        adv_pred = adv_out.argmax(1)
        adv_probs = F.softmax(adv_out, dim=1)

        # accuracy and robustness indicators
        is_acc_batch = nat_pred.eq(targets).int().cpu().numpy()
        is_rob_batch = adv_pred.eq(nat_pred).int().cpu().numpy()
        is_acc[rel_sample_indices] = is_acc_batch
        is_rob[rel_sample_indices] = is_rob_batch

        """
        test model
        """
        if variant == 'mrevadv':
            revadv_test_loss = revadv_loss(nat_out, adv_out, targets, soft, reduction='none')
        elif variant == 'grevadv':
            revadv_test_loss = revadv_gambler_loss(
                nat_out, adv_out, targets, conf=args.revadv_conf, reduction='none'
            )
        elif variant == 'mrevadv_conf':
            revadv_test_loss = revadv_conf_loss(nat_out, adv_out, targets, soft, reduction='none')
        else:
            raise ValueError(f'Error: invalid loss variant {variant}.')

        trades_test_loss, _, _, _, _ = trades_loss(
            nat_out, adv_out, targets, beta=args.trades_beta, reduction='none'
        )
        loss = (revadv_test_loss + beta * trades_test_loss).mean()

        # update metrics
        test_loss.update(loss.item(), inputs.size(0))
        nat_accs, _ = accuracy(nat_out, targets, topk=(1,5))
        adv_acc, _ = adv_accuracy(adv_out, nat_out, targets)
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))
        adv_acc1.update(adv_acc.item(), inputs.size(0))
        nonrob.update(100.0 * np.average(1-is_rob_batch), inputs.size(0))
        rob_inacc.update(100.0 * np.average(is_rob_batch & (1-is_acc_batch)), inputs.size(0))

        pbar.set_description(
            '[V] REVADV epoch={}, loss={:.4f}, nat_acc1={:.4f}, '\
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

    return (test_loss.avg, nat_acc1.avg, adv_acc1.avg, rob_inacc.avg, is_acc, is_rob)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()
    noise_sd_float = convert_floatstr(args.noise_sd) if args.noise_sd is not None else None
    train_eps_float = convert_floatstr(args.train_eps) if args.train_eps is not None else None

    # log directories
    out_dir, exp_dir, exp_id, branch_chkpt_path_best = logging_setup(args, train_mode=args.revadv_loss)
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
    if args.branch_model:
        branch_model, _, branch_chkpt = load_checkpoint(
            args.branch_model, net=None, arch=None, dataset=args.dataset, device=device,
            normalize=not args.no_normalize, optimizer=None, parallel=True
        )
        branch_arch = branch_chkpt['arch']
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        branch_source_dir = os.path.dirname(args.branch_model)
    else:
        # build model
        branch_model = get_net(args.arch, args.dataset, num_classes, device, normalize=not args.no_normalize, parallel=True)
        branch_arch = args.arch
        branch_model_name = os.path.splitext(os.path.basename(branch_chkpt_path_best))[0]
        branch_source_dir = exp_dir


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
        finetune(branch_model, device, feature_extract=True) # freeze all layers except last FC layer

    # loss, optimizer, lr scheduling
    opt = optim.SGD(branch_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = get_lr_scheduler(args, args.lr_sched, opt)

    # load optimizer state
    if args.load_opt and 'optimizer' in branch_chkpt:
        opt = opt.load_state_dict(branch_chkpt['optimizer'])

    # write args config
    train_args = vars(args)
    train_args['exp_dir'] = exp_dir
    train_args['chkpt_path'] = branch_chkpt_path_best
    train_args['logfile'] = log_filename
    write_config(train_args, exp_dir)
    print(f'==> Starting training with config: ', \
        json.dumps(train_args, default=default_serialization, indent=2)
    )

    # get accuracy and robustness indicators of branch model on testset
    if trunk_model:
        # if a trunk model is given, evaluate compositional accuracies
        trunk_model_dir = os.path.dirname(args.trunk_models[0])
        trunk_model_name = os.path.splitext(os.path.basename(args.trunk_models[0]))[0]
        (
            trunk_nat_acc1, trunk_adv_acc1, trunk_rob_inacc,
            trunk_is_acc_test, trunk_is_rob_test, _, _, _, _
        ) = get_acc_rob_indicator(
                args, trunk_model, trunk_model_dir, trunk_model_name, device,
                test_loader, 'test', args.adv_norm, args.test_eps[0],
                args.test_adv_attack, use_existing=True, write_log=True, write_report=True
        )

    if args.branch_model:
        (
            branch_nat_acc1, branch_adv_acc1, branch_rob_inacc,
            branch_is_acc_test, branch_is_rob_test, _, _, _, _
        ) = get_acc_rob_indicator(
                args, branch_model, branch_source_dir, branch_model_name, device,
                test_loader, 'test', args.adv_norm, args.test_eps[0],
                args.test_adv_attack, use_existing=True, write_log=True, write_report=True
        )
        logging.info('Initial branch model test accuracy ({} {} eps={}): ' \
            'nat_acc={:.4f}, adv_acc={:.4f}, rob_inacc={:.4f}'.format(
                args.test_adv_attack, args.adv_norm, args.test_eps[0],
                branch_nat_acc1, branch_adv_acc1, branch_rob_inacc
        ))

        if trunk_model:
            # initial compositional accuracy
            comp_nat_acc, comp_adv_acc = compositional_accuracy(
                branch_is_acc=branch_is_acc_test, branch_is_rob=branch_is_rob_test,
                trunk_is_acc=trunk_is_acc_test, trunk_is_rob=trunk_is_rob_test,
                selector=branch_is_rob_test
            )

            logging.info('Initial trunk model test accuracy ({} {} eps={}): ' \
                'nat_acc={:.4f}, adv_acc={:.4f}, rob_inacc={:.4f}'.format(
                    args.test_adv_attack, args.adv_norm, args.test_eps[0],
                    trunk_nat_acc1, trunk_adv_acc1, trunk_rob_inacc
            ))
            logging.info('Initial compositional test accuracy ({} {} eps={}): ' \
                'nat_acc={:.4f}, adv_acc={:.4f}'.format(
                    args.test_adv_attack, args.adv_norm, args.test_eps[0], comp_nat_acc, comp_adv_acc
            ))
    logging.info(120 * '=')

    # update model name
    branch_model_name = os.path.splitext(os.path.basename(branch_chkpt_path_best))[0]

    # train the model
    best_loss, best_acc, best_rob_inacc = float('inf'), -1, 100
    revadv_beta = args.revadv_beta
    revadv_beta_scheduler = StepScheduler(args.revadv_beta_step, args.revadv_beta_gamma)
    for epoch in range(0, args.epochs):
        """
        Train branch model
        """
        train_loss, train_nat_acc, train_adv_acc, branch_rob_inacc_train, \
            branch_is_acc_train, branch_is_rob_train = train_revadv(
                args, branch_model, device, train_loader, opt, epoch,
                revadv_beta, variant=args.revadv_loss, writer=writer
        )

        """
        Validate branch model
        """
        should_log = (epoch % args.val_freq == 0) or (epoch == args.epochs)
        if should_log:
            val_loss, val_nat_acc, val_adv_acc, branch_rob_inacc_val, \
                branch_is_acc_val, branch_is_rob_val = test_revadv(
                    args, branch_model, device, val_loader, epoch,
                    revadv_beta, variant=args.revadv_loss, writer=writer
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
            should_checkpoint = branch_rob_inacc_val <= best_rob_inacc
            if should_checkpoint:
                add_state = {'nat_prec1': val_nat_acc, 'adv_prec1': val_adv_acc}
                save_checkpoint(branch_chkpt_path_best, branch_model, branch_arch, args.dataset, epoch, opt, add_state)
                best_loss, best_acc, best_rob_inacc = val_loss, val_acc, branch_rob_inacc_val

        """
        Eval model on testset every args.test_freq epochs and save running checkpoints
        """
        if args.test_freq > 0 and epoch % args.test_freq == 0:
            tmp_chkpt_dir = os.path.join(running_chkpt_dir, str(epoch))
            tmp_chkpt_path = os.path.join(tmp_chkpt_dir, f'{branch_model_name}.pt')

            # evaluate branch model on test set and log it
            write_evals = args.running_checkpoint # only write logs when saving running checkpoints
            (
                branch_nat_acc1, branch_adv_acc1, branch_rob_inacc,
                branch_is_acc_test, branch_is_rob_test, _, _, _, _
            ) = get_acc_rob_indicator(
                    args, branch_model, tmp_chkpt_dir, branch_model_name, device,
                    test_loader, 'test', args.adv_norm, args.test_eps[0],
                    args.test_adv_attack, use_existing=False,
                    write_log=write_evals, write_report=write_evals
            )
            writer.add_scalar('test/nat_acc', branch_nat_acc1, epoch)
            writer.add_scalar(f'test/adv_acc{args.test_eps[0]}', branch_adv_acc1, epoch)
            writer.add_scalar('test/rob_inacc', branch_rob_inacc, epoch)

            if args.running_checkpoint:
                add_state = {'nat_prec1': branch_nat_acc1, 'adv_prec1': branch_adv_acc1}
                save_checkpoint(tmp_chkpt_path, branch_model, branch_arch, args.dataset, epoch, opt, add_state)

        # decrease revadv_beta according to scheduler
        writer.add_scalar('revadv_beta', revadv_beta, epoch)
        revadv_beta = revadv_beta_scheduler.step(revadv_beta, epoch)

        lr_scheduler.step()

    # save final model
    split = os.path.splitext(branch_chkpt_path_best)
    add_state = {'nat_prec1': val_nat_acc, 'adv_prec1': val_adv_acc}
    save_checkpoint(f'{split[0]}_last{split[1]}', branch_model, branch_arch, args.dataset, epoch, opt, add_state)
    logging.info(120 * '=')

    # load best checkpoint model
    branch_model, _, _ = load_checkpoint(
        branch_chkpt_path_best, net=branch_model, arch=branch_arch, dataset=args.dataset,
        device=device, normalize=not args.no_normalize, optimizer=None, parallel=True
    )

    # get accuracy and robustness indicators on testset
    logging.info('Evaluating compositional accuracy of best checkpoint model on test set.')
    (
        branch_nat_acc1, branch_adv_acc1, branch_rob_inacc,
        branch_is_acc_test, branch_is_rob_test, _, _, _, _
    ) = get_acc_rob_indicator(
            args, branch_model, exp_dir, branch_model_name, device,
            test_loader, 'test', args.adv_norm, args.test_eps[0],
            args.test_adv_attack, use_existing=False, write_log=True, write_report=True
    )
    logging.info('Branch model test accuracy ({} {} eps={}): ' \
        'nat_acc={:.4f}, adv_acc={:.4f}, rob_inacc={:.4f}'.format(
            args.test_adv_attack, args.adv_norm, args.test_eps[0],
            branch_nat_acc1, branch_adv_acc1, branch_rob_inacc
    ))

    if trunk_model:
        # if a trunk model is given, evaluate compositional accuracies (reuse trunk indicators since trunk is not trained)
        comp_nat_acc, comp_adv_acc = compositional_accuracy(
            branch_is_acc=branch_is_acc_test, branch_is_rob=branch_is_rob_test,
            trunk_is_acc=trunk_is_acc_test, trunk_is_rob=trunk_is_rob_test,
            selector=branch_is_rob_test
        )

        logging.info('Trunk model test accuracy ({} {} eps={}): ' \
            'nat_acc={:.4f}, adv_acc={:.4f}, rob_inacc={:.4f}'.format(
                args.test_adv_attack, args.adv_norm, args.test_eps[0],
                trunk_nat_acc1, trunk_adv_acc1, trunk_rob_inacc
        ))
        logging.info('Compositional test accuracy ({} {} eps={}): ' \
            'nat_acc={:.4f}, adv_acc={:.4f}'.format(
                args.test_adv_attack, args.adv_norm, args.test_eps[0],
                comp_nat_acc, comp_adv_acc
        ))

        # write eval report
        eps_str = args.test_eps[0]
        comp_key = f'comp_{branch_model_name}__{trunk_model_name}'
        selector_key = 'rob'
        comp_accs = {
            eps_str: {
                comp_key: {
                    selector_key: {
                        'comp_nat_acc': comp_nat_acc,
                        'comp_adv_acc': comp_adv_acc,
                        #'comp_cert_acc': comp_cert_acc,
                        'branch_model': args.branch_model,
                        'trunk_model': args.trunk_models[0]
        }}}}
        write_eval_report(args, out_dir=exp_dir, comp_accs=comp_accs)

    logging.info(120 * '=')
    writer.close()


if __name__ == '__main__':
    main()