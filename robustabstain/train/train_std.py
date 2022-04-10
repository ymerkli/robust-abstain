"""
Standard training and training with data augmentation script.
"""

import setGPU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import json
from tqdm import tqdm
import logging
from typing import List, Union

import robustabstain.utils.args_factory as args_factory
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.common import PERTURBATION_REGIONS
from robustabstain.eval.nat import natural_eval
from robustabstain.utils.checkpointing import save_checkpoint, load_checkpoint, get_net
from robustabstain.utils.loaders import get_dataloader, get_indicator_subsample, get_label_weights, get_targets
from robustabstain.utils.log import write_config, default_serialization, logging_setup, init_logging, log
from robustabstain.utils.metrics import accuracy, AverageMeter
from robustabstain.utils.schedulers import get_lr_scheduler


def get_args():
    """argparsing

    Returns:
        object: object subclass exposing 'setattr` and 'getattr'
    """
    parser = args_factory.get_parser(
        description='Standard training for common architectures on cifar10, MTSD, etc.',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS,
            args_factory.ATTACK_ARGS, args_factory.SMOOTHING_ARGS
        ],
        required_args=['dataset', 'arch', 'epochs']
    )
    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    return args


def train_std(
        train_loader: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.modules.loss._Loss,
        opt: optim.Optimizer, epoch: int, device: str, selector: np.ndarray = None, writer: SummaryWriter = None
    ) -> Union[float, float]:
    """Single training iteration function to prevent memory leaks.

    Args:
        train_loader (torch.utils.data.DataLoader): train dataloader
        model (nn.Module): model to train
        criterion (nn.modules.loss._Loss): loss function
        opt (optim.Optimizer): optimizer
        epoch (int): current epoch
        device (str): device
        selector (np.ndarray): Binary indicator for which samples to select
        writer (SummaryWriter): SummaryWriter

    Returns:
        Union[float, float]: nat_acc top1, nat_acc top5
    """
    if selector is not None:
        assert len(selector) == len(train_loader.dataset), \
            "Number of training samples does not match number of sample weights."

    model.train()
    train_loss = AverageMeter()
    nat_acc1 = AverageMeter()
    nat_acc5 = AverageMeter()

    pbar = tqdm(train_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = get_indicator_subsample(train_loader, inputs, targets, sample_indices, selector)
        if inputs.size(0) == 0:
            continue

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        opt.zero_grad()
        loss.backward()
        opt.step()

        nat_accs, _ = accuracy(outputs, targets, topk=(1,5))

        train_loss.update(loss.item(), inputs.size(0))
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))
        nat_acc5.update(nat_accs[1].item(), inputs.size(0))

        pbar.set_description(f'[T] STD-TRAIN epoch=%d, loss=%.4f, acc1=%.4f, acc5=%.4f' % (
            epoch, train_loss.avg, nat_acc1.avg, nat_acc5.avg
        ))

    if writer:
        writer.add_scalar('train/loss', train_loss.avg, epoch)
        writer.add_scalar('train/nat_acc1', nat_acc1.avg, epoch)

    return (train_loss.avg, nat_acc1.avg)


def test_std(
        test_loader: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.modules.loss._Loss,
        epoch: int, device: str, selector: np.ndarray = None, writer: SummaryWriter = None
    ) -> Union[float, float]:
    """Single test iteration.

    Args:
        test_loader (torch.utils.data.DataLoader): test loader
        model (nn.Module): model to test
        criterion (nn.modules.loss._Loss): loss function
        epoch (int): current epoch
        device (str): device
        selector (np.ndarray): Binary indicator for which samples to select
        writer (SummaryWriter): SummaryWriter

    Returns:
        Union[float, float]: nat_acc top1, nat_acc top5
    """
    if selector is not None:
        assert len(selector) == len(test_loader.dataset), \
            "Number of training samples does not match number of sample weights."

    model.eval()
    test_loss = AverageMeter()
    nat_acc1 = AverageMeter()
    nat_acc5 = AverageMeter()

    pbar = tqdm(test_loader, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = get_indicator_subsample(test_loader, inputs, targets, sample_indices, selector)
            if inputs.size(0) == 0:
                continue

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            nat_accs, _ = accuracy(outputs, targets, topk=(1,5))

            test_loss.update(loss.item(), inputs.size(0))
            nat_acc1.update(nat_accs[0].item(), inputs.size(0))
            nat_acc5.update(nat_accs[1].item(), inputs.size(0))

            pbar.set_description(f'[V] STD-TRAIN epoch=%d, loss=%.4f, acc1=%.4f, acc5=%.4f' % (
                epoch, test_loss.avg, nat_acc1.avg, nat_acc5.avg
            ))

    if writer:
        writer.add_scalar('val/loss', test_loss.avg, epoch)
        writer.add_scalar('val/nat_acc1', nat_acc1.avg, epoch)


    return (test_loss.avg, nat_acc1.avg)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()

    # Setup logging
    out_dir, exp_dir, exp_id, chkpt_path = logging_setup(args, train_mode='std')
    running_chkpt_dir = os.path.join(exp_dir, 'running_chkpt')
    model_name = os.path.splitext(os.path.basename(chkpt_path))[0]
    log_filename = os.path.join(exp_dir, 'train_log.csv')
    log_list = init_logging(args, exp_dir)
    writer = SummaryWriter(log_dir=exp_dir)

    # build dataset
    train_loader, val_loader, test_loader, _, _, num_classes = get_dataloader(
        args, args.dataset, normalize=False, indexed=True, val_set_source='train', val_split=0.1
    )
    test_dataset = args.dataset
    if args.test_dataset:
        _, _, test_loader, _, _, _ = get_dataloader(
            args, args.test_dataset, normalize=False, indexed=True
        )
        test_dataset = args.test_dataset

    label_weights = None
    if args.weighted_loss:
        label_weights = get_label_weights(train_loader.dataset)
        label_weights = torch.from_numpy(label_weights).to(device).float()

    # build model and load checkpoint if given
    model = get_net(args.arch, args.dataset, num_classes, device, normalize=not args.no_normalize, parallel=True)

    # loss, optimizer, lr scheduling
    criterion = nn.CrossEntropyLoss(weight=label_weights)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = get_lr_scheduler(args, args.lr_sched, opt)

    # checkpointing
    start_epoch = 0
    if args.resume:
        if args.load_opt:
            # load optimizer from checkpoint
            model, opt, checkpoint = load_checkpoint(
                args.resume, model, args.arch, args.dataset, device=device,
                normalize=not args.no_normalize, optimizer=opt, parallel=True
            )
        else:
            model, _, checkpoint = load_checkpoint(
                args.resume, model, args.arch, args.dataset, device=device,
                normalize=not args.no_normalize ,optimizer=None, parallel=True
            )
        start_epoch = checkpoint['epoch']

    # write args config
    train_args = vars(args)
    train_args['exp_dir'] = exp_dir
    train_args['chkpt_path'] = chkpt_path
    train_args['logfile'] = log_filename
    write_config(train_args, exp_dir)
    print(f'==> Starting training with config: ', \
        json.dumps(vars(args), default=default_serialization, indent=2)
    )

    logging.info(120 * '=')
    best_acc = 0
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_loss, train_acc = train_std(train_loader, model, criterion, opt, epoch, device, writer=writer)
        val_loss, val_acc = test_std(val_loader, model, criterion, epoch, device, writer=writer)

        # write to logfile
        log_list = log(log_filename, log_list, log_dict=dict(
            epoch=epoch, lr=lr_scheduler.get_last_lr()[0],
            train_loss=train_loss, train_nat_acc=train_acc,
            test_loss=val_loss, test_nat_acc=val_acc
        ))

        if args.test_freq > 0 and epoch % args.test_freq == 0:
            tmp_chkpt_dir = os.path.join(running_chkpt_dir, str(epoch))
            tmp_chkpt_path = os.path.join(tmp_chkpt_dir, f'{model_name}.pt')

            test_accs, _, _, _, _, _ = natural_eval(
                args, model, device, test_loader
            )
            writer.add_scalar(f'test_{test_dataset}/nat_acc1', test_accs[0], epoch)

            if args.running_checkpoint:
                save_checkpoint(
                    tmp_chkpt_path, model, args.arch, args.dataset, epoch, opt, {'nat_prec1': test_accs[0]}
                )

        # checkpointing
        if val_acc > best_acc:
            save_checkpoint(chkpt_path, model, args.arch, args.dataset, epoch, opt, {'nat_prec1': val_acc})
            best_acc = val_acc

        lr_scheduler.step()

    # save final model
    split = os.path.splitext(chkpt_path)
    chkpt_path = f'{split[0]}_last{split[1]}'
    save_checkpoint(chkpt_path, model, args.arch, args.dataset, epoch, opt, {'nat_prec1': val_acc})

    # load best checkpoint model
    model, _, _ = load_checkpoint(
        chkpt_path, net=model, arch=args.arch, dataset=args.dataset,
        device=device, normalize=not args.no_normalize, optimizer=None, parallel=True
    )
    logging.info('Evaluating nat/adv accuracy of best checkpoint model on test set.')
    if not args.adv_norm:
        args.adv_norm = 'Linf'
    test_eps = PERTURBATION_REGIONS[args.adv_norm]
    if args.test_eps:
        test_eps = args.test_eps

    for eps_str in test_eps:
        _, _, _, _, _, nat_pred, _, _, _ = get_acc_rob_indicator(
            args, model, exp_dir, model_name, device, test_loader,
            'test', args.adv_norm, eps_str, args.test_adv_attack,
            use_existing=False, write_log=True, write_report=True
        )
    logging.info(120 * '=')

    writer.close()


if __name__ == '__main__':
    main()

