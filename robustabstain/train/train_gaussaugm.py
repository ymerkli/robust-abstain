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
import logging
from tqdm import tqdm
from typing import Tuple

import robustabstain.utils.args_factory as args_factory
from robustabstain.loss.noise import noise_loss
from robustabstain.eval.cert import get_acc_cert_indicator
from robustabstain.utils.checkpointing import save_checkpoint, load_checkpoint, get_net
from robustabstain.utils.loaders import get_dataloader, get_indicator_subsample
from robustabstain.utils.log import write_config, default_serialization, logging_setup, init_logging, log
from robustabstain.utils.metrics import accuracy, AverageMeter, adv_accuracy
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.schedulers import get_lr_scheduler


def get_args():
    parser = args_factory.get_parser(
        description='Gaussian noise augmentation training.',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS,
            args_factory.ATTACK_ARGS, args_factory.SMOOTHING_ARGS
        ],
        required_args=['dataset', 'arch', 'epochs', 'noise-sd']
    )
    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    return args


def train_gaussaugm(
        args: object, model: nn.Module, device: str, train_loader: torch.utils.data.DataLoader,
        criterion: nn.modules.loss._Loss, opt: optim.Optimizer, noise_sd: float, epoch: int,
        selector: np.ndarray = None, writer: SummaryWriter = None
    ) -> Tuple[float, float, float]:
    """Single training iteration for gaussian augmentation training.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module
        device (str): device
        train_loader (torch.utils.data.DataLoader): PyTorch loader with train data
        criterion (nn.modules.loss._Loss): loss function
        opt (optim.Optimizer): optimizer
        noise_sd (float): Gaussian noise std
        epoch (int): current epoch
        selector (np.ndarray): Binary indicator for which samples to select
        writer (SummaryWriter): SummaryWriter

    Returns:
        Tuple[float, float, float]: train loss, nat. accuracy, adv. (noise) accuracy
    """
    if selector is not None:
        assert len(selector) == len(train_loader.dataset), \
            "Number of training samples does not match number of sample weights."

    model.train()
    train_loss = AverageMeter()
    nat_acc1 =  AverageMeter()
    nat_acc5 = AverageMeter()
    adv_acc1 =  AverageMeter() # adv accuracy in this case is the accuracy on noisy inputs

    pbar = tqdm(train_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = get_indicator_subsample(train_loader, inputs, targets, sample_indices, selector)
        if inputs.size(0) == 0:
            continue

        opt.zero_grad()

        logits_nat = model(inputs)
        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs, device=device) * noise_sd
        logits_noise = model(inputs)
        loss = criterion(logits_noise, targets)

        loss.backward()
        opt.step()

        # measure accuracy
        nat_accs, _ = accuracy(logits_nat, targets, topk=(1,5))
        adv_acc, _ = adv_accuracy(logits_noise, logits_nat, targets)
        train_loss.update(loss.item(), inputs.size(0))
        nat_acc1.update(nat_accs[0].item(), inputs.size(0))
        nat_acc5.update(nat_accs[1].item(), inputs.size(0))
        adv_acc1.update(adv_acc.item(), inputs.size(0))

        pbar.set_description(
            '[T] GAUSS-AUGM epoch={}, loss={:.4f}, nat_acc1={:.4f}, adv_acc1={:.4f} (noise_sd={:.4f})'.format(
                epoch, train_loss.avg, nat_acc1.avg, adv_acc1.avg, noise_sd)
        )

    if writer:
        writer.add_scalar('loss/train', train_loss.avg, epoch)
        writer.add_scalar('nat_acc1/train', nat_acc1.avg, epoch)
        writer.add_scalar('adv_acc1/train', adv_acc1.avg, epoch)

    return (train_loss.avg, nat_acc1.avg, adv_acc1.avg)


def test_gaussaugm(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        criterion: nn.modules.loss._Loss, noise_sd: float, epoch: int,
        selector: np.ndarray = None, writer: SummaryWriter = None
    ) -> Tuple[float, float, float]:
    """Evaluate natural accuracies on gaussian augmented test data.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module
        device (str): device
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data
        criterion (nn.modules.loss._Loss): loss function
        noise_sd (float): Gaussian noise std
        epoch (int): current epoch
        selector (np.ndarray): Binary indicator for which samples to select
        writer (torch.SummaryWriter): SummaryWriter

    Returns:
        Tuple[float, float, float]: test loss, nat. accuracy, adv. (noise) accuracy
    """
    if selector is not None:
        assert len(selector) == len(test_loader.dataset), \
            "Number of training samples does not match number of sample weights."

    model.eval()
    test_loss = AverageMeter()
    nat_acc1 =  AverageMeter()
    nat_acc5 = AverageMeter()
    adv_acc1 =  AverageMeter() # adv accuracy in this case is the accuracy on noisy inputs

    with torch.no_grad():
        pbar = tqdm(test_loader, dynamic_ncols=True)
        for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = get_indicator_subsample(test_loader, inputs, targets, sample_indices, selector)
            if inputs.size(0) == 0:
                continue

            logits_nat = model(inputs)
            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device=device) * noise_sd
            logits_noise = model(inputs)
            loss = criterion(logits_noise, targets)

            # measure accuracy
            nat_accs, _ = accuracy(logits_nat, targets, topk=(1,5))
            adv_acc, _ = adv_accuracy(logits_noise, logits_nat, targets)
            test_loss.update(loss.item(), inputs.size(0))
            nat_acc1.update(nat_accs[0].item(), inputs.size(0))
            nat_acc5.update(nat_accs[1].item(), inputs.size(0))
            adv_acc1.update(adv_acc.item(), inputs.size(0))

            pbar.set_description(
                '[V] GAUSS-AUGM epoch={}, loss={:.4f}, nat_acc1={:.4f}, adv_acc1={:.4f} (noise_sd={:.4f})'.format(
                    epoch, test_loss.avg, nat_acc1.avg, adv_acc1.avg, noise_sd)
            )

    if writer:
        writer.add_scalar('loss/test', test_loss.avg, epoch)
        writer.add_scalar('nat_acc1/test', nat_acc1.avg, epoch)
        writer.add_scalar('adv_acc1/test', adv_acc1.avg, epoch)

    return (test_loss.avg, nat_acc1.avg, adv_acc1.avg)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()
    noise_sd_float = convert_floatstr(args.noise_sd) if args.noise_sd is not None else None

    # log directories
    train_mode = 'augm'
    out_dir, exp_dir, exp_id, chkpt_path_best = logging_setup(args, train_mode=train_mode)

    # setup logging
    log_filename = os.path.join(exp_dir, 'train_log.csv')
    log_list = init_logging(args, exp_dir)
    writer = SummaryWriter(log_dir=exp_dir)

    # build dataset
    train_loader, _, test_loader, _, _, num_classes = get_dataloader(args, args.dataset, normalize=False, indexed=True)

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
        start_epoch = checkpoint['epoch']

    # write args config
    train_args = vars(args)
    train_args['exp_dir'] = exp_dir
    train_args['chkpt_path'] = chkpt_path_best
    train_args['logfile'] = log_filename
    write_config(train_args, exp_dir)
    print(f'==> Starting training with config: ', \
        json.dumps(train_args, default=default_serialization, indent=2)
    )

    # update model name
    model_name = os.path.splitext(os.path.basename(chkpt_path_best))[0]

    best_loss, best_acc = float('inf'), -1
    for epoch in range(start_epoch, start_epoch+args.epochs):
        # train
        # gaussian augmented training
        train_loss, train_nat_acc, train_adv_acc = train_gaussaugm(
            args, model, device, train_loader, criterion, opt, noise_sd_float, epoch
        )

        # test
        should_log = (epoch % args.val_freq == 0) or (epoch == args.epochs)
        if should_log:
            # gaussian augmented test
            test_loss, test_nat_acc, test_adv_acc = test_gaussaugm(
                args, model, device, test_loader, criterion, noise_sd_float, epoch
            )
            test_acc = test_nat_acc

            # write to logfile
            log_list = log(log_filename, log_list, log_dict=dict(
                epoch=epoch, lr=lr_scheduler.get_last_lr()[0], train_loss=train_loss,
                train_nat_acc=train_nat_acc,  train_adv_acc=train_adv_acc, test_loss=test_loss,
                test_nat_acc=test_nat_acc, test_adv_acc=test_adv_acc
            ))

            # checkpointing
            should_checkpoint = test_acc > best_acc
            if should_checkpoint:
                add_state = {'nat_prec1': test_nat_acc, 'adv_prec1': test_adv_acc}
                save_checkpoint(chkpt_path_best, model, args.arch, args.dataset, epoch, opt, add_state)
                best_loss, best_acc = test_loss, test_acc

        lr_scheduler.step()

    split = os.path.splitext(chkpt_path_best)
    add_state = {'nat_prec1': test_nat_acc, 'adv_prec1': test_adv_acc}
    save_checkpoint(f'{split[0]}_last{split[1]}', model, args.arch, args.dataset, epoch, opt, add_state)

    # load best checkpoint model
    model, _, _ = load_checkpoint(
        chkpt_path_best, net=model, arch=args.arch, dataset=args.dataset, device=device,
        normalize=not args.no_normalize, optimizer=None, parallel=True
    )

    if len(args.test_eps) == 1 and args.adv_norm is not None:
        if not args.smoothing_sigma:
            args.smoothing_sigma = args.noise_sd

        # get accuracy and robustness indicators on testset
        logging.info('Evaluating compositional accuracy of best checkpoint models on test set.')
        (
            nat_acc1, cert_acc1, cert_inacc,
            is_acc_test, is_cert_test, _, _
        ) = get_acc_cert_indicator(
                args, model, exp_dir, model_name, device, test_loader,
                eval_set='test', eps_str=args.test_eps[0], smooth=True, n_smooth_samples=500,
                use_existing=True, write_log=True, write_report=True
        )
        logging.info('Model test accuracy ({} eps={}): nat_acc={:.4f}, ' \
            'cert_acc={:.4f}, cert_inacc={:.4f}'.format(
                args.adv_norm, args.test_eps[0], nat_acc1,
                cert_acc1, cert_inacc
        ))

    writer.close()


if __name__ == '__main__':
    main()