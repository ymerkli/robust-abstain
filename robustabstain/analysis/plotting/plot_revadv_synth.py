import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import logging
from typing import List, Tuple, Callable

import robustabstain.utils.args_factory as args_factory
from robustabstain.analysis.plotting.utils.decision_region import decision_region_plot_2Ddata
from robustabstain.data.synth.datasets import SYNTH_DATASETS
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.cert import get_acc_cert_indicator
from robustabstain.train.train_adv import train_trades
from robustabstain.train.train_gaussaugm import train_gaussaugm
from robustabstain.train.train_revadv import train_revadv
from robustabstain.train.train_std import train_std
from robustabstain.train.train_revcert import train_revcert
from robustabstain.utils.args_factory import get_full_parser
from robustabstain.utils.checkpointing import get_net
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.paths import get_root_package_dir, FIGURES_DIR
from robustabstain.utils.log import init_logging
from robustabstain.utils.loaders import get_dataloader
from robustabstain.utils.schedulers import get_lr_scheduler


def get_args():
    parser = args_factory.get_parser(
        description='Plots for revadv-abstain trained models.',
        arg_lists=['dataset', 'train-eps', 'noise-sd', 'seed'],
    )

    print('==> argparsing')
    args = parser.parse_args()

    return args


def reinit_train_utils(
        args: object, device: str, num_classes: int, seed: int = 0
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Get reinitialized model, optimizer and LR schedulers.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        device (str): device.
        num_classes (int): Number of classes in dataset.
        seed (int): Random seed to reinitalize.

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]: Model, optimizer, LR scheduler.
    """
    torch.manual_seed(seed)
    model = get_net(args.arch, args.dataset, num_classes, device, normalize=False, parallel=True)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = get_lr_scheduler(args, args.lr_sched, opt)

    return model, opt, lr_scheduler


def train_plot_model(
        args: object, device: str, train_loader: torch.utils.data.DataLoader, num_classes: int
    ) -> None:
    criterion = nn.CrossEntropyLoss()
    root_dir = get_root_package_dir()
    train_eps_float = convert_floatstr(args.train_eps)
    noise_sd_float = convert_floatstr(args.noise_sd)

    # train model using standard training
    model, opt, lr_scheduler = reinit_train_utils(args, device, num_classes, args.seed)
    out_dir = os.path.join(
        root_dir, FIGURES_DIR, 'revadv', args.dataset,
        args.adv_norm, args.train_eps, args.arch, 'std'
    )
    for epoch in range(args.epochs):
        train_std(train_loader, model, criterion, opt, epoch, device)
        lr_scheduler.step()

    decision_region_plot_2Ddata(
        model, device, data=train_loader.dataset.data, data_labels=train_loader.dataset.targets,
        savepath=os.path.join(out_dir, 'train_std')
    )
    get_acc_rob_indicator(
        args, model, out_dir, model_name=f'{args.arch}_std', device=device,
        dataloader=train_loader, eval_set='train', adv_norm=args.adv_norm,
        eps_str=args.train_eps, adv_attack='pgd', write_log=False, write_report=True
    )

    # train model using TRADES training
    model, opt, lr_scheduler = reinit_train_utils(args, device, num_classes, args.seed)
    out_dir = os.path.join(
        root_dir, FIGURES_DIR, 'revadv', args.dataset,
        args.adv_norm, args.train_eps, args.arch, 'adv'
    )
    for epoch in range(args.epochs):
        train_trades(args, model, device, train_loader, opt, train_eps_float, epoch)
        lr_scheduler.step()

    decision_region_plot_2Ddata(
        model, device, data=train_loader.dataset.data, data_labels=train_loader.dataset.targets,
        savepath=os.path.join(out_dir, 'train_trades')
    )
    get_acc_rob_indicator(
        args, model, out_dir, model_name=f'{args.arch}_trades{args.train_eps}',
        device=device, dataloader=train_loader, eval_set='train', adv_norm=args.adv_norm,
        eps_str=args.train_eps, adv_attack='pgd', write_log=False, write_report=True
    )

    # train model using revadv training
    model, opt, lr_scheduler = reinit_train_utils(args, device, num_classes, args.seed)
    out_dir = os.path.join(
        root_dir, FIGURES_DIR, 'revadv', args.dataset,
        args.adv_norm, args.train_eps, args.arch, 'mrevadv'
    )
    for epoch in range(args.epochs):
        train_revadv(
            args, model, device, train_loader, opt, epoch,
            beta=args.revadv_beta, soft=True, variant='mrevadv'
        )
        lr_scheduler.step()
    decision_region_plot_2Ddata(
        model, device, data=train_loader.dataset.data, data_labels=train_loader.dataset.targets,
        savepath=os.path.join(out_dir, 'train_mrevadv')
    )
    get_acc_rob_indicator(
        args, model, out_dir, model_name=f'{args.arch}_mrevadv{args.train_eps}',
        device=device, dataloader=train_loader, eval_set='train', adv_norm=args.adv_norm,
        eps_str=args.train_eps, adv_attack='pgd', write_log=False, write_report=True
    )

    # train model using gaussaugm training
    model, opt, lr_scheduler = reinit_train_utils(args, device, num_classes, args.seed)
    out_dir = os.path.join(
        root_dir, FIGURES_DIR, 'revcert', args.dataset,
        'L2', args.noise_sd, args.arch, 'gaussaugm'
    )
    for epoch in range(args.epochs):
        train_gaussaugm(
            args, model, device, train_loader, criterion, opt, noise_sd_float, epoch
        )
        lr_scheduler.step()

    decision_region_plot_2Ddata(
        model, device, data=train_loader.dataset.data, data_labels=train_loader.dataset.targets,
        savepath=os.path.join(out_dir, 'train_gaussaugm')
    )
    get_acc_cert_indicator(
        args, model, model_dir=out_dir, model_name=f'{args.arch}_gaussaugm{args.noise_sd}',
        device=device, dataloader=train_loader, eval_set='train', eps_str=args.train_eps,
        smooth=True, n_smooth_samples=len(train_loader.dataset), write_report=True
    )

    # train model using revcert training
    model, opt, lr_scheduler = reinit_train_utils(args, device, num_classes, args.seed)
    out_dir = os.path.join(
        root_dir, FIGURES_DIR, 'revcert', args.dataset,
        'L2', args.noise_sd, args.arch, 'revcertrad'
    )
    for epoch in range(args.epochs):
        train_revcert(
            args, model, device, train_loader, opt, epoch,
            beta=0.7, variant='revcertrad'
        )
        lr_scheduler.step()

    decision_region_plot_2Ddata(
        model, device, data=train_loader.dataset.data, data_labels=train_loader.dataset.targets,
        savepath=os.path.join(out_dir, 'train_revcertrad')
    )
    get_acc_cert_indicator(
        args, model, model_dir=out_dir, model_name=f'{args.arch}_revcertrad{args.noise_sd}',
        device=device, dataloader=train_loader, eval_set='train', eps_str=args.train_eps,
        smooth=True, n_smooth_samples=len(train_loader.dataset), write_report=True
    )



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()
    init_logging(args)

    train_eps = args.train_eps if args.train_eps else '0.1'
    noise_sd = args.noise_sd if args.noise_sd else '0.2'
    datasets = [args.dataset] if args.dataset else SYNTH_DATASETS
    for dataset in datasets:
        parser = get_full_parser()
        args = parser.parse_args([
            '--dataset', dataset,
            '--arch', 'mininet',
            '--adv-norm', 'Linf',
            '--train-batch', '10',
            '--train-eps', train_eps,
            '--noise-sd', noise_sd,
            '--smoothing-sigma', noise_sd,
            '--epochs', '20',
            '--lr', '0.01',
            '--revadv-beta', '0.25'
        ])

        train_loader, _, _, _, _, num_classes = get_dataloader(
            args, args.dataset, indexed=True, root_prefix='../', shuffle_train=True
        )

        train_plot_model(args, device, train_loader, num_classes)


if __name__ == '__main__':
    main()