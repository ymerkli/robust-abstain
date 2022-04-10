import torch
import torch.nn as nn

import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict

import robustabstain.utils.args_factory as args_factory
from robustabstain.ace.deepTrunk_networks import MyDeepTrunkNet
from robustabstain.ace.networks import translate_net_name
from robustabstain.eval.ace import ace_eval, build_ace_net, get_ace_indicator
from robustabstain.eval.log import write_sample_log, write_eval_report
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.loaders import get_dataloader, get_rel_sample_indices
from robustabstain.utils.metrics import AverageMeter
from robustabstain.utils.log import init_logging
from robustabstain.utils.paths import eval_attack_log_filename


def get_args() -> object:
    """Argparser

    Returns:
        object: object subclass exposing 'setattr` and 'getattr'
    """
    parser = args_factory.get_parser(
        description='Baseline model evaluation',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS, args_factory.ATTACK_ARGS,
            args_factory.COMP_ARGS, args_factory.ACE_ARGS
        ],
        required_args=['dataset', 'model', 'adv-norm', 'test-eps']
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    return args


def eval_ace(
        args: object, ace_cnet_path: str, device: str, test_loader: torch.utils.data.DataLoader,
        test_eps: List[str], log_file: str = None, use_exist_log: bool = False,
        no_report: bool = False, no_log: bool = False
    ) -> None:
    """Evaluate natural accuracy, empirical robustness, certified robustness and gate selection
    of an ACE network with gate network selection.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        ace_cnet_path (str): Filepath to a exported, fully combined ACE model.
        device (str): device
        test_loader (torch.utils.data.DataLoader): Dataloader containing test data
        test_eps (List[str]): List of (stringified) perturbation region epsilons to evaluate.
        log_file (str, optional): Path to per-sample log file. Defaults to None.
        use_exist_log (bool, optional): If set, existing log file is used (if available). Defaults to False.
        no_report (bool, optional): If set, no report file is written. Defaults to False.
        no_log (bool, optional): If set, no per-sample log file is written. Defaults to False.
    """
    model_dir = os.path.dirname(ace_cnet_path)
    model_name = os.path.splitext(os.path.basename(ace_cnet_path))[0]
    log_name = eval_attack_log_filename(args.eval_set, args.dataset, args.adv_norm, 'pgd')
    if log_file is not None:
        log_name = os.path.basename(log_file)
    else:
        log_file = os.path.join(model_dir, log_name)

    # build ACE model
    dTNet = build_ace_net(args, ace_cnet_path, device)

    # evaluate ACE architecture
    for eps_str in test_eps:
        # get ACE evaluation indicators by checking logfiles or evaluating from scratch
        _, _, _, is_acc, is_rob, is_cert, is_select, \
            select_rob, select_cert, nat_preds, indices = get_ace_indicator(
                args, dTNet, model_dir, model_name, device, test_loader, args.eval_set,
                args.adv_norm, eps_str, use_existing=use_exist_log, write_log=True,
                write_report=True
        )
        nat_acc1 = round(100.0 * np.average(is_acc), 2)
        adv_acc1 = round(100.0 * np.average(is_rob & is_acc), 2)
        cert_acc1 = round(100.0 * np.average(is_cert & is_acc), 2)

        if not no_report and args.eval_set == 'test':
            write_eval_report(
                args, out_dir=model_dir, model_path=ace_cnet_path,
                nat_accs=[nat_acc1], adv_accs={eps_str: adv_acc1},
                adv_attack='pgd', dcert_accs={eps_str: cert_acc1}
            )

        if not no_log:
            write_sample_log(
                model_name, model_dir, args.dataset, args.eval_set, args.adv_norm,
                adv_attack='pgd', indices=indices, is_acc=is_acc, preds=nat_preds,
                is_rob=is_rob, is_cert=is_cert, is_select=is_select,
                select_rob=select_rob, select_cert=select_cert, eps=eps_str
            )


def main():
    """Evaluate ACE model.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = get_args()

    # init logging
    init_logging(args)

    # Build dataset: load unnormalized data. Normalization is done in model.
    train_loader, _, test_loader, _, _, _ = get_dataloader(
        args, args.dataset, normalize=False, indexed=True, shuffle_train=False, val_split=0.0
    )
    loader_to_eval = test_loader
    if args.eval_set == 'train':
        loader_to_eval = train_loader
    del train_loader, test_loader

    eval_ace(
        args, args.model, device, loader_to_eval, args.test_eps, use_exist_log=args.use_exist_log,
        no_report=args.no_eval_report, no_log=args.no_sample_log
    )


if __name__ == '__main__':
    main()