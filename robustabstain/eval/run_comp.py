import setGPU
import torch
import torch.nn as nn

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

import robustabstain.utils.args_factory as args_factory
from robustabstain.eval.comp import get_comp_indicator, get_comp_key
from robustabstain.eval.log import write_eval_report
from robustabstain.utils.loaders import get_dataloader
from robustabstain.utils.log import init_logging


def get_args():
    parser = args_factory.get_parser(
        description='Baseline model evaluation',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS, args_factory.ATTACK_ARGS,
            args_factory.COMP_ARGS, args_factory.SMOOTHING_ARGS
        ],
        required_args=['dataset', 'trunk-models', 'branch-model']
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    if args.adv_attack == 'autoattack':
        # autoattack does not take step size or number of steps. Set this args to None to clarify the logs
        args.test_att_n_steps = None
        args.test_att_step_size = None

    if any('smo' in s for s in args.evals) or args.smooth:
        assert args.smoothing_sigma is not None, 'Specify --smoothing-sigma for smoothing evaluation'

    return args


def eval_comp(
        args: object, branch_model_path: str, trunk_model_paths: List[str], device: str,
        test_loader: torch.utils.data.DataLoader, comp_dir: str = None,
        branch_model_logs: List[str] = [], trunk_model_logs: List[str] = []
    ) -> None:
    """Evalate compositional architecture consisting of a branch model and a trunk model.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        branch_model_path (str): Path to branch model checkpoint.
        trunk_model_path (str): Path to trunk model checkpoint.
        device (str): device.
        test_loader (torch.utils.data.DataLoader): test_loader
        comp_dir (str, optional): Compositional experiment directory. Defaults to None.
        branch_model_logs (List[str], optional): Trunk model log. Defaults to [].
        trunk_model_logs (List[str], optional): Branch model log. Defaults to [].
    """
    comp_accs = {}
    for eps_str in args.test_eps:
        comp_nat_acc, comp_adv_acc, _, _, _ = get_comp_indicator(
            args, branch_model_path, trunk_model_paths, device, test_loader,
            args.eval_set, eps_str, abstain_method=args.selector,
            conf_threshold=args.conf_threshold, use_existing=True
        )
        comp_cert_acc = None #TODO

        comp_key = get_comp_key(branch_model_path, trunk_model_paths)
        selector_key = args.selector
        if selector_key == 'conf':
            selector_key += str(args.conf_threshold)

        comp_accs[eps_str] = {
            comp_key: {
                selector_key: {
                    'comp_nat_acc': comp_nat_acc,
                    'comp_adv_acc': comp_adv_acc,
                    #'comp_cert_acc': comp_cert_acc,
                    'branch_model': args.branch_model,
                    'trunk_model': args.trunk_models
        }}}

    if not args.no_eval_report:
        out_dir = args.comp_dir if args.comp_dir else os.path.dirname(args.branch_model)
        write_eval_report(args, out_dir=out_dir, comp_accs=comp_accs)


def main():
    """Evaluate accuracies of a given model.
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

    eval_comp(args, args.branch_model, args.trunk_models, device, loader_to_eval, args.branch_model_log, args.trunk_model_log)


if __name__ == '__main__':
    main()

