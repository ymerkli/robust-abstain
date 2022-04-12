import torch
import torch.nn as nn

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Union

import robustabstain.utils.args_factory as args_factory
from robustabstain.abstain.selector import abstain_selector
from robustabstain.analysis.plotting.utils.colors import COLOR_PAIRS_1 as COLOR_PAIRS_BASELINE
from robustabstain.analysis.plotting.utils.colors import COLOR_PAIRS_2 as COLOR_PAIRS_OURS
from robustabstain.analysis.plotting.utils.colors import normalize_color
from robustabstain.analysis.plotting.utils.helpers import (
    get_model_paths, check_is_abstain_trained, check_is_ace_trained,
    check_det_cert_method, pair_sorted, three_sorted, update_ax_lims)
from robustabstain.analysis.plotting.utils.model_measures import (
    conf_model_measures, adv_robind_model_measures, ace_model_measures, get_running_chkpt_vals)
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.comp import compositional_accuracy, get_comp_indicator, get_comp_key
from robustabstain.utils.checkpointing import load_checkpoint
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.helpers import pretty_floatstr, loggable_floatstr, multiply_eps
from robustabstain.utils.latex import latex_norm
from robustabstain.utils.loaders import get_dataloader
from robustabstain.utils.log import init_logging
from robustabstain.utils.paths import get_root_package_dir, FIGURES_DIR
from robustabstain.utils.regex import EPS_STR_RE
from robustabstain.utils.transforms import DATA_AUG


SELECTOR_ABREVS = {
    'conf': 'SR',
    'emprobind': 'ERI',
    'certrobind': 'CRI',
    'ace': 'SN',
    'noabst': 'non-abstain',
    'nocomp': 'non-comp'
}

TRAINING_ABREVS = {
    'ace': 'ACE',
    'era': 'Abstain',
    'trades': 'Trades',
    'std': 'Std',
    'base': 'Base'
}

DATAAUG_ABREVS = {
    'autoaugment': 'AA',
    'stdaug': 'SA'
}

METRICS_NAMES = {
    'natacc': 'Natural Accuracy $\mathcal{R}_{nat}$ [%]',
    'robacc': 'Robust Accuracy $\mathcal{R}_{rob}$ [%]',
    'robinacc': 'Robust Inaccuracy\n$\mathcal{R}_{robinacc}$ [%]',
    'commitrate': 'Robust Coverage $\Phi_{rob}$ [%]',
    'commitprec': 'Robust Accuracy $\mathcal{R}_{rob}$ [%]',
    'compnatacc': 'Natural Accuracy $\mathcal{R}_{nat}$ [%]',
    'comprobacc': 'Robust Accuracy $\mathcal{R}_{rob}$ [%]',
    'misselect': 'Misselection Rate [%]'
}


def get_args():
    parser = args_factory.get_parser(
        description='Plots for revadv-abstain trained models.',
        arg_lists=[
            args_factory.TESTING_ARGS, args_factory.LOADER_ARGS, args_factory.ATTACK_ARGS,
            args_factory.SMOOTHING_ARGS, args_factory.COMP_ARGS, args_factory.ACE_ARGS
        ],
        required_args=['dataset', 'train-eps', 'test-eps', 'adv-norm', 'trunk-models']
    )
    parser.add_argument(
        '--baseline-model', type=str, required=True, help='Path to baseline model.'
    )
    parser.add_argument(
        '--baseline-train-eps', type=str, required=True, help='Perturbation region for which the baseline was trained.'
    )
    parser.add_argument(
        '--branch-models', type=str, nargs='+', required=True, help='Branch models to plot.'
    )
    parser.add_argument(
        '--branch-models-predict', type=str, nargs='+', required=False, help='Branch models that are evaluated to predict.'
    )
    parser.add_argument(
        '--branch-model-id', type=str, required=True, help='Parent identifier over all branch models.'
    )
    parser.add_argument(
        '--ace-train-eps', type=str, help='Perturbation region for which the ACE model was trained.'
    )
    parser.add_argument(
        '--ace-model-id', type=str, help='Parent identifier over all ACE models.'
    )
    parser.add_argument(
        '--conf-baseline', action='store_true', help='If set, confidence based abstain is added as another baseline.'
    )
    parser.add_argument(
        '--na-baseline', action='store_true', help='If set, a standard model is used as additional abstain baseline. Resulting model has 100% comit rate.'
    )
    parser.add_argument(
        '--na-baseline-comp', action='store_true', help='If set, a standard model is used as baseline in comp plots.'
    )
    parser.add_argument(
        '--no-comp-ace', action='store_true', help='If set, ACE models are not added to compositional plots.'
    )
    parser.add_argument(
        '--no-comp-conf', action='store_true', help='If set, conf (softmax response) models are not added to compositional plots.'
    )
    parser.add_argument(
        '--plot-running-checkpoints', action='store_true', help='If set, each running checkpoint from training is plotted.'
    )
    parser.add_argument(
        '--set-title', action='store_true', help='If set, plots are given a title.'
    )
    parser.add_argument(
        '--only-comp', action='store_true', help='If set, only compositional plots are created.'
    )
    parser.add_argument(
        '--show-ace-cert', action='store_true', help='If set, only compositional plots are created.'
    )
    parser.add_argument(
        '--eval-2xeps', action='store_true', help='If set, RI abstain is done with double perturbation region.'
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)
    args.baseline_train_eps = loggable_floatstr(args.baseline_train_eps)
    if args.ace_train_eps:
        args.ace_train_eps = loggable_floatstr(args.ace_train_eps)
    assert len(args.test_eps) == 1, 'Error: specify 1 test-eps'

    return args


def get_train_eps(args: object, model_path: str) -> str:
    is_abstain_trained = check_is_abstain_trained(model_path)
    is_trades_trained = ('trades' in model_path) and not is_abstain_trained
    is_ace_trained = check_is_ace_trained(model_path)
    is_baseline = model_path == args.baseline_model
    if is_abstain_trained:
        match = re.search(rf"mra({EPS_STR_RE})", model_path)
        if match:
            return match.group(1)
    elif is_trades_trained:
        match = re.search(rf"trades({EPS_STR_RE})", model_path)
        if match:
            return match.group(1)
    elif is_ace_trained:
        return args.ace_train_eps
    elif is_baseline:
        return args.baseline_train_eps
    raise ValueError(f'Error: no training eps to match in {model_path}')


def plot(
        args: object, test_loader: torch.utils.data.DataLoader, device: str,
        annotate_ts: bool = False, sep_branchpred: bool = False
    ) -> None:
    """Create and export all plots on given datasplit.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data.
        device (str): device.
        annotate_ts (bool, optional): If true, annotate abstain models plot points with timestamps.
    """
    pretty_norm = latex_norm(args.adv_norm)
    pretty_norm_p = latex_norm(args.adv_norm, no_l=True)
    pretty_test_eps = pretty_floatstr(args.test_eps[0])
    test_eps2x = multiply_eps(args.test_eps[0], 2)

    # setup plot of compositional nat vs adv adccuracies robustness indicator and confidence thresholding
    comp_nat_adv_acc_fig, comp_nat_adv_acc_ax = plt.subplots()
    comp_nat_adv_acc_ax.set_xlabel(METRICS_NAMES['compnatacc'])
    comp_nat_adv_acc_ax.set_ylabel(METRICS_NAMES['comprobacc'])
    comp_nat_adv_acc_xlim, comp_nat_adv_acc_ylim = [100, 0], [100, 0]
    if args.set_title:
        comp_nat_adv_acc_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of compositional nat vs adv adccuracies robustness indicator and confidence thresholding and ACE
    comp_nat_adv_acc_ace_fig, comp_nat_adv_acc_ace_ax = plt.subplots()
    comp_nat_adv_acc_ace_ax.set_xlabel(METRICS_NAMES['compnatacc'])
    comp_nat_adv_acc_ace_ax.set_ylabel(METRICS_NAMES['comprobacc'])
    comp_nat_adv_acc_ace_xlim, comp_nat_adv_acc_ace_ylim = [100, 0], [100, 0]
    if args.set_title:
        comp_nat_adv_acc_ace_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of compositional nat vs adv adccuracies for robustness indicator and confidence thresholding and ACE
    comp_nat_adv_acc_ace_joined_fig, comp_nat_adv_acc_ace_joined_ax = plt.subplots(nrows=1, ncols=2, figsize=(7,5))
    comp_nat_adv_acc_ace_joined_ax[0].set_xlabel(METRICS_NAMES['compnatacc'])
    comp_nat_adv_acc_ace_joined_ax[0].set_ylabel(METRICS_NAMES['comprobacc'])
    comp_nat_adv_acc_ace_joined_ax[1].set_xlabel(METRICS_NAMES['compnatacc'])
    comp_nat_adv_acc_ace_joined_ax[1].set_ylabel(METRICS_NAMES['comprobacc'])
    comp_nat_adv_acc_ace_joined_xlim, comp_nat_adv_acc_ace_joined_ylim = [[100, 0], [100, 0]], [[100, 0], [100, 0]]
    if args.set_title:
        comp_nat_adv_acc_ace_joined_fig.suptitle(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of compositional nat vs adv adccuracies robustness indicator and confidence thresholding BUT with non-compositional baselines
    nacomp_nat_adv_acc_fig, nacomp_nat_adv_acc_ax = plt.subplots()
    nacomp_nat_adv_acc_ax.set_xlabel(METRICS_NAMES['compnatacc'])
    nacomp_nat_adv_acc_ax.set_ylabel(METRICS_NAMES['comprobacc'])
    nacomp_nat_adv_acc_xlim, nacomp_nat_adv_acc_ylim = [100, 0], [100, 0]
    if args.set_title:
        nacomp_nat_adv_acc_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of compositional nat vs adv adccuracies for robustness indicator and confidence thresholding and ACE + non-comp baselines
    cnacomp_nat_adv_acc_ace_joined_fig, cnacomp_nat_adv_acc_ace_joined_ax = plt.subplots(nrows=1, ncols=2, figsize=(7,5))
    cnacomp_nat_adv_acc_ace_joined_ax[0].set_xlabel(METRICS_NAMES['compnatacc'])
    cnacomp_nat_adv_acc_ace_joined_ax[0].set_ylabel(METRICS_NAMES['comprobacc'])
    cnacomp_nat_adv_acc_ace_joined_ax[1].set_xlabel(METRICS_NAMES['compnatacc'])
    cnacomp_nat_adv_acc_ace_joined_ax[1].set_ylabel(METRICS_NAMES['comprobacc'])
    cnacomp_nat_adv_acc_ace_joined_xlim, cnacomp_nat_adv_acc_ace_joined_ylim = [[100, 0], [100, 0]], [[100, 0], [100, 0]]
    if args.set_title:
        cnacomp_nat_adv_acc_ace_joined_fig.suptitle(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of branch model commit precision vs commit rate when using robustness indicator selector
    robind_branch_commit_prec_rate_fig, robind_branch_commit_prec_rate_ax = plt.subplots()
    robind_branch_commit_prec_rate_ax.set_xlabel(METRICS_NAMES['commitprec'])
    robind_branch_commit_prec_rate_ax.set_ylabel(METRICS_NAMES['commitrate'])
    robind_branch_commit_prec_rate_xlim, robind_branch_commit_prec_rate_ylim = [100, 0], [100, 0]
    if args.set_title:
        robind_branch_commit_prec_rate_ax.set_title(
            f"{args.dataset} {args.branch_model_id} Empirical Robustness Indicator ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )
    # setup plot of branch model commit precision vs commit rate for BOTH robustness indicator and confidence thresholding
    branch_commit_prec_rate_fig, branch_commit_prec_rate_ax = plt.subplots()
    branch_commit_prec_rate_ax.set_xlabel(METRICS_NAMES['commitrate'])
    branch_commit_prec_rate_ax.set_ylabel(METRICS_NAMES['commitprec'])
    branch_commit_prec_rate_xlim, branch_commit_prec_rate_ylim = [100, 0], [100, 0]
    if args.set_title:
        branch_commit_prec_rate_ax.set_title(
            f"{args.dataset} {args.branch_model_id} Empirical Robustness Indicator ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of branch model commit precision vs commit rate for robustness indicator and confidence thresholding and ACE
    branch_commit_prec_rate_ace_fig, branch_commit_prec_rate_ace_ax = plt.subplots()
    branch_commit_prec_rate_ace_ax.set_xlabel(METRICS_NAMES['commitrate'])
    branch_commit_prec_rate_ace_ax.set_ylabel(METRICS_NAMES['commitprec'])
    branch_commit_prec_rate_ace_xlim, branch_commit_prec_rate_ace_ylim = [100, 0], [100, 0]
    if args.set_title:
        branch_commit_prec_rate_ace_ax.set_title(
            f"{args.dataset} {args.branch_model_id} Empirical Robustness Indicator ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of branch model commit precision vs commit rate for robustness indicator and confidence thresholding and ACE
    branch_commit_prec_rate_ace_joined_fig, branch_commit_prec_rate_ace_joined_ax = plt.subplots(nrows=1, ncols=2, figsize=(7,5))
    branch_commit_prec_rate_ace_joined_ax[0].set_xlabel(METRICS_NAMES['commitrate'])
    branch_commit_prec_rate_ace_joined_ax[0].set_ylabel(METRICS_NAMES['commitprec'])
    branch_commit_prec_rate_ace_joined_ax[1].set_xlabel(METRICS_NAMES['commitrate'])
    branch_commit_prec_rate_ace_joined_ax[1].set_ylabel(METRICS_NAMES['commitprec'])
    branch_commit_prec_rate_ace_joined_xlim, branch_commit_prec_rate_ace_joined_ylim = [[100, 0], [100, 0]], [[100, 0], [100, 0]]
    if args.set_title:
        branch_commit_prec_rate_ace_joined_fig.suptitle(
            f"{args.dataset} {args.branch_model_id} Empirical Robustness Indicator ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of branch model (non-abstain) robust accuracy and robust inaccuracy
    branch_robacc_robinacc_fig, branch_robacc_robinacc_ax = plt.subplots()
    branch_robacc_robinacc_ax.set_xlabel(METRICS_NAMES['robinacc'])
    branch_robacc_robinacc_ax.set_ylabel(METRICS_NAMES['robacc'])
    branch_robacc_robinacc_xlim, branch_robacc_robinacc_ylim = [100, 0], [100, 0]
    if args.set_title:
        branch_robacc_robinacc_fig.set_title(
            f"{args.dataset} {args.branch_model_id} Empirical Robustness ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # summarize all figures and axes
    width, height = 3, 3.5
    width_w, height_w = 12, 3 # wide plots
    width_s, height_s = 1, 3 # small plots
    plots = [
        {
            'fig': comp_nat_adv_acc_fig, 'ax': comp_nat_adv_acc_ax,
            'xlim': comp_nat_adv_acc_xlim, 'ylim': comp_nat_adv_acc_ylim,
            'name': 'comp_nat_adv_acc', 'size': (width, height)
        },
        {
            'fig': nacomp_nat_adv_acc_fig, 'ax': nacomp_nat_adv_acc_ax,
            'xlim': nacomp_nat_adv_acc_xlim, 'ylim': nacomp_nat_adv_acc_ylim,
            'name': 'nacomp_nat_adv_acc', 'size': (width, height)
        },
        {
            'fig': cnacomp_nat_adv_acc_ace_joined_fig, 'ax': cnacomp_nat_adv_acc_ace_joined_ax,
            'xlim': cnacomp_nat_adv_acc_ace_joined_xlim, 'ylim': cnacomp_nat_adv_acc_ace_joined_ylim,
            'name': 'cnacomp_nat_adv_acc_ace_joined', 'size': (width_w, height_w)
        },
        {
            'fig': comp_nat_adv_acc_ace_fig, 'ax': comp_nat_adv_acc_ace_ax,
            'xlim': comp_nat_adv_acc_ace_xlim, 'ylim': comp_nat_adv_acc_ace_ylim,
            'name': 'comp_nat_adv_acc_ace', 'size': (width, height)
        },
        {
            'fig': comp_nat_adv_acc_ace_joined_fig, 'ax': comp_nat_adv_acc_ace_joined_ax,
            'xlim': comp_nat_adv_acc_ace_joined_xlim, 'ylim': comp_nat_adv_acc_ace_joined_ylim,
            'name': 'comp_nat_adv_acc_ace_joined', 'size': (width_w, height_w)
        },
        {
            'fig': branch_commit_prec_rate_fig, 'ax': branch_commit_prec_rate_ax,
            'xlim': branch_commit_prec_rate_xlim, 'ylim': branch_commit_prec_rate_ylim,
            'name': 'commit_prec_rate', 'size': (width, height)
        },
        {
            'fig': branch_commit_prec_rate_ace_fig, 'ax': branch_commit_prec_rate_ace_ax,
            'xlim': branch_commit_prec_rate_ace_xlim, 'ylim': branch_commit_prec_rate_ace_ylim,
            'name': 'commit_prec_rate_ace', 'size': (width, height)
        },
        {
            'fig': branch_commit_prec_rate_ace_joined_fig, 'ax': branch_commit_prec_rate_ace_joined_ax,
            'xlim': branch_commit_prec_rate_ace_joined_xlim, 'ylim': branch_commit_prec_rate_ace_joined_ylim,
            'name': 'commit_prec_rate_ace_joined', 'size': (width_w, height_w)
        },
        {
            'fig': branch_robacc_robinacc_fig, 'ax': branch_robacc_robinacc_ax,
            'xlim': branch_robacc_robinacc_xlim, 'ylim': branch_robacc_robinacc_ylim,
            'name': 'branch_robacc_robinacc', 'size': (width_s, height_s)
        }
    ]

    # eval trunk model(s)
    _, _, trunk_is_acc, trunk_is_acc_adv, trunk_is_rob = get_comp_indicator(
        args, branch_model_path=args.trunk_models[0],
        trunk_model_paths=args.trunk_models[1:], device=device,
        dataloader=test_loader, eval_set='test', eps_str=args.test_eps[0],
        abstain_method='rob', use_existing=True
    )

    # extract branchpred model paths
    branchpred_path_noaug, branchpred_path_aug = None, None
    if args.eval_2xeps and sep_branchpred:
        assert len(args.branch_models_predict) == 2
        branchpred_path_aug = [p for p in args.branch_models_predict if any(aug in p for aug in DATA_AUG)]
        assert len(branchpred_path_aug) == 1, 'Error: exactly one branchpred model trained with data augmentations must be present.'
        branchpred_path_aug = branchpred_path_aug[0]
        branchpred_path_noaug = [p for p in args.branch_models_predict if not any(aug in p for aug in DATA_AUG)]
        assert len(branchpred_path_noaug) == 1, 'Error: exactly one branchpred model trained without data augmentations must be present.'
        branchpred_path_noaug = branchpred_path_noaug[0]

    # TRADES trained models metrics
    ria_ra_trades_noaug = [-1, -1]
    ria_ra_trades_aug = [-1, -1]
    cp_cr_trades_noaug = [-1, -1]
    cp_cr_trades_aug = [-1, -1]
    comp_na_ra_trades_aug = [-1, -1]
    comp_na_ra_trades_noaug = [-1, -1]


    # plot settings
    markersize = 6

    # counters for number of plotted abstain trained and all other models
    i_abstain, i_baseline = 0, 0
    # eval branch model(s)
    for branch_model_path in args.branch_models:
        train_eps = get_train_eps(args, branch_model_path)
        pretty_train_eps = pretty_floatstr(train_eps)
        # recover all model paths if given path is a directory to multiple model directories
        branch_model_paths = get_model_paths(args, branch_model_path)
        if len(branch_model_paths) == 0:
            # no models were found under the given directory
            warnings.warn(f'No branch models were found in path {branch_model_path}')
            continue

        # check what kind of model we're dealing with
        branch_is_abstain_trained = check_is_abstain_trained(branch_model_path)
        branch_is_trades_trained = ('trades' in branch_model_path) and not branch_is_abstain_trained
        branch_is_ace_trained = check_is_ace_trained(branch_model_path)
        branch_det_cert_method = check_det_cert_method(branch_model_path)
        branch_is_baseline = branch_model_path == args.baseline_model
        branch_model_name = os.path.splitext(os.path.basename(branch_model_paths[0]))[0]
        if any(aug in branch_model_path for aug in DATA_AUG):
            data_aug = [aug for aug in DATA_AUG if aug in branch_model_path][0]
            data_aug = DATAAUG_ABREVS[data_aug]
            data_aug = '-'+data_aug
        else:
            data_aug = ''

        # skip baseline
        if branch_is_baseline:
            i_baseline += 1
            continue

        # get colors
        if branch_is_abstain_trained:
            color_0, color_1 = COLOR_PAIRS_OURS[i_abstain]
            i_abstain += 1
        else:
            color_0, color_1 = COLOR_PAIRS_BASELINE[i_baseline]
            i_baseline += 1

        if branch_is_ace_trained:
            # no model arch for ACE label
            label = f"{TRAINING_ABREVS['ace']}-{branch_det_cert_method}-{pretty_train_eps}, {SELECTOR_ABREVS['ace']}"
            adv_ext = ''
            if args.show_ace_cert:
                adv_ext = ', adv'

            # ACE markers
            marker_style = dict(color='dimgrey', linestyle='None', markeredgewidth=2, fillstyle='full', markersize=markersize)
            marker_style['marker'] = 's' if branch_det_cert_method == 'COLT' else 'p'

            # get performance measures of each ACE model in branch_model_paths
            (
                branch_nat_acc, branch_adv_acc, branch_cert_acc,
                gate_comp_nat_acc, gate_comp_adv_acc, gate_comp_cert_acc,
                gate_commit_prec_nat, gate_commit_prec_adv, gate_commit_prec_cert,
                gate_commit_rate_nat, gate_commit_rate_adv, gate_commit_rate_cert,
                gate_misselect_nat, gate_misselect_adv, gate_misselect_cert
            ) = ace_model_measures(
                    args, branch_model_paths, test_loader, device, args.test_eps[0],
                    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob
            )

            if not args.no_comp_ace:
                comp_nat_adv_acc_ace_ax.plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_adv_acc, index=1),
                    label=label+adv_ext, **marker_style
                )
                if args.show_ace_cert:
                    comp_nat_adv_acc_ace_ax.plot(
                        *pair_sorted(gate_comp_nat_acc, gate_comp_cert_acc, index=1), marker='p',
                        label=label+', cert', color=color_0, linestyle=linestyle
                    )
                comp_nat_adv_acc_ace_joined_ax[0].plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_adv_acc, index=1),
                    label=label+adv_ext, **marker_style
                )
                if args.show_ace_cert:
                    comp_nat_adv_acc_ace_joined_ax[0].plot(
                        *pair_sorted(gate_comp_nat_acc, gate_comp_cert_acc, index=1), marker='p',
                        label=label+', cert', color=color_0, linestyle=linestyle
                    )
                cnacomp_nat_adv_acc_ace_joined_ax[0].plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_adv_acc, index=1),
                    label=label+adv_ext, **marker_style
                )
                if args.show_ace_cert:
                    cnacomp_nat_adv_acc_ace_joined_ax[0].plot(
                        *pair_sorted(gate_comp_nat_acc, gate_comp_cert_acc, index=1), marker='p',
                        label=label+', cert', color=color_0, linestyle=linestyle
                    )
            branch_commit_prec_rate_ace_ax.plot(
                *pair_sorted(gate_commit_rate_adv, gate_commit_prec_adv, index=1),
                label=label+adv_ext, **marker_style
            )
            if args.show_ace_cert:
                branch_commit_prec_rate_ace_ax.plot(
                    *pair_sorted(gate_commit_rate_cert, gate_commit_prec_cert, index=1), marker='p',
                    label=label+', cert', color=color_0, linestyle=linestyle
                )
            branch_commit_prec_rate_ace_joined_ax[0].plot(
                *pair_sorted(gate_commit_rate_adv, gate_commit_prec_adv, index=1),
                label=label+adv_ext, **marker_style
            )
            if args.show_ace_cert:
                branch_commit_prec_rate_ace_joined_ax[0].plot(
                    *pair_sorted(gate_commit_rate_cert, gate_commit_prec_cert, index=1), marker='p',
                    label=label+', cert', color=color_0, linestyle=linestyle
                )

            # update xlim/ ylim
            comp_nat_adv_acc_ace_xlim = update_ax_lims(comp_nat_adv_acc_ace_xlim, gate_comp_nat_acc, slack=0.01)
            comp_nat_adv_acc_ace_ylim = update_ax_lims(comp_nat_adv_acc_ace_ylim, gate_comp_adv_acc, slack=0.01)
            if args.show_ace_cert:
                comp_nat_adv_acc_ace_ylim = update_ax_lims(comp_nat_adv_acc_ace_ylim, gate_comp_cert_acc, slack=0.01)
            comp_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(comp_nat_adv_acc_ace_joined_xlim[0], gate_comp_nat_acc, slack=0.01)
            comp_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(comp_nat_adv_acc_ace_joined_ylim[0], gate_comp_adv_acc, slack=0.01)
            if args.show_ace_cert:
                comp_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(comp_nat_adv_acc_ace_joined_ylim[0], gate_comp_cert_acc, slack=0.01)
            cnacomp_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_xlim[0], gate_comp_nat_acc, slack=0.01)
            cnacomp_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_ylim[0], gate_comp_adv_acc, slack=0.01)
            if args.show_ace_cert:
                cnacomp_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_ylim[0], gate_comp_cert_acc, slack=0.01)
            branch_commit_prec_rate_ace_xlim = update_ax_lims(branch_commit_prec_rate_ace_xlim, gate_commit_rate_adv, slack=0.01)
            if args.show_ace_cert:
                branch_commit_prec_rate_ace_xlim = update_ax_lims(branch_commit_prec_rate_ace_xlim, gate_commit_rate_cert, slack=0.01)
            branch_commit_prec_rate_ace_ylim = update_ax_lims(branch_commit_prec_rate_ace_ylim, gate_commit_prec_adv, slack=0.01)
            if args.show_ace_cert:
                branch_commit_prec_rate_ace_ylim = update_ax_lims(branch_commit_prec_rate_ace_ylim, gate_commit_prec_cert, slack=0.01)
            branch_commit_prec_rate_ace_joined_xlim[0] = update_ax_lims(branch_commit_prec_rate_ace_joined_xlim[0], gate_commit_rate_adv, slack=0.01)
            if args.show_ace_cert:
                branch_commit_prec_rate_ace_joined_xlim[0] = update_ax_lims(branch_commit_prec_rate_ace_joined_xlim[0], gate_commit_rate_cert, slack=0.01)
            branch_commit_prec_rate_ace_joined_ylim[0] = update_ax_lims(branch_commit_prec_rate_ace_joined_ylim[0], gate_commit_prec_adv, slack=0.01)
            if args.show_ace_cert:
                branch_commit_prec_rate_ace_joined_ylim[0] = update_ax_lims(branch_commit_prec_rate_ace_joined_ylim[0], gate_commit_prec_cert, slack=0.01)
        else:
            if branch_is_abstain_trained:
                label = f"{TRAINING_ABREVS['era']}-{pretty_train_eps}{data_aug}, {SELECTOR_ABREVS['emprobind']} (ours)"
                label_nc = f"{TRAINING_ABREVS['era']}-{pretty_train_eps}{data_aug}, {SELECTOR_ABREVS['nocomp']} (ours)"
                label_ns = f"{TRAINING_ABREVS['era']}-{pretty_train_eps}{data_aug} (ours)"
            elif branch_is_baseline:
                label = f"{TRAINING_ABREVS['base']}-{pretty_train_eps}, {SELECTOR_ABREVS['emprobind']}"
                label_nc = f"{TRAINING_ABREVS['base']}-{pretty_train_eps}, {SELECTOR_ABREVS['nocomp']}"
                label_ns = f"{TRAINING_ABREVS['base']}-{pretty_train_eps}"
            elif branch_is_trades_trained:
                label = f"{TRAINING_ABREVS['trades']}-{pretty_train_eps}{data_aug}, {SELECTOR_ABREVS['emprobind']}"
                label_nc = f"{TRAINING_ABREVS['trades']}-{pretty_train_eps}{data_aug}, {SELECTOR_ABREVS['nocomp']}"
                label_ns = f"{TRAINING_ABREVS['trades']}-{pretty_train_eps}{data_aug}"
            else:
                raise ValueError(f'Model doesnt match.')

            # get branchpred_model
            branchpred_path = None
            if sep_branchpred:
                branchpred_path = branchpred_path_aug if data_aug else branchpred_path_noaug

            # get performance measures of each model in branch_model_paths
            eval_2xeps = False
            if args.eval_2xeps and train_eps == test_eps2x:
                eval_2xeps = True
            (
                branch_nat_acc, branch_adv_acc, branch_rob_inacc,
                robind_comp_nat_acc, robind_comp_adv_acc,
                robind_commit_prec, robind_commit_rate, branch_timestamps
            ) = adv_robind_model_measures(
                    args, branch_model_paths, test_loader, device, args.test_eps[0],
                    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob,
                    branchpred_path, eval_2xeps
            )
            if branch_is_trades_trained:
                if data_aug:
                    ria_ra_trades_aug = [branch_rob_inacc, branch_adv_acc]
                    cp_cr_trades_aug = [robind_commit_prec[0], robind_commit_rate[0]]
                    comp_na_ra_trades_aug = [robind_comp_nat_acc[0], robind_comp_adv_acc[0]]
                else:
                    ria_ra_trades_noaug = [branch_rob_inacc, branch_adv_acc]
                    cp_cr_trades_noaug = [robind_commit_prec[0], robind_commit_rate[0]]
                    comp_na_ra_trades_noaug = [robind_comp_nat_acc[0], robind_comp_adv_acc[0]]

            # plot for robustness indicator selector
            era_color = normalize_color([11, 102, 152])
            marker_style = dict(color='dimgrey', linestyle='-', marker='D', markeredgewidth=1, markersize=markersize+1)
            marker_style['marker'] = 'o' if branch_is_abstain_trained else '^'
            marker_style['color'] = 'dimgrey' if branch_is_trades_trained else era_color
            marker_style['fillstyle'] = 'full' if data_aug else 'none'
            marker_style['linestyle'] = 'None' if len(branch_model_paths) == 1 else '-'
            xtext, ytext = 0 if data_aug else -80, 0 if data_aug else -20 # offset for annotation text
            linestyle = 'None' if len(branch_model_paths) == 1 else '-'

            # plot compositional model
            x, y, timestamps = three_sorted(robind_comp_nat_acc, robind_comp_adv_acc, branch_timestamps, index=1)
            comp_nat_adv_acc_ax.plot(
                x, y, label=label, **marker_style
            )
            comp_nat_adv_acc_ace_ax.plot(x, y, label=label, **marker_style)
            comp_nat_adv_acc_ace_joined_ax[0].plot(x, y, label=label, **marker_style)
            comp_nat_adv_acc_ace_joined_ax[1].plot(x, y, label=label, **marker_style)
            cnacomp_nat_adv_acc_ace_joined_ax[0].plot(x, y, label=label, **marker_style)
            cnacomp_nat_adv_acc_ace_joined_ax[1].plot(x, y, label=label, **marker_style)
            if branch_is_abstain_trained:
                nacomp_nat_adv_acc_ax.plot(x, y, label=label, **marker_style )
                na, ra = x[-1], y[-1]
                trades_na, trades_ra = comp_na_ra_trades_aug if data_aug else comp_na_ra_trades_noaug
                if trades_na < na and trades_ra > ra:
                    tmp_marker = marker_style['marker']
                    marker_style['marker'] = '' # for last point, we dont want marker since a marker is already there
                    comp_nat_adv_acc_ace_ax.plot([na,trades_na], [ra,trades_ra], **marker_style)
                    comp_nat_adv_acc_ace_joined_ax[0].plot([na,trades_na], [ra,trades_ra], **marker_style)
                    comp_nat_adv_acc_ace_joined_ax[1].plot([na,trades_na], [ra,trades_ra], **marker_style)
                    cnacomp_nat_adv_acc_ace_joined_ax[0].plot([na,trades_na], [ra,trades_ra], **marker_style)
                    cnacomp_nat_adv_acc_ace_joined_ax[1].plot([na,trades_na], [ra,trades_ra], **marker_style)
                    marker_style['marker'] = tmp_marker
            if annotate_ts and branch_is_abstain_trained:
                for i, ts in enumerate(timestamps):
                    comp_nat_adv_acc_ax.annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    comp_nat_adv_acc_ace_ax.annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    comp_nat_adv_acc_ace_joined_ax[0].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    comp_nat_adv_acc_ace_joined_ax[1].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    cnacomp_nat_adv_acc_ace_joined_ax[0].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    cnacomp_nat_adv_acc_ace_joined_ax[1].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )

            # plot abstain model commit rate commit precision
            x, y, timestamps = three_sorted(robind_commit_rate, robind_commit_prec, branch_timestamps, index=1)
            robind_branch_commit_prec_rate_ax.plot(y, x, label=label, **marker_style)
            branch_commit_prec_rate_ax.plot(x, y, label=label, **marker_style)
            branch_commit_prec_rate_ace_ax.plot(x, y, label=label, **marker_style)
            branch_commit_prec_rate_ace_joined_ax[0].plot(x, y, label=label, **marker_style)
            branch_commit_prec_rate_ace_joined_ax[1].plot(x, y, label=label, **marker_style)
            if branch_is_abstain_trained:
                cr, cp = x[0], y[0]
                trades_cp, trades_cr = cp_cr_trades_aug if data_aug else cp_cr_trades_noaug
                if trades_cp < cp and trades_cr > cr:
                    tmp_marker = marker_style['marker']
                    marker_style['marker'] = '' # for last point, we dont want marker since a marker is already there
                    robind_branch_commit_prec_rate_ax.plot([cp, trades_cp], [cr, trades_cr], **marker_style)
                    branch_commit_prec_rate_ax.plot([cr, trades_cr], [cp, trades_cp], **marker_style)
                    branch_commit_prec_rate_ace_ax.plot([cr, trades_cr], [cp, trades_cp], **marker_style)
                    branch_commit_prec_rate_ace_joined_ax[0].plot([cr, trades_cr], [cp, trades_cp], **marker_style)
                    branch_commit_prec_rate_ace_joined_ax[1].plot([cr, trades_cr], [cp, trades_cp], **marker_style)
                    marker_style['marker'] = tmp_marker
            if annotate_ts and branch_is_abstain_trained:
                for i, ts in enumerate(timestamps):
                    robind_branch_commit_prec_rate_ax.annotate(
                        text=ts, xy=(y[i], x[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    branch_commit_prec_rate_ax.annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    branch_commit_prec_rate_ace_ax.annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    branch_commit_prec_rate_ace_joined_ax[0].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )
                    branch_commit_prec_rate_ace_joined_ax[1].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                    )

            # plot robacc vs robinacc
            if not branch_is_baseline and not args.eval_2xeps:
                x, y, timestamps = three_sorted(branch_rob_inacc, branch_adv_acc, branch_timestamps, index=1)
                branch_robacc_robinacc_ax.plot(x, y, label=label_ns, **marker_style)
                if annotate_ts and branch_is_abstain_trained:
                    for i, ts in enumerate(timestamps):
                        branch_robacc_robinacc_ax.annotate(
                            text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=marker_style['color']
                        )
                if branch_is_abstain_trained:
                    # get robacc/robinacc of highest point
                    ria, ra = x[-1], y[-1]
                    trades_ria, trades_ra = ria_ra_trades_aug if data_aug else ria_ra_trades_noaug
                    if trades_ria > ria and trades_ra > ra:
                        tmp_marker = marker_style['marker']
                        marker_style['marker'] = '' # for last point, we dont want marker since a marker is already there
                        branch_robacc_robinacc_ax.plot([ria, trades_ria], [ra, trades_ra], **marker_style)
                        marker_style['marker'] = tmp_marker

            # update xlim/ ylim
            comp_nat_adv_acc_xlim = update_ax_lims(comp_nat_adv_acc_xlim, robind_comp_nat_acc, slack=0.01)
            comp_nat_adv_acc_ylim = update_ax_lims(comp_nat_adv_acc_ylim, robind_comp_adv_acc, slack=0.01)
            if not branch_is_abstain_trained:
                nacomp_nat_adv_acc_xlim = update_ax_lims(nacomp_nat_adv_acc_xlim, branch_nat_acc, slack=0.01)
                nacomp_nat_adv_acc_ylim = update_ax_lims(nacomp_nat_adv_acc_ylim, branch_adv_acc, slack=0.01)
            else:
                nacomp_nat_adv_acc_xlim = update_ax_lims(nacomp_nat_adv_acc_xlim, robind_comp_nat_acc, slack=0.01)
                nacomp_nat_adv_acc_ylim = update_ax_lims(nacomp_nat_adv_acc_ylim, robind_comp_adv_acc, slack=0.01)
            comp_nat_adv_acc_ace_xlim = update_ax_lims(comp_nat_adv_acc_ace_xlim, robind_comp_nat_acc, slack=0.01, round_to=1.0)
            comp_nat_adv_acc_ace_ylim = update_ax_lims(comp_nat_adv_acc_ace_ylim, robind_comp_adv_acc, slack=0.01, round_to=1.0)
            comp_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(comp_nat_adv_acc_ace_joined_xlim[0], robind_comp_nat_acc, slack=0.01)
            comp_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(comp_nat_adv_acc_ace_joined_ylim[0], robind_comp_adv_acc, slack=0.01)
            comp_nat_adv_acc_ace_joined_xlim[1] = update_ax_lims(comp_nat_adv_acc_ace_joined_xlim[1], robind_comp_nat_acc, slack=0.01)
            comp_nat_adv_acc_ace_joined_ylim[1] = update_ax_lims(comp_nat_adv_acc_ace_joined_ylim[1], robind_comp_adv_acc, slack=0.01)
            robind_branch_commit_prec_rate_xlim = update_ax_lims(robind_branch_commit_prec_rate_xlim, robind_commit_prec, slack=0.01)
            robind_branch_commit_prec_rate_ylim = update_ax_lims(robind_branch_commit_prec_rate_ylim, robind_commit_rate, slack=0.01)
            branch_commit_prec_rate_xlim = update_ax_lims(branch_commit_prec_rate_xlim, robind_commit_rate, slack=0.01)
            branch_commit_prec_rate_ylim = update_ax_lims(branch_commit_prec_rate_ylim, robind_commit_prec, slack=0.01)
            branch_commit_prec_rate_ace_xlim = update_ax_lims(branch_commit_prec_rate_ace_xlim, robind_commit_rate, slack=0.01)
            branch_commit_prec_rate_ace_ylim = update_ax_lims(branch_commit_prec_rate_ace_ylim, robind_commit_prec, slack=0.01)
            branch_commit_prec_rate_ace_joined_xlim[0] = update_ax_lims(branch_commit_prec_rate_ace_joined_xlim[0], robind_commit_rate, slack=0.01)
            branch_commit_prec_rate_ace_joined_ylim[0] = update_ax_lims(branch_commit_prec_rate_ace_joined_ylim[0], robind_commit_prec, slack=0.01)
            branch_commit_prec_rate_ace_joined_xlim[1] = update_ax_lims(branch_commit_prec_rate_ace_joined_xlim[1], robind_commit_rate, slack=0.01)
            branch_commit_prec_rate_ace_joined_ylim[1] = update_ax_lims(branch_commit_prec_rate_ace_joined_ylim[1], robind_commit_prec, slack=0.01)
            if not branch_is_baseline:
                branch_robacc_robinacc_xlim = update_ax_lims(branch_robacc_robinacc_xlim, branch_rob_inacc, slack=0.01, round_to=1.0)
                branch_robacc_robinacc_ylim = update_ax_lims(branch_robacc_robinacc_ylim, branch_adv_acc, slack=0.01, round_to=1.0)
            cnacomp_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_xlim[0], robind_comp_nat_acc, slack=0.01)
            cnacomp_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_ylim[0], robind_comp_adv_acc, slack=0.01)
            cnacomp_nat_adv_acc_ace_joined_xlim[1] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_xlim[1], robind_comp_nat_acc, slack=0.01)
            cnacomp_nat_adv_acc_ace_joined_ylim[1] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_ylim[1], robind_comp_adv_acc, slack=0.01)

            # plot non-abstain models
            if not branch_is_abstain_trained:
                assert len(branch_model_paths) == 1, 'Error: confidence thresholding abstain only supported for a single model.'
                model_path_ft = None
                if args.eval_2xeps and sep_branchpred:
                    if not branch_is_baseline and not data_aug:
                        model_path_ft =  branchpred_path_noaug
                    elif not branch_is_baseline and data_aug:
                        model_path_ft =  branchpred_path_aug
                    else:
                        model_path_ft =  branch_model_paths[0]
                else:
                    model_path_ft =  branch_model_paths[0]
                train_eps = get_train_eps(args, model_path_ft)
                pretty_train_eps = pretty_floatstr(train_eps)

                # plot non-compositional model
                branchpred_nat_acc, branchpred_adv_acc, _, _, _,_, _, _  = adv_robind_model_measures(
                    args, [model_path_ft], test_loader, device, args.test_eps[0],
                    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob,
                )
                # non-abstain markers
                marker_style = dict(color='dimgrey', linestyle='-', marker='d', markeredgewidth=1, markersize=markersize)
                marker_style['fillstyle'] = 'full' if data_aug else 'none'

                x, y, timestamps = three_sorted(branchpred_nat_acc, branchpred_adv_acc, branch_timestamps, index=1)
                nacomp_nat_adv_acc_ax.plot(x, y, label=label_nc, **marker_style)
                cnacomp_nat_adv_acc_ace_joined_ax[0].plot(x, y, label=label_nc, **marker_style)
                cnacomp_nat_adv_acc_ace_joined_ax[1].plot(x, y, label=label_nc, **marker_style)
                cnacomp_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_xlim[0], branchpred_nat_acc, slack=0.01)
                cnacomp_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_ylim[0], branchpred_adv_acc, slack=0.01)

            # plot softmax response (as another baseline)
            if args.conf_baseline and not branch_is_abstain_trained:
                assert len(branch_model_paths) == 1, 'Error: confidence thresholding abstain only supported for a single model.'
                train_eps = get_train_eps(args, branch_model_paths[0])
                pretty_train_eps = pretty_floatstr(train_eps)

                # get branchpred_model
                branchpred_path = None
                if sep_branchpred:
                    branchpred_path = branchpred_path_aug if data_aug else branchpred_path_noaug

                # produce label for softmax response
                if branch_is_baseline:
                    label = f"{TRAINING_ABREVS['base']}-{pretty_train_eps}, {SELECTOR_ABREVS['conf']}"
                elif branch_is_trades_trained:
                    label = f"{TRAINING_ABREVS['trades']}-{pretty_train_eps}{data_aug}, {SELECTOR_ABREVS['conf']}"
                else:
                    raise ValueError(f'Model doesnt match.')

                eval_2xeps = False
                if args.eval_2xeps and train_eps == test_eps2x:
                    eval_2xeps = True
                conf_comp_nat_acc, conf_comp_adv_acc, _, conf_commit_prec_adv, _, conf_commit_rate_adv, _, _ = conf_model_measures(
                    args, branch_model_paths[0], test_loader, device, args.test_eps[0],
                    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob, n_conf_steps=10,
                    branchpred_path=branchpred_path, eval_2xeps=eval_2xeps
                )

                # softmax response markers
                marker_style = dict(color='dimgrey', linestyle='-', marker='D', markeredgewidth=1, markersize=markersize-1)
                marker_style['fillstyle'] = 'full' if data_aug else 'none'

                # plotting
                if not args.no_comp_conf:
                    x, y = pair_sorted(conf_comp_nat_acc, conf_comp_adv_acc, index=1)
                    comp_nat_adv_acc_ax.plot(x, y, label=label, **marker_style)
                    comp_nat_adv_acc_ace_ax.plot(x, y, label=label, **marker_style)
                    comp_nat_adv_acc_ace_joined_ax[0].plot(x, y, label=label, **marker_style)
                    comp_nat_adv_acc_ace_joined_ax[1].plot(x, y, label=label, **marker_style)
                    cnacomp_nat_adv_acc_ace_joined_ax[0].plot(x, y, label=label, **marker_style)
                    cnacomp_nat_adv_acc_ace_joined_ax[1].plot(x, y, label=label, **marker_style)

                x, y = pair_sorted(conf_commit_rate_adv, conf_commit_prec_adv, index=1)
                branch_commit_prec_rate_ax.plot(x, y, label=label, **marker_style)
                branch_commit_prec_rate_ace_ax.plot(x, y, label=label, **marker_style)
                branch_commit_prec_rate_ace_joined_ax[0].plot(x, y, label=label, **marker_style)
                branch_commit_prec_rate_ace_joined_ax[1].plot(x, y, label=label, **marker_style)

    # plot the trunk model in the na-comp plot (only for non-compositional trunk models)
    if len(args.trunk_models) == 1:
        trunk_model_name = os.path.splitext(os.path.basename(args.trunk_models[0]))[0]
        trunk_model_dir = os.path.dirname(args.trunk_models[0])
        trunk_nat_acc, trunk_adv_acc, _, _, _, _, _, _, _ = get_acc_rob_indicator(
            args, None, trunk_model_dir, trunk_model_name, device,
            test_loader, 'test', args.adv_norm, args.test_eps[0],
            args.test_adv_attack, use_existing=True,
            write_log=True, write_report=True
        )
        label_nc = f"{TRAINING_ABREVS['std']}, {SELECTOR_ABREVS['nocomp']}"
        marker_style = dict(color='dimgrey', linestyle='-', marker='d', markeredgewidth=1, fillstyle='full', markersize=markersize)
        nacomp_nat_adv_acc_ax.plot(trunk_nat_acc, trunk_adv_acc, label=label_nc, **marker_style)
        cnacomp_nat_adv_acc_ace_joined_ax[0].plot(trunk_nat_acc, trunk_adv_acc, label=label_nc, **marker_style)

        nacomp_nat_adv_acc_xlim = update_ax_lims(nacomp_nat_adv_acc_xlim, trunk_nat_acc, slack=0.01)
        nacomp_nat_adv_acc_ylim = update_ax_lims(nacomp_nat_adv_acc_ylim, trunk_adv_acc, slack=0.01)
        cnacomp_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_xlim[0], trunk_nat_acc, slack=0.01)
        cnacomp_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacomp_nat_adv_acc_ace_joined_ylim[0], trunk_adv_acc, slack=0.01)

    # make pretty plots, set optimal xlim and ylim and export plots
    comp_id = get_comp_key(args.branch_model_id, args.trunk_models)
    root_dir = get_root_package_dir()
    comp_dirname = comp_id
    solo_dirname = args.branch_model_id
    if args.eval_2xeps:
        if sep_branchpred:
            comp_dirname = comp_dirname.replace('comp', 'comp2xsepbr')
            solo_dirname += '_2xsepbr'
        else:
            comp_dirname = comp_dirname.replace('comp', 'comp2x')
            solo_dirname += '_2x'
    comp_plot_dir = os.path.join(
        root_dir, FIGURES_DIR, 'revadv', args.dataset,
        args.eval_set, args.adv_norm, args.test_eps[0], comp_dirname
    )
    solo_plot_dir = os.path.join(
        root_dir, FIGURES_DIR, 'revadv', args.dataset,
        args.eval_set, args.adv_norm, args.test_eps[0], solo_dirname
    )
    if annotate_ts:
        comp_plot_dir = os.path.join(comp_plot_dir, 'annotated')
        solo_plot_dir = os.path.join(solo_plot_dir, 'annotated')

    if not os.path.isdir(comp_plot_dir):
        os.makedirs(comp_plot_dir)
    if not os.path.isdir(solo_plot_dir):
        os.makedirs(solo_plot_dir)

    ax_label_fs = 12 # ax label font size
    for plot_dict in plots:
        is_comp_plot = 'comp' in plot_dict['name']
        if not is_comp_plot and args.only_comp:
            # avoid overwriting existing abstain model plots for a different composition
            logging.info(f"Skipping plot {plot_dict['name']} since --only-comp CLI arg was set")
            continue

        fig, ax = plot_dict['fig'], plot_dict['ax']
        plot_width, plot_height = plot_dict['size']
        if annotate_ts:
            plot_height *= 2 # increase space for annotations
        fig.tight_layout()
        fig_leg = None # handle to figure legend
        if 'joined' in plot_dict['name']:
            fig.set_size_inches(plot_width, plot_height)
            legend_loc = 'below'
            ax[0].grid(True)
            ax[1].grid(True)
            handles, labels = ax[0].get_legend_handles_labels()
            if legend_loc == 'below':
                anchor = (0.525, -0.6) if 'cna' in plot_dict['name'] else (0.525, -0.5)
                fig_leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=anchor, prop={'size': 14}, ncol=3)
            else:
                fig_leg = ax[0].legend(loc='center right', bbox_to_anchor=(-0.17, 0.5), prop={'size': 13}, ncol=1)
            ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=ax_label_fs)
            ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=ax_label_fs)
            ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=ax_label_fs)
            ax[1].set_ylabel(ax[1].get_ylabel(), fontsize=ax_label_fs)
            ax[0].set_xlim(plot_dict['xlim'][0])
            ax[0].set_ylim(plot_dict['ylim'][0])
            ax[1].set_xlim(plot_dict['xlim'][1])
            ax[1].set_ylim(plot_dict['ylim'][1])
        else:
            fig.set_size_inches(plot_width, plot_height)
            ax.grid(True)
            if 'ace' in plot_dict['name']:
                fig_leg = ax.legend(loc='center right', bbox_to_anchor=(-0.1, 0.5), prop={'size': 14})
            elif plot_dict['name'] == 'branch_robacc_robinacc':
                if args.eval_2xeps:
                    continue # dont eval branch_robacc_robinacc at 2x region
                ax_label_fs = 14
                handles, labels = ax.get_legend_handles_labels()
                fig_leg = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-3.00, 0.5), prop={'size': 10}, ncol=1)
            else:
                fig_leg = ax.legend(loc='lower left', prop={'size': 13})
            ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fs)
            ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fs)
            if plot_dict['name'] == 'branch_robacc_robinacc':
                ax.set_xlabel('$\mathcal{R}_{robinacc}$ [%]', fontsize=ax_label_fs)
                ax.set_ylabel('$\mathcal{R}_{rob}$ [%]', fontsize=ax_label_fs)
            ax.set_xlim(plot_dict['xlim'])
            ax.set_ylim(plot_dict['ylim'])

        # export figure with legend
        if is_comp_plot:
            plot_dir = comp_plot_dir
        else:
            plot_dir = solo_plot_dir
        plot_fp = os.path.join(plot_dir, plot_dict['name'])
        fig.savefig(plot_fp+'.png', dpi=300, bbox_inches="tight")
        fig.savefig(plot_fp+'.pdf', bbox_inches="tight")
        logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')

        # re-export robacc-robinacc plot with wider x-axis
        if plot_dict['name'] == 'branch_robacc_robinacc':
            fig.set_size_inches(2 * plot_width, plot_height)
            fig.savefig(plot_fp+'_wider.png', dpi=300, bbox_inches="tight")
            fig.savefig(plot_fp+'_wider.pdf', bbox_inches="tight")
            logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')
            fig.set_size_inches(plot_width, plot_height)


        # export figure without legend and without axis labels
        fig_leg.set_visible(False)
        axs = ax
        if type(ax) != np.ndarray:
            axs = [ax]
        for a in axs:
            a.set_xlabel(None)
            a.set_ylabel(None)
        plot_dir = os.path.join(plot_dir, 'nolegend')
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        plot_fp = os.path.join(plot_dir, plot_dict['name'])
        fig.savefig(plot_fp+'.png', dpi=300, bbox_inches="tight")
        fig.savefig(plot_fp+'.pdf', bbox_inches="tight")
        logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')

        # re-export robacc-robinacc plot with wider x-axis
        if plot_dict['name'] == 'branch_robacc_robinacc':
            fig.set_size_inches(2 * plot_width, plot_height)
            fig.savefig(plot_fp+'_wider.png', dpi=300, bbox_inches="tight")
            fig.savefig(plot_fp+'_wider.pdf', bbox_inches="tight")
            logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()
    init_logging(args)

    # build dataset
    _, _, test_loader, _, _, num_classes = get_dataloader(args, args.dataset, normalize=False, indexed=True)

    plot(args, test_loader, device, annotate_ts=False, sep_branchpred=False)
    plot(args, test_loader, device, annotate_ts=True, sep_branchpred=False)


if __name__ == '__main__':
    main()
