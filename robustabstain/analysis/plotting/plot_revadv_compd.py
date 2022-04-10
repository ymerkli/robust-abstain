import torch
import torch.nn as nn

import os
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
from robustabstain.analysis.plotting.utils.helpers import (
    get_model_paths, check_is_abstain_trained, check_is_ace_trained,
    check_det_cert_method, pair_sorted, three_sorted, update_ax_lims)
from robustabstain.analysis.plotting.utils.model_measures import (
    conf_model_measures, adv_robind_model_measures, ace_model_measures)
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.comp import compositional_accuracy, get_comp_indicator, get_comp_key
from robustabstain.utils.checkpointing import load_checkpoint
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.helpers import pretty_floatstr, loggable_floatstr
from robustabstain.utils.latex import latex_norm
from robustabstain.utils.loaders import get_dataloader
from robustabstain.utils.log import init_logging
from robustabstain.utils.paths import get_root_package_dir, FIGURES_DIR
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
    'natacc': 'Natural Accuracy $\mathcal{A}_{nat}$ [%]',
    'robacc': 'Robust Accuracy $\mathcal{A}_{rob}$ [%]',
    'commitrate': 'Robust Coverage $\Phi_{rob}$ [%]',
    'commitprec': 'Robust Accuracy $\mathcal{A}_{rob}$ [%]',
    'compnatacc': 'Natural Accuracy $\mathcal{A}_{nat}$ [%]',
    'comprobacc': 'Robust Accuracy $\mathcal{A}_{rob}$ [%]',
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
        '--abst-train-eps', type=str, required=True, help='Perturbation region for which the ERI abstaining models were trained.'
    )
    parser.add_argument(
        '--pred-train-eps', type=str, required=True, help='Perturbation region for which the predicting models were trained.'
    )
    parser.add_argument(
        '--branch-models-predict', type=str, nargs='+', required=True, help='Branch models that are evaluated to predict.'
    )
    parser.add_argument(
        '--branch-models', type=str, nargs='+', required=True, help='Branch models whose robustness is evaluated for abstaining.'
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
        '--set-title', action='store_true', help='If set, plots are given a title.'
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)
    args.baseline_train_eps = loggable_floatstr(args.baseline_train_eps)
    if args.ace_train_eps:
        args.ace_train_eps = loggable_floatstr(args.ace_train_eps)
    assert len(args.test_eps) == 1, 'Error: specify 1 test-eps'
    assert len(args.branch_models_predict) == 2, 'Error: provide two predict branch models.'

    return args


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
        sep_branchpred (bool, optional): If true, use the separate models in args.branch_models_predict for the
            branch predictions. Defaults to False.
    """
    pretty_norm = latex_norm(args.adv_norm)
    pretty_norm_p = latex_norm(args.adv_norm, no_l=True)
    pretty_test_eps = pretty_floatstr(args.test_eps[0])
    pretty_train_eps = pretty_floatstr(args.train_eps)
    pretty_baseline_train_eps = pretty_floatstr(args.baseline_train_eps)
    pretty_abstainmodel_train_eps = pretty_floatstr(args.abst_train_eps)
    pretty_predmodel_train_eps = pretty_floatstr(args.pred_train_eps)
    pretty_ace_train_eps = pretty_floatstr(args.ace_train_eps) if args.ace_train_eps else None

    # setup plot of compositional nat vs adv adccuracies robustness indicator and confidence thresholding
    compd_nat_adv_acc_fig, compd_nat_adv_acc_ax = plt.subplots()
    compd_nat_adv_acc_ax.set_xlabel(METRICS_NAMES['compnatacc'])
    compd_nat_adv_acc_ax.set_ylabel(METRICS_NAMES['comprobacc'])
    compd_nat_adv_acc_xlim, compd_nat_adv_acc_ylim = [100, 0], [100, 0]
    if args.set_title:
        compd_nat_adv_acc_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of compositional nat vs adv adccuracies robustness indicator and confidence thresholding and ACE
    compd_nat_adv_acc_ace_fig, compd_nat_adv_acc_ace_ax = plt.subplots()
    compd_nat_adv_acc_ace_ax.set_xlabel(METRICS_NAMES['compnatacc'])
    compd_nat_adv_acc_ace_ax.set_ylabel(METRICS_NAMES['comprobacc'])
    compd_nat_adv_acc_ace_xlim, compd_nat_adv_acc_ace_ylim = [100, 0], [100, 0]
    if args.set_title:
        compd_nat_adv_acc_ace_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of compositional nat vs adv adccuracies for robustness indicator and confidence thresholding and ACE
    compd_nat_adv_acc_ace_joined_fig, compd_nat_adv_acc_ace_joined_ax = plt.subplots(nrows=1, ncols=2, figsize=(7,5))
    compd_nat_adv_acc_ace_joined_ax[0].set_xlabel(METRICS_NAMES['compnatacc'])
    compd_nat_adv_acc_ace_joined_ax[0].set_ylabel(METRICS_NAMES['comprobacc'])
    compd_nat_adv_acc_ace_joined_ax[1].set_xlabel(METRICS_NAMES['compnatacc'])
    compd_nat_adv_acc_ace_joined_ax[1].set_ylabel(METRICS_NAMES['comprobacc'])
    compd_nat_adv_acc_ace_joined_xlim, compd_nat_adv_acc_ace_joined_ylim = [[100, 0], [100, 0]], [[100, 0], [100, 0]]
    if args.set_title:
        compd_nat_adv_acc_ace_joined_fig.suptitle(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of compositional nat vs adv adccuracies for robustness indicator and confidence thresholding and ACE + non-comp baselines
    cnacompd_nat_adv_acc_ace_joined_fig, cnacompd_nat_adv_acc_ace_joined_ax = plt.subplots(nrows=1, ncols=2, figsize=(7,5))
    cnacompd_nat_adv_acc_ace_joined_ax[0].set_xlabel(METRICS_NAMES['compnatacc'])
    cnacompd_nat_adv_acc_ace_joined_ax[0].set_ylabel(METRICS_NAMES['comprobacc'])
    cnacompd_nat_adv_acc_ace_joined_ax[1].set_xlabel(METRICS_NAMES['compnatacc'])
    cnacompd_nat_adv_acc_ace_joined_ax[1].set_ylabel(METRICS_NAMES['comprobacc'])
    cnacompd_nat_adv_acc_ace_joined_xlim, cnacompd_nat_adv_acc_ace_joined_ylim = [[100, 0], [100, 0]], [[100, 0], [100, 0]]
    if args.set_title:
        cnacompd_nat_adv_acc_ace_joined_fig.suptitle(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # setup plot of compositional nat vs adv adccuracies robustness indicator and confidence thresholding BUT with non-compositional baselines
    nacompd_nat_adv_acc_fig, nacompd_nat_adv_acc_ax = plt.subplots()
    nacompd_nat_adv_acc_ax.set_xlabel(METRICS_NAMES['compnatacc'])
    nacompd_nat_adv_acc_ax.set_ylabel(METRICS_NAMES['comprobacc'])
    nacompd_nat_adv_acc_xlim, nacompd_nat_adv_acc_ylim = [100, 0], [100, 0]
    if args.set_title:
        nacompd_nat_adv_acc_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_test_eps}$)"
        )

    # summarize all figures and axes
    plots = [
        {
            'fig': compd_nat_adv_acc_fig, 'ax': compd_nat_adv_acc_ax,
            'xlim': compd_nat_adv_acc_xlim, 'ylim': compd_nat_adv_acc_ylim,
            'name': 'compd_nat_adv_acc'
        },
        {
            'fig': compd_nat_adv_acc_ace_fig, 'ax': compd_nat_adv_acc_ace_ax,
            'xlim': compd_nat_adv_acc_ace_xlim, 'ylim': compd_nat_adv_acc_ace_ylim,
            'name': 'compd_nat_adv_acc_ace'
        },
        {
            'fig': compd_nat_adv_acc_ace_joined_fig, 'ax': compd_nat_adv_acc_ace_joined_ax,
            'xlim': compd_nat_adv_acc_ace_joined_xlim, 'ylim': compd_nat_adv_acc_ace_joined_ylim,
            'name': 'compd_nat_adv_acc_ace_joined'
        },
        {
            'fig': cnacompd_nat_adv_acc_ace_joined_fig, 'ax': cnacompd_nat_adv_acc_ace_joined_ax,
            'xlim': cnacompd_nat_adv_acc_ace_joined_xlim, 'ylim': cnacompd_nat_adv_acc_ace_joined_ylim,
            'name': 'cnacompd_nat_adv_acc_ace_joined'
        },
        {
            'fig': nacompd_nat_adv_acc_fig, 'ax': nacompd_nat_adv_acc_ax,
            'xlim': nacompd_nat_adv_acc_xlim, 'ylim': nacompd_nat_adv_acc_ylim,
            'name': 'nacompd_nat_adv_acc'
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
    branchpred_path_aug = [p for p in args.branch_models_predict if any(aug in p for aug in DATA_AUG)]
    assert len(branchpred_path_aug) == 1, 'Error: exactly one branchpred model trained with data augmentations must be present.'
    branchpred_path_aug = branchpred_path_aug[0]
    branchpred_path_noaug = [p for p in args.branch_models_predict if not any(aug in p for aug in DATA_AUG)]
    assert len(branchpred_path_noaug) == 1, 'Error: exactly one branchpred model trained without data augmentations must be present.'
    branchpred_path_noaug = branchpred_path_noaug[0]

    # counters for number of plotted abstain trained and all other models
    i_abstain, i_baseline = 0, 0
    # eval branch model(s)
    for branch_model_path in args.branch_models:
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

        # get colors
        if branch_is_abstain_trained:
            color_0, color_1 = COLOR_PAIRS_OURS[i_abstain]
            i_abstain += 1
        else:
            color_0, color_1 = COLOR_PAIRS_BASELINE[i_baseline]
            i_baseline += 1

        if branch_is_ace_trained:
            # no model arch for ACE label
            #label = f"{args.ace_model_id}-{branch_det_cert_method}-ACE{pretty_ace_train_eps}, {SELECTOR_ABREVS['ace']}"
            label = f"{TRAINING_ABREVS['ace']}-{branch_det_cert_method}-{pretty_ace_train_eps}, {SELECTOR_ABREVS['ace']}"
            linestyle = 'None'

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
                compd_nat_adv_acc_ace_ax.plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_adv_acc, index=1), marker='s',
                    label=label+', adv', color=color_0, linestyle=linestyle
                )
                compd_nat_adv_acc_ace_ax.plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_cert_acc, index=1), marker='p',
                    label=label+', cert', color=color_0, linestyle=linestyle
                )
                compd_nat_adv_acc_ace_joined_ax[0].plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_adv_acc, index=1), marker='s',
                    label=label+', adv', color=color_0, linestyle=linestyle
                )
                compd_nat_adv_acc_ace_joined_ax[0].plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_cert_acc, index=1), marker='p',
                    label=label+', cert', color=color_0, linestyle=linestyle
                )
                cnacompd_nat_adv_acc_ace_joined_ax[0].plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_adv_acc, index=1), marker='s',
                    label=label+', adv', color=color_0, linestyle=linestyle
                )
                cnacompd_nat_adv_acc_ace_joined_ax[0].plot(
                    *pair_sorted(gate_comp_nat_acc, gate_comp_cert_acc, index=1), marker='p',
                    label=label+', cert', color=color_0, linestyle=linestyle
                )

            # update xlim/ ylim
            compd_nat_adv_acc_ace_xlim = update_ax_lims(compd_nat_adv_acc_ace_xlim, gate_comp_nat_acc, slack=0.01)
            compd_nat_adv_acc_ace_ylim = update_ax_lims(compd_nat_adv_acc_ace_ylim, gate_comp_adv_acc, slack=0.01)
            compd_nat_adv_acc_ace_ylim = update_ax_lims(compd_nat_adv_acc_ace_ylim, gate_comp_cert_acc, slack=0.01)
            compd_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(compd_nat_adv_acc_ace_joined_xlim[0], gate_comp_nat_acc, slack=0.01)
            compd_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(compd_nat_adv_acc_ace_joined_ylim[0], gate_comp_adv_acc, slack=0.01)
            compd_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(compd_nat_adv_acc_ace_joined_ylim[0], gate_comp_cert_acc, slack=0.01)
            cnacompd_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_xlim[0], gate_comp_nat_acc, slack=0.01)
            cnacompd_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_ylim[0], gate_comp_adv_acc, slack=0.01)
            cnacompd_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_ylim[0], gate_comp_cert_acc, slack=0.01)
        else:
            if branch_is_abstain_trained:
                label = f"{TRAINING_ABREVS['era']}-{pretty_abstainmodel_train_eps}{data_aug}, {SELECTOR_ABREVS['emprobind']} (ours)"
                label_nc = f"{TRAINING_ABREVS['era']}-{pretty_abstainmodel_train_eps}{data_aug}, {SELECTOR_ABREVS['nocomp']} (ours)"
            elif branch_is_baseline:
                label = f"{TRAINING_ABREVS['base']}-{pretty_baseline_train_eps}, {SELECTOR_ABREVS['emprobind']}"
                label_nc = f"{TRAINING_ABREVS['base']}-{pretty_baseline_train_eps}, {SELECTOR_ABREVS['nocomp']}"
            elif branch_is_trades_trained:
                label = f"{TRAINING_ABREVS['trades']}-{pretty_abstainmodel_train_eps}{data_aug}, {SELECTOR_ABREVS['emprobind']}"
                label_nc = f"{TRAINING_ABREVS['trades']}-{pretty_predmodel_train_eps}{data_aug}, {SELECTOR_ABREVS['nocomp']}"
            else:
                raise ValueError(f'Model doesnt match.')

            # get branchpred_model
            branchpred_path = None 
            if sep_branchpred:
                branchpred_path = branchpred_path_aug if data_aug else branchpred_path_noaug 

            # get performance measures of each model in branch_model_paths
            (
                branch_nat_acc, branch_adv_acc, branch_rob_inacc,
                robind_comp_nat_acc, robind_comp_adv_acc,
                robind_commit_prec, robind_commit_rate, branch_timestamps
            ) = adv_robind_model_measures(
                    args, branch_model_paths, test_loader, device, args.test_eps[0],
                    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob,
                    branchpred_path, eval_2xeps=True
            )

            # plot for robustness indicator selector
            marker = 'o' if branch_is_abstain_trained else '^'
            xtext, ytext = 0 if data_aug else -80, 0 if data_aug else -20 # offset for annotation text
            linestyle = 'None' if len(branch_model_paths) == 1 else '-'

            # plot compositional model
            x, y, timestamps = three_sorted(robind_comp_nat_acc, robind_comp_adv_acc, branch_timestamps, index=1)
            compd_nat_adv_acc_ax.plot(
                x, y, marker=marker, label=label, color=color_0, linestyle=linestyle
            )
            if branch_is_abstain_trained:
                nacompd_nat_adv_acc_ax.plot(
                    x, y, marker=marker, label=label, color=color_0, linestyle=linestyle
                )
            compd_nat_adv_acc_ace_ax.plot(
                x, y, marker=marker, label=label, color=color_0, linestyle=linestyle
            )
            compd_nat_adv_acc_ace_joined_ax[0].plot(
                x, y, marker=marker, label=label, color=color_0, linestyle=linestyle
            )
            compd_nat_adv_acc_ace_joined_ax[1].plot(
                x, y, marker=marker, label=label, color=color_0, linestyle=linestyle
            )
            cnacompd_nat_adv_acc_ace_joined_ax[0].plot(
                x, y, marker=marker, label=label, color=color_0, linestyle=linestyle
            )
            cnacompd_nat_adv_acc_ace_joined_ax[1].plot(
                x, y, marker=marker, label=label, color=color_0, linestyle=linestyle
            )
            if annotate_ts and branch_is_abstain_trained:
                for i, ts in enumerate(timestamps):
                    compd_nat_adv_acc_ax.annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=color_0
                    )
                    compd_nat_adv_acc_ace_ax.annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=color_0
                    )
                    compd_nat_adv_acc_ace_joined_ax[0].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=color_0
                    )
                    compd_nat_adv_acc_ace_joined_ax[1].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=color_0
                    )
                    cnacompd_nat_adv_acc_ace_joined_ax[0].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=color_0
                    )
                    cnacompd_nat_adv_acc_ace_joined_ax[1].annotate(
                        text=ts, xy=(x[i], y[i]), xytext=(xtext, ytext), textcoords='offset points', color=color_0
                    )

            if not branch_is_abstain_trained:
                # plot non-compositional adversarial/baseline trained models
                if not branch_is_baseline and not data_aug:
                    branchpred_path_na =  branchpred_path_noaug
                elif not branch_is_baseline and data_aug:
                    branchpred_path_na =  branchpred_path_aug
                else:
                    branchpred_path_na =  branch_model_paths[0]

                branchpred_nat_acc, branchpred_adv_acc, _, _, _, _, _, _ = adv_robind_model_measures(
                    args, [branchpred_path_na], test_loader, device, args.test_eps[0],
                    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob,
                )
                marker_nc = 'd'
                x, y, timestamps = three_sorted(branchpred_nat_acc, branchpred_adv_acc, branch_timestamps, index=1)
                nacompd_nat_adv_acc_ax.plot(
                    x, y, marker=marker_nc, label=label_nc, color=color_0, linestyle=linestyle
                )
                cnacompd_nat_adv_acc_ace_joined_ax[0].plot(
                    x, y, marker=marker_nc, label=label_nc, color=color_0, linestyle=linestyle
                )
                cnacompd_nat_adv_acc_ace_joined_ax[1].plot(
                    x, y, marker=marker_nc, label=label_nc, color=color_0, linestyle=linestyle
                )

                # plot compositional accuracies for confidence selector (as another baseline)
                if args.conf_baseline and not branch_is_abstain_trained:
                    assert len(branch_model_paths) == 1, 'Error: confidence thresholding abstain only supported for a single model.'

                    # produce label
                    if branch_is_baseline:
                        label = f"{TRAINING_ABREVS['base']}-{pretty_baseline_train_eps}, {SELECTOR_ABREVS['conf']}"
                    elif branch_is_trades_trained:
                        label = f"{TRAINING_ABREVS['trades']}-{pretty_predmodel_train_eps}{data_aug}, {SELECTOR_ABREVS['conf']}"
                    else:
                        raise ValueError(f'Model doesnt match.')

                    conf_comp_nat_acc, conf_comp_adv_acc,_, _, _, _, _, _ = conf_model_measures(
                        args, branchpred_path_na, test_loader, device, args.test_eps[0],
                        trunk_is_acc, trunk_is_acc_adv, trunk_is_rob, n_conf_steps=20
                    )

                    # plotting
                    if not args.no_comp_conf:
                        x, y = pair_sorted(conf_comp_nat_acc, conf_comp_adv_acc, index=1)
                        compd_nat_adv_acc_ax.plot(x, y, label=label, color=color_1)
                        compd_nat_adv_acc_ace_ax.plot(x, y, label=label, color=color_1)
                        compd_nat_adv_acc_ace_joined_ax[0].plot(x, y, label=label, color=color_1)
                        compd_nat_adv_acc_ace_joined_ax[1].plot(x, y, label=label, color=color_1)
                        cnacompd_nat_adv_acc_ace_joined_ax[0].plot(x, y, label=label, color=color_1)
                        cnacompd_nat_adv_acc_ace_joined_ax[1].plot(x, y, label=label, color=color_1)

            # update xlim/ ylim
            compd_nat_adv_acc_xlim = update_ax_lims(compd_nat_adv_acc_xlim, robind_comp_nat_acc, slack=0.01)
            compd_nat_adv_acc_ylim = update_ax_lims(compd_nat_adv_acc_ylim, robind_comp_adv_acc, slack=0.01)
            if not branch_is_abstain_trained:
                nacompd_nat_adv_acc_xlim = update_ax_lims(nacompd_nat_adv_acc_xlim, branch_nat_acc, slack=0.01)
                nacompd_nat_adv_acc_ylim = update_ax_lims(nacompd_nat_adv_acc_ylim, branch_adv_acc, slack=0.01)
                cnacompd_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_xlim[0], branchpred_nat_acc, slack=0.01)
                cnacompd_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_ylim[0], branchpred_adv_acc, slack=0.01)
            else:
                nacompd_nat_adv_acc_xlim = update_ax_lims(nacompd_nat_adv_acc_xlim, robind_comp_nat_acc, slack=0.01)
                nacompd_nat_adv_acc_ylim = update_ax_lims(nacompd_nat_adv_acc_ylim, robind_comp_adv_acc, slack=0.01)
            compd_nat_adv_acc_ace_xlim = update_ax_lims(compd_nat_adv_acc_ace_xlim, robind_comp_nat_acc, slack=0.01)
            compd_nat_adv_acc_ace_ylim = update_ax_lims(compd_nat_adv_acc_ace_ylim, robind_comp_adv_acc, slack=0.01)
            compd_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(compd_nat_adv_acc_ace_joined_xlim[0], robind_comp_nat_acc, slack=0.01)
            compd_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(compd_nat_adv_acc_ace_joined_ylim[0], robind_comp_adv_acc, slack=0.01)
            compd_nat_adv_acc_ace_joined_xlim[1] = update_ax_lims(compd_nat_adv_acc_ace_joined_xlim[1], robind_comp_nat_acc, slack=0.01)
            compd_nat_adv_acc_ace_joined_ylim[1] = update_ax_lims(compd_nat_adv_acc_ace_joined_ylim[1], robind_comp_adv_acc, slack=0.01)
            cnacompd_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_xlim[0], robind_comp_nat_acc, slack=0.01)
            cnacompd_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_ylim[0], robind_comp_adv_acc, slack=0.01)
            cnacompd_nat_adv_acc_ace_joined_xlim[1] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_xlim[1], robind_comp_nat_acc, slack=0.01)
            cnacompd_nat_adv_acc_ace_joined_ylim[1] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_ylim[1], robind_comp_adv_acc, slack=0.01)
            
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
        marker = 'd'
        color_0, color_1 = COLOR_PAIRS_BASELINE[i_baseline]
        nacompd_nat_adv_acc_ax.plot(
            trunk_nat_acc, trunk_adv_acc, marker=marker, label=label_nc, color=color_0, linestyle='None'
        )
        nacompd_nat_adv_acc_xlim = update_ax_lims(nacompd_nat_adv_acc_xlim, trunk_nat_acc, slack=0.01)
        nacompd_nat_adv_acc_ylim = update_ax_lims(nacompd_nat_adv_acc_ylim, trunk_adv_acc, slack=0.01)

        cnacompd_nat_adv_acc_ace_joined_ax[0].plot(
            trunk_nat_acc, trunk_adv_acc, marker=marker, label=label_nc, color=color_0, linestyle='None'
        )
        cnacompd_nat_adv_acc_ace_joined_ax[1].plot(
            trunk_nat_acc, trunk_adv_acc, marker=marker, label=label_nc, color=color_0, linestyle='None'
        )
        cnacompd_nat_adv_acc_ace_joined_xlim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_xlim[0], trunk_nat_acc, slack=0.01)
        cnacompd_nat_adv_acc_ace_joined_ylim[0] = update_ax_lims(cnacompd_nat_adv_acc_ace_joined_ylim[0], trunk_adv_acc, slack=0.01)

    # make pretty plots, set optimal xlim and ylim and export plots
    comp_id = get_comp_key(args.branch_model_id, args.trunk_models)
    if sep_branchpred:
        comp_id = comp_id.replace('comp', 'compdsepbr')
    else:
        comp_id = comp_id.replace('comp', 'compd')
    root_dir = get_root_package_dir()
    comp_plot_dir = os.path.join(
        root_dir, FIGURES_DIR, 'revadv', args.dataset,
        args.eval_set, args.adv_norm, args.test_eps[0], comp_id
    )
    if annotate_ts:
        comp_plot_dir = os.path.join(comp_plot_dir, 'annotated')

    if not os.path.isdir(comp_plot_dir):
        os.makedirs(comp_plot_dir)

    for plot_dict in plots:
        fig, ax = plot_dict['fig'], plot_dict['ax']
        fig.tight_layout()
        fig_leg = None # handle to figure legend
        if 'joined' in plot_dict['name']:
            plot_width, plot_height = 12, 3
            if annotate_ts:
                plot_height *= 2 # increase space for annotations
            fig.set_size_inches(plot_width, plot_height)
            legend_loc = 'below'
            ax[0].grid(True)
            ax[1].grid(True)
            handles, labels = ax[0].get_legend_handles_labels()
            if legend_loc == 'below':
                anchor = (0.525, -0.5)
                if 'cna' in plot_dict['name']:
                    anchor = (0.525, -0.7)
                fig_leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=anchor, prop={'size': 14}, ncol=3)
            else:
                fig_leg = ax[0].legend(loc='center right', bbox_to_anchor=(-0.17, 0.5), prop={'size': 13}, ncol=1)
            ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=12)
            ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=12)
            ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=12)
            ax[1].set_ylabel(ax[1].get_ylabel(), fontsize=12)
            ax[0].set_xlim(plot_dict['xlim'][0])
            ax[0].set_ylim(plot_dict['ylim'][0])
            ax[1].set_xlim(plot_dict['xlim'][1])
            ax[1].set_ylim(plot_dict['ylim'][1])
        else:
            plot_width, plot_height = 6, 3.5
            if annotate_ts:
                plot_height *= 2 # increase space for annotations
            fig.set_size_inches(plot_width, plot_height)
            ax.grid(True)
            if 'ace' in plot_dict['name']:
                fig_leg = ax.legend(loc='center right', bbox_to_anchor=(-0.1, 0.5), prop={'size': 14})
            else:
                fig_leg = ax.legend(loc='lower left', prop={'size': 13})
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.set_xlim(plot_dict['xlim'])
            ax.set_ylim(plot_dict['ylim'])

        # export figure with legend
        plot_dir = comp_plot_dir
        plot_fp = os.path.join(plot_dir, plot_dict['name'])
        fig.savefig(plot_fp+'.png', dpi=300, bbox_inches="tight")
        fig.savefig(plot_fp+'.pdf', bbox_inches="tight")
        logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')

        # export figure without legend
        fig_leg.set_visible(False)
        plot_dir = os.path.join(plot_dir, 'nolegend')
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        plot_fp = os.path.join(plot_dir, plot_dict['name'])
        fig.savefig(plot_fp+'.png', dpi=300, bbox_inches="tight")
        fig.savefig(plot_fp+'.pdf', bbox_inches="tight")
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
    plot(args, test_loader, device, annotate_ts=False, sep_branchpred=True)
    plot(args, test_loader, device, annotate_ts=True, sep_branchpred=True)


if __name__ == '__main__':
    main()
