import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import re
import warnings
from typing import List, Tuple, Dict

import robustabstain.utils.args_factory as args_factory
from robustabstain.analysis.plotting.utils.colors import COLOR_PAIRS_1 as COLOR_PAIRS_BASELINE
from robustabstain.analysis.plotting.utils.colors import COLOR_PAIRS_2 as COLOR_PAIRS_OURS
from robustabstain.analysis.plotting.utils.helpers import (
    find_smoothing_log, get_model_paths, check_is_abstain_trained, check_is_ace_trained,
    pair_sorted, update_ax_lims)
from robustabstain.analysis.plotting.utils.model_measures import (
    cert_robind_model_measures, cert_robind_model_measures_varrad, ace_model_measures)
from robustabstain.analysis.plotting.utils.smoothing import (
    CertAcc, CertInacc, CommitPrec, CommitRate, Line,
    plot_cert_acc, plot_cert_inacc, plot_commit_prec, plot_commit_rate)
from robustabstain.eval.comp import get_comp_indicator, get_comp_key
from robustabstain.utils.helpers import pretty_floatstr, convert_floatstr, loggable_floatstr
from robustabstain.utils.latex import latex_norm
from robustabstain.utils.loaders import get_dataloader
from robustabstain.utils.log import init_logging
from robustabstain.utils.model_utils import get_model_name
from robustabstain.utils.paths import get_root_package_dir, eval_smoothing_log_filename, FIGURES_DIR
from robustabstain.utils.transforms import DATA_AUG

SELECTOR_ABREVS = {
    'conf': 'CTS',
    'rob': 'ERIS',
    'cert': 'CRIS',
    'ace': 'GNS'
}

LINF_EPS_RANGE = ['0.00', '16/255']
L2_EPS_RANGE = ['0.00', '2.00']
LINF_EPS_FLOAT_RANGE = [0, 16/255]
L2_EPS_FLOAT_RANGE = [0, 2]


def get_args():
    parser = args_factory.get_parser(
        description='Plots for revadv-abstain trained models.',
        arg_lists=[
            args_factory.TESTING_ARGS, args_factory.LOADER_ARGS, args_factory.ATTACK_ARGS,
            args_factory.SMOOTHING_ARGS, args_factory.COMP_ARGS, args_factory.ACE_ARGS
        ],
        required_args=['dataset', 'test-eps', 'adv-norm', 'trunk-models', 'smoothing-sigma', 'revcert-loss', 'noise-sd']
    )
    parser.add_argument(
        '--baseline-model', type=str, help='Model to use as baseline when plotting individual abstain models.'
    )
    parser.add_argument(
        '--baseline-noise-sd', type=str, required=True, help='Perturbation region for which the baseline was trained.'
    )
    parser.add_argument(
        '--branch-models', type=str, nargs='+', required=True, help='Branch models to plot.'
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
        '--plot-running-checkpoints', action='store_true', help='If set, each running checkpoint from training is plotted.'
    )
    parser.add_argument(
        '--plot-varrad', action='store_true', help='If set, commit rate / commit precision plot with varying radii is produced.'
    )
    parser.add_argument(
        '--set-title', action='store_true', help='If set, plots are given a title.'
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)
    args.baseline_noise_sd = loggable_floatstr(args.baseline_noise_sd)
    assert len(args.test_eps) == 1, 'Error: specify 1 test-eps'

    return args


def plot(args: object, test_loader: torch.utils.data.DataLoader, device: str) -> None:
    """Create and export all plots on given datasplit.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data.
        device (str): device.
    """
    pretty_norm = latex_norm(args.adv_norm)
    pretty_norm_p = latex_norm(args.adv_norm, no_l=True)
    pretty_eps = pretty_floatstr(args.test_eps[0])
    pretty_noise_sd = pretty_floatstr(args.noise_sd)
    pretty_baseline_noise_sd = pretty_floatstr(args.baseline_noise_sd)
    pretty_ace_train_eps = pretty_floatstr(args.ace_train_eps) if args.ace_train_eps else None
    eps_range = LINF_EPS_RANGE if args.adv_norm == 'Linf' else L2_EPS_RANGE
    eps_float_range = LINF_EPS_FLOAT_RANGE if args.adv_norm == 'Linf' else L2_EPS_FLOAT_RANGE

    # setup plot of compositional nat vs cert adccuracies robustness indicator
    comp_nat_cert_acc_fig, comp_nat_cert_acc_ax = plt.subplots()
    comp_nat_cert_acc_ax.set_xlabel('Compositional Natural Accuracy [%]')
    comp_nat_cert_acc_ax.set_ylabel('Compositional Certified Accuracy [%]')
    comp_nat_cert_acc_xlim, comp_nat_cert_acc_ylim = [100, 0], [100, 0]
    if args.set_title:
        comp_nat_cert_acc_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_eps}, \sigma={args.smoothing_sigma}$)"
        )

    # setup plot of compositional nat vs cert adccuracies robustness indicator and ACE
    comp_nat_cert_acc_ace_fig, comp_nat_cert_acc_ace_ax = plt.subplots()
    comp_nat_cert_acc_ace_ax.set_xlabel('Compositional Natural Accuracy [%]')
    comp_nat_cert_acc_ace_ax.set_ylabel('Compositional Certified Accuracy [%]')
    comp_nat_cert_acc_ace_xlim, comp_nat_cert_acc_ace_ylim = [100, 0], [100, 0]
    if args.set_title:
        comp_nat_cert_acc_ace_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_eps}, \sigma={args.smoothing_sigma}$)"
        )

    # setup plot of branch model abstain precision vs cert accuracy when using robustness indicator selector
    ri_branch_prec_cert_acc_fig, ri_branch_prec_cert_acc_ax = plt.subplots()
    ri_branch_prec_cert_acc_ax.set_xlabel('Commit Precision [%]')
    ri_branch_prec_cert_acc_ax.set_ylabel('Certified Accuracy [%]')
    ri_branch_prec_cert_acc_xlim, ri_branch_prec_cert_acc_ylim = [100, 0], [100, 0]
    if args.set_title:
        ri_branch_prec_cert_acc_ax.set_title(
            f"{args.dataset} Certified Robustness Indicator ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_eps}, \sigma={args.smoothing_sigma}$)"
        )

    # setup plot of branch model commit precision vs commit rate when using robustness indicator selector
    ri_branch_commit_prec_rate_fig, ri_branch_commit_prec_rate_ax = plt.subplots()
    ri_branch_commit_prec_rate_ax.set_xlabel('Commit Precision [%]')
    ri_branch_commit_prec_rate_ax.set_ylabel('Commit Rate [%]')
    ri_branch_commit_prec_rate_xlim, ri_branch_commit_prec_rate_ylim = [100, 0], [100, 0]
    if args.set_title:
        ri_branch_commit_prec_rate_ax.set_title(
            f"{args.dataset} Certified Robustness Indicator ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_eps}, \sigma={args.smoothing_sigma}$)"
        )

    # setup plot of branch model commit precision vs commit rate when using robustness indicator selector and ACE
    ri_branch_commit_prec_rate_ace_fig, ri_branch_commit_prec_rate_ace_ax = plt.subplots()
    ri_branch_commit_prec_rate_ace_ax.set_xlabel('Commit Precision [%]')
    ri_branch_commit_prec_rate_ace_ax.set_ylabel('Commit Rate [%]')
    ri_branch_commit_prec_rate_ace_xlim, ri_branch_commit_prec_rate_ace_ylim = [100, 0], [100, 0]
    if args.set_title:
        ri_branch_commit_prec_rate_ace_ax.set_title(
            f"{args.dataset} Certified Robustness Indicator ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon={pretty_eps}, \sigma={args.smoothing_sigma}$)"
        )

    # setup plot of compositional nat vs cert accuracies robustness indicator for varying perturbation region radius
    comp_nat_cert_acc_varrad_fig, comp_nat_cert_acc_varrad_ax = plt.subplots()
    comp_nat_cert_acc_varrad_ax.set_xlabel('Compositional Natural Accuracy [%]')
    comp_nat_cert_acc_varrad_ax.set_ylabel('Compositional Certified Accuracy [%]')
    comp_nat_cert_acc_varrad_xlim, comp_nat_cert_acc_varrad_ylim = [100, 0], [100, 0]
    if args.set_title:
        comp_nat_cert_acc_varrad_ax.set_title(
            f"{args.dataset} Compositional Accuracies ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon \in [{eps_range[0]}, {eps_range[1]}], \sigma={args.smoothing_sigma}$)"
        )

    # setup plot of branch model commit precision vs commit rate robustness indicator selector for varying perturbation region radius
    ri_branch_commit_prec_rate_varrad_fig, ri_branch_commit_prec_rate_varrad_ax = plt.subplots()
    ri_branch_commit_prec_rate_varrad_ax.set_xlabel('Commit Precision [%]')
    ri_branch_commit_prec_rate_varrad_ax.set_ylabel('Commit Rate [%]')
    ri_branch_commit_prec_rate_varrad_xlim, ri_branch_commit_prec_rate_varrad_ylim = [100, 0], [100, 0]
    if args.set_title:
        ri_branch_commit_prec_rate_varrad_ax.set_title(
            f"{args.dataset} Certified Robustness Indicator ($\mathcal{{B}}_{{{pretty_norm_p}}}^{{\epsilon}}, \epsilon \in [{eps_range[0]}, {eps_range[1]}], \sigma={args.smoothing_sigma}$)"
        )

    # summarize all figures and axes
    plots = [
        {
            'fig': comp_nat_cert_acc_fig, 'ax': comp_nat_cert_acc_ax,
            'xlim': comp_nat_cert_acc_xlim, 'ylim': comp_nat_cert_acc_ylim,
            'name': 'comp_nat_cert_acc'
        },
        {
            'fig': comp_nat_cert_acc_ace_fig, 'ax': comp_nat_cert_acc_ace_ax,
            'xlim': comp_nat_cert_acc_ace_xlim, 'ylim': comp_nat_cert_acc_ace_ylim,
            'name': 'comp_nat_cert_acc_ace'
        },
        {
            'fig': ri_branch_prec_cert_acc_fig, 'ax': ri_branch_prec_cert_acc_ax,
            'xlim': ri_branch_prec_cert_acc_xlim, 'ylim': ri_branch_prec_cert_acc_ylim,
            'name': 'robind_cert_acc_prec'
        },
        {
            'fig': ri_branch_commit_prec_rate_fig, 'ax': ri_branch_commit_prec_rate_ax,
            'xlim': ri_branch_commit_prec_rate_xlim, 'ylim': ri_branch_commit_prec_rate_ylim,
            'name':'robind_commit_prec_rate'
        },
        {
            'fig': ri_branch_commit_prec_rate_ace_fig, 'ax': ri_branch_commit_prec_rate_ace_ax,
            'xlim': ri_branch_commit_prec_rate_ace_xlim, 'ylim': ri_branch_commit_prec_rate_ace_ylim,
            'name':'robind_commit_prec_rate_ace'
        },
        {
            'fig': comp_nat_cert_acc_varrad_fig, 'ax': comp_nat_cert_acc_varrad_ax,
            'xlim': comp_nat_cert_acc_varrad_xlim, 'ylim': comp_nat_cert_acc_varrad_ylim,
            'name': 'comp_nat_cert_acc_varrad'
        },
        {
            'fig': ri_branch_commit_prec_rate_varrad_fig, 'ax': ri_branch_commit_prec_rate_varrad_ax,
            'xlim': ri_branch_commit_prec_rate_varrad_xlim, 'ylim': ri_branch_commit_prec_rate_varrad_ylim,
            'name':'robind_commit_prec_rate_ace_varrad'
        }
    ]

    # eval trunk model(s)
    _, _, trunk_is_acc, trunk_is_acc_adv, trunk_is_rob = get_comp_indicator(
        args, branch_model_path=args.trunk_models[0],
        trunk_model_paths=args.trunk_models[1:], device=device,
        dataloader=test_loader, eval_set='test', eps_str=args.test_eps[0],
        abstain_method='rob', use_existing=True
    )

    # fix baseline model vars
    baseline_model_path = args.baseline_model
    baseline_model_name = get_model_name(baseline_model_path)
    baseline_smooth_logfile = find_smoothing_log(args, os.path.dirname(baseline_model_path))

    # Save model paths for per-model plots 
    augm_model_paths = []
    abstain_model_paths = []

    # counters for number of plotted abstain trained and all other models
    i_abstain, i_baseline = 0, 0

    # eval branch model(s)
    for i, branch_model_path in enumerate(args.branch_models):
        # recover all model paths if given path is a directory to multiple model directories
        branch_model_paths = get_model_paths(args, branch_model_path)
        if len(branch_model_paths) == 0:
            # no models were found under the given directory
            warnings.warn(f'No branch models were found in path {branch_model_path}')
            continue

        # check what kind of model we're dealing with
        branch_is_abstain_trained = check_is_abstain_trained(branch_model_path)
        branch_is_augm_trained = ('augm' in branch_model_path) and not branch_is_abstain_trained
        branch_is_ace_trained = check_is_ace_trained(branch_model_path)
        branch_is_baseline = branch_model_path == args.baseline_model
        branch_model_name = get_model_name(branch_model_paths[0])
        if any(aug in branch_model_path for aug in DATA_AUG):
            data_aug = [aug for aug in DATA_AUG if aug in branch_model_path][0]
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
            label = f"{args.ace_model_id}-ACE{pretty_ace_train_eps}, {SELECTOR_ABREVS['ace']}"
            # get performance measures of each ACE model in branch_model_paths
            (
                branch_nat_acc, branch_adv_acc, branch_cert_acc,
                gate_comp_nat_acc, gate_comp_adv_acc, gate_comp_cert_acc,
                gate_commit_prec_adv, gate_commit_prec_cert,
                gate_commit_rate_adv, gate_commit_rate_cert
            ) = ace_model_measures(
                    args, branch_model_paths, test_loader, device, args.test_eps[0],
                    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob
            )

            comp_nat_cert_acc_ace_ax.plot(
                *pair_sorted(gate_comp_nat_acc, gate_comp_adv_acc, index=1), marker='s',
                label=label+', adv', color=color_0
            )
            comp_nat_cert_acc_ace_ax.plot(
                *pair_sorted(gate_comp_nat_acc, gate_comp_cert_acc, index=1), marker='p',
                label=label+', cert', color=color_0
            )
            ri_branch_commit_prec_rate_ace_ax.plot(
                *pair_sorted(gate_commit_prec_adv, gate_commit_rate_adv, index=1), marker='s',
                label=label+', adv', color=color_0
            )
            ri_branch_commit_prec_rate_ace_ax.plot(
                *pair_sorted(gate_commit_prec_cert, gate_commit_rate_cert, index=1), marker='p',
                label=label+', cert', color=color_0
            )

            # update xlim/ ylim
            comp_nat_cert_acc_ace_xlim = update_ax_lims(comp_nat_cert_acc_ace_xlim, gate_comp_nat_acc, slack=0.1)
            comp_nat_cert_acc_ace_ylim = update_ax_lims(comp_nat_cert_acc_ace_ylim, gate_comp_adv_acc)
            comp_nat_cert_acc_ace_ylim = update_ax_lims(comp_nat_cert_acc_ace_ylim, gate_comp_cert_acc)
            ri_branch_commit_prec_rate_ace_xlim = update_ax_lims(ri_branch_commit_prec_rate_ace_xlim, gate_commit_prec_adv, slack=0.1)
            ri_branch_commit_prec_rate_ace_xlim = update_ax_lims(ri_branch_commit_prec_rate_ace_xlim, gate_commit_prec_cert, slack=0.1)
            ri_branch_commit_prec_rate_ace_ylim = update_ax_lims(ri_branch_commit_prec_rate_ace_ylim, gate_commit_rate_adv)
            ri_branch_commit_prec_rate_ace_ylim = update_ax_lims(ri_branch_commit_prec_rate_ace_ylim, gate_commit_rate_cert)
        else:
            if branch_is_abstain_trained:
                label = f"Abstain{pretty_noise_sd}{data_aug}, {SELECTOR_ABREVS['cert']}"
                for model_path in branch_model_paths:
                    smoothing_logfile = find_smoothing_log(args, os.path.dirname(model_path))
                    abstain_model_paths.append([model_path, label, smoothing_logfile])
            elif branch_is_baseline:
                label = f"Baseline{pretty_baseline_noise_sd}, {SELECTOR_ABREVS['cert']}"
            elif branch_is_augm_trained:
                label = f"GaussAugm{pretty_noise_sd}{data_aug}, {SELECTOR_ABREVS['cert']}"
                for model_path in branch_model_paths:
                    smoothing_logfile = find_smoothing_log(args, os.path.dirname(model_path))
                    augm_model_paths.append([model_path, label, smoothing_logfile])
            else:
                raise ValueError(f'Model doesnt match.')

            # get performance measures of each model in branch_model_paths
            (
                branch_nat_acc, branch_cert_acc, branch_cert_inacc,
                robind_comp_nat_acc, robind_comp_cert_acc,
                robind_commit_prec, robind_commit_rate
            ) = cert_robind_model_measures(
                    args, branch_model_paths, test_loader, device, args.test_eps[0],
                    trunk_is_acc, trunk_is_acc_adv, trunk_is_rob
            )

            # plot for robustness indicator selector
            marker = 'o' if branch_is_abstain_trained else '^'
            comp_nat_cert_acc_ax.plot(
                *pair_sorted(robind_comp_nat_acc, robind_comp_cert_acc, index=1),
                marker=marker, label=label, color=color_0
            )
            comp_nat_cert_acc_ace_ax.plot(
                *pair_sorted(robind_comp_nat_acc, robind_comp_cert_acc, index=1),
                marker=marker, label=label, color=color_0
            )
            ri_branch_prec_cert_acc_ax.plot(
                *pair_sorted(robind_commit_prec, branch_cert_acc, index=1),
                marker=marker, label=label, color=color_0
            )
            ri_branch_commit_prec_rate_ax.plot(
                *pair_sorted(robind_commit_prec, robind_commit_rate, index=1),
                marker=marker, label=label, color=color_0
            )
            ri_branch_commit_prec_rate_ace_ax.plot(
                *pair_sorted(robind_commit_prec, robind_commit_rate, index=1),
                marker=marker, label=label, color=color_0
            )

            # update xlim/ ylim
            comp_nat_cert_acc_xlim = update_ax_lims(comp_nat_cert_acc_xlim, robind_comp_nat_acc, slack=0.1)
            comp_nat_cert_acc_ylim = update_ax_lims(comp_nat_cert_acc_ylim, robind_comp_cert_acc)
            comp_nat_cert_acc_ace_xlim = update_ax_lims(comp_nat_cert_acc_ace_xlim, robind_comp_nat_acc, slack=0.1)
            comp_nat_cert_acc_ace_ylim = update_ax_lims(comp_nat_cert_acc_ace_ylim, robind_comp_cert_acc)
            ri_branch_prec_cert_acc_xlim = update_ax_lims(ri_branch_prec_cert_acc_xlim, robind_commit_prec, slack=0.1)
            ri_branch_prec_cert_acc_ylim = update_ax_lims(ri_branch_prec_cert_acc_ylim, branch_cert_acc)
            ri_branch_commit_prec_rate_xlim = update_ax_lims(ri_branch_commit_prec_rate_xlim, robind_commit_prec, slack=0.1)
            ri_branch_commit_prec_rate_ylim = update_ax_lims(ri_branch_commit_prec_rate_ylim, robind_commit_rate)
            ri_branch_commit_prec_rate_xlim = update_ax_lims(ri_branch_commit_prec_rate_xlim, robind_commit_prec, slack=0.1)
            ri_branch_commit_prec_rate_ylim = update_ax_lims(ri_branch_commit_prec_rate_ylim, robind_commit_rate)
            ri_branch_commit_prec_rate_ace_xlim = update_ax_lims(ri_branch_commit_prec_rate_ace_xlim, robind_commit_prec, slack=0.1)
            ri_branch_commit_prec_rate_ace_ylim = update_ax_lims(ri_branch_commit_prec_rate_ace_ylim, robind_commit_rate)

            if len(branch_model_paths) == 1 and args.plot_varrad:
                (
                    branch_nat_acc, branch_cert_acc, branch_cert_inacc,
                    robind_comp_nat_acc, robind_comp_cert_acc,
                    robind_commit_prec, robind_commit_rate
                ) = cert_robind_model_measures_varrad(
                        args, branch_model_paths[0], test_loader, device, eps_float_range,
                        trunk_is_acc, trunk_is_acc_adv, trunk_is_rob
                )

                comp_nat_cert_acc_varrad_ax.plot(
                    *pair_sorted(robind_comp_nat_acc, robind_comp_cert_acc, index=1),
                    marker=marker, label=label, color=color_0
                )
                ri_branch_commit_prec_rate_varrad_ax.plot(
                    *pair_sorted(robind_commit_prec, robind_commit_rate, index=1),
                    marker=marker, label=label, color=color_0
                )

                # update xlim/ ylim
                comp_nat_cert_acc_varrad_xlim = update_ax_lims(comp_nat_cert_acc_varrad_xlim, robind_comp_nat_acc, slack=0.1)
                comp_nat_cert_acc_varrad_ylim = update_ax_lims(comp_nat_cert_acc_varrad_ylim, robind_comp_cert_acc)
                ri_branch_commit_prec_rate_varrad_xlim = update_ax_lims(ri_branch_commit_prec_rate_varrad_xlim, robind_commit_prec, slack=0.1)
                ri_branch_commit_prec_rate_varrad_ylim = update_ax_lims(ri_branch_commit_prec_rate_varrad_ylim, robind_commit_rate)

    # plot per-model plots
    for abstain_model_path, abstain_model_label, abstain_model_smooth_logfile in abstain_model_paths:
        model_dir = os.path.dirname(model_path)
        model_name = get_model_name(model_path)
        plot_dir = os.path.join(model_dir, 'plots')
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        smoothing_sigma = convert_floatstr(args.smoothing_sigma)
        max_radius = 1.0
        if smoothing_sigma >= 0.5:
            max_radius = 2.0

        # plot certified accuracy
        plot_cert_acc(
            outfile=os.path.join(plot_dir, 'cert_acc'),
            title=f"{args.dataset} Certified Accuray, $\sigma={smoothing_sigma}$",
            max_radius=max_radius, lines=[
                Line(CertAcc(augm_smooth_logfile), augm_label)
                for _, augm_label, augm_smooth_logfile in augm_model_paths
            ] + [Line(CertAcc(abstain_model_smooth_logfile), abstain_model_label)]
        )

        # plot cert inacc
        plot_cert_inacc(
            outfile=os.path.join(plot_dir, 'cert_inacc'),
            title=f"{args.dataset} Certified Inaccurate, $\sigma={smoothing_sigma}$",
            max_radius=max_radius, lines=[
                Line(CertInacc(augm_smooth_logfile), augm_label)
                for _, augm_label, augm_smooth_logfile in augm_model_paths
            ] + [Line(CertInacc(abstain_model_smooth_logfile), abstain_model_label)]
        )

        # plot commit prec
        plot_commit_prec(
            outfile=os.path.join(plot_dir, 'commit_prec'),
            title=f"{args.dataset} Commit Precision, $\sigma={smoothing_sigma}$",
            max_radius=max_radius, lines=[
                Line(CommitPrec(augm_smooth_logfile), augm_label)
                for _, augm_label, augm_smooth_logfile in augm_model_paths
            ] + [Line(CommitPrec(abstain_model_smooth_logfile), abstain_model_label)]
        )

        # plot commit rate
        plot_commit_rate(
            outfile=os.path.join(plot_dir, 'commit_rate'),
            title=f"{args.dataset} Commit Rate, $\sigma={smoothing_sigma}$",
            max_radius=max_radius, lines=[
                Line(CommitRate(augm_smooth_logfile), augm_label)
                for _, augm_label, augm_smooth_logfile in augm_model_paths
            ] + [Line(CommitRate(abstain_model_smooth_logfile), abstain_model_label)]
        )


    # make pretty plots, set optimal xlim and ylim and export plots
    comp_id = get_comp_key(args.branch_model_id, args.trunk_models)
    root_dir = get_root_package_dir()
    comp_plot_dir = os.path.join(
        root_dir, FIGURES_DIR, args.revcert_loss, args.dataset,
        args.adv_norm, args.smoothing_sigma, args.test_eps[0], comp_id
    )
    solo_plot_dir = os.path.join(
        root_dir, FIGURES_DIR, args.revcert_loss, args.dataset,
        args.adv_norm, args.smoothing_sigma, args.test_eps[0], args.branch_model_id
    )
    if not os.path.isdir(comp_plot_dir):
        os.makedirs(comp_plot_dir)
    if not os.path.isdir(solo_plot_dir):
        os.makedirs(solo_plot_dir)

    for plot_dict in plots:
        plot_dict['ax'].grid(True)
        plot_dict['ax'].legend(loc='best', prop={'size': 6})
        plot_dict['ax'].set_xlim(plot_dict['xlim'])
        plot_dict['ax'].set_ylim(plot_dict['ylim'])
        plot_dir = comp_plot_dir if 'comp' in plot_dict['name'] else solo_plot_dir
        plot_fp = os.path.join(plot_dir, plot_dict['name'])
        plot_dict['fig'].savefig(plot_fp+'.png', dpi=300)
        plot_dict['fig'].savefig(plot_fp+'.pdf')
        logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()
    init_logging(args)

    # build dataset
    _, _, test_loader, _, _, num_classes = get_dataloader(
        args, args.dataset, normalize=False, indexed=True, val_set_source='test', val_split=0.0
    )

    plot(args, test_loader, device)


if __name__ == '__main__':
    main()