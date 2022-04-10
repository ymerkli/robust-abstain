import torch
import numpy as np
import pandas as pd
import logging
import os
from typing import List, Tuple, Dict
from plotly.subplots import make_subplots
from PIL import Image

import robustabstain.utils.args_factory as args_factory
from robustabstain.analysis.plotting.utils.robacc_heatmap import get_robacc_heatmap
from robustabstain.utils.helpers import pretty_floatstr, convert_floatstr
from robustabstain.utils.log import init_logging
from robustabstain.utils.paths import (
    eval_attack_log_filename, get_root_package_dir, FIGURES_DIR)


def get_args():
    parser = args_factory.get_parser(
        description='Robustness/Accuracy heatmap plots.',
        arg_lists=['dataset', 'test-eps', 'adv-norm', 'eval-set', 'test-adv-attack'],
        required_args=['dataset', 'test-eps', 'adv-norm']
    )
    parser.add_argument(
        '--branch-models', type=str, nargs='+', required=True, action='append',
        help='Branch models to plot.'
    )
    parser.add_argument(
        '--branch-model-ids', type=str, nargs='+', required=True, action='append',
        help='Identifier for each branch model. Needs to be in same order as --branch-models.'
    )
    parser.add_argument(
        '--horizontal', action='store_true', help='If set, plot orientation is horizontal.'
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)
    assert len(args.branch_models) == len(args.branch_model_ids), 'Error: specify as many branch-model-ids as branch-models.'
    assert len(args.branch_models) % 2 == 0, 'Error: please specify an even number of subplots.'

    return args


def plot(args: object) -> None:
    """Create and export all plots on given datasplit.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data.
        device (str): device.
    """
    heatmaps = []
    for eps_str, branch_models, branch_model_ids in zip(args.test_eps, args.branch_models, args.branch_model_ids):
        branch_model_log_fps = []
        for branch_model_fp in branch_models:
            model_dir = os.path.dirname(branch_model_fp)
            eval_log_fp = eval_attack_log_filename(args.eval_set, args.dataset, args.adv_norm, args.test_adv_attack)
            branch_model_log_fps.append(os.path.join(model_dir, eval_log_fp))

        fig, heatmap, _ = get_robacc_heatmap(
            args.dataset, args.eval_set, eps_str, model_logs=branch_model_log_fps,
            model_ids=branch_model_ids, horizontal=args.horizontal
        )
        heatmaps.append(heatmap)

    # build subplots figure
    if args.horizontal:
        ncols = 1
        nrows = int(len(heatmaps)/ncols)
        x_title = 'Sample Index'
        y_title = ''
    else:
        nrows = 1
        ncols = int(len(heatmaps)/nrows)
        x_title = ''
        y_title = 'Sample Index'

    # plotly pdf export with math gives mathjax log message in exported plot
    #subplot_titles = [f'$\mathcal{{B}}_{{{pretty_floatstr(eps_str)}}}^{{\infty}}$' for eps_str in args.test_eps]
    subplot_titles = [pretty_floatstr(eps_str) for eps_str in args.test_eps]
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=subplot_titles, horizontal_spacing=0.075,
        x_title=x_title, y_title=y_title
    )
    row, col = 1, 1
    for heatmap in heatmaps:
        fig.add_trace(heatmap, row=row, col=col)
        col += 1
        if col > ncols:
            col = 1
            row += 1
    fig.update_layout(font=dict(size=16))
    fig.update_annotations(font_size=20) # subplot titles are annotations
    fig.update_coloraxes(colorbar=dict(len=0.3, lenmode='fraction'))
    if args.horizontal:
        fig.update_xaxes(showticklabels=False) 
        width = 1200
        height = 400 * len(args.test_eps)
    else:
        fig.update_yaxes(showticklabels=False) # dont show sample indices
        fig.update_xaxes(tickangle=45)
        width = 1200
        height = 800

    # export figure
    filename = f'robacc_heatmap_{args.dataset}_{args.adv_norm}'
    for eps_str in args.test_eps:
        filename += f'_{eps_str}'
    out_dir = os.path.join(
        get_root_package_dir(), FIGURES_DIR, 'illustration',
        'robacc_heatmap', args.dataset
    )
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    filepath = os.path.join(out_dir, filename)
    fig.write_image(filepath+'.png', width=width, height=height)
    fig.write_image(filepath+'.pdf', width=width, height=height)
    fig.write_image(filepath+'.svg', width=width, height=height)
    fig.write_image(filepath+'.eps', width=width, height=height)

    # plotly .pdf static image export has a bug
    # instead, use latex epspdf to convert .eps image to .pdf
    try:
        os.system(f'epspdf {filepath}.eps {filepath}.pdf')
    except:
        pass


def main():
    # argparse
    args = get_args()
    init_logging(args)

    plot(args)


if __name__ == '__main__':
    main()
