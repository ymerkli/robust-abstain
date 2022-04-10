import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from textwrap import wrap
from typing import Tuple, List, Union, Optional

import robustabstain.utils.args_factory as args_factory
from robustabstain.analysis.plotting.utils.colors import COLOR_PAIRS_1 as COLOR_PAIRS_BASELINE
from robustabstain.analysis.plotting.utils.colors import COLOR_PAIRS_2 as COLOR_PAIRS_OURS
from robustabstain.analysis.plotting.utils.helpers import (
    get_model_paths, check_is_abstain_trained, check_is_ace_trained,
    pair_sorted, update_ax_lims)
from robustabstain.analysis.plotting.utils.model_measures import (
    conf_model_measures, adv_robind_model_measures, ace_model_measures)
from robustabstain.utils.data_utils import get_dataset_stats, dataset_label_names
from robustabstain.utils.helpers import pretty_floatstr, loggable_floatstr
from robustabstain.utils.latex import latex_norm
from robustabstain.utils.loaders import get_dataset, get_sampling_order
from robustabstain.utils.log import init_logging
from robustabstain.utils.paths import get_root_package_dir, FIGURES_DIR


def get_args():
    parser = args_factory.get_parser(
        description='Plots for revadv-abstain trained models.',
        arg_lists=[args_factory.LOADER_ARGS],
        required_args=['dataset']
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    return args


def plot_label_distribution(
        bincount: List[int], label_ids: List[int], label_names: List[str],
        topk_labels: int = None
    ) -> matplotlib.figure.Figure:
    """Plot label distribution.

    Args:
        bincount (List[int]): Label bincount. 
        label_ids (List[int]): List of label ids.
        label_names (List[str]): List of label names.
        topk_labels (int, optional): Top labels to consider. Defaults to None.

    Returns:
        plt.figure.Figure: Barplot figure.
    """
    if topk_labels:
        topk_labels = min(topk_labels, len(bincount))
    else:
        topk_labels = len(bincount)
    label_names = [
        name.replace('_', ' ') for _, name in
        sorted(zip(bincount, label_names), key=lambda pair: pair[0], reverse=True)
    ]
    label_ids = [id for _, id in sorted(zip(bincount, label_ids), key=lambda pair: pair[0], reverse=True)]
    bincount = sorted(bincount, reverse=True)
    label_names, bincount = label_names[:topk_labels], bincount[:topk_labels]

    rotation = 0
    max_labelname = max([len(name) for name in label_names])
    if max_labelname >= 20:
        rotation = 90
    fig, ax = plt.subplots(figsize=(10,8))
    ax.bar(label_names, bincount)
    ax.tick_params(axis='x', rotation=rotation, labelsize=18)
    ax.set_xlabel('Label', fontsize=20)
    ax.set_ylabel('Sample Count', fontsize=20)
    if topk_labels > 10:
        ax.get_xaxis().set_ticks([])
    fig.tight_layout()

    return fig


def plot_image_grid(
        dataset: torch.utils.data.Dataset,
        label_ids: List[int], label_names: List[str],
        topk_labels: int = 5, samples_per_class: int = 3,
        n_row: int = None, n_col: int = None
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """Utility function for plotting images in a grid.

    Image grid dimensions are dynamically computed given the number of input images.
    """
    topk_labels = min(topk_labels, len(label_ids))
    label_ids = label_ids[:topk_labels]
    label_names = label_names[:topk_labels]

    image_size: Tuple[int, int] = (4, 4)
    fig_size = (samples_per_class * image_size[0], topk_labels * image_size[1])

    # find samples
    sampling_order = np.random.permutation([idx for idx in range(len(dataset.targets))])
    samples = {}
    images_and_labels = []
    for sample_idx in sampling_order:
        x, y = dataset[sample_idx]
        if y in label_ids:
            if y not in samples:
                samples[y] = []
            if len(samples[y]) < samples_per_class:
                npimg = np.transpose(x, (1,2,0)) # change dimension order
                samples[y].append(npimg)
    for label_id in label_ids: 
        for img in samples[label_id]: 
            images_and_labels.append((img, label_names[label_ids.index(label_id)]))

    # create plot
    if not n_row:
        n_row = topk_labels
    if not n_col:
        n_col = samples_per_class
    n_images = n_row * n_col
    if n_images == 0:
        return None
    r_row = min(int(np.ceil(n_images / n_col)), n_row)

    if r_row == 1:
        r_col = n_images
    else:
        r_col = n_col

    fig_size = (r_col * image_size[0], 2 * r_row * image_size[1])
    fig, axes = plt.subplots(nrows=r_row, ncols=r_col, figsize=fig_size, sharey=True)
    axs = axes
    if r_row > 1 and r_col > 1:
        axs = axes.flatten()
    if r_row == 1 and r_col == 1:
        axs = [axes]

    for (image, label), ax in zip(images_and_labels, axs):
        label = label.replace('_', ' ')
        title = "\n".join(wrap(label.replace('--', ' '), 15))
        ax.imshow(image)
        ax.set_title(title, fontdict={'fontsize': 24})

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()

    return fig
 

def plot(
        args: object, train_set: torch.utils.data.Dataset, test_set: torch.utils.data.Dataset,
        topk_labels: int = 5
    ) -> None:
    """Create and export all plots on given datasplit.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        train_set (torch.utils.data.Dataset): PyTorch dataset with all train data.
        test_set (torch.utils.data.Dataset): PyTorch dataset with all test data.
        topk_labels (int, optional): Number of most frequent labels to consider.
    """
    np.random.seed(args.seed)
    dataset = ConcatDataset([train_set, test_set]) 
    root_dir = get_root_package_dir()
    plot_dir = os.path.join(root_dir, FIGURES_DIR, 'illustration', 'dataset', args.dataset)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    # dataset stats
    bincount = np.bincount([target for d in dataset.datasets for target in d.targets])
    label_ids = [i for i in range(len(bincount))]
    label_names = dataset_label_names(args.dataset)
    label_names = [name for _, name in sorted(zip(bincount, label_names), key=lambda pair: pair[0], reverse=True)]
    label_ids = [id for _, id in sorted(zip(bincount, label_ids), key=lambda pair: pair[0], reverse=True)]
    bincount = sorted(bincount, reverse=True)

    # plot topk label distribution
    fig = plot_label_distribution(bincount, label_ids, label_names, topk_labels)
    plot_fp = os.path.join(plot_dir, f'label_distr_top{topk_labels}')
    fig.savefig(plot_fp+'.png', dpi=300, bbox_inches="tight")
    fig.savefig(plot_fp+'.pdf', bbox_inches="tight")
    logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')

    # plot topk label distribution
    fig = plot_label_distribution(bincount, label_ids, label_names)
    plot_fp = os.path.join(plot_dir, 'label_distr')
    fig.savefig(plot_fp+'.png', dpi=300, bbox_inches="tight")
    fig.savefig(plot_fp+'.pdf', bbox_inches="tight")
    logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')

    return

    # plot some samples from the dataset
    fig = plot_image_grid(test_set, label_ids, label_names, topk_labels=5, samples_per_class=3)
    plot_fp = os.path.join(plot_dir, 'samples_grid_5_3')
    fig.savefig(plot_fp+'.png', dpi=300, bbox_inches="tight")
    fig.savefig(plot_fp+'.pdf', bbox_inches="tight")
    logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')

    # plot some samples from the dataset
    fig = plot_image_grid(test_set, label_ids, label_names, topk_labels=5, samples_per_class=1, n_row=1, n_col=5)
    plot_fp = os.path.join(plot_dir, 'samples_grid_5_1')
    fig.savefig(plot_fp+'.png', dpi=300, bbox_inches="tight")
    fig.savefig(plot_fp+'.pdf', bbox_inches="tight")
    logging.info(f'Exported plot to {plot_fp}[.png/.pdf]')


def main():
    # argparse
    args = get_args()
    init_logging(args)

    # build dataset
    train_set, _, test_set, _, _, _, _, _ = get_dataset(args.dataset, args.eval_set)

    plot(args, train_set, test_set, topk_labels=5)


if __name__ == '__main__':
    main()