import torch
import torch.nn as nn

import os
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from typing import Tuple, Union, List

from robustabstain.analysis.plotting.utils.colors import ROBACC_COLORS
from robustabstain.analysis.plotting.utils.helpers import pair_sorted 
from robustabstain.attacker.wrapper import AttackerWrapper
from robustabstain.utils.distributions import rademacher
from robustabstain.utils.helpers import convert_floatstr, normstr2normp
from robustabstain.utils.data_utils import dataset_label_names


def save(img, fp):
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1,2,0))
    matplotlib.image.imsave(fp, npimg)


def region_grid_pred(
        model: nn.Module, x: torch.tensor, y: torch.tensor, device: str,
        eps: float, adv_attack: str = 'pgd', adv_norm: str = 'Linf',
        att_n_steps: int = 40, rel_step_size: float = 0.1,
        nx: int = 500, ny: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce a grid of predictions. The grid iterates over the scale of a
    random Rademacher vector and the scale of a adversarial perturbation vector.

    Args:
        model (nn.module): Model to evaluate decision region for.
        x (torch.tensor): Single input.
        y (torch.tensor): Label.
        device (str): device.
        eps (float): Perturbation region size.
        adv_attack (str, optional): Adversarial attack type. Defaults to 'pgd'.
        adv_norm (str, optional): Adversarial perturbation norm. Defaults to 'Linf'.
        att_n_steps (int, optional): Number of attack steps. Defaults to 40.
        rel_step_size (float, optional): Relative step size. Defaults to 0.1.
        nx (int, optional): Number of x grid steps. Defaults to 500.
        ny (int, optional): Number of y grid steps. Defaults to 500.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: xaxis grid (rademacher direction),
            yaxis grid (adversarial direction), labels matrice
    """
    if len(x.shape) != 4:
        x = x.unsqueeze(0)
    nat_pred = model(x).argmax(1)

    attacker = AttackerWrapper(
        adv_attack, adv_norm, eps, 100, rel_step_size=rel_step_size, device=device
    )
    model.eval()
    x_adv = attacker.attack(model, x, nat_pred)
    rade = rademacher(x.shape).to(device)
    adv_delta = -1 * (x - x_adv) / torch.norm(x - x_adv, p=normstr2normp(adv_norm))

    xgrid = eps * np.linspace(start=-1, stop=1, num=nx)
    ygrid = eps * np.linspace(start=-1, stop=1, num=ny)

    rade_batch = torch.from_numpy(xgrid).float().to(device).view(-1,1,1,1) * rade
    adv_delta_batch = torch.from_numpy(ygrid).float().to(device).view(-1,1,1,1) * adv_delta

    labels = np.zeros((ny, nx), dtype=int)
    for i in range(adv_delta_batch.size(0)):
        pred = model(x + rade_batch + adv_delta_batch[i].unsqueeze(0)).argmax(1)
        pred[pred == 8] = 6
        labels[i] = pred.cpu().numpy()

    return xgrid, ygrid, labels


def decision_region_plot(
        model: nn.Module, x: torch.tensor, y: torch.tensor, device: str,
        eps: float, eps_plot: float = None, adv_attack: str = 'pgd', adv_norm: str = 'Linf',
        att_n_steps: int = 40, rel_step_size: float = 0.1, nx: int = 500, ny: int = 500,
        dataset: str = None, data_dir: str = None, plotlib='plt',
        savepath: str = './decision_region_plot'
    ) -> Union[plt.figure, go.Figure]:
    """Produces a matplotlib plot of the adversarial decision region
    of a neural network for a given sample. The decision region spans
    over the scale of a random Rademacher vector and the scale of an
    adversarial perturbation vector.

    Args:
        model (nn.module): Model to evaluate decision region for.
        x (torch.tensor): Single input.
        y (torch.tensor): Label.
        device (str): device.
        eps (float): Base perturbation region size.
        eps_plot (float, optional): Size of the pertubation region that is actually plotted.
            Allows surrounding regions of base perturbation region. Defaults to eps.
        adv_attack (str, optional): Adversarial attack type. Defaults to 'pgd'.
        adv_norm (str, optional): Adversarial perturbation norm. Defaults to 'Linf'.
        att_n_steps (int, optional): Number of attack steps. Defaults to 40.
        rel_step_size (float, optional): Relative step size. Defaults to 0.1.
        nx (int, optional): Number of x grid steps. Defaults to 500.
        ny (int, optional): Number of y grid steps. Defaults to 500.
        dataset (str, optional): Name of the dataset being analyzed. Defaults to None.
        data_dir (str, optional): Path to the root data directory. Defaults to './data'.
        plotlib (str, optional): Plotting library to use ('plt | matplotlib', 'plotly').
            Defaults to 'plt'.
        savepath (str, optional): Path to save figure to. Defaults to 'decision_region_plot'.

    Returns:
        plt.figure: matplotlib decision region contour plot.
    """
    savedir = os.path.dirname(savepath)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    if not eps_plot:
        eps_plot = eps
    assert eps_plot >= eps, 'Error: eps_plot needs to be equal or larger than eps.'

    xgrid, ygrid, labels = region_grid_pred(
        model, x, y, device, eps_plot, adv_attack, adv_norm, att_n_steps, rel_step_size, nx, ny
    )
    xmesh, ymesh = np.meshgrid(xgrid, ygrid)
    unique_labels = np.unique(labels)

    if plotlib in ['plt', 'matplotlib']:
        plt.autoscale()
        fig, ax = plt.subplots()

        # create colormap legend
        cmap = plt.cm.get_cmap('Paired', lut=len(unique_labels))
        #cmap = plt.cm.get_cmap('tab20', lut=len(unique_labels))

        # plot a pcolormesh, use rasterized to have smaller pdf files
        ax.pcolormesh(xmesh, ymesh, labels, cmap=cmap, rasterized=True) 
        patch_names = [f'{label}' for label in unique_labels]
        if dataset:
            label_dict = dataset_label_names(dataset)
            patch_names = [f'{label} ({label_dict[int(label)]})' for label in unique_labels]

        ax.legend(
            [mpatches.Patch(color=cmap(i)) for i in range(len(unique_labels))],
            patch_names, fontsize=12
        )

        # draw points for natural and adversarial sample
        ax.scatter(0, 0, s=15, c='blue', zorder=2) # nat sample
        ax.scatter(0, eps, s=15, c='red', zorder=2) # nat sample

        # draw base perturbation region
        if eps_plot > eps:
            if adv_norm == 'Linf':
                points = [[0, eps], [eps, 0], [0, -eps], [-eps, 0]]
                base_region = plt.Polygon(points, fill=False, edgecolor='black')
            elif adv_norm == 'L2':
                base_region = plt.Circle((0,0), eps, fill=False, edgecolor='black')
            else:
                raise ValueError(f'Error: unknown adv_norm {adv_norm}.')

            ax.add_patch(base_region)

        # labeling
        ax.set_xlabel('Random Rademacher Direction')
        ax.set_ylabel('Adversarial Direction')

        # save figure
        out_dir = os.path.dirname(savepath)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        savepath = os.path.splitext(savepath)[0]
        fig.savefig(savepath + '.png', dpi=300, bbox_inches="tight")
        fig.savefig(savepath + '.pdf', bbox_inches="tight")

    else:
        #TODO: make the plot pretty
        fig = go.Figure(data=go.Heatmap(x=xgrid, y=ygrid, z=labels))

    return fig


def decision_region_plot_2Ddata(
        model: nn.Module, device: str, xrange: List[float] = [0, 1], yrange: List[float] = [0, 1],
        data: np.ndarray = np.array([]), data_labels: List[int] = [], savepath: str = './plot',
        version: str = 'color', is_acc: List[int] = None, is_rob: List[int] = None, ax: plt.Axes = None,
        set_ticks: bool = True
    ) -> Tuple[plt.Figure,plt.axes]:
    """Plot the decision region of a neural network taking 2D points as inputs.

    Args:
        model (nn.module): Model to evaluate decision region for.
        device (str): device.
        xrange (List[float], optional): xrange to probe. Defaults to [0, 1].
        yrange (List[float], optional): yrange to probe. Defaults to [0, 1].
        data (np.ndarray, optional): Data points to plot. Defaults to np.array([]).
        data_labels (List[int], optional): Labels associated with data. Defaults to [].
        savepath (str, optional): Path to output file. Defaults to './plot'.
        is_acc (List[int], optional): Indicate which samples in data are accurate.
        is_rob (List[int], optional): Indicate which samples in data are robust.
        version (str, optional): Version of visualization ('color' or 'shape'). Defaults to 'color'.
        ax (plt.Axes, optional): If given, plotting will be done on this axis. Defaults to None.
        set_ticks (bool, optional): If set, axis ticks are shown. Defaults to True.

    Returns:
        Tuple[plt.Figure,plt.axes]: Plot figure, axes
    """
    if version == 'shape':
        assert is_acc is not None, "Error: for version='shape', is_acc is required argument"
        assert is_rob is not None, "Error: for version='shape', is_rob is required argument"
    if data is not None:
        assert data_labels is not None, 'Error: provide data_labels'

    nx, ny = 500, 500
    if version == 'shape':
        nx, ny = 1000, 1000
    xgrid = np.linspace(*xrange, num=nx)
    ygrid = np.linspace(*yrange, num=ny)
    labels = np.zeros((ny, nx), dtype=int)
    # colors
    light_blue = np.array([115, 190, 255]) / 255
    light_orange = np.array([255, 215, 140]) / 255
    light_green = np.array([150, 224, 158]) / 255
    colors = ['blue', 'orange', 'green']
    light_colors = np.array([light_blue, light_orange, light_green])
    light_cmap = ListedColormap(light_colors)
    # markers
    markers = ['s', 'v', 'o']

    plt.autoscale()
    fig = None
    if not ax:
        fig, ax = plt.subplots()

    pbar = tqdm(ygrid, dynamic_ncols=True)
    for y_idx, y in enumerate(pbar):
        inputs = np.column_stack((xgrid, np.repeat(y, len(xgrid))))
        inputs = torch.from_numpy(inputs).float().to(device)
        out_batch = model(inputs)
        pred_batch = out_batch.argmax(1)
        labels[y_idx] = pred_batch.cpu().numpy()

    xmesh, ymesh = np.meshgrid(xgrid, ygrid)
    if version == 'color':
        # add a fake predicted label to get all colors in the legend :)
        labels[400,300] = 2
        # plot a pcolormesh, use rasterized to have smaller pdf files
        ax.pcolormesh(xmesh, ymesh, labels, cmap=light_cmap, rasterized=True, shading='auto')

        if data_labels:
            unique_labels = np.unique(data_labels)
            assert len(unique_labels) <= 3, 'Error: only up to 3 classes supported'
            patch_names = [f'{label}' for label in unique_labels]
            ax.legend(
                [mpatches.Patch(color=light_colors[i]) for i in range(len(unique_labels))], patch_names, prop={'size': 14}
            )
    else:
        for y_idx, y in enumerate(ygrid):
            for x_idx, x in enumerate(xgrid): 
                shifts = [[-1,0], [0,-1]] 
                for shift in shifts:
                    neigh = (y_idx+shift[0], x_idx+shift[1])
                    if neigh[0] < 0 or neigh[0] >= ny or neigh[1] < 0 or neigh[1] >= nx:
                        continue
                    if labels[neigh] != labels[(y_idx, x_idx)]:
                        ax.scatter(x, y, c='black', marker=',', s=1)

        if data_labels:
            unique_labels = np.unique(data_labels)
            assert len(unique_labels) <= 3, 'Error: only up to 3 classes supported'
            patch_names = [f'{label}' for label in unique_labels]
            ax.legend(
                [
                    mlines.Line2D([], [], color='grey', marker=markers[i], linestyle='None', markersize=12)
                    for i in range(len(unique_labels))
                ], patch_names, prop={'size': 14}, loc='upper left'
            )

    # plot given data points
    for idx, (point, label) in enumerate(zip(data, data_labels)):
        c = colors[label]
        if version == 'color':
            ax.scatter(point[0], point[1], c=c, marker=markers[label])
        else:
            if is_acc[idx] and is_rob[idx]:
                c = ROBACC_COLORS['ra']
            elif is_acc[idx] and not is_rob[idx]:
                c = ROBACC_COLORS['nra']
            elif not is_acc[idx] and is_rob[idx]:
                c = ROBACC_COLORS['ria']
            else:
                c = ROBACC_COLORS['nria']
            ax.scatter(point[0], point[1], c=np.array([c]), marker=markers[label], s=16)

    # labeling
    ax.set_xlabel('x1', fontsize=14)
    ax.set_ylabel('x2', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if set_ticks:
        ax.set_xticks(np.arange(0, 1.1, step=0.1))
        ax.set_yticks(np.arange(0, 1.1, step=0.1))
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # save figure
    if fig:
        out_dir = os.path.dirname(savepath)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        savepath = os.path.splitext(savepath)[0]
        fig.savefig(savepath + '.png', dpi=300, bbox_inches="tight")
        fig.savefig(savepath + '.pdf', bbox_inches="tight")

    return fig, ax
