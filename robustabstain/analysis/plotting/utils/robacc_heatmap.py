import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Tuple, Union

from robustabstain.analysis.plotting.utils.helpers import get_model_df_from_dfs, get_model_name_from_df
from robustabstain.utils.data_utils import DATASETS


def discrete_colorscale(bvals: List[float], colors: List[str]) -> List[List[Union[float, str]]]:
    """Creates a plotly discrete colorscale.

    Args:
        bvals (List[float]): List of values bounding intervals/ranges of interest
        colors (List[str]): List of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1

    Returns:
        List[List[Union[float, str]]]: Discrete colorscale mapping bval bounds to colors.
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values

    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])

    return dcolorscale


def get_robacc_heatmap(
        dataset: str, evalset: str, eps: str, sort_ref: str = None,
        subsample: int = None, model_names: List[str] = None,
        model_dfs: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = None,
        model_logs: List[str] = None, model_ids: List[str] = None, title: str = None,
        horizontal: bool = False
    ) -> Tuple[go.Figure, pd.DataFrame]:
    """Creates a heatmap over samples of a given dataset split, indicating for each sample whether given models
    are (in-)accurate and/or (non-)robust for the given epsilon.

    Args:
        dataset (str): the dataset to examine
        evalset (str): the split of the dataset evaluated ('train', 'test')
        eps (str): size of perturbation region (e.g. '1_255')
        sort_ref (str): the model according to which samples are sorted
        subsample (int, optional): Number of samples to subsample. Defaults to None.
        model_names (List[str], optional): List of model names to plot. Defaults to None.
        model_dfs (Dict[str, Dict[str, Dict[str, pd.DataFrame]]], optional): Dict that maps
            train_mode -> model_name -> evalset -> sample_log. Must be set if model_names is set. Defaults to None.
        model_logs (List[str], optional): List of eval log paths to plot. Defaults to None.
        model_ids (List[str], optional): List of unique model identifers to label plots.
            Must be in the same order as model_dfs/model_logs. Defaults to None.
        horizontal (bool, optional): If set, heatmaps are horizontal.

    Returns:
        Tuple[go.Figure, pd.DataFrame]: Plotly heatmap plot, summary dataframe.
    """
    x, y, z, z_text, num_inacc_rob = [], [], [], [], []
    summary_df = pd.DataFrame(
        index=model_names, columns=[
            'Nat. Acc.', f'Adv. Acc (eps = {eps})',
            'Fraction of inaccurate & robust samples', 'Fraction of accurate & non-robust samples'
    ])

    # construct dict of model logs
    model_dicts: List[Dict[str, Union[str, pd.DataFrame]]] = []
    if model_names:
        assert model_dfs is not None, 'Error: in model_names is given, model_dfs must be given.'
        for model_name in model_names:
            model_log = get_model_df_from_dfs(model_dfs, evalset, model_name)
            model_dicts.append({'name': model_name, 'log': model_log})
    else:
        assert model_logs is not None, 'Error: if no model_names are given, model_logs must be specified'
        model_names = []
        for i, logfile in enumerate(model_logs):
            assert os.path.isfile(logfile), f'Error: log file {logfile} does not exist'
            model_log = pd.read_csv(logfile, index_col=0)
            model_name = get_model_name_from_df(model_log)
            model_names.append(model_name)
            model_dicts.append({'name': model_name, 'log': model_log})

    for i, d in enumerate(model_dicts):
        model_name = d['name']
        model_id = model_name if not model_ids else model_ids[i]
        if model_id in y:
            model_id = f'{model_id}_{i}'
        log_df = d['log']
        # if given, subsample the dataframe
        subsample = len(log_df) if subsample is None else subsample
        log_df = log_df.sample(subsample).sort_index() if subsample is not None else log_df
        x = [str(sample_idx) for sample_idx in log_df['sample_idx'].to_numpy()]

        data = np.zeros(len(log_df))
        anno = np.empty(len(log_df), dtype='<U256')

        is_acc = log_df[f'{model_name}_is_acc'].to_numpy(dtype=np.int64)
        is_rob = log_df[f'{model_name}_is_rob{eps}'].to_numpy(dtype=np.int64)

        noacc_norob_index = ((log_df[f'{model_name}_is_acc'] == 0) & (log_df[f'{model_name}_is_rob{eps}'] == 0)).to_numpy()
        noacc_rob_index = ((log_df[f'{model_name}_is_acc'] == 0) & (log_df[f'{model_name}_is_rob{eps}'] == 1)).to_numpy()
        acc_norob_index = ((log_df[f'{model_name}_is_acc'] == 1) & (log_df[f'{model_name}_is_rob{eps}'] == 0)).to_numpy()
        acc_rob_index = ((log_df[f'{model_name}_is_acc'] == 1) & (log_df[f'{model_name}_is_rob{eps}'] == 1)).to_numpy()

        nat_acc = 100.0 * np.average(is_acc)
        rob_acc = 100.0 * np.average(is_acc & is_rob)
        inacc_rob_frac = 100.0 * np.average((1-is_acc) & is_rob)
        acc_nonrob_frac = 100.0 * np.average(is_acc * (1-is_rob))

        # evenly spaced values to set color levels in plotly heatmap
        data[acc_rob_index] = 0
        data[acc_norob_index] = 0.25
        data[noacc_rob_index] = 0.5
        data[noacc_norob_index] = 0.75

        # text annoatations
        anno[acc_rob_index] = 'robust & accurate'
        anno[acc_norob_index] = 'non-robust & accurate'
        anno[noacc_rob_index] = 'robust & inaccurate'
        anno[noacc_norob_index] = 'non-robust & inaccurate'

        z.append(list(data))
        z_text.append(anno)
        num_inacc_rob.append(noacc_rob_index.sum())
        y.append(model_id)
        summary_df.loc[model_name] = [nat_acc, rob_acc, inacc_rob_frac, acc_nonrob_frac]

    # sort all samples according to reference model (reference model is the model with most inaccurate, robust samples)
    sort_ref = sort_ref if sort_ref is not None else model_names[0]
    sort_index = z[model_names.index(sort_ref)]
    x = [elem for _, elem in sorted(zip(sort_index, x), key=lambda pair: pair[0])]
    for i in range(len(z)):
        z[i] = [elem for _, elem in sorted(zip(sort_index, z[i]), key=lambda pair: pair[0])]
        z_text[i] = [elem for _, elem in sorted(zip(sort_index, z_text[i]), key=lambda pair: pair[0])]

    # color settings
    bvals = [0.0, 0.25, 0.5, 0.75, 1]
    tickvals = [0.1, 0.275, 0.475, 0.675]
    ticktext = ['robust & accurate', 'non-robust & accurate', 'robust & inaccurate', 'non-robust & inaccurate']
    colors = ['rgb(0, 90, 181)', 'rgb(153, 204, 255)', 'rgb(179, 41, 25)', 'rgb(242, 173, 166)'] # light red / dark red / light blue / dark blue
    colorscale = discrete_colorscale(bvals, colors)
    colorbar = dict(
        thickness=15, ticktext=ticktext, tickvals=tickvals, len=0.3, lenmode='fraction',
        tickfont=dict(size=16)
    )

    # prepare plot
    if horizontal:
        heatmap = go.Heatmap(x=x, y=y, z=z, zmin=0, zmax=0.75, text=z_text, colorscale=colorscale, colorbar=colorbar)
        xaxis = go.layout.XAxis(title=dict(text='Model'))
        yaxis = go.layout.YAxis(title=dict(text='Sample Index'))
    else:
        z = np.array(z).transpose()
        heatmap = go.Heatmap(x=y, y=x, z=z, zmin=0, zmax=0.75, text=z_text, colorscale=colorscale, colorbar=colorbar)
        xaxis = go.layout.XAxis(title=dict(text='Sample Index'))
        yaxis = go.layout.YAxis(title=dict(text='Model'))

    title = f'{dataset} per-sample accuracy & robustness model analysis, eps = {eps}' if title == None else title
    layout = go.Layout(title=title, xaxis=xaxis, yaxis=yaxis)
    fig = go.Figure(data=heatmap, layout=layout)

    return fig, heatmap, summary_df


def plot_comp_models(
        log_dir: str, evalset: str, eps: str, adv_norm: str = None, adv_attack: str = None, sort_ref: str = None
    ) -> Tuple[go.Figure, pd.DataFrame]:
    """Get robacc heatmap plot for compositional models.

    Args:
        log_dir (str): Path to directory containing compositional model logs.
        evalset (str): Dataset split being evaluated ('train', 'test').
        eps (str): Perturbation region size.
        adv_norm (str, optional): Adversarial norm. Defaults to None.
        adv_attack (str, optional): Adversarial attack type. Defaults to None.
        sort_ref (str, optional): Model to use as reference for samples order. Defaults to None.

    Returns:
        Tuple[go.Figure, pd.DataFrame]: Plotly heatmap plot, summary dataframe.
    """
    try:
        dataset = [dataset for dataset in DATASETS if dataset in log_dir][0]
    except IndexError:
        raise ValueError(f'Error: no dataset found on path {log_dir}')

    core_log_file, rob_log_file = None, None
    if adv_norm and adv_attack:
        core_log_file = f'{evalset}set_core_log_{adv_norm}_{adv_attack}.csv'
        rob_log_file = f'{evalset}set_rob_log_{adv_norm}_{adv_attack}.csv'
    else:
        for file in Path(log_dir).iterdir():
            if f'{evalset}set_core_log' in file.name:
                core_log_file = file.absolute()
            elif f'{evalset}set_rob_log' in file.name:
                rob_log_file = file.absolute()
            else:
                continue

    if not core_log_file or not rob_log_file:
        raise ValueError(f'Error: not all logs found in {log_dir}')

    return get_robacc_heatmap(dataset, evalset, eps, sort_ref=sort_ref, model_logs=[core_log_file, rob_log_file])

