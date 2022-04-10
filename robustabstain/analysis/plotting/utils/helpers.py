import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Dict, Union

from robustabstain.loss.revadv import REVADV_LOSSES
from robustabstain.loss.revcert import REVCERT_LOSSES
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.math import roundupto, rounddownto
from robustabstain.utils.paths import eval_smoothing_log_filename


def find_model_names(dir_to_search: str) -> List[str]:
    """Find names of checkpointed models in a directory.

    Args:
        dir_to_search (str): Path to directory with model folders

    Returns:
        List[str]: List of model names found in dir.
    """
    dir_to_search = Path(dir_to_search)
    model_dirs = [p for p in dir_to_search.iterdir() if p.is_dir()]
    model_names = []
    for model_dir in model_dirs:
        model_name = re.match(r'^(\S+?)(?:__\S+?)?$', model_dir.name).group(1)
        model_names.append(model_name)

    return model_names


def find_model_logs(
        dir_to_search: str, attack: str, adv_norm: str, smooth: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Finds evaluation sample_logs for all found models in dir_to_search
    and returns a dict that maps model_name -> evalset -> sample_log_df.

    Args:
        dir_to_search (str): Path to directory with model folders.
        attack (str): Attack type used in evaluation to consider.
        adv_norm (str): Norm of the attack.
        smooth (bool): If set, smoothed logs are returned. Defaults to False.

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Dict mapping model_name (str) -> evalset -> sample_log
    """
    dir_to_search = Path(dir_to_search)
    model_dirs = [p for p in dir_to_search.iterdir() if p.is_dir()]
    model_dfs = {}
    for model_dir in model_dirs:
        model_name = re.match(r'^(\S+?)(?:__\S+?)?$', model_dir.name).group(1)

        df_trainset = get_model_df_from_dir(model_dir, 'train', attack, adv_norm)
        df_testset = get_model_df_from_dir(model_dir, 'test', attack, adv_norm)
        if df_trainset is None and df_testset is None:
            continue

        model_dfs[model_name] = {}
        if df_trainset is not None:
            model_dfs[model_name]['train'] = df_trainset
        if df_testset is not None:
            model_dfs[model_name]['test'] = df_testset

    return model_dfs


def get_model_df_from_dir(
        model_dir: str, evalset: str, attack: str, adv_norm: str, smooth: bool = False
    ) -> pd.DataFrame:
    """Returns model eval logs from model dir.

    Args:
        model_dir (str): Directory logs are stored in
        evalset (str): the split of the dataset evaluated ('train', 'test')
        attack (str): Attack type used in evaluation to consider
        adv_norm (str): Norm of the attack
        smooth (bool): If set, smoothed logs are returned

    Returns:
        pd.DataFrame: DataFrame of the log
    """
    logfile = model_dir / Path(f'{evalset}set_log_{adv_norm}_{attack}.csv')

    df = None
    if logfile.exists():
        df = pd.read_csv(logfile, index_col=0)

    return df


def get_model_df_from_dfs(
        model_dfs: Dict[str, Dict[str, pd.DataFrame]], evalset: str, model_name: str
    ) -> pd.DataFrame:
    """Finds the model log dataframe in dataset_dfs

    Args:
        model_dfs (dict): Dict that maps train_mode -> model_name -> evalset -> sample_log
        evalset (str): the split of the dataset evaluated ('train', 'test')
        model_name (str): Name of the model

    Returns:
        pd.DataFrame: Sample log dataframe for the given model name
    """
    for train_mode, models_dict in model_dfs.items():
        if model_name in models_dict and evalset in models_dict[model_name]:
            return models_dict[model_name][evalset]

    raise ValueError(f'Error: {model_name} {evalset}set eval not found in given model_dfs')


def get_model_name_from_df(log_df: pd.DataFrame) -> str:
    """Returns the name of the model from a model eval log

    Args:
        log_df (pd.DataFrame): Model eval log dataframe

    Returns:
        str: Found model name.
    """
    for col in log_df.columns:
        match = re.match(r'^(\S+?)_is_acc$', col)
        if match:
            return match.group(1)

    raise ValueError("Error: no model_name found, assure that a column '(model_name)_is_acc exists")


def analyze_dataset_dfs(
        model_dfs: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, int]]:
    """Analyze df_dicts by finding all present epsilons (size of perturbation region),
    the number of evaluated samples and the evaluated model names. Further, assert that
    all model sample_logs have the same list of epsilon evaluations
    and the same number of evaluated samples.

    Args:
        model_dfs (dict): Dict that maps train_mode -> model_name -> evalset -> sample_log

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, int]]: Dict mapping evalset -> list of test eps,
            Dict mapping evalset -> list of model names, Dict mapping evalset -> number of samples.
    """
    epsilons = {}
    model_names = {}
    num_samples = {}
    for train_mode, models_dict in model_dfs.items():
        model_names[train_mode] = model_names[train_mode] if train_mode in model_names else list(models_dict.keys())
        assert list(models_dict.keys()) == model_names[train_mode], \
                f'Error: {evalset}set eval does not have all {train_mode} models evaluated'

        for model_name, model_dict in models_dict.items():
            for evalset, model_df in model_dict.items():
                if evalset not in epsilons or evalset not in num_samples:
                    # if no model_df was analyzed yet, analyze it
                    num_samples[evalset] = len(model_df)
                    epsilons[evalset] = []
                    for col in model_df.columns:
                        # match epsilon from the column name
                        match = re.match(r'^(\S+)_is_rob((?:\d+(?:\.\d+)?)|(?:\d+(?:_|\/)\d+))$', col)
                        if match:
                            epsilons[evalset].append(match.group(2))
                else:
                    # if some model_df was already analyzed, check that the current model_df is correct
                    assert num_samples[evalset] == len(model_df), \
                        f'Error: {model_name} {evalset}set eval should have {num_samples} samples, has {len(model_df)} instead.'
                    for eps in epsilons[evalset]:
                        assert f'{model_name}_is_rob{eps}' in model_df.columns, \
                            f'Error: {model_name} {evalset}set eval does not have an evaluation for {eps}'

    return epsilons, model_names, num_samples


def get_rob_inacc_df(
        model_dfs: Dict[str, Dict[str, pd.DataFrame]], evalset: str, model_names: List[str], eps: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract an indicator whether the given models are robust and an indicator
    whether the given models are robust AND inaccurate, for the given epsilon.

    Args:
        model_dfs (Dict[str, Dict[str, pd.DataFrame]]): Dict that maps train_mode -> model_name -> evalset -> sample_log
        evalset (str): the split of the dataset evaluated ('train', 'test')
        model_names (List[str]): List of model names
        eps (str): Perturbation region size

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataframe containing columns '{model_name}_is_rob{eps}' for each model_name,
            Dataframe containing columns column '{model_name}_inacc_rob{eps}' for each model_name
    """
    rob_df = pd.DataFrame()
    inacc_rob_df = pd.DataFrame()
    for model_name in model_names:
        model_df = get_model_df_from_dfs(model_dfs, evalset, model_name)
        rob_df[f'{model_name}_is_rob{eps}'] = model_df[f'{model_name}_is_rob{eps}']
        # add column that describes whether a sample is inaccurate BUT robust on a given model
        inacc_rob_df[f'{model_name}_inacc_rob{eps}'] = (1 - model_df[f'{model_name}_is_acc']) * (model_df[f'{model_name}_is_rob{eps}'])

    return rob_df, inacc_rob_df


def get_accs_from_sample_log(log_path: str) -> Dict[str, float]:
    """Get accuracies from a sample log.

    Args:
        log_path (str): Path to .csv sample log

    Returns:
        Dict[str, float]: Dict containing accuracies.
    """
    log_df = pd.read_csv(log_path, index_col=0)
    try:
        is_acc = log_df[[col for col in log_df.columns if 'is_acc' in col][0]]
    except IndexError:
        raise ValueError(f'Error: is_acc column not found')

    res = {'nat_acc': 100.0 * np.average(is_acc)}
    for col in log_df.columns:
        match = re.match(r'^(\S+)_is_rob((?:\d+(?:\.\d+)?)|(?:\d+(?:_|\/)\d+))$', col)
        if match:
            res[f'adv_acc{match.group(2)}'] = 100.0 * np.average(is_acc & log_df[col].to_numpy())

    print('Accuracies: ', res)
    return res


def get_accs_from_smoothing_log(log_path: str, adv_norm: str, eps: str) -> Dict[str, float]:
    """Get accuracies of a smoothed model from smoothing log.

    Args:
        log_path (str): Path to .csv smoothing log
        adv_norm (str): Adversarial norm.
        eps (str): Size of perturbation region.

    Returns:
        Dict[str, float]: Dict containing accuracies.
    """
    eps_float = convert_floatstr(eps)
    log_df = pd.read_csv(log_path, index_col=0)
    radius_col = 'l2_radius' if adv_norm == 'L2' else 'linf_radius'
    is_acc = (log_df['label'].to_numpy() == log_df['prediction'].to_numpy()).astype(int)
    is_cert = (log_df[radius_col].to_numpy() >= eps_float).astype(int)

    res = {
        'nat_acc': 100.0 * np.average(is_acc),
        'cert_acc': 100.0 * np.average(is_acc & is_cert),
        'rob_inacc': 100.0 * np.average((1-is_acc) & is_cert)
    }

    return res


def find_smoothing_log(args: object, model_dir: str) -> str:
    """Finds smoothing log in model directory.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model_dir (str): Model directory path.

    Returns:
        str: Path to smoothing log file.
    """
    smoothing_logfile = os.path.join(
        model_dir, 'smoothing', eval_smoothing_log_filename(
            args.eval_set, args.smoothing_sigma, args.smoothing_N0, args.smoothing_N
    ))
    if not os.path.isfile(smoothing_logfile):
        raise ValueError(f'Error: no smoothing log found in {model_dir}.')

    return smoothing_logfile


def parse_multi_model_dir(multi_model_dir: str) -> List[str]:
    """Get a checkpointed model paths from a multi model directory containing multiple
    expirement directories.

    Args:
        multi_model_dir (str): Path to multi model directory.
    Returns:
        List[str]: List of model checkpoint paths.
    """
    model_paths = []
    multi_model_dir = Path(multi_model_dir)
    model_dirs = [d for d in multi_model_dir.iterdir() if d.is_dir()]
    for model_dir in model_dirs:
        for f in Path(model_dir).iterdir():
            if str(f).endswith('.pt') and 'last' not in str(f):
                model_paths.append(str(f))
                break

    return model_paths


def get_model_paths(args: object, base_path: str) -> List[str]:
    """Parse a path and return all respective model paths under the base_path.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        base_path (str): Base path to model or top directory.

    Returns:
        List[str]: List of exported .pt models.
    """
    model_paths = []
    is_abstain_trained = ('mrevadv' in base_path) or ('grevadv' in base_path)
    if not base_path.endswith('.pt'):
        # path is a multi model directory, parse it
        model_paths = parse_multi_model_dir(base_path)
    else:
        model_paths = [base_path]
        model_name = os.path.splitext(os.path.basename(base_path))[0]
        model_dir = os.path.dirname(base_path)

        tradeoff_dir = os.path.join(model_dir, 'running_chkpt')
        if os.path.isdir(tradeoff_dir) and is_abstain_trained and args.plot_running_checkpoints:
            # model has multiple checkpointed tradeoff models, evaluated each
            tradeoff_dir = Path(tradeoff_dir)
            tradeoff_model_dirs = sorted(
                [d for d in tradeoff_dir.iterdir() if d.is_dir()],
                key=lambda path: int(os.path.basename(path))
            )
            for tradeoff_model_dir in tradeoff_model_dirs:
                tradeoff_model_path = os.path.join(tradeoff_model_dir, model_name+'.pt')
                model_paths.append(tradeoff_model_path)

    return model_paths


def check_is_abstain_trained(model_path: str) -> bool:
    """Check if model was abstain trained.

    Args:
        model_path (str): Path to model.

    Returns:
        bool: True if model was abstain trained.
    """
    return any(train_mode in model_path for train_mode in REVADV_LOSSES+REVCERT_LOSSES)


def check_is_ace_trained(model_path: str) -> bool:
    """Check if model was ACE trained.

    Args:
        model_path (str): Path to model.

    Returns:
        bool: True if model was ACE trained.
    """
    return 'models/ace' in model_path


def check_det_cert_method(model_path: str) -> str:
    """Check what deterministic certification method was used (if any).

    Args:
        model_path (str): Path to model.

    Returns:
        str: Certification method if found, else ''
    """
    methods = ['COLT', 'IBP']
    used = [meth for meth in methods if meth in model_path]
    if len(used) > 1:
        raise ValueError(f'Error: invalid path {model_path}, more than one cert method was found.')
    elif len(used) == 1:
        return used[0]
    return ''


def pair_sorted(a: List, b: List, index: int) -> Tuple[List, List]:
    """Sort two lists together with one of the two lists dictating the sorting order.

    Args:
        a (List): List 0.
        b (List): List 1.

    Returns:
        Tuple[List, List]: Pair sorted lists.
    """
    assert index in [0,1], 'Error: index must be in [0,1].'
    sa, sb = zip(*sorted(zip(a, b), key=lambda x: x[index]))
    return list(sa), list(sb)


def three_sorted(a: List, b: List, c: List, index: int) -> Tuple[List, List, List]:
    """Sort three lists together with one of the three lists dictating the sorting order.

    Args:
        a (List): List 0.
        b (List): List 1.
        c (List): List 2.

    Returns:
        Tuple[List, List, List]: Three sorted lists.
    """
    assert index in [0,1,2], 'Error: index must be in [0,1].'
    sa, sb, sc = zip(*sorted(zip(a, b, c), key=lambda x: x[index]))
    return list(sa), list(sb), list(sc)


def update_ax_lims(
        lims: List[int], val: Union[float, List[float], Tuple[float]],
        slack: float = 0.0, round_to: float = 5.0
    ) -> List[int]:
    """Update axis limit list to tighten axis limits while still including all values.

    Args:
        lims (List[int]): Axis lower and upper limit.
        val (Union[float, List[float], Tuple[float]]): Newly added value(s).
        slack (float, optional): Relative amount of slack to add to the upper limit.
            Defaults to 0.0.
        round_to (float, optional): Round to next multiple of. Defaults to 5.0.

    Returns:
        List[int]: Updated axis limits.
    """
    min_val, max_val = val, val
    if isinstance(val, list) or isinstance(val, np.ndarray) or isinstance(val, tuple):
        min_val = min(val)
        max_val = max(val)

    lims[0] = min(rounddownto(min_val, round_to), lims[0])
    lims[1] = max(roundupto(max_val, round_to), lims[1])
    lims[1] = min(lims[1], 100)

    lims[1] += (lims[1] - lims[0]) * slack

    return lims