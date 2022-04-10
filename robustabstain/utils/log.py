import os
import datetime
import inspect
import json
import logging
import pandas as pd
from typing import Tuple, List
from cox.utils import Parameters

from robustabstain.utils.paths import model_out_dir


def default_serialization(obj: object) -> str:
    """Pretty print unserializable object for json dumping.

    Args:
        obj (object): Any python object.

    Returns:
        str: Serializable informative string.
    """
    qualifier = None
    if inspect.ismethod(obj) or inspect.isfunction(obj):
        qualifier = obj.__qualname__
    else:
        qualifier = obj.__class__.__qualname__

    return f'<non-serializable: {qualifier}>'


def write_config(args: object, path: str) -> None:
    """Write training config (argparse arguments) to file.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        path (str): path to store to
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    if type(args) == dict:
        pass
    elif type(args) == Parameters:
        args = vars(args)['params']
    else:
        args = vars(args)

    args_file = os.path.join(path, 'args.json')
    with open(args_file, 'w') as fid:
        json.dump(args, fid, indent=4, default=default_serialization)


def logging_setup(args: object, train_mode: str) -> Tuple[str, str, str, str]:
    """Construct logging strings and paths

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        train_mode (str): Employed training mode

    Returns:
        Tuple[str, str, str, str]: Output directory path, experiment directory path,
        experiment identifier, checkpoint path
    """
    out_dir = model_out_dir(train_mode, args.dataset)
    time_log = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    exp_id, chkpt_name = None, None
    # indicate when a model is finetuned instead of being trained from scratch
    finetune = 'ft' if args.finetune else ''
    weighted_loss = 'w' if args.weighted_loss else ''

    if train_mode == 'std':
        exp_id = f'{args.arch}_{args.dataset}_{weighted_loss}{train_mode}__{time_log}'
        chkpt_name = f'{args.arch}_{args.dataset}_{weighted_loss}{train_mode}.pt'
        if args.data_aug:
            exp_id = f'{args.arch}_{args.dataset}_{args.data_aug}_{weighted_loss}{train_mode}_{time_log}'
            chkpt_name = f'{args.arch}_{args.dataset}_{args.data_aug}_{weighted_loss}{train_mode}.pt'
    elif train_mode == 'adv':
        out_dir = os.path.join(out_dir, args.adv_norm)
        exp_id = f'{args.arch}_{args.dataset}_{weighted_loss}{args.defense}{args.train_eps}{finetune}__{time_log}'
        chkpt_name = f'{args.arch}_{args.dataset}_{weighted_loss}{args.defense}{args.train_eps}.pt'
        if args.data_aug:
            exp_id = f'{args.arch}_{args.dataset}_{args.data_aug}_{weighted_loss}{args.defense}{args.train_eps}{finetune}__{time_log}'
            chkpt_name = f'{args.arch}_{args.dataset}_{args.data_aug}_{weighted_loss}{args.defense}{args.train_eps}.pt'
    elif train_mode == 'smoothadv':
        out_dir = os.path.join(out_dir, args.adv_norm)
        exp_id = f'{args.arch}_{args.dataset}_{weighted_loss}smoothadv{args.train_eps}{finetune}__{time_log}'
        chkpt_name = f'{args.arch}_{args.dataset}_{weighted_loss}smoothadv{args.train_eps}.pt'
        if args.data_aug:
            exp_id = f'{args.arch}_{args.dataset}_{args.data_aug}_{weighted_loss}smoothadv{args.train_eps}{finetune}__{time_log}'
            chkpt_name = f'{args.arch}_{args.dataset}_{args.data_aug}_{weighted_loss}smoothadv{args.train_eps}.pt'
    elif train_mode == 'augm':
        exp_id = f'{args.arch}_{args.dataset}_{weighted_loss}gaussaugm{args.noise_sd}{finetune}__{time_log}'
        chkpt_name = f'{args.arch}_{args.dataset}_{weighted_loss}gaussaugm{args.noise_sd}.pt'
        if args.data_aug:
            exp_id = f'{args.arch}_{args.dataset}_{args.data_aug}_{weighted_loss}gaussaugm{args.noise_sd}{finetune}__{time_log}'
            chkpt_name = f'{args.arch}_{args.dataset}_{args.data_aug}_{weighted_loss}gaussaugm{args.noise_sd}.pt'
    elif train_mode == 'mrevadv':
        out_dir = os.path.join(out_dir, args.adv_norm)
        if args.branch_model:
            branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        else:
            branch_model_name = f'{args.arch}_{args.dataset}_FS{args.train_eps}'
        exp_id = f'mra{args.train_eps}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_mra{args.train_eps}.pt'
        if args.data_aug:
            exp_id = f'mra{args.train_eps}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_mra{args.train_eps}.pt'
    elif train_mode == 'grevadv':
        out_dir = os.path.join(out_dir, args.adv_norm)
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'gra{args.train_eps}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_gra{args.train_eps}.pt'
        if args.data_aug:
            exp_id = f'gra{args.train_eps}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_gra{args.train_eps}.pt'
    elif train_mode == 'mrevadv_conf':
        out_dir = os.path.join(out_dir, args.adv_norm)
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'mrac{args.train_eps}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_mrac{args.train_eps}.pt'
        if args.data_aug:
            exp_id = f'mrac{args.train_eps}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_mrac{args.train_eps}.pt'
    elif train_mode == 'smoothmrevadv':
        out_dir = os.path.join(out_dir, args.adv_norm)
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'smra{args.train_eps}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_smra{args.train_eps}.pt'
        if args.data_aug:
            exp_id = f'smra{args.train_eps}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_smra{args.train_eps}.pt'
    elif train_mode == 'smoothgrevadv':
        out_dir = os.path.join(out_dir, args.adv_norm)
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'sgra{args.train_eps}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_sgra{args.train_eps}.pt'
        if args.data_aug:
            exp_id = f'sgra{args.train_eps}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_sgra{args.train_eps}.pt'
    elif train_mode == 'revnoise':
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'rn{args.noise_sd}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_rn{args.noise_sd}.pt'
        if args.data_aug:
            exp_id = f'rn{args.noise_sd}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_rn{args.noise_sd}.pt'
    elif train_mode == 'mrevcert':
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'mrc{args.noise_sd}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_mrc{args.noise_sd}.pt'
        if args.data_aug:
            exp_id = f'mrc{args.noise_sd}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_mrc{args.noise_sd}.pt'
    elif train_mode == 'nrevcert':
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'nrc{args.noise_sd}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_nrc{args.noise_sd}.pt'
        if args.data_aug:
            exp_id = f'nrc{args.noise_sd}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_nrc{args.noise_sd}.pt'
    elif train_mode == 'selector':
        out_dir = os.path.join(out_dir, args.adv_norm)
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'sel{args.train_eps}__{branch_model_name}__{time_log}'
        chkpt_name = f'sel{args.train_eps}__{branch_model_name}.pt'
        if args.data_aug:
            exp_id = f'sel{args.train_eps}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'sel{args.train_eps}_{args.data_aug}__{branch_model_name}.pt'
    elif train_mode == 'revcertrad':
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'rcr{args.noise_sd}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_rcr{args.noise_sd}.pt'
        if args.data_aug:
            exp_id = f'rcr{args.noise_sd}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_rcr{args.noise_sd}.pt'
    elif train_mode == 'revcertnoise':
        branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
        exp_id = f'rcn{args.noise_sd}__{branch_model_name}__{time_log}'
        chkpt_name = f'{branch_model_name}_rcn{args.noise_sd}.pt'
        if args.data_aug:
            exp_id = f'rcn{args.noise_sd}_{args.data_aug}__{branch_model_name}__{time_log}'
            chkpt_name = f'{branch_model_name}_{args.data_aug}_rcn{args.noise_sd}.pt'
    else:
        raise ValueError(f'Error: unknown traininig mode {train_mode}.')

    exp_dir = os.path.join(out_dir, exp_id)
    chkpt_path = os.path.join(exp_dir, chkpt_name)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    return out_dir, exp_dir, exp_id, chkpt_path


def init_logging(args: object, exp_dir: str = '', logfile: str = 'training.log') -> List:
    """Initialize log file and log.csv.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        exp_dir (str): Experiment directory
        logfile (str, optional)

    Returns:
        List: Empty array for adding log object into which can then be used in
            combination with robustabstain.utils.log.log().
    """
    # Remove all handlers associated with the root logger object
    # Required to reset a basicConfig that may have been set previously
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.StreamHandler()]
    if exp_dir:
        handlers.append(logging.FileHandler(os.path.join(exp_dir, logfile)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=handlers
    )
    logging.info('Args: %s', args)

    return []


def log(filename: str, log_list: List[dict], log_dict: dict) -> List[dict]:
    """ Append log row to log_list, write update log to csv table and to
    pretty print markdown table.

    Args:
        filename (str): Path to file.
        log (List[dict]): List with log entries.
        log_dict (dict): Dictionary with row to log.

    Returns:
        List[dict]: log_list with new logging row appended.
    """
    log_list.append(log_dict)
    log_df = pd.DataFrame(log_list)
    log_df.to_csv(filename, sep=',', index=False, float_format='%.5f')

    md_filename = f'{os.path.splitext(filename)[0]}.md'
    md_table = log_df.to_markdown(index=False, floatfmt=".5f")
    with open(md_filename, 'w') as f:
        f.write(md_table)

    return log_list
