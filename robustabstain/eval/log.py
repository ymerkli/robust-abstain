import numpy as np
import pandas as pd
import json
import logging
import os
import re
from typing import Tuple, List, Dict

from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.paths import (
    eval_report_filename, eval_attack_log_filename,
    eval_smoothing_log_filename
)


def write_eval_report(
        args: object, out_dir: str, model_path: str = '', nat_accs: List[float] = [],
        pc_nat_accs: List[float] = [], adv_accs: Dict[str, Dict[str, float]] = {},
        pc_adv_accs: Dict[str, List[Dict[str, float]]] = {}, adv_attack: str = '',
        pcert_accs: Dict[str, Dict[str, float]] = {}, dcert_accs: Dict[str, Dict[str, float]] = {},
        comp_accs: Dict[str, Dict[str, float]] = {}, label_names: List[str] = [], per_class: bool = False
    ) -> None:
    """Write .json report of evaluated accuracies.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        out_dir (str): Directory to write report file to.
        model_path (str, optional): File path to model checkpoint. Defaults to ''.
        nat_accs (List[float], optional): List with topk natural accuracies. Defaults to [].
        pc_nat_accs (List[float], optional): Per-class top1 natural accuracies. Defaults to [].
        adv_accs (Dict[str, Dict[str, float]], optional): Dict with eps and topk adversarial accuracies. Defaults to {}.
        pc_adv_accs (Dict[str, Dict[str, List[float]]], optional): Dict with eps and per-class top1 adversarial accuracies. Defaults to {}.
        adv_attack (str, optional): Adversarial attack used. Defaults to ''.
        pcert_accs (Dict[str, Dict[str, float]], optional): Dict with probabilistic certification accuracies. Defaults to {}.
        dcert_accs (Dict[str, Dict[str, float]], optional): Dict with deterministic certification accuracies. Defaults to {}.
        comp_accs (Dict[str, Dict[str, float]]): Dict with eps and comp natural, adversarial and certified accuracy
        label_names (List[str], optional): List mapping label index to label name. Defaults to [].
        per_class (bool, optional): If set, per-class accuracies are written (if given). Defaults to False.
    """
    report = {}
    report_file = os.path.join(out_dir, eval_report_filename(args.dataset, args.eval_set))
    if os.path.isfile(report_file):
        with open(report_file, 'r') as fid:
            report = json.load(fid)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    report['info'] = {'model': model_path, 'dataset': args.dataset}

    # write natural accuracies to report
    if nat_accs:
        report['nat_acc'] = {'top1': round(nat_accs[0], 2)}
        if len(nat_accs) > 1:
            report['nat_acc']['top5'] = round(nat_accs[1], 2)

    # write per-class natural accuracies to report
    if pc_nat_accs and per_class:
        if 'per_class' not in report:
            report['per_class'] = {}
        if 'nat_acc' not in report['per_class']:
            report['per_class']['nat_acc'] = {}

        for label_id, pc_nat_acc in enumerate(pc_nat_accs):
            label_name = label_names[label_id] if label_names else str(label_id)
            report['per_class']['nat_acc'][label_name] = round(pc_nat_acc, 2)

    # write adversarial accuracies to report
    if adv_accs:
        assert adv_attack, 'Error: specify the adversarial attack.'
        assert args.adv_norm, 'Error: adv_norm needs to be specified in args'
        if 'adv_acc' not in report:
            report['adv_acc'] = {}
        if args.adv_norm not in report['adv_acc']:
            report['adv_acc'][args.adv_norm] = {}
        if adv_attack not in report['adv_acc'][args.adv_norm]:
            report['adv_acc'][args.adv_norm][adv_attack] = {}

        report['adv_acc'][args.adv_norm][adv_attack]['info'] = {
            'adv_attack': adv_attack,
            'adv_norm': args.adv_norm,
            'test_att_n_steps': args.test_att_n_steps,
            'test_att_step_size': args.test_att_step_size
        }
        for eps, valdict in adv_accs.items():
            if isinstance(valdict, dict):
                for metric, val in valdict.items():
                    if eps not in report['adv_acc'][args.adv_norm][adv_attack]:
                        report['adv_acc'][args.adv_norm][adv_attack][eps] = {}
                    report['adv_acc'][args.adv_norm][adv_attack][eps][metric] = val
            else:
                report['adv_acc'][args.adv_norm][adv_attack][eps] = val

    # write per-class adversarial accuracies to report
    if pc_adv_accs and per_class:
        if 'per_class' not in report:
            report['per_class'] = {}
        if 'adv_acc' not in report['per_class']:
            report['per_class']['adv_acc'] = {}
        if args.adv_norm not in report['per_class']['adv_acc']:
            report['per_class']['adv_acc'][args.adv_norm] = {}
        if adv_attack not in report['per_class']['adv_acc'][args.adv_norm]:
            report['per_class']['adv_acc'][args.adv_norm][adv_attack] = {}

        for eps, adv_accs in pc_adv_accs.items():
            if eps not in report['per_class']['adv_acc'][args.adv_norm][adv_attack]:
                report['per_class']['adv_acc'][args.adv_norm][adv_attack][eps] = {}

            for label_id, val in enumerate(adv_accs):
                label_name = label_names[label_id] if label_names else str(label_id)
                report['per_class']['adv_acc'][args.adv_norm][adv_attack][eps][label_name] = val

    # write smoothing certified accuracies to report
    if pcert_accs:
        if 'cert_acc' not in report:
            report['cert_acc'] = {}
        if args.smoothing_sigma not in report['cert_acc']:
            report['cert_acc'][args.smoothing_sigma] = {}
        if args.adv_norm not in report['cert_acc'][args.smoothing_sigma]:
            report['cert_acc'][args.smoothing_sigma][args.adv_norm] = {}

        report['cert_acc'][args.smoothing_sigma]['info'] = {
            'sigma': args.smoothing_sigma,
            'N0': args.smoothing_N0,
            'N': args.smoothing_N,
            'alpha': args.smoothing_alpha
        }
        for eps, val in pcert_accs.items():
            report['cert_acc'][args.smoothing_sigma][args.adv_norm][eps] = val

    if dcert_accs:
        if 'cert_acc' not in report:
            report['cert_acc'] = {}
        if args.cert_domain not in report['cert_acc']:
            report['cert_acc'][args.cert_domain] = {}
        if args.adv_norm not in report['cert_acc'][args.cert_domain]:
            report['cert_acc'][args.cert_domain][args.adv_norm] = {}

        for eps, val in dcert_accs.items():
            report['cert_acc'][args.cert_domain][args.adv_norm][eps] = val

    # write compositional accuracies to report
    if comp_accs:
        if 'comp_acc' not in report:
            report['comp_acc'] = {}
        if adv_attack not in report['comp_acc']:
            report['comp_acc'][adv_attack] = {}

        for eps, comp_eval in comp_accs.items():
            if eps not in report['comp_acc'][adv_attack]:
                report['comp_acc'][adv_attack][eps] = {}
            for comp_key, selector_eval in comp_eval.items():
                if comp_key not in report['comp_acc'][adv_attack][eps]:
                    report['comp_acc'][adv_attack][eps][comp_key] = {}
                for selector_method, valdict in selector_eval.items():
                    if isinstance(valdict, dict):
                        for metric, val in valdict.items():
                            report['comp_acc'][adv_attack][eps][comp_key][selector_method][metric] = val
                    else:
                        report['comp_acc'][adv_attack][eps][comp_key][selector_method] = val

    logging.info(f'Writing evaluation report to {report_file}')
    with open(report_file, 'w') as fid:
        json.dump(report, fid, sort_keys=True, indent=4, separators=(',', ': '))


def write_sample_log(
        model_name: str, log_dir: str, dataset: str, eval_set: str, adv_norm: str,
        adv_attack: str, indices: np.ndarray, is_acc: np.ndarray = None, preds: np.ndarray = None,
        nat_predconf: np.ndarray = None, adv_predconf: np.ndarray = None,
        is_rob: np.ndarray = None, is_cert: np.ndarray = None,
        is_select: np.ndarray = None, select_rob: np.ndarray = None, select_cert: np.ndarray = None,
        eps: str = '', log_filename: str = None
    ) -> None:
    """Write one log per test_eps that for each test sample contains binary variables that
    tell whether each is sample is accurate, robust on each given model.

    Args:
        model_name (str): Model identifier.
        log_dir (str): Directory to write the log to.
        dataset (str): Name of the evaluated dataset.
        eval_set (str): Dataset split that is evaluated ('train' or 'test').
        adv_norm (str): Norm of the adv attack.
        adv_attack (str): Adversarial attack used.
        indices (np.ndarray): Sample indices that were evaluated.
        is_acc (np.ndarray, optional): Binary indicator array indicating
            natural accurate samples. Defaults to None.
        preds (np.ndarray, optional): Natural predictions. Defaults to None.
        nat_predconf (np.ndarray, optional): Confidences of the top1 natural prediction. Defaults to None.
        adv_predconf (np.ndarray, optional): Confidences of the top1 adversarial prediction. Defaults to None.
        is_rob (np.ndarray, optional): Binary indicator array indicating
            robust samples (for given attack/adv norm/eps). Defaults to None.
        is_cert (np.ndarray, optional): Binary indicator array indicating
            certified samples (for given norm/eps). Defaults to None.
        is_select (np.ndarray, optional): Binary indicator array indicating
            selected samples in a compositional architecture (used for ACE). Defaults to None.
        select_rob (np.ndarray, optional): Binary indicator array indicating
            samples for which is_select is empirically robust under adversarial attack.
            Defaults to None.
        select_cert (np.ndarray, optional): Binary indicator array indicating
            samples for which is_select is certifiably robust. Defaults to None.
        eps (str): Perturbation region size (stringified). Must be specified
            when writing is_rob. Defaults to ''.
        log_filename (str, optional): Custom log filename. Defaults to None.
    """
    if not log_filename:
        log_filename = eval_attack_log_filename(eval_set, dataset, adv_norm, adv_attack)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, log_filename)
    res_df = pd.DataFrame() # log file to append to
    if os.path.isfile(log_file):
        res_df = pd.read_csv(log_file, index_col=0)
        if len(res_df) == len(indices):
            # existing log has same number of samples, check for equal sample order
            assert (res_df['sample_idx'].to_numpy() == indices).all(), \
                f'Error: existing log file {log_file} and provided log have non-agreeing sample_idx.'
        else:
            # existing log has different number of samples - overwrite with new log
            logging.info(f'Existing log {log_file} has different samples evaluated, overwriting existing log')
            res_df = pd.DataFrame() # log file to append to

    res_df['sample_idx'] = indices
    # log natural evaluations
    if is_acc is not None:
        res_df[f'{model_name}_is_acc'] = is_acc
    if preds is not None:
        res_df[f'{model_name}_pred'] = preds
    if nat_predconf is not None:
        res_df[f'{model_name}_nat_conf'] = nat_predconf

    # log adversarial evaluations
    if is_rob is not None:
        assert eps, 'Error: specify perturbation region size eps when writing is_rob to log.'
        res_df[f'{model_name}_is_rob{eps}'] = is_rob
    if adv_predconf is not None:
        res_df[f'{model_name}_adv_conf{eps}'] = adv_predconf

    # log certification evaluations
    if is_cert is not None:
        assert eps, 'Error: specify perturbation region size eps when writing is_cert to log.'
        res_df[f'{model_name}_is_cert{eps}'] = is_cert

    # log selector decision
    if is_select is not None:
        res_df[f'{model_name}_is_select'] = is_select

    # log selector empirical robustness
    if select_rob is not None:
        res_df[f'{model_name}_select_rob'] = select_rob

    # log selector empirical robustness
    if is_select is not None:
        res_df[f'{model_name}_select_cert'] = select_cert

    # sort columns with correct numerical order of perturbation region size
    def sort_friendly_eps_str(elem: str) -> str:
        """Replace fraction representations (e.g. 1_255) of epsilon strings by their
        float representation to ensure correct numerical sorting.
        """
        match_rob = re.match(r'^\S+?_is_rob(\d+(?:\/|_|\.)\d+)$', elem)
        match_cert = re.match(r'^\S+?_is_cert(\d+(?:\/|_|\.)\d+)$', elem)
        match_advconf = re.match(r'^\S+?_adv_predconf(\d+(?:\/|_|\.)\d+)$', elem)
        if match_rob:
            eps_str = match_rob.group(1)
            return elem.replace(f'is_rob{eps_str}', f'is_rob{convert_floatstr(eps_str)}')
        elif match_cert:
            eps_str = match_cert.group(1)
            return elem.replace(f'is_cert{eps_str}', f'is_cert{convert_floatstr(eps_str)}')
        elif match_advconf:
            eps_str = match_advconf.group(1)
            return elem.replace(f'adv_predconf{eps_str}', f'adv_predconf{convert_floatstr(eps_str)}')
        else:
            return elem

    cols_order = list(res_df.columns)
    cols_order.remove('sample_idx')
    cols_order.sort(key=sort_friendly_eps_str)
    cols_order.insert(0, 'sample_idx')
    res_df = res_df[cols_order]

    logging.info(f'Writing sample log {log_filename} to {log_file}')
    res_df.to_csv(log_file, sep=',')


def write_smoothing_log(
        args: object, log_dir: str, eval_set: str, linf_radii: np.ndarray, l2_radii: np.ndarray,
        base_predictions: np.ndarray, predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray
    ) -> None:
    """ Write log file for smoothing certification.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        log_dir (str): Directory to write the log to
        eval_set (str): Dataset split that is evaluated ('train' or 'test')
        linf_radii (np.ndarray): List of certified Linf radii of test samples
        l2_radii (np.ndarray): List of certified L2 radii of test samples
        base_predictions (np.ndarray): Predictions of test samples by base classifier
        predictions (np.ndarray): Predictions of test samples by the smoothed classifier
        labels (np.ndarray): [description]
        indices (np.ndarray): [description]
    """
    out_dir = os.path.join(log_dir, 'smoothing')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    log_name = eval_smoothing_log_filename(eval_set, args.smoothing_sigma, args.smoothing_N0, args.smoothing_N)
    log_file = os.path.join(out_dir, log_name)
    log_df = pd.DataFrame()

    log_df['sample_idx'] = indices.astype(int)
    log_df['label'] = labels.astype(int)
    log_df['base_prediction'] = base_predictions.astype(int)
    log_df['prediction'] = predictions.astype(int)
    log_df['linf_radius'] = linf_radii
    log_df['l2_radius'] = l2_radii

    logging.info(f'Writing smoothing log {log_name} to {log_file}')
    log_df.to_csv(log_file, sep=',')