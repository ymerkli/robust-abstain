import setGPU
import torch
import torch.nn as nn

import datetime
import logging
import os
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple, Dict, List

from robustabstain.eval.nat import natural_eval
from robustabstain.eval.log import write_smoothing_log, write_eval_report
from robustabstain.certify.smoothing import Smooth
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.loaders import get_subsampled_dataset
from robustabstain.utils.paths import eval_smoothing_log_filename


def smoothing_eval_batch(
        args: object, model: nn.Module, smoothed_model: Smooth, device: str, curr_idx: int,
        inputs: torch.tensor, targets: torch.tensor, sample_indices: torch.tensor,
        log_indices: np.ndarray = np.array([])
    ):
    idx = curr_idx
    l2_radii_batch = np.array([], dtype=np.float64)
    linf_radii_batch = np.array([], dtype=np.float64)
    labels_batch = np.array([], dtype=np.int64)
    indices_batch = np.array([], dtype=np.int64)
    base_predictions_batch = np.array([], dtype=np.int64)
    predictions_batch = np.array([], dtype=np.int64)

    inputs, targets = inputs.to(device), targets.to(device)
    for x, target, sample_index in zip(inputs, targets, sample_indices):
        if sample_index.item() in log_indices:
            # samples that were already certified can be skipped
            assert log_indices[idx] == sample_index, "sample_idx order of read smoothing_log does not match."
            idx += 1
            continue

        before_time = time.time()
        base_prediction = model(x.unsqueeze(0)).argmax(1).cpu().item()
        prediction, radius, counts = smoothed_model.certify_counts(
            x, args.smoothing_N0, args.smoothing_N, args.smoothing_alpha, batch_size=args.smoothing_batch
        )
        time_elapsed = str(datetime.timedelta(seconds=(time.time() - before_time)))
        correct = int(prediction == target)

        indices_batch = np.append(indices_batch, sample_index.cpu().numpy())
        labels_batch = np.append(labels_batch, target.cpu().numpy())
        base_predictions_batch = np.append(base_predictions_batch, base_prediction)
        predictions_batch = np.append(predictions_batch, prediction)
        l2_radii_batch = np.append(l2_radii_batch, radius)
        linf_radii_batch = np.append(linf_radii_batch, radius / math.sqrt(x.numel()))

        logging.info("{idx:<7}\t{sample_idx:<10}\t{label:<10}\t{base_predict:<15}\t{predict:<10}\t{radius:<10.4f}\t{correct:<10}\t{time:<10}".format(
            idx=idx, sample_idx=sample_index, label=target, base_predict=base_prediction,
            predict=prediction, radius=radius, correct=correct, time=time_elapsed
        ))
        idx += 1

    return linf_radii_batch, l2_radii_batch, base_predictions_batch, predictions_batch, labels_batch, indices_batch


def smoothing_eval(
        args: object, model: nn.Module, model_dir: str, device: str,
        test_loader: torch.utils.data.DataLoader, test_eps: List[str],
        eval_set: str = 'test', use_exist_log: bool = False
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate smoothed model on given test data.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): Model to apply randomized smoothing to.
        model_dir (str): The model directory. Required to check for existing logs.
        device (str): device
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data.
        test_eps (List[str]): Test eps (in string form, e.g. '1_255')
        eval_set (str):  Dataset split to evaluate. Defaults to 'test'.
        use_exist_log (bool, optional): If set, existing log file is used (if available). Defaults to False.

    Returns:
        Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Certified accuracies for each test eps, Linf radii, L2 radii, base model predictions,
            smooth model predictions, labels, sample indices
    """
    logging.info(f'Evaluating smoothing certified robustness using sigma={args.smoothing_sigma}, N0={args.smoothing_N0}, N={args.smoothing_N}')
    model.eval()

    _, _, num_classes = get_dataset_stats(args.dataset)
    smoothing_sigma_float = convert_floatstr(args.smoothing_sigma)

    smoothed_model = Smooth(model, num_classes, smoothing_sigma_float, device)
    l2_radii, linf_radii, labels, indices = np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    base_predictions, predictions = np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # if smoothing log is found, use logged data
    smoothing_log_filename = eval_smoothing_log_filename(eval_set, args.smoothing_sigma, args.smoothing_N0, args.smoothing_N)
    smoothing_log_file = os.path.join(model_dir, 'smoothing', smoothing_log_filename)
    if os.path.isfile(smoothing_log_file) and use_exist_log:
        logging.info(f'==> Reading existing smoothing log {smoothing_log_filename}')
        smoothing_log_df = pd.read_csv(smoothing_log_file, index_col=0)

        indices = smoothing_log_df['sample_idx'].to_numpy()
        labels = smoothing_log_df['label'].to_numpy()
        base_predictions = smoothing_log_df['base_prediction'].to_numpy()
        predictions = smoothing_log_df['prediction'].to_numpy()
        linf_radii = smoothing_log_df['linf_radius'].to_numpy()
        l2_radii = smoothing_log_df['l2_radius'].to_numpy()

    logging.info(f'==> Running smoothing certification with sigma={args.smoothing_sigma}, N={args.smoothing_N}, N0={args.smoothing_N0}, alpah={args.smoothing_alpha}')
    logging.info("{idx:<7}\t{sample_idx:<10}\t{label:<10}\t{base_predict:<15}\t{predict:<10}\t{radius:<10}\t{correct:<10}\t{time:<10}".format(
        idx='idx', sample_idx='sample_idx', label='label', base_predict='base_predict',
        predict='predict', radius='radius', correct='correct', time='time'
    ))
    idx = 0
    for batch_idx, (inputs, targets, sample_indices) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        for x, target, sample_index in zip(inputs, targets, sample_indices):
            if sample_index.item() in indices:
                # samples that were already certified can be skipped
                assert indices[idx] == sample_index, "sample_idx order of read smoothing_log does not match."
                linf_radii[idx] = l2_radii[idx] / math.sqrt(x.numel())
                idx += 1
                continue

            before_time = time.time()
            base_prediction = model(x.unsqueeze(0)).argmax(1).cpu().item()
            prediction, radius = smoothed_model.certify(
                x, args.smoothing_N0, args.smoothing_N, args.smoothing_alpha, batch_size=args.smoothing_batch
            )
            time_elapsed = str(datetime.timedelta(seconds=(time.time() - before_time)))
            correct = int(prediction == target)

            indices = np.append(indices, sample_index.cpu().numpy())
            labels = np.append(labels, target.cpu().numpy())
            base_predictions = np.append(base_predictions, base_prediction)
            predictions = np.append(predictions, prediction)
            l2_radii = np.append(l2_radii, radius)
            linf_radii = np.append(linf_radii, radius / math.sqrt(x.numel()))

            logging.info("{idx:<7}\t{sample_idx:<10}\t{label:<10}\t{base_predict:<15}\t{predict:<10}\t{radius:<10.4f}\t{correct:<10}\t{time:<10}".format(
                idx=idx, sample_idx=sample_index, label=target, base_predict=base_prediction,
                predict=prediction, radius=radius, correct=correct, time=time_elapsed
            ))
            idx += 1
            if args.dry_run:
                break

    cert_accs = {}
    for eps_str in test_eps:
        eps_float = convert_floatstr(eps_str)
        if args.adv_norm == 'Linf':
            cert_accs[eps_str] = {
                'cert_acc': round(100. * np.average((predictions == labels) & (linf_radii >= eps_float)), 2),
                'cert_inacc': round(100. * np.average((predictions != labels) & (predictions != -1) & (linf_radii >= eps_float)), 2)
            }
        elif args.adv_norm == 'L2':
            cert_accs[eps_str] = {
                'cert_acc': round(100. * np.average((predictions == labels) & (l2_radii >= eps_float)), 2),
                'cert_inacc': round(100. * np.average((predictions != labels) & (predictions != -1) & (l2_radii >= eps_float)), 2)
            }
        else:
            raise ValueError(f'Unsupported adv norm {args.adv_norm} for smoothing.')

    return cert_accs, linf_radii, l2_radii, base_predictions, predictions, labels, indices


def get_indicator_from_smoothing_log(
        log_path: str, adv_norm: str, eps_str: str
    ) -> Tuple[bool, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Extract accuracy and certified indicators from smoothing log.

    Args:
        log_path (str): Path to log file.
        adv_norm (str): Adversarial norm.
        eps_str (str): Perturbation region size.

    Returns:
        Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Indicators found, accuracy indicator,
            certicication indicator for each eps, natural predictions, sample indices
    """
    indicators_found, is_acc, is_cert, pred, indices = False, None, None, None, None
    if not os.path.isfile(log_path):
        logging.info(f'No log file {log_path} found, evaluating instead')
        return indicators_found, is_acc, is_cert, pred, indices

    log_df = pd.read_csv(log_path, index_col=0)
    indicators_found = True
    eps = convert_floatstr(eps_str)
    radius_col = 'linf_radius' if adv_norm == 'Linf' else 'l2_radius'

    is_acc = (log_df['label'] == log_df['prediction']).to_numpy().astype(int)
    is_cert = (log_df[radius_col] >= eps).to_numpy().astype(int)
    pred = log_df['prediction'].to_numpy().astype(int)
    indices = log_df['sample_idx'].to_numpy().astype(int)

    return indicators_found, is_acc, is_cert, pred, indices


def get_acc_cert_indicator(
        args: object, model: nn.Module, model_dir: str, model_name: str, device: str,
        dataloader: torch.utils.data.DataLoader, eval_set: str, eps_str: str,
        smooth: bool = True, n_smooth_samples: int = 500, use_existing: bool = False,
        write_log: bool = False, write_report: bool = False
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get 0-1 indicators for each sample on whether the given model is accurate and robust
    for the given perturbation region. If no existing sample log is found in the model_dir,
    the model is evaluated from scratch.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): PyTorch module.
        model_dir (str): Directory in which the model is stored (and associated eval logs).
        model_name (str): Name of the model to evaluate.
        device (str): device.
        dataloader (torch.utils.data.DataLoader): Dataloader to evaluate.
        eval_set (str): Dataset split that is evaluated ('train' or 'test').
        eps_str (str): Perturbation region size.
        smooth (bool, optional): If set, smoothed model is evaluated. Defaults to True.
        n_smooth_samples (int, optional): Number of samples to certify. Defaults to 500.
        use_existing (bool, optional): If set, an existing model eval log will be used (if such a log exists).
        write_log (bool, optional): If set, the evaluation log will be written to file. Defaults to False.
        write_report (bool, optional): If set, evaluation report will be written. Defaults to False.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: nat accuracy, cert accuracy, cert inacc,
            accuracy indicator, certification indicator, natural predictions, sample indices
    """
    # check first whether a log exists
    log_name = eval_smoothing_log_filename(eval_set, args.smoothing_sigma, args.smoothing_N0, args.smoothing_N)
    log_path = os.path.join(model_dir, 'smoothing', log_name)
    indicators_found, is_acc, is_cert, predictions, indices = get_indicator_from_smoothing_log(log_path, args.adv_norm, eps_str)

    if indicators_found and use_existing:
        is_acc = is_acc if len(is_acc) <= n_smooth_samples else is_acc[:n_smooth_samples]
        is_cert = is_cert if len(is_cert) <= n_smooth_samples else is_cert[:n_smooth_samples]
        predictions = predictions if len(predictions) <= n_smooth_samples else predictions[:n_smooth_samples]
        indices = indices if len(indices) <= n_smooth_samples else indices[:n_smooth_samples]

        nat_acc1 = round(100.0 * np.average(is_acc), 2)
        cert_acc1 = round(100.0 * np.average(is_cert & is_acc), 2)
        cert_inacc = round(100.0 * np.average(is_cert & (1-is_acc)), 2)

        return nat_acc1, cert_acc1, cert_inacc, is_acc, is_cert, predictions, indices

    # it is possible to run this function without a model given that a logfile is available
    if model is None:
        raise ValueError(f'Error: no model provided and indicators not found in log {log_path}.')

    """Put dataset into sequential dataloader to get deterministic sample order
    (if a shuffled train_loader is evaluated). Only n_smooth_samples samples are evaluated.
    """
    dataset = get_subsampled_dataset(args.dataset, dataloader.dataset, n_samples=n_smooth_samples, balanced=True)
    seq_dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=dataloader.num_workers
    )
    eps = convert_floatstr(eps_str)
    if smooth:
        # probabilistic certification via randomized smoothing
        cert_accs, linf_radii, l2_radii, base_pred, predictions, labels, indices = smoothing_eval(
            args, model, model_dir, device, seq_dataloader,
            [eps_str], eval_set=eval_set, use_exist_log=True
        )
        is_acc = (predictions == labels).astype(int)
        if args.adv_norm == 'Linf':
            is_cert = (linf_radii >= eps).astype(int)
        elif args.adv_norm == 'L2':
            is_cert = (l2_radii >= eps).astype(int)
        else:
            raise ValueError(f'Unsupported adv norm {args.adv_norm} for smoothing.')

        nat_acc1 = round(100.0 * np.average(is_acc), 2)
        cert_acc1 = cert_accs[eps_str]['cert_acc']
        cert_inacc = cert_accs[eps_str]['cert_inacc']

        if write_log:
            write_smoothing_log(
                args, model_dir, eval_set, linf_radii, l2_radii, base_pred, predictions, labels, indices
            )
        if write_report:
            write_eval_report(args, out_dir=model_dir, pcert_accs=cert_accs)
    else:
        # only evaluated natural accuracy
        nat_accs, _, _, predictions, is_acc, indices = natural_eval(args, model, device, seq_dataloader)

        nat_acc1 = round(nat_accs[0], 2)
        cert_acc1, cert_inacc = 0, 0
        is_cert = np.zeros_like(is_acc) # if not smoothing, assume not certifiable
    del seq_dataloader

    return nat_acc1, cert_acc1, cert_inacc, is_acc, is_cert, predictions, indices
