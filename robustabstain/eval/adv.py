import setGPU
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple, Dict, List

from robustabstain.attacker.wrapper import AttackerWrapper
from robustabstain.eval.log import write_sample_log, write_eval_report
from robustabstain.eval.nat import natural_eval
from robustabstain.utils.data_utils import get_dataset_stats, dataset_label_names
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.loaders import get_rel_sample_indices
from robustabstain.utils.metrics import adv_accuracy, AverageMeter, rob_inacc_from_ind
from robustabstain.utils.paths import eval_attack_log_filename


def empirical_robustness_eval(
        args: object, model: nn.Module, device: str,
        test_loader: torch.utils.data.DataLoader, adv_attack: str,
        adv_norm: str, test_eps: str, best_loss: bool = False
    ) -> Tuple[Dict[str, float], List[Dict[str, float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate empirical robustness for given adversarial perturbation region
    with a given adversarial attack.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): PyTorch module to evaluate.
        device (str): device.
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data.
        adv_attack (str): Type of adversarial attack to use.
        adv_norm (str): Norm of the perturbation region.
        test_eps (str): Size of adversarial perturbation region (stringified).
        best_loss (bool, optional): If True, the points attaining highest loss are returned,
            otherwise adversarial examples. Only supported in APGD. Defaults to False.

    Returns:
        Tuple[Dict[str, float], List[Dict[str, float]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Dict with rob_acc, rob_inacc, List of dicts with rob_acc, rob_inacc for each class,
            adv confidences, adv predictions, binary indicator array indicating per-sample robustness,
            sample_indices array.
    """
    logging.info(f'Evaluating empirical robustness using {adv_norm} {adv_attack}')
    model.eval()

    test_eps_float = convert_floatstr(test_eps)
    smoothing_sigma_float = convert_floatstr(args.smoothing_sigma) if args.smoothing_sigma else None
    n_samples = len(test_loader.dataset)
    _, _, num_classes = get_dataset_stats(args.dataset)
    attacker = AttackerWrapper(
        adv_type=adv_attack, adv_norm=adv_norm, eps=test_eps_float, steps=args.test_att_n_steps,
        rel_step_size=args.test_att_step_size, version=args.autoattack_version,
        gamma_ddn=args.gamma_ddn, init_norm_ddn=args.init_norm_ddn, device=device
    )

    adv_confidences = np.zeros(n_samples, dtype=np.float64)
    adv_predictions = np.zeros(n_samples, dtype=np.int64)
    is_rob = np.zeros(n_samples, dtype=np.int64)
    indices = np.zeros(n_samples, dtype=np.int64)

    rob_acc = AverageMeter()
    rob_inacc = AverageMeter()
    pc_rob_acc = [AverageMeter() for _ in range(num_classes)]
    pc_rob_inacc = [AverageMeter() for _ in range(num_classes)]

    pbar = tqdm(test_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        rel_sample_indices = get_rel_sample_indices(test_loader, sample_indices)
        inputs, targets = inputs.to(device), targets.to(device)

        nat_out = model(inputs)
        nat_pred = nat_out.argmax(1)

        """GT Labels for attack are the predicted labels.
        This is necessary for the attack to also search
        for adversarial samples for inaccurate samples.
        """
        clipped_advs = attacker.attack(
            model=model, inputs=inputs, labels=nat_pred, noise=smoothing_sigma_float,
            num_noise_vectors=args.num_noise_vec, no_grad=args.no_grad_attack,
            best_loss=best_loss
        )
        adv_out = model(clipped_advs)
        adv_probs = F.softmax(adv_out, dim=1)
        adv_conf, adv_pred = adv_probs.max(1)

        adv_confidences[rel_sample_indices] = adv_conf.cpu().detach().numpy()
        adv_predictions[rel_sample_indices] = adv_pred.cpu().detach().numpy()
        is_acc_batch = nat_pred.eq(targets).int().cpu().numpy()
        is_rob_batch = adv_pred.eq(nat_pred).int().cpu().numpy()
        is_rob[rel_sample_indices] = is_rob_batch
        indices[rel_sample_indices] = sample_indices

        adv_acc_batch, pc_adv_acc_batch = adv_accuracy(adv_out, nat_out, targets)
        rob_inacc_batch, pc_rob_inacc_batch = rob_inacc_from_ind(
            is_acc_batch, is_rob_batch, targets.cpu().numpy(), num_classes
        )

        # update meters
        rob_acc.update(adv_acc_batch.item(), inputs.size(0))
        rob_inacc.update(rob_inacc_batch, inputs.size(0))
        for label, meter in enumerate(pc_rob_acc):
            pc_batch_size = targets[targets == label].size(0)
            meter.update(pc_adv_acc_batch[label].item(), pc_batch_size)
        for label, meter in enumerate(pc_rob_inacc):
            pc_batch_size = targets[targets == label].size(0)
            meter.update(pc_rob_inacc_batch[label], pc_batch_size)

        pbar.set_description('[V] adversarial accuracy ({} {} eps={}): acc1={:.4f}, rob_inacc={:.4f}'.format(
            adv_attack, adv_norm, test_eps, rob_acc.avg, rob_inacc.avg)
        )
        if args.dry_run:
            break

    adv_accs = {'adv_acc': round(rob_acc.avg, 2), 'rob_inacc': round(rob_inacc.avg, 2)}
    adv_accs['robind_commit_rate'] = adv_accs['adv_acc'] + adv_accs['rob_inacc']
    adv_accs['robind_commit_prec'] = round(100.0 * adv_accs['adv_acc'] / adv_accs['robind_commit_rate'], 2)
    pc_adv_accs = [
        {'adv_acc': round(adv_acc_meter.avg, 2), 'rob_inacc': round(rob_inacc_meter.avg, 2)}
        for adv_acc_meter, rob_inacc_meter in zip(pc_rob_acc, pc_rob_inacc)
    ]

    return adv_accs, pc_adv_accs, adv_confidences, adv_predictions, is_rob, indices


def get_indicator_from_log(
        log_path: str, eps_str: str, model_name: str
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract '{model_name}_is_acc' and '{model_name}_is_rob{eps}' columns from
    sample log.

    Args:
        log_path (str): Path to log file.
        eps_str (str): Perturbation region size.
        model_name (str): Name of the model to evaluate.

    Returns:
        Tuple[bool, np.ndarray, Dict[str, np.ndarray], np.ndarray]: Indicators found, accuracy indicator,
            robustness indicator, natural predictions, natural confidences, adversarial confidences
    """
    indicators_found, is_acc, is_rob, pred, nat_conf, adv_conf = False, None, None, None, None, None
    if not os.path.isfile(log_path):
        logging.info(f'No log file {log_path} found, evaluating instead')
        return indicators_found, is_acc, is_rob, pred, nat_conf, adv_conf

    log_df = pd.read_csv(log_path, index_col=0)
    is_acc_col = f'{model_name}_is_acc'
    is_rob_col = f'{model_name}_is_rob{eps_str}'
    pred_col = f'{model_name}_pred'
    nat_conf_col = f'{model_name}_nat_conf'
    adv_conf_col = f'{model_name}_adv_conf{eps_str}'

    cols_required = [is_acc_col, is_rob_col, pred_col]
    if 'apgdconf' in log_path:
        cols_required += [nat_conf_col, adv_conf_col]
    if any(col not in log_df.columns for col in cols_required):
        logging.info(f'Not all required columns found, evaluating instead')
        return indicators_found, is_acc, is_rob, pred, nat_conf, adv_conf

    indicators_found = True
    is_acc = log_df[is_acc_col].to_numpy()
    is_rob = log_df[is_rob_col].to_numpy()
    pred = log_df[pred_col].to_numpy()
    nat_conf = log_df[nat_conf_col].to_numpy() if nat_conf_col in log_df.columns else None
    adv_conf = log_df[adv_conf_col].to_numpy() if adv_conf_col in log_df.columns else None

    return indicators_found, is_acc, is_rob, pred, nat_conf, adv_conf


def get_acc_rob_indicator(
        args: object, model: nn.Module, model_dir: str, model_name: str,
        device: str, dataloader: torch.utils.data.DataLoader, eval_set: str,
        adv_norm: str, eps_str: str, adv_attack: str, use_existing: bool = False,
        write_log: bool = False, write_report: bool = False,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get 0-1 indicators for each sample on whether the given model is accurate and robust
    for the given perturbation region. If no existing sample log is found in the model_dir,
    the model is evaluated from scratch.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): PyTorch module. Maybe be equal to None if a logfile is available.
        model_dir (str): Directory in which the model is stored (and associated eval logs).
        model_name (str): Name of the model to evaluate.
        device (str): device.
        dataloader (torch.utils.data.DataLoader): Dataloader to evaluate.
        eval_set (str): Dataset split that is evaluated ('train', 'val', 'test').
        adv_norm (str): Norm of the adversarial perturbation region.
        eps_str (str): Perturbation region size.
        adv_attack (str): Type of adversarial attack to perform.
        use_existing (bool, optional): If set, an existing model eval log will be used (if such a log exists).
        write_log (bool, optional): If set, the evaluation log will be written to file. Defaults to False.
        write_report (bool, optional): If set, evaluation report will be written. Defaults to False.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            nat accuracy, adv accuracy, rob inacc, accuracy indicator, robustness indicator,
            natural predictions, natural prediction confidences, adversarial prediction confidences, indices
    """
    # get indices order of samples in dataloader
    dataset_indices = np.arange(len(dataloader.dataset))
    if type(dataloader.dataset) == torch.utils.data.dataset.Subset:
        # Subset dataset have random index order
        dataset_indices = dataloader.dataset.indices

    # load label names
    label_names = dataset_label_names(args.dataset)

    # check first whether a log exists
    log_name = eval_attack_log_filename(eval_set, args.dataset, adv_norm, adv_attack)
    log_path = os.path.join(model_dir, log_name)
    indicators_found, is_acc, is_rob, predictions, nat_conf, adv_conf = get_indicator_from_log(log_path, eps_str, model_name)
    if indicators_found and use_existing:
        logging.info(f'Using logged evaluation from logfile {log_path}')
        nat_acc1 = round(100.0 * np.average(is_acc[dataset_indices]), 2)
        adv_acc1 = round(100.0 * np.average(is_acc[dataset_indices] & is_rob[dataset_indices]), 2)
        rob_inacc = round(100.0 * np.average(is_rob[dataset_indices] & (1 - is_acc[dataset_indices])), 2)
        is_acc = is_acc[dataset_indices]
        is_rob = is_rob[dataset_indices]
        predictions = predictions if predictions is None else predictions[dataset_indices]
        nat_conf = nat_conf if nat_conf is None else nat_conf[dataset_indices]
        adv_conf = adv_conf if adv_conf is None else adv_conf[dataset_indices]

        return nat_acc1, adv_acc1, rob_inacc, is_acc, is_rob, predictions, nat_conf, adv_conf, dataset_indices

    # it is possible to run this function without a model given that a logfile is available
    if model is None:
        raise ValueError(f'Error: no model provided and indicators not found in log {log_path}.')

    # put dataset into sequential dataloader to get deterministic sample order
    seq_dataloader = torch.utils.data.DataLoader(
        dataset=dataloader.dataset, batch_size=dataloader.batch_size,
        shuffle=False, num_workers=dataloader.num_workers
    )

    logging.info(f'Evaluating model {model_name} from scratch.')
    nat_accs, pc_nat_accs, nat_conf, predictions, is_acc, indices = natural_eval(args, model, device, seq_dataloader)
    assert (indices == dataset_indices).all(), 'Indices of accuracy indicators are not in expected order.'

    adv_accs, pc_adv_accs, adv_conf, _, is_rob, indices = empirical_robustness_eval(
        args, model, device, seq_dataloader, adv_attack, adv_norm, eps_str
    )
    assert (indices == dataset_indices).all(), 'Indices of robustness indicators are not in expected order.'
    del seq_dataloader

    nat_accs = [round(acc, 2) for acc in nat_accs]
    pc_nat_accs = [round(acc, 2) for acc in pc_nat_accs]
    nat_acc1 = nat_accs[0]
    adv_acc1 = adv_accs['adv_acc']
    rob_inacc = adv_accs['rob_inacc']

    if write_report:
        per_class = 'sbb' in args.dataset # per-class accuracies are mostly interesting for SBB dataset
        write_eval_report(
            args, out_dir=model_dir, nat_accs=nat_accs, pc_nat_accs=pc_nat_accs,
            adv_accs={eps_str: adv_accs}, pc_adv_accs={eps_str: pc_adv_accs},
            adv_attack=adv_attack, label_names=label_names, per_class=per_class
        )

    if write_log:
        if 'conf' in adv_attack:
            # for conf adv_attacks, write confidences into log
            write_sample_log(
                model_name, model_dir, args.dataset, eval_set, adv_norm, adv_attack,
                indices=indices, is_acc=is_acc, preds=predictions, nat_predconf=nat_conf,
                adv_predconf=adv_conf, is_rob=is_rob, eps=eps_str
            )
        else:
            write_sample_log(
                model_name, model_dir, args.dataset, eval_set, adv_norm, adv_attack,
                indices=indices, is_acc=is_acc, preds=predictions, is_rob=is_rob, eps=eps_str
            )

    return nat_acc1, adv_acc1, rob_inacc, is_acc, is_rob, predictions, nat_conf, adv_conf, dataset_indices

