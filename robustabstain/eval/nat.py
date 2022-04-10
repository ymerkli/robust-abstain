import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os
import numpy as np
from tqdm import tqdm
from typing import Union, Tuple, Dict, List

from robustabstain.certify.smoothing import Smooth
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.loaders import get_rel_sample_indices
from robustabstain.utils.metrics import accuracy, AverageMeter


def natural_eval(
        args: object, model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader, smooth: bool = False
    ) -> Tuple[Tuple[float, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate top1, top5 natural accuracy of a model.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module to evaluate
        device (str): device
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data
        smooth (bool, optional): If set, model is smoothed and evaluated. Defaults to False.

    Returns:
        Tuple[List[float], List[float], np.ndarray, np.ndarray, np.ndarray]: (top1,top5) nat accuracy,
            per class top1 nat accuracies, array of confidences, array of predictions,
            binary indicator array indicating per-sample accuracy, array of sample indices
    """
    logging.info(f'Evaluating natural accuracy of model')
    model.eval()
    n_samples = len(test_loader.dataset)
    _, _, num_classes = get_dataset_stats(args.dataset)

    nat_confidences = np.zeros(n_samples, dtype=np.float64)
    nat_predictions = np.zeros(n_samples, dtype=np.int64)
    is_acc = np.zeros(n_samples, dtype=np.int64)
    indices = np.zeros(n_samples, dtype=np.int64)

    nat_acc1 = AverageMeter()
    nat_acc5 = AverageMeter()
    pc_nat_accs = [AverageMeter() for _ in range(num_classes)]

    pbar = tqdm(test_loader, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
            rel_sample_indices = get_rel_sample_indices(test_loader, sample_indices)
            inputs, targets = inputs.to(device), targets.to(device)
            if smooth:
                # smoothed classifier evaluation
                nat_conf, nat_pred = smooth_predict(args, model, device, inputs)
                accs, pc_accs = accuracy(nat_pred, targets, topk=(1,5), num_classes=num_classes)
                acc1, acc5 = accs
            else:
                nat_out = model(inputs)
                nat_pred = nat_out.argmax(1)
                nat_probs = F.softmax(nat_out, dim=1)
                nat_conf, nat_pred = nat_probs.max(1)
                accs, pc_accs = accuracy(nat_out, targets, topk=(1,5))
                acc1, acc5 = accs

            nat_confidences[rel_sample_indices] = nat_conf.cpu().numpy()
            nat_predictions[rel_sample_indices] = nat_pred.cpu().numpy()
            is_acc[rel_sample_indices] = targets.eq(nat_pred).cpu().numpy()
            indices[rel_sample_indices] = sample_indices

            nat_acc1.update(acc1.item(), inputs.size(0))
            nat_acc5.update(acc5.item(), inputs.size(0))
            for label in range(num_classes):
                pc_batch_size = targets[targets == label].size(0)
                pc_nat_accs[label].update(pc_accs[label].item(), pc_batch_size)

            pbar.set_description(f'[V] natural accuracy: acc1=%.4f, acc5=%.4f' % (
                nat_acc1.avg, nat_acc5.avg
            ))
            if args.dry_run:
                break

    nat_accs = [nat_acc1.avg, nat_acc5.avg]
    pc_nat_accs = [meter.avg for meter in pc_nat_accs]

    return nat_accs, pc_nat_accs, nat_confidences, nat_predictions, is_acc, indices



def smooth_predict(
        args: object, model: nn.Module, device: str, inputs: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
    """Get predictions of the smoothed model on an input batch.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model (nn.Module): PyTorch module to evaluate
        device (str): device
        inputs (torch.tensor): Input batch to predict labels for

    Returns:
        Tuple[torch.tensor, torch.tensor]: Base classifier confidences,
            smoothed classifier predictions.
    """

    _, _, num_classes = get_dataset_stats(args.dataset)
    smoothing_sigma_float = convert_floatstr(args.smoothing_sigma)

    smooth_model = Smooth(model, num_classes, smoothing_sigma_float, device)
    nat_pred = torch.tensor([], dtype=torch.int64)
    n = args.smoothing_N
    alpha = args.smoothing_alpha
    if args.eval_set == 'train':
        # sacrifice exactness for speed on train set
        n /= 5
        alpha *= 5

    for x in inputs:
        pred = smooth_model.predict(x, n=n, alpha=alpha, batch_size=args.smoothing_batch)
        nat_pred = torch.cat([nat_pred, torch.tensor([pred], dtype=torch.int64)])

    nat_pred = nat_pred.to(device)
    nat_out = model(inputs)
    nat_probs = F.softmax(nat_out, dim=1)
    nat_conf, _ = nat_probs.max(1)

    return nat_conf, nat_pred
