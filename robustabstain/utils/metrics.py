import torch
import numpy as np
from typing import Tuple, Union, List, Dict

from robustabstain.utils.helpers import check_indicator


class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[int, float], n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def accuracy(
        output: torch.tensor, targets: torch.tensor, topk: tuple = (1,),
        num_classes: int = 0, include: torch.tensor = None
    ) -> List[torch.tensor]:
    """Computes the precision@k for the specified values of k, both averaged over
    all classes and for each class individually.
    NOTE: If a class does not occur in targets, it is assigned 0% accuracy.

    Args:
        output (torch.tensor): predicted output
        targets (torch.tensor): ground truth labels
        topk (tuple, optional): top-k value. Defaults to (1,).
        num_classes (int, optional): Number of classes in dataset. Defaults to 0.
        include (torch.tensor): Binary tensor, only nonzero indices are considered

    Returns:
        List[torch.tensor]: topk accuracies
    """
    nat_accs = []
    pc_nat_accs = [] # per class natural accuracies
    batch_size = targets.size(0)
    if output.ndim == 1:
        assert topk[0] == 1, 'Error: if labels given, only top1 accuracy are returned.'
        # if output is a 1d tensor, it is assumed that the tensor contains labels, not logits
        correct = output.eq(targets)
        nat_accs.append(correct.float().sum(0).mul_(100.0 / batch_size))
        nat_accs.append([torch.tensor([0])] * len(topk[1:]))

        # calculate per-class accuracies
        for label in range(num_classes):
            pc_batch_size = targets[targets == label].size(0)
            if pc_batch_size == 0:
                pc_nat_accs.append(torch.tensor[0.0])
            else:
                pc_nat_accs.append(correct[targets == label].float().sum(0).mul_(100.0 / pc_batch_size))

        return nat_accs, pc_nat_accs

    if num_classes > 0:
        assert num_classes == output.size(1), f'Error: num_classes={num_classes} was passed but output has size(1)={output.size(1)}.'
    else:
        num_classes = output.size(1)

    maxk = max(topk)
    if maxk > num_classes:
        maxk = num_classes

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    if include is not None:
        correct = correct[:, torch.nonzero(include, as_tuple=True)[0]]
        batch_size = include.sum().item()

    for k in topk:
        if k <= maxk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            nat_accs.append(correct_k.mul_(100.0 / batch_size))
        else:
            nat_accs.append(torch.tensor([100.0]))

    # per class accuracies
    for label in range(num_classes):
        pc_batch_size = targets[targets == label].size(0)
        if pc_batch_size > 0:
            pc_correct = correct[:1, targets == label].reshape(-1).float().sum(0)
            pc_nat_accs.append(pc_correct.mul_(100.0 / pc_batch_size))
        elif pc_batch_size == 0:
            pc_nat_accs.append(torch.tensor([0.0]))

    return nat_accs, pc_nat_accs


def adv_accuracy(
        adv_out: torch.tensor, nat_out: torch.tensor, targets: torch.tensor
    ) -> Tuple[torch.tensor, List[torch.tensor]]:
    """Computes the adversarial top1 precision@1. An adversarial sample is only
    considered robust and accurate if its predicted label equals the predicted
    label of the natural sample AND the ground truth label.
    NOTE: If a class does not occur in targets, it is assigned 0% accuracy.

    Args:
        adv_out (torch.tensor): predicted output on adversarial samples
        nat_out (torch.tensor): predicted output on natural samples
        targets (torch.tensor): ground truth labels

    Returns:
        Tuple[torch.tensor, List[torch.tensor]]: Adversarial accuracy,
            per class adversarial accuracies
    """
    batch_size = targets.size(0)
    num_classes = adv_out.size(1)
    adv_pred = adv_out.argmax(1)
    nat_pred = nat_out.argmax(1)
    correct = (adv_pred.eq(targets) & adv_pred.eq(nat_pred))
    adv_acc = correct.float().sum(0).mul_(100.0 / batch_size)
    pc_adv_acc = []
    for label in range(num_classes):
        pc_batch_size = targets[targets == label].size(0)
        if pc_batch_size > 0:
            pc_adv_acc.append(correct[targets == label].float().sum(0).mul_(100.0 / pc_batch_size))
        else:
            pc_adv_acc.append(torch.tensor([0.0]))

    return adv_acc, pc_adv_acc


def accuracy_from_ind(
        is_acc: np.ndarray, targets: np.ndarray, num_classes: int
    ) -> Tuple[float, List[float]]:
    """Get overall and per-class natural accuracy from indicator arrays.

    Args:
        is_acc (np.ndarray): Indicatory array indicating accuracy.
        targets (np.ndarray): GT targets of the evaluated samples.
        num_classes (int): Number of classes in dataset.

    Returns:
        Tuple[float, List[float]]: overall nat accuracy, per-class nat accuracy.
    """
    is_acc = check_indicator(is_acc)
    nat_acc = 100.0 * np.average(is_acc)
    pc_nat_accs = [0.0] * num_classes
    for label in range(num_classes):
        if targets[targets == label].shape[0] > 0:
            pc_nat_accs[label] = 100.0 * np.average(is_acc[targets == label])

    return nat_acc, pc_nat_accs


def adv_accuracy_from_ind(
        is_acc: np.ndarray, is_rob: np.ndarray, targets: np.ndarray, num_classes: int
    ) -> Tuple[float, List[float]]:
    """Get overall and per-class adversarial accuracy from indicator arrays.

    Args:
        is_acc (np.ndarray): Indicatory array indicating accuracy.
        is_rob (np.ndarray): Indicatory array indicating robustness.
        targets (np.ndarray): GT targets of the evaluated samples.
        num_classes (int): Number of classes in dataset.

    Returns:
        Tuple[float, List[float]]: overall adv accuracy, per-class adv accuracy.
    """
    is_acc = check_indicator(is_acc)
    is_rob = check_indicator(is_rob)
    adv_acc = 100.0 * np.average(is_rob & is_acc)
    pc_adv_accs = [0.0] * num_classes
    for label in range(num_classes):
        if targets[targets == label].shape[0] > 0:
            pc_adv_accs[label] = 100.0 * np.average(is_rob[targets == label] & is_acc[targets == label])

    return adv_acc, pc_adv_accs


def rob_inacc_from_ind(
        is_acc: np.ndarray, is_rob: np.ndarray, targets: np.ndarray, num_classes: int
    ) -> Tuple[float, List[float]]:
    """Get overall and per-class robust-inaccuracy from indicator arrays.

    Args:
        is_acc (np.ndarray): Indicatory array indicating accuracy.
        is_rob (np.ndarray): Indicatory array indicating robustness.
        targets (np.ndarray): GT targets of the evaluated samples.
        num_classes (int): Number of classes in dataset.

    Returns:
        Tuple[float, List[float]]: overall robust-inaccurate, per-class robust inaccurate.
    """
    is_acc = check_indicator(is_acc)
    is_rob = check_indicator(is_rob)
    rob_inacc = 100.0 * np.average(is_rob & (1-is_acc))
    pc_rob_inaccs = [0.0] * num_classes
    for label in range(num_classes):
        if targets[targets == label].shape[0] > 0:
            pc_rob_inaccs[label] = 100.0 * np.average(is_rob[targets == label] & (1-is_acc[targets == label]))

    return rob_inacc, pc_rob_inaccs


def rob_inacc_perc(is_acc: Union[torch.tensor, np.ndarray], is_rob: Union[torch.tensor, np.ndarray]) -> float:
    """Helper function to get the percentage of inaccurate but robust samples.

    Args:
        is_acc (Union[torch.tensor, np.ndarray]): Binary array indicating which samples are accurate
        is_rob (Union[torch.tensor, np.ndarray]): Binary array indicating which samples are robust

    Returns:
        float: Fraction of inaccurate robust samples.
    """
    is_acc = check_indicator(is_acc)
    is_rob = check_indicator(is_rob)

    return float(100.0 * np.average((1-is_acc) & is_rob))


def get_iou(bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
    """Calculate the Intersection of Union (IoU) of two bounding boxes.

    Args:
        bbox1 (Dict[str, int]): BBox 1, keys 'xmin', 'ymin', 'xmax', 'ymax'.
        bbox2 (Dict[str, int]): BBox 2, keys 'xmin', 'ymin', 'xmax', 'ymax'.

    Returns:
        float: IoU score.
    """
    assert bbox1['xmin'] <= bbox1['xmax']
    assert bbox1['ymin'] <= bbox1['ymax']
    assert bbox2['xmin'] <= bbox2['xmax']
    assert bbox2['ymin'] <= bbox2['ymax']

    # determine coordinates of the intersection bbox
    xmin = max(bbox1['xmin'], bbox2['xmin'])
    ymin = max(bbox1['ymin'], bbox2['ymin'])
    xmax = min(bbox1['xmax'], bbox2['xmax'])
    ymax = min(bbox1['ymax'], bbox2['ymax'])

    intersection_area = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    bbox1_area = (bbox1['xmax'] - bbox1['xmin'] + 1) * (bbox1['ymax'] - bbox1['ymin'] + 1)
    bbox2_area = (bbox2['xmax'] - bbox2['xmin'] + 1) * (bbox2['ymax'] - bbox2['ymin'] + 1)

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou