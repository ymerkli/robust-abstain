import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

import robustabstain.utils.args_factory as args_factory
from robustabstain.abstain.confidence_threshold import get_confidences
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.comp import compositional_accuracy
from robustabstain.utils.checkpointing import load_checkpoint
from robustabstain.utils.helpers import pretty_floatstr
from robustabstain.utils.latex import latex_norm
from robustabstain.utils.loaders import get_dataloader
from robustabstain.utils.log import init_logging


def get_args():
    parser = args_factory.get_parser(
        description='Plots for confidence thresholding abstain mechanism.',
        arg_lists=[
            args_factory.TESTING_ARGS, args_factory.LOADER_ARGS, args_factory.ATTACK_ARGS,
            args_factory.SMOOTHING_ARGS, args_factory.COMP_ARGS
        ],
        required_args=['dataset', 'test-eps', 'adv-norm', 'branch-model', 'trunk-model']
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)
    assert len(args.test_eps) == 1, 'Error: specify 1 test-eps'

    return args


def plot_roc(
        args: object, tpr: np.ndarray, fpr: np.ndarray, out_filename: str = 'roc_curve'
    ) -> None:
    """Plots a receiver operating characteristic curve for an confidence thresholding based
    abstain mechanism.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        tpr (np.ndarray): True positive rate for each tested confidence threshold.
        fpr (np.ndarray): False positive rate for each tested confidence threshold.
        out_filename (str, optional): Filename of the saved plot. Defaults to 'roc_curve'.
    """
    auc_value = round(auc(fpr, tpr), 4)

    fig, ax = plt.subplots()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    pretty_norm = latex_norm(args.adv_norm)
    pretty_eps = pretty_floatstr(args.test_eps[0])
    ax.set_title(f"Abstain ROC Curve (${pretty_norm}, \epsilon={pretty_eps}$)")


    ax.plot([0, 1], [0, 1], color='blue', linestyle='--')
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {auc_value})')
    plt.legend(loc='lower right')
    plt.grid(True)

    # save figure
    branch_model_dir = os.path.dirname(args.branch_model)
    plot_dir = os.path.join(branch_model_dir, 'plot', 'conf_threshold', args.test_eps[0])
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, out_filename+'.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, out_filename+'.pdf'))


def plot_comp_accs_conf(
        args: object, comp_nat_acc: np.ndarray, comp_adv_acc: np.ndarray,
        precision: np.ndarray, conf_thresholds: np.ndarray, out_filename: str = 'comp_accs_conf'
    ) -> None:
    """Plots natural, adversarial accuracy and the precision of the abstain mechanism for a
    confidence threshold based abstaining compositional architecture.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        comp_nat_acc (np.ndarray): Compositional natural accuracy for each confidence threshold.
        comp_adv_acc (np.ndarray): Compositional adversarial accuracy for each confidence threshold.
        precision (np.ndarray): Precision (=PPV) of the abstention for each confidence threshold.
        conf_thresholds (np.ndarray): Confidence thresholds array (x-axis of the plot).
        out_filename (str, optional): Filename of the saved plot. Defaults to 'conf_acc_plot'.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Confidence Threshold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 100])
    ax.set_yticks(np.linspace(start=0, stop=100, num=11))
    pretty_norm = latex_norm(args.adv_norm)
    pretty_eps = pretty_floatstr(args.test_eps[0])
    ax.set_title(f"Compositional accuracies (${pretty_norm}, \epsilon={pretty_eps}$)")

    ax.plot(conf_thresholds, comp_nat_acc, color='blue', label='Compositional Natural Accuracy [%]')
    ax.plot(conf_thresholds, comp_adv_acc, color='darkorange', label='Compositional Adversarial Accuracy [%]')
    ax.plot(conf_thresholds, precision, color='darkgreen', label='Abstain Precision')
    plt.legend(loc='lower left')
    plt.grid(True)

    # save figure
    branch_model_dir = os.path.dirname(args.branch_model)
    plot_dir = os.path.join(branch_model_dir, 'plot', 'conf_threshold', args.test_eps[0])
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, out_filename+'.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, out_filename+'.pdf'))


def plot_comp_nat_adv_acc(
        args: object, comp_nat_acc: np.ndarray, comp_adv_acc: np.ndarray,
        precision: np.ndarray, conf_thresholds: np.ndarray, out_filename: str = 'comp_nat_adv_acc'
    ) -> None:

    fig, ax = plt.subplots()
    ax.set_xlabel('Compositional Natural Accuracy [%]')
    ax.set_ylabel('Compositional Adversarial Accuracy [%]')
    ax.set_xlim([80, 100])
    ax.set_ylim([50, 100])
    pretty_norm = latex_norm(args.adv_norm)
    pretty_eps = pretty_floatstr(args.test_eps[0])
    ax.set_title(f"Compositional accuracies (${pretty_norm}, \epsilon={pretty_eps}$)")

    ax.plot(comp_nat_acc, comp_adv_acc, color='blue', label='')
    plt.legend(loc='lower left')
    plt.grid(True)

    # save figure
    branch_model_dir = os.path.dirname(args.branch_model)
    plot_dir = os.path.join(branch_model_dir, 'plot', 'conf_threshold', args.test_eps[0])
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, out_filename+'.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, out_filename+'.pdf'))



def plot_conf_threshold_curves(
        args: object, branch_model: nn.Module, device: str, test_loader: torch.utils.data.DataLoader,
        branch_is_acc: np.ndarray, branch_is_rob: np.ndarray, trunk_is_acc: np.ndarray,
        trunk_is_rob: np.ndarray, n_steps: int = 50
    ) -> None:
    """Create plots for a confidence threshold based compositional abstain architecture. The abstain mechanism is
    based on confidence thresholding and evaluated for various thresholds.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        branch_model (nn.Module): The branch model in the comp architecture.
        device (str): device.
        test_loader (torch.utils.data.DataLoader): Dataloader containing test data.
        branch_is_acc (np.ndarray): Binary array indicating for which test samples branch model is accurate.
        branch_is_rob (np.ndarray): Binary array indicating for which test samples branch model is robust.
        trunk_is_acc (np.ndarray): Binary array indicating for which test samples trunk model is accurate.
        trunk_is_rob (np.ndarray): Binary array indicating for which test samples trunk model is robust.
        n_steps (int, optional): Number of confidence steps in interval [0,1]. Defaults to 50.
    """
    branch_is_robacc = branch_is_acc & branch_is_rob
    nat_conf, adv_conf, _, _, _ = get_confidences(args, branch_model, device, test_loader, args.test_eps[0])
    conf_thresholds = np.linspace(start=0, stop=0.99, num=n_steps)

    # NOTE: 'true' = robust & accurate, 'false' = NOT robust & inaccurate
    tpr = np.zeros(n_steps) # true positive rate (= recall)
    fpr = np.zeros(n_steps) # false positive rate
    prec = np.zeros(n_steps) # precision
    comp_nat_acc = np.zeros(n_steps)
    comp_adv_acc = np.zeros(n_steps)
    for i, conf_t in enumerate(conf_thresholds):
        nat_is_confident = (nat_conf >= conf_t).astype(np.int64)
        adv_is_confident = (adv_conf >= conf_t).astype(np.int64)

        tp = np.sum(branch_is_robacc & adv_is_confident)
        tn = np.sum((1 - branch_is_robacc) & (1 - adv_is_confident))
        fp = np.sum((1 - branch_is_robacc) & adv_is_confident)
        fn = np.sum(branch_is_robacc & (1 - adv_is_confident))
        tpr[i] = tp / (tp + fn)
        fpr[i] = fp / (fp + tn)
        prec[i] = 100.0 * tp / (tp + fp)

        comp_nat_acc[i], comp_adv_acc[i] = compositional_accuracy(
            branch_is_acc=branch_is_acc, branch_is_rob=branch_is_rob,
            trunk_is_acc=trunk_is_acc, trunk_is_rob=trunk_is_rob,
            selector=nat_is_confident, adv_selector=adv_is_confident
        )

    # plotting
    plot_roc(args, tpr, fpr)
    plot_comp_accs_conf(args, comp_nat_acc, comp_adv_acc, prec, conf_thresholds)
    plot_comp_nat_adv_acc(args, comp_nat_acc, comp_adv_acc, prec, conf_thresholds)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()
    init_logging(args)

    # build dataset
    _, _, test_loader, _, _, num_classes = get_dataloader(
        args, args.dataset, normalize=False, indexed=True, val_set_source='test', val_split=0.0
    )

    # load branch model
    branch_model, _, branch_chkpt = load_checkpoint(
        args.branch_model, net=None, arch=None, dataset=args.dataset, device=device,
        normalize=not args.no_normalize, optimizer=None, parallel=True
    )
    branch_model_name = os.path.splitext(os.path.basename(args.branch_model))[0]
    branch_model_dir = os.path.dirname(args.branch_model)

    # get accuracy and robustness indicators of branch model on testset
    (
        branch_nat_acc1, branch_adv_acc1, branch_rob_inacc,
        branch_is_acc_test, branch_is_rob_test, _, _, _, _
    ) = get_acc_rob_indicator(
            args, branch_model, branch_model_dir, branch_model_name, device,
            test_loader, 'test', args.adv_norm, args.test_eps[0],
            args.test_adv_attack, use_existing=True, write_log=True, write_report=True
    )

    # load trunk model
    trunk_model, _, _ = load_checkpoint(
        args.trunk_model, net=None, arch=None, dataset=args.dataset, device=device,
        normalize=not args.no_normalize, optimizer=None, parallel=True
    )
    trunk_model_name = os.path.splitext(os.path.basename(args.trunk_model))[0]
    trunk_model_dir = os.path.dirname(args.trunk_model)

    # get accuracy and robustness indicators of branch model on testset
    (
        trunk_nat_acc1, trunk_adv_acc1, trunk_rob_inacc,
        trunk_is_acc_test, trunk_is_rob_test, _, _, _, _
    ) = get_acc_rob_indicator(
            args, trunk_model, trunk_model_dir, trunk_model_name,
            device, test_loader, 'test', args.adv_norm, args.test_eps[0],
            args.test_adv_attack, use_existing=True, write_log=True, write_report=True
    )

    # plotting
    plot_conf_threshold_curves(
        args, branch_model, device, test_loader, branch_is_acc_test, branch_is_rob_test,
        trunk_is_acc_test, trunk_is_rob_test, n_steps=20
    )


if __name__ == '__main__':
    main()
