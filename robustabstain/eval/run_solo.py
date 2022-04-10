import setGPU
import torch

import logging
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Union, Tuple, Dict, List

import robustabstain.utils.args_factory as args_factory
from robustabstain.analysis.plotting.utils.confusion_matrix import plot_confusion_matrix
from robustabstain.eval.log import write_sample_log, write_smoothing_log, write_eval_report
from robustabstain.eval.nat import natural_eval
from robustabstain.eval.adv import empirical_robustness_eval
from robustabstain.eval.cert import smoothing_eval
from robustabstain.utils.checkpointing import  get_net, load_checkpoint
from robustabstain.utils.data_utils import dataset_label_names, get_dataset_stats
from robustabstain.utils.loaders import get_dataloader, get_targets
from robustabstain.utils.log import init_logging
from robustabstain.utils.metrics import accuracy_from_ind, adv_accuracy_from_ind, rob_inacc_from_ind
from robustabstain.utils.paths import eval_attack_log_filename


def get_args() -> object:
    """Argparser

    Returns:
        object: object subclass exposing 'setattr` and 'getattr'
    """
    parser = args_factory.get_parser(
        description='Baseline model evaluation',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS, args_factory.ATTACK_ARGS,
            args_factory.COMP_ARGS, args_factory.SMOOTHING_ARGS
        ],
        required_args=['dataset', 'model']
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    if args.test_adv_attack == 'autoattack':
        # autoattack does not take step size or number of steps. Set this args to None to clarify the logs
        args.test_att_n_steps = None
        args.test_att_step_size = None

    if any('smo' in s for s in args.evals) or args.smooth:
        assert args.smoothing_sigma is not None, 'Specify --smoothing-sigma for smoothing evaluation'

    return args


def eval_model(
        args: object, model_path: str, device: str, test_loader: torch.utils.data.DataLoader,
        test_eps: List[str], evals: List[str] = [], log_file: str = None,
        use_exist_log: bool = False, no_report: bool = False, no_log: bool = False
    ) -> None:
    """Evaluate natural and robust accuracy of a given models.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        model_path (str): Filepath to model to evaluate
        device (str): device
        test_loader (torch.utils.data.DataLoader): Dataloader containing test data
        test_eps (List[str]): List of (stringified) perturbation region epsilons to evaluate.
        evals (str, optional): The evaluations to perform. Defaults to [].
        log_file (str, optional): Path to per-sample log file. Defaults to None.
        use_exist_log (bool, optional): If set, existing log file is used (if available). Defaults to False.
        no_report (bool, optional): If set, no report file is written. Defaults to False.
        no_log (bool, optional): If set, no per-sample log file is written. Defaults to False.
    """
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    log_name = eval_attack_log_filename(args.eval_set, args.dataset, args.adv_norm, args.test_adv_attack)
    if log_file is not None:
        log_name = os.path.basename(log_file)
    else:
        log_file = os.path.join(model_dir, log_name)

    # build model
    model, arch = None, args.arch
    _, _, num_classes = get_dataset_stats(args.dataset)
    if args.arch:
        model = get_net(args.arch, args.dataset, num_classes, device, normalize=not args.no_normalize, parallel=True)

    model, _, checkpoint = load_checkpoint(
        model_path, model, args.arch, args.dataset, device=device,
        normalize=not args.no_normalize, optimizer=None, parallel=True
    )
    arch = checkpoint['arch'] if not arch else arch

    # load GT targets from test_loader
    targets = get_targets(test_loader)

    # load label names
    label_names = dataset_label_names(args.dataset)

    # check for existing log file
    adv_log = None
    if os.path.isfile(log_file) and use_exist_log:
        # if existing log is found, reuse that evaluation
        logging.info(f'Using existing log file {log_file}, reusing results from this file.')
        adv_log = pd.read_csv(log_file, index_col=0)

    # natural evaluation
    if any('nat' in s for s in evals) or (adv_log is not None and f'{model_name}_is_acc' not in adv_log.columns):
        nat_accs, pc_nat_accs, _, nat_preds, is_acc, nat_idx = natural_eval(args, model, device, test_loader, smooth=args.smooth)
        nat_accs = [round(acc, 2) for acc in nat_accs]
        pc_nat_accs = [round(acc, 2) for acc in pc_nat_accs]
        logging.info(f'Natural accuracy (top1, top5) = {nat_accs}.')

        # plot confusion matrix of natural predictions
        conf_mat = confusion_matrix(y_true=targets, y_pred=nat_preds)
        outdir = os.path.join(model_dir, 'plots')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        fp = os.path.join(outdir, f'conf_mat_{args.dataset}')
        plot_confusion_matrix(conf_mat, label_names, title=f'{args.dataset} (all samples)', savepath=fp)

        if not no_report:
            per_class = 'sbb' in args.dataset # per-class accuracies are mostly interesting for SBB dataset
            write_eval_report(
                args, out_dir=model_dir, model_path=model_path,
                nat_accs=nat_accs, pc_nat_accs=pc_nat_accs,
                label_names=label_names, per_class=per_class
            )

        if not no_log:
            write_sample_log(
                model_name, model_dir, args.dataset, args.eval_set,
                args.adv_norm, args.test_adv_attack, indices=nat_idx,
                is_acc=is_acc, preds=nat_preds
            )

    # empirical robustness evaluation
    if any('adv' in s for s in evals):
        for eps_str in test_eps:
            adv_accs = {}
            if adv_log is not None and f'{model_name}_is_rob{eps_str}' in adv_log.columns:
                is_acc = adv_log[f'{model_name}_is_acc'].to_numpy()
                is_rob = adv_log[f'{model_name}_is_rob{eps_str}'].to_numpy()
                rob_acc, pc_rob_accs = adv_accuracy_from_ind(is_acc, is_rob, targets, num_classes)
                rob_inacc, pc_rob_inaccs = rob_inacc_from_ind(is_acc, is_rob, targets, num_classes)

                adv_accs = {'adv_acc': round(rob_acc, 2), 'rob_inacc': round(rob_inacc, 2)}
                # add commit rate and commit precision to adv_accs dict
                adv_accs['robind_commit_rate'] = adv_accs['adv_acc'] + adv_accs['rob_inacc']
                adv_accs['robind_commit_prec'] = round(100.0 * adv_accs['adv_acc'] / adv_accs['robind_commit_rate'], 2)
                pc_adv_accs = [
                    {'adv_acc': round(pc_adv_acc, 2), 'rob_inacc': round(pc_rob_inacc, 2)}
                    for pc_adv_acc, pc_rob_inacc in zip(pc_rob_accs, pc_rob_inaccs)
                ]
            else:
                adv_accs, pc_adv_accs, _, _, is_rob, adv_idx = empirical_robustness_eval(
                    args, model, device, test_loader, args.test_adv_attack, args.adv_norm, eps_str
                )
                if not no_log:
                    write_sample_log(
                        model_name, model_dir, args.dataset, args.eval_set,
                        args.adv_norm, args.test_adv_attack, indices=adv_idx,
                        is_rob=is_rob, eps=eps_str
                    )

            logging.info(f'Adversarial accuracies ({eps_str} {args.adv_norm}) = {adv_accs}.')
            if not no_report:
                per_class = 'sbb' in args.dataset # per-class accuracies are mostly interesting for SBB dataset
                write_eval_report(
                    args, out_dir=model_dir, model_path=model_path,
                    adv_accs={eps_str: adv_accs}, pc_adv_accs={eps_str: pc_adv_accs},
                    adv_attack=args.test_adv_attack, label_names=label_names, per_class=per_class
                )

    # certified robustness via randomized smoothing evaluation
    if any('smo' in s for s in evals):
        cert_accs, linf_radii, l2_radii, base_predictions, predictions, labels, indices = smoothing_eval(
            args, model, model_dir, device, test_loader, test_eps, use_exist_log=args.use_exist_log
        )
        write_smoothing_log(args, model_dir, args.eval_set, linf_radii, l2_radii, base_predictions, predictions, labels, indices)
        logging.info(f'Smoothing certified accuracy (top1) = {cert_accs}')

        if not no_report:
            write_eval_report(args, model_dir, model_path, pcert_accs=cert_accs)


def main():
    """Evaluate accuracies of a given model.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = get_args()

    # init logging
    init_logging(args)

    # Build dataset: load unnormalized data. Normalization is done in model.
    train_loader, _, test_loader, _, _, _ = get_dataloader(
        args, args.dataset, normalize=False, indexed=True, shuffle_train=False, val_split=0.0
    )
    loader_to_eval = test_loader
    if args.eval_set == 'train':
        loader_to_eval = train_loader
    del train_loader, test_loader

    eval_model(
        args, args.model, device, loader_to_eval, test_eps=args.test_eps, evals=args.evals,
        use_exist_log=args.use_exist_log, no_report=args.no_eval_report, no_log=args.no_sample_log
    )


if __name__ == '__main__':
    main()


