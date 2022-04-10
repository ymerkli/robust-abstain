import torch
import torch.nn as nn

import os
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union

from robustabstain.abstain.selector import abstain_selector
from robustabstain.abstain.confidence_threshold import get_confidences
from robustabstain.eval.ace import build_ace_net
from robustabstain.eval.ace import get_ace_indicator
from robustabstain.eval.adv import get_acc_rob_indicator
from robustabstain.eval.cert import get_acc_cert_indicator
from robustabstain.eval.comp import compositional_accuracy, get_comp_indicator, get_comp_key
from robustabstain.eval.log import write_eval_report
from robustabstain.utils.checkpointing import load_checkpoint
from robustabstain.utils.helpers import multiply_eps
from robustabstain.utils.regex import ABSTAIN_MODEL_DIR_RE


def commit_prec(rob_acc: Union[float, np.float], commit_rate: Union[float, np.float]) -> float:
    """Get commit precision. Commit precision is set to 0 for 0 commit rate.

    Args:
        rob_acc (float): Fraction of samples that are robust and accurate.
        commit_rate (float): Fraction of samples that a committed.

    Returns:
        float: Commit precision.
    """
    commit_prec = 0
    if commit_rate != 0:
        commit_prec = rob_acc / commit_rate
    return commit_prec


def conf_model_measures(
        args: object, model_path: str, test_loader: torch.utils.data.DataLoader,
        device: str,  test_eps: str, trunk_is_acc: np.ndarray,
        trunk_is_acc_adv: np.ndarray, trunk_is_rob: np.ndarray, n_conf_steps: int = 20,
        branchpred_path: str = None, eval_2xeps: bool = False, adv_attack: str = 'apgdconf'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get abstain model measures using confidence thresholding abstain.
    A single model is evaluated for various confidence threshold steps.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model_path (str): Model path.
        test_loader (torch.utils.data.DataLoader): Test dataloader
        device (str): device
        test_eps (str): Perturbation region size to get model measures for.
        trunk_is_acc (np.ndarray): Binary indicator indicating accuracy of trunk.
        trunk_is_acc_adv (np.ndarray): Binary indicator indicating accuracy under an adversary,
            (when the trunk is compositional).
        trunk_is_rob (np.ndarray): BInary indicator indicating robustness of the trunk.
        n_conf_steps (int, optional): Number of confidence threshold steps in range [0,0.99]. Defaults to 20.
        branchpred_path (str, optional): Path to branch model that makes prediction. If set, this model is
            used to evaluate predictions, while models from model_paths are used for abstain. Defaults to None.
        eval_2xeps (bool, optional): If set, abstain branch models from model_paths are evaluated at 2x
            the perturbation region size. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Compositional natural accuracy, compositional adversarial accuracy,
            commit precision for natural accuracy, commit precision for adversarial accuracy,
            natural commit rate (no adversary), adversarial commit rate (adversary),
            natural misselected (confident but inaccurate),
            adversarially misselected (robustly confident but prediction is non-robust or inaccurate)
    """
    branch_model = None
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_dir = os.path.dirname(model_path)
    if os.path.isfile(model_path):
        branch_model, _, _ = load_checkpoint(
            model_path, net=None, arch=None, dataset=args.dataset, device=device,
            normalize=not args.no_normalize, optimizer=None, parallel=True
        )
    
    evalabst_eps = test_eps
    if eval_2xeps:
        # for compositional eval, abstain branch must be evaluated at double the perturbation region
        # to guarantee selection for every point in the region
        evalabst_eps = multiply_eps(test_eps, 2)
        (
            _, _, _, branchabst_is_acc, branchabst_is_rob, _,
            branchabst_nat_conf, branchabst_adv_conf, _
        ) = get_acc_rob_indicator(
            args, branch_model, model_dir, model_name, device,
            test_loader, 'test', args.adv_norm, evalabst_eps,
            adv_attack=adv_attack, use_existing=True,
            write_log=True, write_report=False
        )
    else:
        (
            _, _, _, branchabst_is_acc, branchabst_is_rob, _,
            branchabst_nat_conf, branchabst_adv_conf, _
        ) = get_acc_rob_indicator(
            args, branch_model, model_dir, model_name, device,
            test_loader, 'test', args.adv_norm, evalabst_eps,
            adv_attack=adv_attack, use_existing=True,
            write_log=True, write_report=False
        )

    if branchpred_path:
        branchpred_name = os.path.splitext(os.path.basename(branchpred_path))[0]
        branchpred_dir = os.path.dirname(branchpred_path)
        if os.path.isfile(branchpred_path):
            branchpred_model, _, _ = load_checkpoint(
                branchpred_path, net=None, arch=None, dataset=args.dataset, device=device,
                normalize=not args.no_normalize, optimizer=None, parallel=True
            )
        _, _, _, branchpred_is_acc, branchpred_is_rob, _, _, _, _ = get_acc_rob_indicator(
            args, branchpred_model, branchpred_dir, branchpred_name, device,
            test_loader, 'test', args.adv_norm, test_eps,
            args.test_adv_attack, use_existing=True,
            write_log=True, write_report=True
        )
    else:
        _, _, _, branchpred_is_acc, branchpred_is_rob, _, _, _, _ = get_acc_rob_indicator(
            args, branch_model, model_dir, model_name, device,
            test_loader, 'test', args.adv_norm, test_eps,
            args.test_adv_attack, use_existing=True,
            write_log=True, write_report=False
        )

    conf_thresholds = np.linspace(start=0, stop=0.99, num=n_conf_steps)
    conf_comp_nat_acc = np.zeros(n_conf_steps)
    conf_comp_adv_acc = np.zeros(n_conf_steps)
    conf_commit_prec_nat = np.zeros(n_conf_steps)
    conf_commit_prec_adv = np.zeros(n_conf_steps)
    conf_commit_rate_nat = np.zeros(n_conf_steps)
    conf_commit_rate_adv = np.zeros(n_conf_steps)
    conf_misselect_nat = np.zeros(n_conf_steps)
    conf_misselect_adv = np.zeros(n_conf_steps)

    for k, conf_t in enumerate(conf_thresholds):
        branchabst_is_confident_nat = (branchabst_nat_conf >= conf_t).astype(np.int64)
        branchabst_is_confident_adv = (branchabst_adv_conf >= conf_t).astype(np.int64)

        # committed adversarial accuracy on branch
        conf_commit_nat_acc = 100.0 * np.average(branchabst_is_confident_nat & branchpred_is_acc)
        conf_commit_adv_acc = 100.0 * np.average(branchabst_is_confident_adv & branchpred_is_acc & branchpred_is_rob)

        # commit precision and rates are evaluated on adversarial selector
        conf_commit_rate_nat[k] = 100.0 * np.average(branchabst_is_confident_nat)
        conf_commit_rate_adv[k] = 100.0 * np.average(branchabst_is_confident_adv)
        conf_commit_prec_nat[k] = 100.0 * commit_prec(conf_commit_nat_acc, conf_commit_rate_nat[k])
        conf_commit_prec_adv[k] = 100.0 * commit_prec(conf_commit_adv_acc, conf_commit_rate_adv[k])

        # committed samples that are inaccurate or non-robust
        conf_misselect_nat[k] = 100.0 * np.average(branchabst_is_confident_nat & (1-branchpred_is_acc))
        conf_misselect_adv[k] = 100.0 * np.average(branchabst_is_confident_adv & ((1-branchpred_is_rob) | (1-branchpred_is_acc)))

        if len(args.trunk_models) > 1:
            # if trunk is compositional architecture, re-evaluate
            _, _, trunk_is_acc, trunk_is_acc_adv, trunk_is_rob = get_comp_indicator(
                args, branch_model_path=args.trunk_models[0],
                trunk_model_paths=args.trunk_models[1:], device=device,
                dataloader=test_loader, eval_set='test', eps_str=test_eps,
                abstain_method='conf', conf_threshold=conf_t, use_existing=True
            )

        conf_comp_nat_acc[k], conf_comp_adv_acc[k] = compositional_accuracy(
            branch_is_acc=branchpred_is_acc, branch_is_rob=branchpred_is_rob,
            trunk_is_acc=trunk_is_acc, trunk_is_rob=trunk_is_rob,
            selector=branchabst_is_confident_nat, adv_selector=branchabst_is_confident_adv,
            trunk_is_acc_adv=trunk_is_acc_adv
        )

    return (
        conf_comp_nat_acc, conf_comp_adv_acc,
        conf_commit_prec_nat, conf_commit_prec_adv,
        conf_commit_rate_nat, conf_commit_rate_adv,
        conf_misselect_nat, conf_misselect_adv
    )


def adv_robind_model_measures(
        args: object, model_paths: List[str], test_loader: torch.utils.data.DataLoader, device: str,
        test_eps: str, trunk_is_acc: np.ndarray, trunk_is_acc_adv: np.ndarray, trunk_is_rob: np.ndarray,
        branchpred_path: str = None, eval_2xeps: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Get abstain model measures using empirical robustness indicator abstain for a list of models.

    Args:
        args (object): [description]
        model_paths (List[str]): List of model paths.
        test_loader (torch.utils.data.DataLoader): Test dataloader
        device (str): device
        test_eps (str): Perturbation region size to get model measures for.
        trunk_is_acc (np.ndarray): Binary indicator indicating accuracy of trunk.
        trunk_is_acc_adv (np.ndarray): Binary indicator indicating accuracy under an adversary,
            (when the trunk is compositional).
        trunk_is_rob (np.ndarray): Binary indicator indicating robustness of the trunk.
        branchpred_path (str, optional): Path to branch model that makes prediction. If set, this model is
            used to evaluate predictions, while models from model_paths are used for abstain. Defaults to None.
        eval_2xeps (bool, optional): If set, abstain branch models from model_paths are evaluated at 2x
            the perturbation region size. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Natural accuracies, adversarial accuracies, robust inaccurate fraction, compositional natural
            accuracy, compositional adversarial accuracies, commit precision, commit rate, model timestamps.
    """
    if branchpred_path:
        branchpred_name = os.path.splitext(os.path.basename(branchpred_path))[0]
        branchpred_dir = os.path.dirname(branchpred_path)
        if os.path.isfile(branchpred_path):
            branchpred_model, _, _ = load_checkpoint(
                branchpred_path, net=None, arch=None, dataset=args.dataset, device=device,
                normalize=not args.no_normalize, optimizer=None, parallel=True
            )

        _, _, _, branchpred_is_acc, branchpred_is_rob, _, _, _, _ = get_acc_rob_indicator(
            args, branchpred_model, branchpred_dir, branchpred_name, device,
            test_loader, 'test', args.adv_norm, test_eps,
            args.test_adv_attack, use_existing=True,
            write_log=True, write_report=True
        )

    n_models = len(model_paths)
    branchpred_nat_acc = np.zeros(n_models)
    branchpred_adv_acc = np.zeros(n_models)
    branchpred_rob_inacc = np.zeros(n_models)
    robind_comp_nat_acc = np.zeros(n_models)
    robind_comp_adv_acc = np.zeros(n_models)
    robind_commit_prec = np.zeros(n_models)
    robind_commit_rate = np.zeros(n_models)
    model_ts = [''] * n_models
    for j, model_path in enumerate(model_paths):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_dir = os.path.dirname(model_path)
        match = re.match(ABSTAIN_MODEL_DIR_RE, model_dir)
        if match:
            model_ts[j] = match.group('timestamp')
        else:
            model_ts[j] = model_dir

        branch_model = None
        if os.path.isfile(model_path):
            branch_model, _, _ = load_checkpoint(
                model_path, net=None, arch=None, dataset=args.dataset, device=device,
                normalize=not args.no_normalize, optimizer=None, parallel=True
            )

        # eval branch model
        if not branchpred_path:
            _, _, _, branchpred_is_acc, branchpred_is_rob, _, _, _, _ = get_acc_rob_indicator(
                args, branch_model, model_dir, model_name, device,
                test_loader, 'test', args.adv_norm, test_eps,
                args.test_adv_attack, use_existing=True,
                write_log=True, write_report=True
            )

        (
            _, branchabst_adv_acc, branchabst_rob_inacc,
            branchabst_is_acc, branchabst_is_rob, _, _, _, _
        ) = get_acc_rob_indicator(
            args, branch_model, model_dir, model_name, device,
            test_loader, 'test', args.adv_norm, test_eps,
            args.test_adv_attack, use_existing=True,
            write_log=True, write_report=True
        )
        evalabst_eps = test_eps
        if eval_2xeps:
            # for compositional eval, abstain branch must be evaluated at double the perturbation region
            # to guarantee selection for every point in the region
            evalabst_eps = multiply_eps(test_eps, 2)
            _, _, _, branchabst_is_acc, branchabst_is_rob, _, _, _, _ = get_acc_rob_indicator(
                args, branch_model, model_dir, model_name, device,
                test_loader, 'test', args.adv_norm, evalabst_eps,
                args.test_adv_attack, use_existing=True,
                write_log=True, write_report=True
            )

        branchpred_nat_acc[j] = 100.0 * np.average(branchpred_is_acc)
        branchpred_adv_acc[j] = 100.0 * np.average(branchpred_is_rob & branchpred_is_acc)
        branchpred_rob_inacc[j] = 100.0 * np.average(branchpred_is_rob & (1 - branchpred_is_acc))

        robind_commit_rate[j] = 100.0 * np.average(branchabst_is_rob)
        committed_adv_acc = 100.0 * np.average(branchabst_is_rob & branchpred_is_rob & branchpred_is_acc)
        robind_commit_prec[j] = 100.0 * commit_prec(committed_adv_acc, robind_commit_rate[j])

        # get robustness selector
        selector, adv_selector = abstain_selector(
            args, branch_model, model_dir, model_name, device,
            test_loader, method='rob', eps=evalabst_eps, is_rob=branchabst_is_rob
        )

        # evaluate branch and trunk in compositional architecture
        robind_comp_nat_acc[j], robind_comp_adv_acc[j] = compositional_accuracy(
            branch_is_acc=branchpred_is_acc, branch_is_rob=branchpred_is_rob,
            trunk_is_acc=trunk_is_acc, trunk_is_rob=trunk_is_rob,
            selector=selector, adv_selector=adv_selector,
            trunk_is_acc_adv=trunk_is_acc_adv
        )

        # write report
        comp_key = get_comp_key(model_path, args.trunk_models)
        # report keys
        if eval_2xeps and branchpred_path:
            cna = 'comp_nat_acc_2xsepbr'
            cra = 'comp_adv_acc_2xsepbr'
            cp = 'robind_commit_prec_2xsepbr'
            cr = 'robind_commit_rate_2xsepbr'
        elif eval_2xeps:
            cna = 'comp_nat_acc_2x'
            cra = 'comp_adv_acc_2x'
            cp = 'robind_commit_prec_2x'
            cr = 'robind_commit_rate_2x'
        else:
            cna = 'comp_nat_acc'
            cra = 'comp_adv_acc'
            cp = 'robind_commit_prec'
            cr = 'robind_commit_rate'
        selector_key = 'rob'
        comp_accs = {
            test_eps: {
                comp_key: {
                    selector_key: {
                        cna: robind_comp_nat_acc[j],
                        cra: robind_comp_adv_acc[j],
                        'branch_model': model_path,
                        'trunk_model': args.trunk_models
        }}}}
        adv_accs = {
            test_eps: {
                'adv_acc': branchabst_adv_acc,
                'rob_inacc': branchabst_rob_inacc,
                cp: robind_commit_prec[j],
                cr: robind_commit_rate[j]
            }
        }
        write_eval_report(
            args, out_dir=model_dir, adv_attack=args.test_adv_attack,
            adv_accs=adv_accs, comp_accs=comp_accs
        )

    return (
        branchpred_nat_acc, branchpred_adv_acc, branchpred_rob_inacc, robind_comp_nat_acc,
        robind_comp_adv_acc, robind_commit_prec, robind_commit_rate, model_ts
    )


def ace_model_measures(
        args: object, model_paths: List[str], test_loader: torch.utils.data.DataLoader, device: str,
        test_eps: str, trunk_is_acc: np.ndarray, trunk_is_acc_adv: np.ndarray, trunk_is_rob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get performance measures for list of ACE models. Compositional accuracies are evaluated
    under gate network selection.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model_paths (List[str]): List of ACE model paths.
        test_loader (torch.utils.data.DataLoader): Test dataloader
        device (str): device
        test_eps (str): Perturbation region size to get model measures for.
        trunk_is_acc (np.ndarray): Binary indicator indicating accuracy of trunk.
        trunk_is_acc_adv (np.ndarray): Binary indicator indicating accuracy under an adversary,
            (when the trunk is compositional).
        trunk_is_rob (np.ndarray): BInary indicator indicating robustness of the trunk.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Natural accuracies, adversarial accuracies, certified accuracies, compositional natural accuracies,
            compositional adversarial accuracies, compositional certified accuracies,
            gate commit precision empirical robust accuracy, gate precision certified robust accuracy,
            gate commit rate empirical robustness, gate commit rate certified robustness.
    """
    n_models = len(model_paths)
    model_nat_acc = np.zeros(n_models) # branch is selected and natural accurate
    model_adv_acc = np.zeros(n_models) # branch is selected, accurate and empirically robust
    model_cert_acc = np.zeros(n_models) # branch is selected, accurate and certifably robust
    gate_comp_nat_acc = np.zeros(n_models) # compositional natural accuracy under gate selection
    gate_comp_adv_acc = np.zeros(n_models) # compositional adversarial accuracy under gate selection
    gate_comp_cert_acc = np.zeros(n_models) # compositional certified accuracy under gate selection
    gate_commit_prec_nat = np.zeros(n_models) # Precision of gate selecting naturally accurate samples
    gate_commit_prec_adv = np.zeros(n_models) # Precision of gate selecting adversarially robust accurate samples
    gate_commit_prec_cert = np.zeros(n_models) # Precision of gate selecting certifably robust accurate samples
    gate_commit_rate_nat = np.zeros(n_models) # Fraction of samples samples being selected by gate
    gate_commit_rate_adv = np.zeros(n_models) # Fraction of samples being robustly selected by gate
    gate_commit_rate_cert = np.zeros(n_models) # Fraction of samples being certifiablt selected by gate
    gate_misselect_nat = np.zeros(n_models) # Fraction of inaccurate samples being selected by gate
    gate_misselect_adv = np.zeros(n_models) # Fraction of inaccurate samples being robustly selected by gate
    gate_misselect_cert = np.zeros(n_models) # Fraction of inaccurate samples being certifiably selected by gate

    # iterate over all given ACE models
    for j, model_path in enumerate(model_paths):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_dir = os.path.dirname(model_path)

        # build ACE model
        dTNet = None
        if os.path.isfile(model_path):
            dTNet = build_ace_net(args, model_path, device)

        # eval ACE model
        _, _, _, branch_is_acc, branch_is_rob, branch_is_cert, \
            branch_is_select, select_rob, select_cert, _, _ = get_ace_indicator(
                args, dTNet, model_dir, model_name, device, test_loader,
                'test', args.adv_norm, test_eps, use_existing=True,
                write_log=True, write_report=True
        )
        model_nat_acc[j] = 100.0 * np.average(branch_is_acc & branch_is_select)
        model_adv_acc[j] = 100.0 * np.average(branch_is_rob & branch_is_acc & branch_is_select & select_rob)
        model_cert_acc[j] = 100.0 * np.average(branch_is_cert & branch_is_acc & branch_is_select & select_cert)
        gate_commit_rate_nat[j] = 100.0 * np.average(branch_is_select)
        gate_commit_rate_adv[j] = 100.0 * np.average(branch_is_select & select_rob)
        gate_commit_rate_cert[j] = 100.0 * np.average(branch_is_select & select_cert)
        gate_commit_prec_nat[j] = 100.0 * commit_prec(100.0 * np.average(branch_is_acc & branch_is_select), gate_commit_rate_nat[j])
        gate_commit_prec_adv[j] = 100.0 * commit_prec(100.0 * np.average(branch_is_rob & branch_is_acc & branch_is_select & select_rob), gate_commit_rate_adv[j])
        gate_commit_prec_cert[j] = 100.0 * commit_prec(100.0 * np.average(branch_is_cert & branch_is_acc & branch_is_select & select_cert), gate_commit_rate_cert[j])
        gate_misselect_nat[j] = 100.0 * np.average(1-branch_is_acc & branch_is_select)
        gate_misselect_adv[j] = 100.0 * np.average((1-branch_is_rob | 1-branch_is_acc) & branch_is_select & select_rob)
        gate_misselect_cert[j] = 100.0 * np.average((1-branch_is_cert | 1-branch_is_acc) & branch_is_select & select_cert)

        # evaluate branch and trunk in compositional architecture for both empirical and certified robustness
        gate_comp_nat_acc[j], gate_comp_adv_acc[j] = compositional_accuracy(
            branch_is_acc=branch_is_acc, branch_is_rob=branch_is_rob,
            trunk_is_acc=trunk_is_acc, trunk_is_rob=trunk_is_rob,
            selector=branch_is_select, adv_selector=branch_is_select&select_rob,
            trunk_is_acc_adv=trunk_is_acc_adv
        )
        _, gate_comp_cert_acc[j] = compositional_accuracy(
            branch_is_acc=branch_is_acc, branch_is_rob=branch_is_cert,
            trunk_is_acc=trunk_is_acc, trunk_is_rob=trunk_is_rob,
            selector=branch_is_select, adv_selector=branch_is_select&select_cert,
            trunk_is_acc_adv=trunk_is_acc_adv
        )

        # write report
        comp_key = get_comp_key(model_path, args.trunk_models)
        selector_key = 'selnet'
        comp_accs = {
            test_eps: {
                comp_key: {
                    selector_key: {
                        'comp_nat_acc': gate_comp_nat_acc[j],
                        'comp_adv_acc': gate_comp_adv_acc[j],
                        'comp_cert_acc': gate_comp_cert_acc[j],
                        'branch_model': model_path,
                        'trunk_model': args.trunk_models
        }}}}
        write_eval_report(args, out_dir=model_dir, adv_attack='cert', comp_accs=comp_accs)

    return (
        model_nat_acc, model_adv_acc, model_cert_acc,
        gate_comp_nat_acc, gate_comp_adv_acc, gate_comp_cert_acc,
        gate_commit_prec_nat, gate_commit_prec_adv, gate_commit_prec_cert,
        gate_commit_rate_nat, gate_commit_rate_adv, gate_commit_rate_cert,
        gate_misselect_nat, gate_misselect_adv, gate_misselect_cert
    )


def cert_robind_model_measures(
        args: object, model_paths: List[str], test_loader: torch.utils.data.DataLoader, device: str,
        test_eps: str, trunk_is_acc: np.ndarray, trunk_is_acc_adv: np.ndarray, trunk_is_rob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get abstain model measures using probabilistic certified robustness indicator
    abstain for a list of models.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model_paths (List[str]): List of model paths.
        test_loader (torch.utils.data.DataLoader): Test dataloader
        device (str): device
        test_eps (str): Perturbation region size to get model measures for.
        trunk_is_acc (np.ndarray): Binary indicator indicating accuracy of trunk.
        trunk_is_acc_adv (np.ndarray): Binary indicator indicating accuracy under an adversary,
            (when the trunk is compositional).
        trunk_is_cert (np.ndarray): BInary indicator indicating robustness of the trunk.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Natural accuracies, adversarial accuracies, robust inaccurate fraction, compositional natural
            accuracy, compositional adversarial accuracies, commit precision, commit rate.
    """
    n_models = len(model_paths)
    model_nat_acc = np.zeros(n_models)
    model_cert_acc = np.zeros(n_models)
    model_cert_inacc = np.zeros(n_models)
    robind_comp_nat_acc = np.zeros(n_models)
    robind_comp_cert_acc = np.zeros(n_models)
    robind_commit_prec = np.zeros(n_models)
    robind_commit_rate = np.zeros(n_models)
    for j, model_path in enumerate(model_paths):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_dir = os.path.dirname(model_path)

        branch_model = None
        if os.path.isfile(model_path):
            branch_model, _, _ = load_checkpoint(
                model_path, net=None, arch=None, dataset=args.dataset, device=device,
                normalize=not args.no_normalize, optimizer=None, parallel=True
            )

        # eval branch model
        _, cert_acc, cert_inacc, branch_is_acc, branch_is_cert, _, indices = get_acc_cert_indicator(
            args, branch_model, model_dir, model_name, device,
            test_loader, 'test', test_eps, smooth=True, n_smooth_samples=500,
            use_existing=True, write_log=True, write_report=True
        )
        model_nat_acc[j] = 100.0 * np.average(branch_is_acc)
        model_cert_acc[j] = 100.0 * np.average(branch_is_cert & branch_is_acc)
        model_cert_inacc[j] = 100.0 * np.average(branch_is_cert & 1-branch_is_acc)
        robind_commit_rate[j] = 100.0 * np.average(branch_is_cert)
        robind_commit_prec[j] = 100.0 * commit_prec(model_cert_acc[j], robind_commit_rate[j])

        # get robustness selector
        selector, adv_selector = abstain_selector(
            args, branch_model, model_dir, model_name, device,
            test_loader, method='rob', eps=test_eps, is_rob=branch_is_cert
        )

        # evaluate branch and trunk in compositional architecture
        robind_comp_nat_acc[j], robind_comp_cert_acc[j] = compositional_accuracy(
            branch_is_acc=branch_is_acc, branch_is_rob=branch_is_cert,
            trunk_is_acc=trunk_is_acc[indices], trunk_is_rob=trunk_is_rob[indices],
            selector=selector, adv_selector=adv_selector,
            trunk_is_acc_adv=trunk_is_acc_adv[indices]
        )

        # write report
        comp_key = get_comp_key(model_path, args.trunk_models)
        selector_key = 'rob'
        comp_accs = {
            test_eps: {
                comp_key: {
                    selector_key: {
                        'comp_nat_acc': robind_comp_nat_acc[j],
                        'comp_cert_acc': robind_comp_cert_acc[j],
                        'branch_model': model_path,
                        'trunk_model': args.trunk_models
        }}}}
        write_eval_report(args, out_dir=model_dir, adv_attack='cert', comp_accs=comp_accs)

    return (
        model_nat_acc, model_cert_acc, model_cert_inacc,
        robind_comp_nat_acc, robind_comp_cert_acc,
        robind_commit_prec, robind_commit_rate
    )


def cert_robind_model_measures_varrad(
        args: object, model_path: str, test_loader: torch.utils.data.DataLoader, device: str,
        eps_range: List[float], trunk_is_acc: np.ndarray, trunk_is_acc_adv: np.ndarray,
        trunk_is_rob: np.ndarray, n_eps_steps: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get abstain model measures using probabilistic certified robustness indicator
    abstain for a single model evaluated at a range of perturbation region radii.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model_path (str): Path to model to evaluate.
        test_loader (torch.utils.data.DataLoader): Test dataloader
        device (str): device
        eps_range (List[float]): Lower and upper bound of perturbation region size to consider.
        trunk_is_acc (np.ndarray): Binary indicator indicating accuracy of trunk.
        trunk_is_acc_adv (np.ndarray): Binary indicator indicating accuracy under an adversary,
            (when the trunk is compositional).
        trunk_is_cert (np.ndarray): BInary indicator indicating robustness of the trunk.
        n_eps_steps (int, optional): Number of steps in eps_range. Defaults to 20.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Natural accuracies, adversarial accuracies, robust inaccurate fraction, compositional natural
            accuracy, compositional adversarial accuracies, commit precision, commit rate.
    """
    assert len(eps_range) == 2, 'Error: eps_range must be of length 2: [lb, ub]'
    model_nat_acc = np.zeros(n_eps_steps)
    model_cert_acc = np.zeros(n_eps_steps)
    model_cert_inacc = np.zeros(n_eps_steps)
    robind_comp_nat_acc = np.zeros(n_eps_steps)
    robind_comp_cert_acc = np.zeros(n_eps_steps)
    robind_commit_prec = np.zeros(n_eps_steps)
    robind_commit_rate = np.zeros(n_eps_steps)

    test_epss = np.linspace(start=eps_range[0], stop=eps_range[1], num=n_eps_steps)
    for i, test_eps in enumerate(test_epss):
        (
            model_nat_acc[i], model_cert_acc[i], model_cert_inacc[i],
            robind_comp_nat_acc[i], robind_comp_cert_acc[i],
            robind_commit_prec[i], robind_commit_rate[i]
        ) = cert_robind_model_measures(
                args, [model_path], test_loader, device, str(test_eps),
                trunk_is_acc, trunk_is_acc_adv, trunk_is_rob
        )

    return (
        model_nat_acc, model_cert_acc, model_cert_inacc, robind_comp_nat_acc,
        robind_comp_cert_acc, robind_commit_prec, robind_commit_rate
    )


def get_running_chkpt_vals(
        args: object, test_loader: torch.utils.data.DataLoader, device: str,
        eps: str, branch_model_name: str, branch_running_chkpt_dir: str,
        trunk_is_acc: np.ndarray, trunk_is_rob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get model figures for all running checkpointed models.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data.
        device (str): device.
        eps (str): Perturbation region size.
        branch_model_name (str): Name of the branch model.
        branch_running_chkpt_dir (str): Directory of running checkpoints.
        trunk_is_acc (np.ndarray): Binary array indicating nat accuracy on trunk model.
        trunk_is_rob (np.ndarray): Binary array indicating robustness on trunk model.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: For each found checkpoint:
            nat accuracy, adv accuracy, rob_inacc, commit precision, comp nat accuracy, comp adv accuracy
    """
    branch_running_chkpt_dir = Path(branch_running_chkpt_dir)
    branch_chkpt_dirs = sorted(
        [d for d in branch_running_chkpt_dir.iterdir() if d.is_dir()],
        key=lambda path: int(os.path.basename(path))
    )
    n_chkpts = len(branch_chkpt_dirs)
    branch_nat_accs = np.zeros(n_chkpts)
    branch_adv_accs = np.zeros(n_chkpts)
    branch_rob_inaccs = np.zeros(n_chkpts)
    commit_precisions = np.zeros(n_chkpts)
    comp_nat_accs = np.zeros(n_chkpts)
    comp_adv_accs = np.zeros(n_chkpts)

    # iterate over checkpointed models
    for i, branch_chkpt_dir in enumerate(branch_chkpt_dirs):
        branch_chkpt_path = os.path.join(branch_chkpt_dir, branch_model_name+'.pt')
        # load checkpointed branch model
        branch_model, _, _ = load_checkpoint(
            branch_chkpt_path, net=None, arch=None, dataset=args.dataset, device=device,
            normalize=not args.no_normalize, optimizer=None, parallel=True
        )
        _, _, _, branch_is_acc, branch_is_rob, _, _, _, _ = get_acc_rob_indicator(
                args, branch_model, branch_chkpt_dir, branch_model_name, device,
                test_loader, 'test', args.adv_norm, eps, args.test_adv_attack,
                use_existing=True, write_log=True, write_report=True
        )
        branch_nat_accs[i] = 100.0 * np.average(branch_is_acc)
        branch_adv_accs[i] = 100.0 * np.average(branch_is_rob & branch_is_acc)
        branch_rob_inaccs[i] = 100.0 * np.average(branch_is_rob & (1 - branch_is_acc))
        commit_precisions[i] = 100.0 * branch_adv_accs[i] / (branch_adv_accs[i] + branch_rob_inaccs[i])

        selector = abstain_selector(
            args, branch_model, device, test_loader, args.selector,
            eps, is_rob=branch_is_rob, conf_threshold=args.conf_threshold
        )
        comp_nat_accs[i], comp_adv_accs[i] = compositional_accuracy(
            branch_is_acc=branch_is_acc, branch_is_rob=branch_is_rob,
            trunk_is_acc=trunk_is_acc, trunk_is_rob=trunk_is_rob,
            selector=selector
        )

    return branch_nat_accs, branch_adv_accs, branch_rob_inaccs, commit_precisions, comp_nat_accs, comp_adv_accs
