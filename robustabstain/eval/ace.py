import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple, Dict, List

from robustabstain.ace.cert_deepTrunk import ai_cert_sample, ai_cert_sample_single_branch
from robustabstain.ace.deepTrunk_networks import MyDeepTrunkNet
from robustabstain.ace.networks import translate_net_name
from robustabstain.ace.utils import AdvAttack
from robustabstain.attacker.wrapper import AttackerWrapper
from robustabstain.eval.log import write_sample_log, write_eval_report
from robustabstain.utils.data_utils import get_dataset_stats
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.loaders import get_rel_sample_indices
from robustabstain.utils.metrics import AverageMeter
from robustabstain.utils.paths import eval_attack_log_filename



def build_ace_net(args: object, ace_model_path: str, device: str) -> MyDeepTrunkNet:
    """Build an ACE network given by model path args.model.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        ace_model_path (str): Path to saved ACE model.
        device (str): device.

    Returns:
        MyDeepTrunkNet: Loaded ACE network (=Deeptrunk network)
    """
    args.branch_nets = [translate_net_name(net) for net in args.branch_nets] if isinstance(args.branch_nets,list) else translate_net_name(args.branch_nets)
    args.gate_nets = None if args.gate_nets is None else [translate_net_name(net) for net in args.gate_nets]

    lossFn = nn.CrossEntropyLoss(reduction='none')
    def evalFn(x): return torch.max(x, dim=1)[1]
    input_dim, num_channels, num_classes = get_dataset_stats(args.dataset)
    dTNet = MyDeepTrunkNet.get_deepTrunk_net(
        args, device, lossFn, evalFn, input_dim, num_channels, num_classes, model_path=ace_model_path
    )

    return dTNet


def ace_eval(
        args: object, dTNet: MyDeepTrunkNet, device: str,
        test_loader: torch.utils.data.DataLoader, adv_norm: str,
        test_eps: str, cert_domain: str
    ) -> Tuple[
        Dict[str, float], np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
    """Evaluate natural accuracy, empirical robustness, certified robustness and gate selection on a given
    ACE architecture. This evaluation assumes that only a single branch model is present in the ACE model.
    Further, a trunk model that may be present is ignored, meaning only the gate (selector) network and the
    branch





  network are evaluated.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        dTNet (MyDeepTrunkNet): Deeptrunk network (ACE architecture).
        device (str): device.
        test_loader (torch.utils.data.DataLoader): PyTorch loader with test data.
        adv_norm (str): Norm of the perturbation region.
        test_eps (str): Size of adversarial perturbation region (stringified).
        cert_domain (str): Certification domain. Must be 'zono' for COLT trained models
            and 'box' for IBP trained models.

    Returns:
        Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Dict containing model measures, natural predictions, array indicating accuracy,
                array indicating empirical robustness, array indicating certified robustness,
                array indicating selection by gate net, array indicating empirical robustness of
                selector, array indicating certified robustness of selector, sample indices.
    """
    assert len(dTNet.gate_nets.keys()) == 1, 'Only evaluation of single branch ACE models supported.'
    exit_idx = 0
    dTNet.trunk_net = None # we dont care about the trunk
    dTNet.eval()
    dTNet.gate_nets[exit_idx].eval()
    dTNet.branch_nets[exit_idx].eval()

    test_eps_float = convert_floatstr(test_eps)
    n_samples = len(test_loader.dataset)
    nat_predictions = np.zeros(n_samples, dtype=np.int64)
    is_acc = np.zeros(n_samples, dtype=np.int64)
    is_rob = np.zeros(n_samples, dtype=np.int64)
    is_cert = np.zeros(n_samples, dtype=np.int64)
    is_select = np.zeros(n_samples, dtype=np.int64)
    select_rob = np.zeros(n_samples, dtype=np.int64) # gate predictions that are empirically robust
    select_cert = np.zeros(n_samples, dtype=np.int64) # gate predictions that are certifiable robust
    indices = np.zeros(n_samples, dtype=np.int64)

    adv_attack_gate = AdvAttack(
        eps=test_eps_float, n_steps=args.test_att_n_steps,
        step_size=args.test_att_step_size, adv_type="pgd"
    )
    attacker = AttackerWrapper(
        adv_type=args.test_adv_attack, adv_norm=adv_norm, eps=test_eps_float, steps=args.test_att_n_steps,
        rel_step_size=args.test_att_step_size, version=args.autoattack_version,
        gamma_ddn=args.gamma_ddn, init_norm_ddn=args.init_norm_ddn, device=device
    )

    nat_acc = AverageMeter()
    rob_acc = AverageMeter()
    cert_acc = AverageMeter()
    selection_rate = AverageMeter()
    selection_rate_adv = AverageMeter()
    selection_rate_cert = AverageMeter()

    pbar = tqdm(test_loader, dynamic_ncols=True)
    for batch_idx, (inputs, targets, sample_indices) in enumerate(pbar):
        rel_sample_indices = get_rel_sample_indices(test_loader, sample_indices)
        inputs, targets = inputs.to(device), targets.to(device)

        gate_nat_out = dTNet.gate_nets[exit_idx](inputs).squeeze()
        nat_select = (gate_nat_out >= dTNet.threshold[exit_idx]).int()
        nat_out = dTNet.branch_nets[exit_idx](inputs)
        nat_pred = nat_out.argmax(1)

        """GT Labels for attack are the predicted labels.
        This is necessary for the attack to also search
        for adversarial samples for inaccurate samples.
        """
        _, _, _, adv_inputs, _ = dTNet.get_adv_loss(inputs, nat_pred, adv_attack_gate)
        gate_adv_out = dTNet.gate_nets[exit_idx](adv_inputs).squeeze()
        adv_select = gate_adv_out >= dTNet.threshold[exit_idx]

        adv_inputs = attacker.attack(dTNet.branch_nets[exit_idx], inputs, nat_pred)
        adv_out = dTNet.branch_nets[exit_idx](adv_inputs)
        adv_pred = adv_out.argmax(1)

        is_acc_batch = nat_pred.eq(targets).int().cpu().numpy()
        is_select_batch = nat_select.int().cpu().numpy()
        is_rob_batch = adv_pred.eq(nat_pred).int().cpu().numpy()
        select_rob_batch = adv_select.eq(nat_select).int().cpu().numpy()

        nat_predictions[rel_sample_indices] = nat_pred.cpu().numpy()
        is_acc[rel_sample_indices] = is_acc_batch
        is_rob[rel_sample_indices] = is_rob_batch
        is_select[rel_sample_indices] = is_select_batch
        select_rob[rel_sample_indices] = select_rob_batch
        indices[rel_sample_indices] = sample_indices

        for input, target, pred, select, rel_sample_idx in zip(inputs, targets, nat_pred, nat_select, rel_sample_indices):
            input, target = input.unsqueeze(0), target.unsqueeze(0)
            pred, select = pred.unsqueeze(0), select.unsqueeze(0)
            with torch.no_grad():
                is_cert_gate, is_cert_branch = ai_cert_sample_single_branch(
                    dTNet, input, pred, select, cert_domain, test_eps_float
                )
                is_cert[rel_sample_idx] = is_cert_branch.item()
                select_cert[rel_sample_idx] = is_cert_gate.item()

                cert_acc.update(100 * is_cert_branch.item() * target.eq(pred).item(), 1)
                selection_rate_cert.update(100 * select.item() * is_cert_gate.item(), 1)
        nat_acc.update(100 * np.average(is_acc_batch), inputs.size(0))
        rob_acc.update(100 * np.average(is_rob_batch & is_acc_batch), inputs.size(0))
        selection_rate.update(100 * np.average(is_select_batch), inputs.size(0))
        selection_rate_adv.update(100 * np.average(is_select_batch & select_rob_batch), inputs.size(0))

        pbar.set_description('[V] ACE ({} eps={}): nat_acc={:.4f}, adv_acc={:.4f}, cert_acc={:.4f}, ' \
            'sel_rate_nat={:.4f}, sel_rate_adv={:.4f}, sel_rate_cert={:.4f},'.format(
                args.adv_norm, test_eps, nat_acc.avg, rob_acc.avg, cert_acc.avg,
                selection_rate.avg, selection_rate_adv.avg, selection_rate_cert.avg
        ))

    accs = {
        'nat_acc': round(nat_acc.avg, 2),
        'adv_acc': round(rob_acc.avg, 2),
        'cert_acc': round(cert_acc.avg, 2),
    }

    return accs, nat_predictions, is_acc, is_rob, is_cert, is_select, select_rob, select_cert, indices


def get_indicator_from_log(
        log_path: str, eps_str: str
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract '{model_name}_is_acc', '{model_name}_is_rob{eps}', '{model_name}_is_cert{eps}',
    '{model_name}_is_select' columns from sample log.

    Args:
        log_path (str): Path to log file.
        eps_str (str): Perturbation region size.


    Returns:
        Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Indicators found, accuracy indicator, robustness indicator, certification indicator,
            ACE selector indicator, empirical robustness of selector, certified robustness of selector,
            natural predictions of branch
    """
    indicators_found, is_acc, is_rob, is_cert = False, None, None, None
    is_select, select_rob, select_cert, pred = None, None, None, None
    if not os.path.isfile(log_path):
        logging.info(f'No log file {log_path} found, evaluating instead')
        return indicators_found, is_acc, is_rob, is_cert, is_select, select_rob, select_cert, pred

    log_df = pd.read_csv(log_path, index_col=0)
    indices_col = [col for col in log_df.columns if 'sample_idx' in col]
    is_acc_col = [col for col in log_df.columns if 'is_acc' in col]
    is_rob_col = [col for col in log_df.columns if f'is_rob{eps_str}' in col]
    is_cert_col = [col for col in log_df.columns if f'is_cert{eps_str}' in col]
    is_select_col = [col for col in log_df.columns if 'is_select' in col]
    select_rob_col = [col for col in log_df.columns if 'select_rob' in col]
    select_cert_col = [col for col in log_df.columns if 'select_cert' in col]
    pred_col = [col for col in log_df.columns if 'pred' in col]

    if (len(is_acc_col) != 1 or len(is_rob_col) != 1 or len(is_cert_col) != 1 or
        len(is_select_col) != 1 or len(select_rob_col) != 1 or len(select_cert_col) != 1 or
        len(pred_col) != 1):
        logging.info(f'Not all columns found in logfile {log_path}, evaluating instead.')

        return indicators_found, is_acc, is_rob, is_cert, is_select, select_rob, select_cert, pred

    indicators_found = True
    indices = log_df[indices_col[0]].to_numpy()
    is_acc = log_df[is_acc_col[0]].to_numpy()
    is_rob = log_df[is_rob_col[0]].to_numpy()
    is_cert = log_df[is_cert_col[0]].to_numpy()
    is_select = log_df[is_select_col[0]].to_numpy()
    select_rob = log_df[select_rob_col[0]].to_numpy()
    select_cert = log_df[select_cert_col[0]].to_numpy()
    pred = log_df[pred_col[0]].to_numpy()

    return indicators_found, is_acc, is_rob, is_cert, is_select, select_rob, select_cert, pred


def get_ace_indicator(
        args: object, dTNet: MyDeepTrunkNet, model_dir: str, model_name: str, device: str,
        dataloader: torch.utils.data.DataLoader, eval_set: str, adv_norm: str, eps_str: str,
        use_existing: bool = False, write_log: bool = False, write_report: bool = False
    ) -> Tuple[
        float, float, float, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
    """Get 0-1 indicators for each sample on whether the given ACE model is accurate,
    empirically robust, certifably robust, selected by gate network, for the given perturbation region.
    If no existing sample log is found in the model_dir, the model is evaluated from scratch.
    This evaluation assumes that only a single branch model is present in the ACE model
    and a trunk model that may be present is ignored, meaning only the gate (selector) network
    and the branch network are evaluated.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        dTNet (MyDeepTrunkNet): Deeptrunk network (ACE architecture).
        model_dir (str): Directory in which the model is stored (and associated eval logs).
        model_name (str): Name of the model to evaluate.
        device (str): device.
        dataloader (torch.utils.data.DataLoader): Dataloader to evaluate.
        eval_set (str): Dataset split that is evaluated ('train', 'val', 'test').
        adv_norm (str): Norm of the adversarial perturbation region.
        eps_str (str): Perturbation region size.
        use_existing (bool, optional): If set, an existing model eval log will be used (if such a log exists).
        write_log (bool, optional): If set, the evaluation log will be written to file. Defaults to False.
        write_report (bool, optional): If set, evaluation report will be written. Defaults to False.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                nat. accuracy of branch, adv. accuracy of branch, cert. accuracy of branch, accuracy indicator,
                empirical robustness indicator, certified robustness indicator, gate selection indicator,
                gate empirical robustness, gate certified robustness, branch natural predictions
    """
    # get indices order of samples in dataloader
    dataset_indices = np.arange(len(dataloader.dataset))
    if type(dataloader.dataset) == torch.utils.data.dataset.Subset:
        # Subset dataset have random index order
        dataset_indices = dataloader.dataset.indices

    # check first whether a log exists
    log_name = eval_attack_log_filename(eval_set, args.dataset, adv_norm, 'pgd')
    log_path = os.path.join(model_dir, log_name)
    indicators_found, is_acc, is_rob, is_cert, is_select, \
        select_rob, select_cert, predictions = get_indicator_from_log(log_path, eps_str)

    if indicators_found and use_existing:
        nat_acc1 = round(100.0 * np.average(is_acc[dataset_indices]), 2)
        adv_acc1 = round(100.0 * np.average(is_acc[dataset_indices] & is_rob[dataset_indices]), 2)
        cert_acc1 = round(100.0 * np.average(is_acc[dataset_indices] & is_cert[dataset_indices]), 2)
        select_rate = round(100.0 * np.average(is_select[dataset_indices]), 2)
        is_acc = is_acc[dataset_indices]
        is_rob = is_rob[dataset_indices]
        is_cert = is_cert[dataset_indices]
        is_select = is_select[dataset_indices]
        select_rob = select_rob[dataset_indices]
        select_cert = select_cert[dataset_indices]
        predictions = predictions if predictions is None else predictions[dataset_indices]

        return nat_acc1, adv_acc1, cert_acc1, is_acc, is_rob, is_cert, \
            is_select, select_rob, select_cert, predictions, dataset_indices

    # it is possible to run this function without a model given that a logfile is available
    if dTNet is None:
        raise ValueError(f'Error: no model provided and indicators not found in log {log_path}.')

    # put dataset into sequential dataloader to get deterministic sample order
    seq_dataloader = torch.utils.data.DataLoader(
        dataset=dataloader.dataset, batch_size=dataloader.batch_size,
        shuffle=False, num_workers=dataloader.num_workers
    )

    cert_domain = 'zono' if 'COLT' in model_name else 'box'
    accs, nat_preds, is_acc, is_rob, is_cert, is_select, select_rob, select_cert, indices = ace_eval(
        args, dTNet, device, seq_dataloader, args.adv_norm, eps_str, cert_domain
    )
    assert (indices == dataset_indices).all(), 'Indices of ACE evaluation are not in expected order.'
    nat_acc1 = accs['nat_acc']
    adv_acc1 = accs['adv_acc']
    cert_acc1 = accs['cert_acc']

    if write_report:
        write_eval_report(
            args, out_dir=model_dir, nat_accs=[nat_acc1], adv_accs={eps_str: adv_acc1},
            adv_attack='pgd', dcert_accs={eps_str: cert_acc1}
        )

    if write_log:
        write_sample_log(
            model_name, model_dir, args.dataset, args.eval_set, args.adv_norm, 'pgd',
            indices=indices, is_acc=is_acc, preds=nat_preds, is_rob=is_rob,
            is_cert=is_cert, is_select=is_select, select_rob=select_rob,
            select_cert=select_cert, eps=eps_str
        )

    return nat_acc1, adv_acc1, cert_acc1, is_acc, is_rob, is_cert, \
        is_select, select_rob, select_cert, predictions, dataset_indices