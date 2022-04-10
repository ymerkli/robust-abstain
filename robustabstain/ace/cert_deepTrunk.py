import torch
import numpy as np
import random
from warnings import warn
from typing import Tuple

from robustabstain.ace.deepTrunk_networks import MyDeepTrunkNet


seed = 100

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.set_printoptions(precision=10)
np.random.seed(seed)
random.seed(seed)


def ai_cert_sample_single_branch(
        dTNet: MyDeepTrunkNet, inputs: torch.tensor, target: torch.tensor,
        selection_target: torch.tensor, domain: str, eps: float
    ) -> Tuple[torch.tensor, torch.tensor]:
    """Certify the branch and gate of a Deeptrunknetwork with only a single
    branch network.

    Args:
        dTNet (MyDeepTrunkNet): Combined Deeptrunk network.
        inputs (torch.tensor): Inputs to certify.
        target (torch.tensor): Targets to certify inputs for.
        selection_target (torch.tensor): Targets of the gate network.
        domain (str): Certification domain.
        eps (float): Perturbation region size.

    Returns:
        Tuple[torch.tensor, torch.tensor]: Tensor indicating certified samples on gate, tensor
            indicating certified samples on branch.
    """
    assert len(dTNet.gate_nets.keys()) == 1, 'Only evaluation of single branch ACE models supported.'
    exit_idx = 0
    dTNet.eval()

    # verify whether the gate is certified
    is_cert_gate, _, _ = dTNet.gate_cnets[exit_idx].get_abs_loss(
        inputs, selection_target, eps, domain, dTNet.threshold[exit_idx], beta=1
    )

    # verify whether the prediction of the branch is certified robust
    is_cert_branch, _, _ = dTNet.branch_cnets[exit_idx].get_abs_loss(
        inputs, target, eps, domain, dTNet.branch_cnets[exit_idx].threshold_min, beta=1
    )

    return is_cert_gate, is_cert_branch


def ai_cert_sample(dTNet, inputs, target, branch_p, domain, eps, break_on_failure, cert_trunk=True):
    dTNet.eval()
    ver_corr = torch.ones_like(target).byte()
    ver_not_trunk = False
    gate_threshold_s = {}
    n_class = dTNet.gate_nets[dTNet.exit_ids[1]].blocks[-1].out_features

    for k, exit_idx in enumerate(dTNet.exit_ids[1:]):
        if branch_p[k+1] == 0:
            ver_not_trunk = False
            ver_not_branch = True
            continue

        # try:
        ver_not_branch, non_selection_threshold, _ = dTNet.gate_cnets[exit_idx].get_abs_loss(inputs,
                                                                       torch.zeros_like(target).int(), eps, domain,
                                                                       dTNet.threshold[exit_idx],beta=1)
        ver_not_trunk, selection_threshold, _ = dTNet.gate_cnets[exit_idx].get_abs_loss(inputs,
                                                                       torch.ones_like(target).int(), eps, domain,
                                                                       dTNet.threshold[exit_idx],beta=1)
        # gate_threshold_s[exit_idx] = torch.stack((-non_selection_threshold,selection_threshold),dim=0).gather(dim=0, index=targets_abs.view(1, -1, 1)).squeeze(0)

        try:
            pass
        except:
            print("Verification of gate failed")
            # ver = torch.zeros_like(target).byte()
            ver_not_branch = torch.zeros_like(target).byte()
            # gate_threshold_s[exit_idx] = -np.inf
        # ver_not_trunk = ver - ver_not_branch

        if ver_not_branch:
            # Sample can not reach branch
            branch_p[k + 1] = 0
        else:
            # Sample can reach branch
            if branch_p[k+1] == 2:
                # Already certified
                ver_corr_branch = torch.ones_like(target).byte()
                # ver = torch.ones_like(target).byte()
            elif branch_p[k+1] == -1:
                # Already certified as incorrect
                ver_corr_branch = torch.zeros_like(target).byte()
                # ver = torch.ones_like(target).byte()
            else:
                branch_p[k + 1] = 1

                ver_corr_branch, _, _ = dTNet.branch_cnets[exit_idx].get_abs_loss(inputs, target, eps, domain,
                                                                                    dTNet.branch_cnets[exit_idx].threshold_min,beta=1)
                ver_corr_branch = ver_corr_branch.byte()
                # ver_branch = ver_branch.byte()


                if ver_corr_branch:
                    branch_p[k + 1] = 2
                # elif ver_branch:
                #     branch_p[k + 1] = -1
            # ver corr if all reachable branches are correct
            ver_corr = ver_corr_branch & ver_corr

        if ver_not_trunk:
            # Will definitely branch => all later branches cannot be reached
            branch_p[0] = 0
            if k+2 < len(branch_p):
                for i, reachability in enumerate(branch_p[k+2:]):
                    branch_p[k + 2 + i] = 0
            break

        if not ver_corr and break_on_failure:
            # assume that all branches can be reached if the opposite was not certified
            if not ver_not_trunk:
                for i, reachability in enumerate(branch_p[k+1:]):
                    if reachability == 0:
                        branch_p[k + 1 + i] = 1
            break

    if not ver_not_trunk and not branch_p[0] == 0:
        if ver_corr or not break_on_failure:
            if cert_trunk:
                try:
                    ver_corr_trunk, threshold_n, _ = dTNet.trunk_cnet.get_abs_loss(inputs, target, eps, domain,
                                                                                    dTNet.trunk_cnet.threshold_min,kappa=1)
                    if ver_corr_trunk:
                        branch_p[0] = 2
                    # elif ver_trunk:
                    #     branch_p[0] = -1
                    ver_corr = ver_corr_trunk & ver_corr
                except:
                    warn("Certification of trunk failed critically.")
                    ver_corr[:] = False
            else:
                ver_corr[:] = False
    return branch_p, ver_corr, gate_threshold_s
