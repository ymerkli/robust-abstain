import torch
import torch.nn as nn
from autoattack import AutoAttack as AutoAttackBase

from robustabstain.attacker.base import Attacker


class AutoAttack(Attacker):
    """ Wrapper class around AutoAttack

    Args:
        adv_norm (str): The norm of the perturbation region (Linf, L2)
        version (str, optional): Version of AutoAttack to use (standard, reduced)
        verbose (bool, optional): Verbose. Defaults to True.
        device (torch.device, optional): Device on which to perform the attack. Defaults to 'cpu'.
    """

    def __init__(
            self,
            adv_norm: str,
            version: str = 'standard',
            verbose: bool = False,
            device: torch.device = torch.device('cpu')
        ) -> None:

        self.adv_norm = adv_norm
        self.version = version
        self.verbose = verbose
        self.device = device

    def attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, eps: float
        ) -> torch.Tensor:
        auto_attack = AutoAttackBase(
            model, norm=self.adv_norm, eps=eps, version='standard', device=self.device, verbose=self.verbose
        )
        if self.version == 'reduced':
            auto_attack.attacks_to_run = ['apgd-ce']

        return auto_attack.run_standard_evaluation(inputs, labels, bs=inputs.size(0))

