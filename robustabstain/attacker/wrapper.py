import torch
import torch.nn as nn
from typing import Optional
import warnings

from robustabstain.attacker.autoattack import AutoAttack
from robustabstain.attacker.autopgd import APGDAttack
from robustabstain.attacker.pgd import PGD_FB_Linf, PGD_FB_L2
from robustabstain.attacker.pgdconf import PGDCONF_Linf, PGDCONF_L2
from robustabstain.attacker.smoothadv import SmoothAdv_PGD_L2, SmoothAdv_DDN


ATTACKS = ['pgd', 'apgd', 'autoattack', 'smoothpgd', 'smoothddn']
ADVERSARIAL_NORMS = ['Linf', 'L2']


class AttackerWrapper(object):
    """ Wrapper class around different adversarial attack implementations.

    Args:
        adv_type (str): Attack type
        adv_norm (str): The norm of the perturbation region (Linf, L2)
        eps (float): Perturbation region epsilon
        steps (int): Number of attack steps
        targeted (bool, optional): Whether the attacks is (un-)targeted. Defaults to False.
        restarts (int, optional): Number of attack restarts. Defaults to 1.
        random_start (bool, optional): Whether to start with a random delta. Defaults to True.
        rel_step_size (float, optional): Relative attack step size. Defaults to 0.033.
        abs_step_size (float, optional): Absolute attack step size. Defaults to None.
        version (str, optional): Version of AutoAttack to use (standard, reduced)
        device (torch.device, optional): Device on which to perform the attack
    """
    def __init__(
        self, adv_type: str, adv_norm: str, eps: float, steps: int, targeted: bool = False,
        restarts: int = 1, random_start: bool = True, rel_step_size: float = 0.01 / 0.3,
        abs_step_size: Optional[float] = None, version: str = 'standard',
        gamma_ddn: float = 0.05, init_norm_ddn: float = 1.0,
        device: torch.device = torch.device('cpu')
    ) -> None:

        self.adv_type = adv_type
        self.adv_norm = adv_norm
        self.eps = eps
        self.steps = steps
        self.targeted = targeted
        self.restarts = restarts
        self.random_start = random_start
        self.rel_step_size = rel_step_size
        self.abs_step_size = abs_step_size
        self.version = version
        self.gamma_ddn = gamma_ddn
        self.init_norm_ddn = init_norm_ddn
        self.device = device

        self.attacker = None
        if self.adv_type == 'pgd':
            if self.adv_norm == 'Linf':
                self.attacker = PGD_FB_Linf(self.steps, self.random_start, self.device)
            elif self.adv_norm == 'L2':
                self.attacker = PGD_FB_L2(self.steps, self.random_start, self.device)
            else:
                raise NotImplementedError(f'{self.adv_norm}-PGD not implemented')
        elif self.adv_type == 'pgdconf':
            if self.adv_norm == 'Linf':
                self.attacker = PGDCONF_Linf(self.steps, self.random_start, self.device)
            elif self.adv_norm == 'L2':
                self.attacker = PGDCONF_L2(self.steps, self.random_start, self.device)
            else:
                raise NotImplementedError(f'{self.adv_norm}-PGD not implemented')
        elif self.adv_type == 'apgd':
            self.attacker = APGDAttack(self.steps, self.adv_norm, verbose=False, device=self.device)
        elif self.adv_type == 'apgdconf':
            self.attacker = APGDAttack(self.steps, self.adv_norm, loss='conf', verbose=False, device=self.device)
        elif self.adv_type == 'autoattack':
            self.attacker = AutoAttack(self.adv_norm, self.version, verbose=False, device=self.device)
        elif self.adv_type == 'smoothpgd':
            if self.adv_norm == 'L2':
                self.attacker = SmoothAdv_PGD_L2(self.steps, self.random_start, self.eps, self.device)
            else:
                raise ValueError(f'Error: DDN is an L2 attack, but {self.adv_norm} was specified.')
        elif self.adv_type == 'smoothddn':
            if self.adv_norm == 'L2':
                self.attacker = SmoothAdv_DDN(
                    self.steps, gamma=self.gamma_ddn, max_norm=self.eps,
                    init_norm=self.init_norm_ddn, device=self.device
                )
            else:
                raise ValueError(f'Error: DDN is an L2 attack, but {self.adv_norm} was specified.')
        else:
            raise ValueError(f'Error: unknown attack {self.adv_type}')


    def attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            conf_threshold: float = 0.0, noise: torch.Tensor = None, num_noise_vectors: int = 1,
            no_grad: bool = False, best_loss: bool = False
        ) -> torch.Tensor:
        """Unified attack method.

        Args:
            model (nn.Module): Model to attack.
            inputs (torch.Tensor): Batch of samples to attack. Values should be in the [0, 1] range.
            labels (torch.Tensor): Labels of the samples to attack if untargeted, else labels of targets.
            conf_threshold (float, optional): Confidence threshold for PGDCONF attacks. Defaults to 0.0.
            noise (torch.Tensor, optional): Noise for SmoothAdv attacks. Defaults to None.
            num_noise_vectors (int, optional): Number of noise vectors for SmoothAdv attacks. Defaults to 1.
            no_grad (bool, optional): no_grad option for SmoothAdv attacks. Defaults to False.
            best_loss (bool, optional): If True, the points attaining highest loss are returned,
                otherwise adversarial examples. Only supported in APGD. Defaults to False.

        Returns:
            torch.Tensor: Batch of samples modified to be adversarial to the model.
        """
        model.eval()
        if type(self.attacker) in [PGD_FB_Linf, PGD_FB_L2]:
            if best_loss:
                warnings.warn('best_loss=True is only supported for APGD attack.')
            return self.attacker.attack(
                model, inputs, labels, self.eps, self.targeted, self.restarts,
                self.rel_step_size, self.abs_step_size
            )
        elif type(self.attacker) in [PGDCONF_Linf, PGDCONF_L2]:
            if best_loss:
                warnings.warn('best_loss=True is only supported for APGD attack.')
            return self.attacker.attack(
                model, inputs, labels, self.eps, conf_threshold, self.restarts,
                self.rel_step_size, self.abs_step_size
            )
        elif type(self.attacker) == AutoAttack:
            if best_loss:
                warnings.warn('best_loss=True is only supported for APGD attack.')
            return self.attacker.attack(model, inputs, labels, self.eps)
        elif type(self.attacker) == APGDAttack:
            return self.attacker.attack(
                model, inputs, labels, self.eps, targeted=False,
                restarts=self.restarts, best_loss=best_loss
            )
        elif type(self.attacker) in [SmoothAdv_PGD_L2, SmoothAdv_DDN]:
            if best_loss:
                warnings.warn('best_loss=True is only supported for APGD attack.')
            return self.attacker.attack(
                model, inputs, labels, noise, num_noise_vectors, self.targeted, no_grad
            )
        else:
            raise NotImplementedError
