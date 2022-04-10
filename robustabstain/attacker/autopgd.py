import torch
import torch.nn as nn

from robustabstain.attacker.autopgd_base import APGDAttack as APGDAttack_base
from robustabstain.attacker.autopgd_base import APGDAttack_targeted as APGDAttack_targeted_base
from robustabstain.attacker.base import Attacker


class APGDAttack(Attacker):
    """ Wrapper class around APGDAttack

    Args:
        steps (int): Number of steps for the optimization.
        adv_norm (str): The norm of the perturbation region (Linf, L2).
        loss (str): Type of loss to use.
        verbose (bool, optional): Verbose. Defaults to True.
        device (torch.device, optional): Device on which to perform the attack
    """

    def __init__(
            self,
            steps: int,
            adv_norm: str,
            loss: str = 'ce',
            verbose: bool = False,
            device: torch.device = torch.device('cpu')
        ) -> None:

        super(APGDAttack, self).__init__()
        self.steps = steps
        self.adv_norm = adv_norm
        self.loss = loss
        self.verbose = verbose
        self.device = device

    def attack(
            self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, eps: float,
            targeted: bool = False, restarts: int = 1, best_loss: bool = False
        ) -> torch.Tensor:
        """Find adversarial samples for a given model, inputs and labels using APGD.

        Args:
            model (nn.Module): Model to attack.
            inputs (torch.Tensor): Batch of samples to attack. Values should be in the [0, 1] range.
            labels (torch.Tensor): Labels of the samples to attack if untargeted, else labels of targets.
            eps (float): Perturbation region epsilon
            targeted (bool, optional): Whether the attacks is (un-)targeted. Defaults to False.
            restarts (int, optional): Number of attack restarts. Defaults to 1.
            best_loss (bool, optional): If True the points attaining highest loss are returned,
                otherwise adversarial examples. Defaults to False.

        Returns:
            torch.Tensor: Adversarial samples.
        """

        attack = None
        if targeted:
            attack = APGDAttack_targeted_base(
                model, n_iter=self.steps, norm=self.adv_norm, n_restarts=restarts,
                eps=eps, loss=self.loss, eot_iter=1, rho=0.75,
                verbose=self.verbose, device=self.device
            )
        else:
            attack = APGDAttack_base(
                model, n_iter=self.steps, norm=self.adv_norm, n_restarts=restarts,
                eps=eps, loss=self.loss, eot_iter=1, rho=0.75,
                verbose=self.verbose, device=self.device
            )

        if self.loss == 'conf':
            best_loss = True

        return attack.perturb(inputs, labels, best_loss=best_loss)

