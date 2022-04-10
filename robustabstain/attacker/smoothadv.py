import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

from robustabstain.attacker.base import Attacker


class SmoothAdv_PGD_L2(Attacker):
    """ SmoothAdv PGD L2 attack
    (Source: https://github.com/Hadisalman/smoothing-adversarial, https://github.com/jeromerony/fast_adversarial)

    Args:
        steps (int): Number of steps for the optimization.
        max_norm (float or None, optional): If specified, the norms of the perturbations will not be greater than
            this value which might lower success rate.
        device (torch.device, optional): Device on which to perform the attack.
    """

    def __init__(self,
                 steps: int,
                 random_start: bool = True,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super(SmoothAdv_PGD_L2, self).__init__()
        self.steps = steps
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors: int = 1, targeted: bool = False,
               no_grad: bool = False) -> torch.Tensor:
        if num_noise_vectors == 1:
            return self._attack(model, inputs, labels, noise, targeted)
        else:
            if no_grad:
                with torch.no_grad():
                    return self._attack_mutlinoise_no_grad(model, inputs, labels, noise, num_noise_vectors, targeted)
            else:
                    return self._attack_mutlinoise(model, inputs, labels, noise, num_noise_vectors, targeted)


    def _attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Args:
            model (nn.Module): Model to attack.
            inputs (torch.Tensor): Batch of samples to attack. Values should be in the [0, 1] range.
            labels (torch.Tensor): Labels of the samples to attack if untargeted, else labels of targets.
            targeted (bool, optional): Whether to perform a targeted attack or not.

        Returns
            torch.Tensor: Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm/self.steps*2)

        for i in range(self.steps):
            adv = inputs + delta
            if noise is not None:
                adv = adv + noise
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return inputs + delta


    def _attack_mutlinoise(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors: int = 1, targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Args:
            model (nn.Module): Model to attack.
            inputs (torch.Tensor): Batch of samples to attack. Values should be in the [0, 1] range.
            labels (torch.Tensor): Labels of the samples to attack if untargeted, else labels of targets.
            targeted (bool, optional): Whether to perform a targeted attack or not.

        Returns:
            torch.Tensor: Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros((len(labels), *inputs.shape[1:]), requires_grad=True, device=self.device)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm/self.steps*2)

        for i in range(self.steps):

            adv = inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)
            if noise is not None:
                adv = adv + noise
            logits = model(adv)

            pred_labels = logits.argmax(1).reshape(-1, num_noise_vectors).mode(1)[0]
            # safe softamx
            softmax = F.softmax(logits, dim=1)
            # average the probabilities across noise
            average_softmax = softmax.reshape(-1, num_noise_vectors, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)
            logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
            ce_loss = F.nll_loss(logsoftmax, labels)

            loss = multiplier * ce_loss

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs[::num_noise_vectors])
            delta.data.clamp_(0, 1).sub_(inputs[::num_noise_vectors])

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)


    def _attack_mutlinoise_no_grad(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors: int = 1,targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Args:
            model (nn.Module): Model to attack.
            inputs (torch.Tensor): Batch of samples to attack. Values should be in the [0, 1] range.
            labels (torch.Tensor): Labels of the samples to attack if untargeted, else labels of targets.
            targeted (bool, optional): Whether to perform a targeted attack or not.

        Returns
            torch.Tensor: Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros((len(labels), *inputs.shape[1:]), requires_grad=True, device=self.device)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm/self.steps*2)

        for i in range(self.steps):

            adv = inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)
            if noise is not None:
                adv = adv + noise
            logits = model(adv)

            pred_labels = logits.argmax(1).reshape(-1, num_noise_vectors).mode(1)[0]
            # safe softamx

            softmax = F.softmax(logits, dim=1)
            grad = F.nll_loss(softmax,  labels.unsqueeze(1).repeat(1,1,num_noise_vectors).view(batch_size*num_noise_vectors),
                            reduction='none').repeat(*noise.shape[1:],1).permute(3,0,1,2)*noise

            grad = grad.reshape(-1, num_noise_vectors, *inputs.shape[1:]).mean(1)
            # average the probabilities across noise

            grad_norms = grad.view(batch_size, -1).norm(p=2, dim=1)
            grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                grad[grad_norms == 0] = torch.randn_like(grad[grad_norms == 0])

            # optimizer.step()
            delta = delta + grad*self.max_norm/self.steps*2

            delta.data.add_(inputs[::num_noise_vectors])
            delta.data.clamp_(0, 1).sub_(inputs[::num_noise_vectors])

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)


class SmoothAdv_DDN(Attacker):
    """ DDN attack: decoupling the direction and norm of the perturbation to achieve a small L2 norm in few steps.
    (Source: https://github.com/Hadisalman/smoothing-adversarial, https://github.com/jeromerony/fast_adversarial)

    Args:
        steps (int): Number of steps for the optimization.
        gamma (float, optional): Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma).
        init_norm (float, optional): Initial value for the norm.
        quantize (bool, optional): If True, the returned adversarials will have quantized values to the specified number of levels.
        levels (int, optional): Number of levels to use for quantization (e.g. 256 for 8 bit images).
        max_norm (float or None, optional): If specified, the norms of the perturbations will not be greater than
            this value which might lower success rate.
        device (torch.device, optional): Device on which to perform the attack.
        callback (object, optional): Visdom callback to display various metrics.
    """

    def __init__(self,
                 steps: int,
                 gamma: float = 0.05,
                 init_norm: float = 1.,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional[object] = None) -> None:
        super(SmoothAdv_DDN, self).__init__()
        self.steps = steps
        self.gamma = gamma
        self.init_norm = init_norm

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm

        self.device = device
        self.callback = callback

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors=1, targeted: bool = False, no_grad=False) -> torch.Tensor:
        if num_noise_vectors == 1:
            return self._attack(model, inputs, labels, noise, targeted)
            # return self._attack_mutlinoise(model, inputs, labels, noise, num_noise_vectors, targeted)
        else:
            if no_grad:
                raise NotImplementedError
            else:
                return self._attack_mutlinoise(model, inputs, labels, noise, num_noise_vectors, targeted)


    def _attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Args:
            model (nn.Module): Model to attack.
            inputs (torch.Tensor): Batch of samples to attack. Values should be in the [0, 1] range.
            labels (torch.Tensor): Labels of the samples to attack if untargeted, else labels of targets.
            targeted (bool, optional): Whether to perform a targeted attack or not.

        Returns
            torch.Tensor: Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)
        norm = torch.full((batch_size,), self.init_norm, device=self.device, dtype=torch.float)
        worst_norm = torch.max(inputs, 1 - inputs).view(batch_size, -1).norm(p=2, dim=1)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps, eta_min=0.01)

        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(inputs)
        adv_found = torch.zeros(inputs.size(0), dtype=torch.uint8, device=self.device)

        for i in range(self.steps):
            scheduler.step()

            l2 = delta.data.view(batch_size, -1).norm(p=2, dim=1)
            adv = inputs + delta
            if noise is not None:
                adv = adv + noise
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss

            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            adv_found[is_both] = 1
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            if self.callback:
                cosine = F.cosine_similarity(-delta.grad.view(batch_size, -1),
                                             delta.data.view(batch_size, -1), dim=1).mean().item()
                self.callback.scalar('ce', i, ce_loss.item() / batch_size)
                self.callback.scalars(
                    ['max_norm', 'l2', 'best_l2'], i,
                    [norm.mean().item(), l2.mean().item(),
                     best_l2[adv_found].mean().item() if adv_found.any() else norm.mean().item()]
                )
                self.callback.scalars(['cosine', 'lr', 'success'], i,
                                      [cosine, optimizer.param_groups[0]['lr'], adv_found.float().mean().item()])

            optimizer.step()

            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)
            norm = torch.min(norm, worst_norm)

            delta.data.mul_((norm / delta.data.view(batch_size, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(inputs)
            if self.quantize:
                delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
            delta.data.clamp_(0, 1).sub_(inputs)

        if self.max_norm is not None:
            best_delta.renorm_(p=2, dim=0, maxnorm=self.max_norm)
            if self.quantize:
                best_delta.mul_(self.levels - 1).round_().div_(self.levels - 1)
        return inputs + best_delta


    def _attack_mutlinoise(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               noise: torch.Tensor = None, num_noise_vectors: int = 1,targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Args:
            model (nn.Module): Model to attack.
            inputs (torch.Tensor): Batch of samples to attack. Values should be in the [0, 1] range.
            labels (torch.Tensor): Labels of the samples to attack if untargeted, else labels of targets.
            targeted (bool, optional): Whether to perform a targeted attack or not.

        Returns:
            torch.Tensor: Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        batch_size = labels.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros((len(labels), *inputs.shape[1:]), requires_grad=True, device=self.device)
        norm = torch.full((batch_size,), self.init_norm, device=self.device, dtype=torch.float)
        worst_norm = torch.max(inputs[::num_noise_vectors], 1 - inputs[::num_noise_vectors]).view(batch_size, -1).norm(p=2, dim=1)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps, eta_min=0.01)

        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(inputs[::num_noise_vectors])
        adv_found = torch.zeros(inputs[::num_noise_vectors].size(0), dtype=torch.uint8, device=self.device)

        for i in range(self.steps):
            scheduler.step()

            l2 = delta.data.view(batch_size, -1).norm(p=2, dim=1)
            adv = inputs + delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)
            if noise is not None:
                adv = adv + noise
            logits = model(adv)

            pred_labels = logits.argmax(1).reshape(-1, num_noise_vectors).mode(1)[0]
            # safe softamx
            softmax = F.softmax(logits, dim=1)
            # average the probabilities across noise
            average_softmax = softmax.reshape(-1, num_noise_vectors, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)

            logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
            ce_loss = F.nll_loss(logsoftmax, labels)

            loss = multiplier * ce_loss

            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            adv_found[is_both] = 1
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            if self.callback:
                cosine = F.cosine_similarity(-delta.grad.view(batch_size, -1),
                                             delta.data.view(batch_size, -1), dim=1).mean().item()
                self.callback.scalar('ce', i, ce_loss.item() / batch_size)
                self.callback.scalars(
                    ['max_norm', 'l2', 'best_l2'], i,
                    [norm.mean().item(), l2.mean().item(),
                     best_l2[adv_found].mean().item() if adv_found.any() else norm.mean().item()]
                )
                self.callback.scalars(['cosine', 'lr', 'success'], i,
                                      [cosine, optimizer.param_groups[0]['lr'], adv_found.float().mean().item()])

            optimizer.step()

            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)
            norm = torch.min(norm, worst_norm)

            delta.data.mul_((norm / delta.data.view(batch_size, -1).norm(2, 1)).view(-1, 1, 1, 1))

            delta.data.add_(inputs[::num_noise_vectors])
            if self.quantize:
                delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
            delta.data.clamp_(0, 1).sub_(inputs[::num_noise_vectors])

        if self.max_norm is not None:
            best_delta.renorm_(p=2, dim=0, maxnorm=self.max_norm)
            if self.quantize:
                best_delta.mul_(self.levels - 1).round_().div_(self.levels - 1)
        return inputs + best_delta.repeat(1,num_noise_vectors,1,1).view_as(inputs)

