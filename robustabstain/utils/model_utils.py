import torch
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn

import os

import robustbench.model_zoo.architectures as rb_archs

import robustabstain.archs as archs
from robustabstain.utils.transforms import NormalizeLayer


def requires_grad_(model: nn.Module, requires_grad: bool) -> None:
    """Set requires_grad for each parameter in a model.

    Args:
        model (nn.Module): Model to set requires_grad in.
        requires_grad (bool): Requires grad?
    """
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def finetune(
        model: nn.Module, device: str, feature_extract: bool = True,
        reset_fc: bool = False, num_classes: int = None
    ) -> None:
    """Prepare finetuning for a model. If feature_extract is set, all layers
    except for the last one will have requires_grad set to False.

    Args:
        model (nn.Module): PyTorch module
        device (str): device
        feature_extract (bool, optional): If set, only last layer has requires_grad=True. Defaults to True.
        reset_fc (bool, optional): If set, weights/bias of final FC layers are reset. Defaults to False.
        num_classes (int, optional): Reinit final FC layer to have num_classes out_features. Defaults to None.
    """
    if feature_extract:
        requires_grad_(model, False)

    base_model = model
    if type(base_model) == nn.parallel.DataParallel:
        # extract base model from DataParallel model
        base_model = model.module
    if type(base_model) == nn.Sequential and type(base_model[0]) == NormalizeLayer:
        # model prepended with NormalizeLayer
        base_model = base_model[1]
    model_arch = type(base_model)

    # depending on the architecture, a different named module/layer is finetuned
    if any(issubclass(model_arch, arch) for arch in [
        archs.ResNet, archs.WideResNet, rb_archs.wide_resnet.WideResNet
    ]):
        # for these models, module to finetune is .fc
        if reset_fc:
            if not num_classes:
                num_classes = base_model.fc.out_features
            base_model.fc = nn.Linear(
                in_features=base_model.fc.in_features, out_features=num_classes, bias=True
            )
        requires_grad_(base_model.fc, True)
    elif any(issubclass(model_arch, arch) for arch in [
        archs.DLA, rb_archs.resnet.ResNet, rb_archs.resnet.PreActResNet
    ]):
        # for these models, module to finetune is .linear
        if reset_fc:
            if not num_classes:
                num_classes = base_model.linear.out_features
            base_model.linear = nn.Linear(
                in_features=base_model.linear.in_features, out_features=num_classes, bias=True
            )
        requires_grad_(base_model.linear, True)
    elif any(issubclass(model_arch, arch) for arch in [
        archs.VGG, rb_archs.resnext.CifarResNeXt
    ]):
        # for these models, module to finetune is .classifier
        if reset_fc:
            if not num_classes:
                num_classes = base_model.classifier.out_features
            base_model.classifier = nn.Linear(
                in_features=base_model.classifier.in_features, out_features=num_classes, bias=True
            )
        requires_grad_(base_model.classifier, True)
    elif any(issubclass(model_arch, arch) for arch in [
        rb_archs.dm_wide_resnet.DMWideResNet
    ]):
        # for these models, module to finetune is .logits
        if reset_fc:
            if not num_classes:
                num_classes = base_model.logits.out_features
            base_model.logits = nn.Linear(
                in_features=base_model.logits.in_features, out_features=num_classes, bias=True
            )
        requires_grad_(base_model.logits, True)
    else:
        raise ValueError(f'Error: finetune not yet supported for arch {type(model)}')

    model.to(device)


def get_model_name(model_path: str) -> str:
    """Get the model name from its path.

    Args:
        model_path (str): Path to exported model.

    Returns:
        str: Name of the model.
    """
    return os.path.splitext(os.path.basename(model_path))[0]