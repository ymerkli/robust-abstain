import torch
import torchvision.transforms as transforms

from typing import Tuple, Dict, List

from robustabstain.data.synth.datasets import SYNTH_DATASETS
from robustabstain.utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from robustabstain.utils.data_utils import get_dataset_stats, DATASET_MEAN_STD


# available data augmentation strategies
DATA_AUG = ['stdaug', 'autoaugment']


class NoneTransform(object):
    """None transform, just returns the image.
    """
    def __call__(self, sample):
        return sample


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.

    Args:
        mean (torch.tensor): the channel means
        std (torch.tensor): the channel standard deviations
        device (str): device
    """

    def __init__(self, mean: torch.tensor, std: torch.tensor, device: str) -> None:
        super(NormalizeLayer, self).__init__()
        self.mean = mean.to(device)
        self.std = std.to(device)

    def forward(self, input: torch.tensor) -> torch.tensor:
        (batch_size, num_channels, height, width) = input.shape
        mean = self.mean.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        std = self.std.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - mean) / std


def get_normalize_layer(device: str, dataset: str, set01: bool = False) -> NormalizeLayer:
    """Return the datasets normalization layer.

    Args:
        device (str): 'cpu' or 'cuda'.
        dataset (str): Name of the dataset.
        set01 (bool, optional): If set, mean=0, std=1 is returned. Defaults to False.

    Returns:
        NormalizeLayer: Normalization layer for the dataset.
    """
    mean, std = get_mean_sigma(device, dataset, set01)

    return NormalizeLayer(mean, std, device)


def get_mean_sigma(device: str, dataset: str, set01: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Return mean and std torch.Tensor of given dataset

    Args:
        device (str): 'cpu' or 'cuda'
        dataset (str): Name of the dataset
        set01 (bool, optional): If set, mean=0, std=1 is returned. Defaults to False.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: dataset mean, std deviation.
    """
    try:
        mean = torch.FloatTensor(DATASET_MEAN_STD[dataset]['mean'])
        std = torch.FloatTensor(DATASET_MEAN_STD[dataset]['std'])
    except KeyError:
        raise ValueError(f"Error: dataset {dataset} is unknown.")

    if set01:
        mean.fill_(0)
        std.fill_(1)

    return mean.to(device), std.to(device)


def get_sizing_transform(dataset: str) -> transforms.Resize:
    """Image resizing transform for given dataset

    Args:
        dataset (str): Name of the dataset

    Returns:
        transforms.Resize: sizing transform.
    """
    base_transform = None
    dim, _, _ = get_dataset_stats(dataset)
    if dataset not in SYNTH_DATASETS:
        # these datasets need to be resized
        base_transform = transforms.Resize((dim, dim))

    return base_transform


def get_normalize_transform(dataset: str) -> transforms.Normalize:
    """Normalization transform for given dataset

    Args:
        dataset (str): Name of the dataset

    Returns:
        transforms.Normalize: Normalize transform.
    """
    normalize_tranform = None
    if dataset not in SYNTH_DATASETS:
        normalize_tranform = transforms.Normalize(*get_mean_sigma('cpu', dataset))

    return normalize_tranform


def get_base_augment(dataset: str, normalize: bool = False) -> transforms.Compose:
    """Basic augmentation transforms for training

    Args:
        dataset (str): Name of the dataset
        normalize (bool, optional): If set, normalization transform is added. Defaults to False.

    Returns:
        transforms.Compose: Composed base augmemtation transforms.
    """
    base_augment = None
    dim, _, _ = get_dataset_stats(dataset)
    sizing_transform = get_sizing_transform(dataset)
    normalize_transform = get_normalize_transform(dataset) if normalize else None

    base_augment = []
    base_augment.extend([sizing_transform] if sizing_transform else [])
    if dataset in ['cifar10', 'cifar10_h3']:
        base_augment.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif dataset == 'cifar100':
        base_augment.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif dataset == 'gstrb':
        base_augment.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    elif 'mtsd' in dataset:
        base_augment.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    elif 'sbb' in dataset:
        base_augment.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    elif dataset in SYNTH_DATASETS:
        pass
    else:
        raise ValueError(f"Error: dataset {dataset} is unknown.")

    base_augment.extend([normalize_transform] if normalize_transform else [])

    return transforms.Compose(base_augment)


def get_stdaugment(dataset: str, normalize: bool = False) -> transforms.Compose:
    """Get augmentation transforms for augmenting a dataset.

    Args:
        dataset (str): Name of the dataset.
        normalize (bool, optional): If set, normalization transform is added. Defaults to False.

    Returns:
        transforms.Compose: Composed augmentation transforms.
    """
    dim, _, _ = get_dataset_stats(dataset)
    sizing_transform = get_sizing_transform(dataset)
    normalize_transform = get_normalize_transform(dataset) if normalize else None

    augm_transforms = []
    augm_transforms.extend([sizing_transform] if sizing_transform else [])
    if dataset in ['cifar10', 'cifar10_h3']:
        augm_transforms.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
    elif dataset == 'cifar100':
        augm_transforms.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
    elif 'mtsd' in dataset:
        augm_transforms.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
    elif 'sbb' in dataset: 
        augm_transforms.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Error: dataset {dataset} is unknown.")

    augm_transforms.extend([normalize_transform] if normalize_transform else [])

    return transforms.Compose(augm_transforms)


def get_autoaugment(dataset: str, normalize: bool = False) -> transforms.Compose:
    """Get autoaugment transforms for augmenting a dataset.

    Args:
        dataset (str): Name of the dataset.
        normalize (bool, optional): If set, normalization transform is added. Defaults to False.

    Returns:
        transforms.Compose: Composed autoaugment transforms.
    """
    dim, _, _ = get_dataset_stats(dataset)
    sizing_transform = get_sizing_transform(dataset)
    normalize_transform = get_normalize_transform(dataset) if normalize else None

    augm_transforms = []
    augm_transforms.extend([sizing_transform] if sizing_transform else [])
    if dataset in ['cifar10', 'cifar10_h3']:
        augm_transforms.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor()
        ])
    elif dataset == 'cifar100':
        # cifar10 autoaugment policy still gives non-trivial improvements on cifar100
        augm_transforms.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor()
        ])
    elif 'mtsd' in dataset:
        augm_transforms.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor()
        ])
    elif 'sbb' in dataset: 
        augm_transforms.extend([
            transforms.RandomCrop(dim, padding=4),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f'Error: unknown dataset {dataset}')

    augm_transforms.extend([normalize_transform] if normalize_transform else [])

    return transforms.Compose(augm_transforms)


def get_transforms(
        dataset: str, augment: str = None, normalize: bool = False
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns composed train and test transforms for given dataset.

    Args:
        dataset (str): Name of the dataset.
        augment (str, optional): If set, specified data augmentation transforms are used. Defaults to None.
        normalize (bool, optional): If set, data is normalized. Defaults to False.
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: train transforms, test transforms
    """
    test_transforms = []
    sizing_transform = get_sizing_transform(dataset)
    normalize_transform = get_normalize_transform(dataset) if normalize else None

    test_transforms.extend([sizing_transform] if sizing_transform else [])
    test_transforms.extend([transforms.ToTensor()])
    test_transforms.extend([normalize_transform] if normalize_transform else [])
    test_transforms = transforms.Compose(test_transforms)

    train_transforms = get_base_augment(dataset, normalize)
    if augment == 'stdaug':
        train_transforms = get_stdaugment(dataset, normalize)
    elif augment == 'autoaugment':
        train_transforms = get_autoaugment(dataset, normalize)

    return train_transforms, test_transforms

