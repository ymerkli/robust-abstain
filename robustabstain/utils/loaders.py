import torch
import torchvision

import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Union
from robustbench.model_zoo import model_dicts as rb_model_dict

from robustabstain.utils.transforms import get_transforms
from robustabstain.utils.paths import default_data_dir
from robustabstain.utils.datasets import DATASET_CLASSES, ImageFolderGenPatchConcat, Synth, ImageFolderGenPatch
from robustabstain.utils.data_utils import get_dataset_stats


SEED = 0
torch.manual_seed(SEED)


def get_label_weights(dataset: torch.utils.data.Dataset) -> np.ndarray:
    """Get balanced label weights for dataset. Useful for weighted losses.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample.

    Returns:
        np.ndarray: Per label weights.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        targets = np.array(dataset.dataset.targets)
    else:
        targets = np.array(dataset.targets)

    label_bincount = np.bincount(np.array(targets))
    label_weights = np.divide(
        np.ones_like(label_bincount), label_bincount,
        out=np.zeros_like(label_bincount, dtype='float'), where=(label_bincount!=0)
    )

    # renormalize to sum to 1
    label_weights = label_weights / label_weights.sum()

    return label_weights


def get_sampling_order(dataset: torch.utils.data.Dataset, balanced: bool = True) -> List[int]:
    """Returns a fixed sampling order, for subsampling datasets.
    If desired, the sampling order is balanced according to class counts.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample.
        balanced (bool, optional): If set, examples are sampled weight by class bincount. Defaults to True.

    Returns:
        List[int]: List with sample indices in order of balaned sampling.
    """
    torch.manual_seed(SEED)
    if isinstance(dataset, torch.utils.data.Subset):
        targets = np.array(dataset.dataset.targets)
    else:
        targets = np.array(dataset.targets)

    label_bincount = np.bincount(targets)
    num_classes = len(label_bincount)
    label_weights = np.divide(np.ones_like(label_bincount), num_classes)
    if balanced:
        label_weights = np.divide(
            np.ones_like(label_bincount), label_bincount,
            out=np.zeros_like(label_bincount, dtype='float'), where=(label_bincount!=0)
        )

    samples_weight = np.array([label_weights[target] for target in targets])
    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight, num_samples=len(dataset), replacement=False,
        generator=torch.Generator().manual_seed(SEED)
    )

    return list(sampler)


def indexed_dataset(cls: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    """In the standard PyTorch dataset class, the __getitem__() function return a tuple (sample, target).
    In some cases (e.g. when analyzing specific test samples), we'd like to have a persistent sample index.
    The function indexed_dataset takes a PyTorch dataset class (not an instance!) and replaces its __getitem__()
    function with a version that returns a tuple (sample, target, index).

    Args:
        cls (torch.utils.data.Dataset): A subclass of torch.utils.data.Dataset (not an instance!).

    Returns:
        torch.utils.data.Dataset: Same class as cls but with its __getitem__() function replaced.
    """

    def __getitem__(self, index):
        sample, target = cls.__getitem__(self, index)
        return sample, target, index

    return type(cls.__name__, (cls,), {'__getitem__': __getitem__})


def get_val_split(
        data_set: torch.utils.data.Dataset, val_set: torch.utils.data.Dataset,
        val_split: float, sampling_order: List[int], overlap: bool = False
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Splits a data set into data and validation set, ensuring no samples are overlapping.

    Args:
        data_set (torch.utils.data.Dataset): Dataset from which val split is taken.
        val_set (torch.dataset): Validation set (must be equal to data set).
        val_split (float): Fraction of data samples to use for val_set.
        sampling_order (List[int]): Sampling order.
        overlap (bool, optional): If set, validation samples are also present in data_set. Defaults to False.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Dataset with val_split removed, validation set.
    """
    assert len(data_set) == len(val_set), 'Error: initially, train_set and val_set should have equal length'
    n_val_samples = int(val_split * len(data_set))
    if n_val_samples == 0:
        return data_set, None

    if not overlap:
        data_set = torch.utils.data.Subset(data_set, sampling_order[n_val_samples:])
    val_set = torch.utils.data.Subset(val_set, sampling_order[:n_val_samples])

    return data_set, val_set


def get_dataloader(
        args: object, dataset: str = None, normalize: bool = False, root: str = '', root_prefix: str = '',
        indexed: bool = False, shuffle_train: bool = True, val_set_source: str = 'train', val_split: float = 0.0
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int, int]:
    """ Build dataloaders for the given dataset

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        dataset (str, optional): The name of the dataset to load. If not given, args.dataset is used. Defaults to None.
        normalize (bool, optional): If set, data is normalized. Defaults to True.
        root (str, optional): Path to dataset root folder. Defaults to ''.
        root_prefix (str, optional): Prefix path to default dataset root folder. Defaults to ''.
        indexed (bool, optional): If set, underlying datasets are indexed, meaning
            __getitem__() returns tuple (sample, target, index). Defaults to False.
        shuffle_train (bool, optional): If set, train_loader is shuffled. Defaults to True.
        val_set_source (str, optional): Whether to take the val set from the train or test set.
        val_split (float, optional): The fraction of the trainset to use as validation set.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int, int]:
            Train dataloader, validation dataloader, test dataloader, image size, image channels, number of classes.
    """
    if not dataset:
        dataset = args.dataset

    logging.info(f'==> Building dataset {dataset}')
    (
        train_set, val_set, test_set, train_sampling_order,
        test_sampling_order, dim, num_channels, num_classes
    ) = get_dataset(
            dataset, args.eval_set, args.data_aug, normalize,
            root, root_prefix, indexed, val_set_source, val_split,
            args.n_train_samples, args.n_test_samples
    )

    # subsample the train and test sets if required
    train_set = get_subsampled_dataset(dataset, train_set, args.n_train_samples, balanced=True)
    test_set = get_subsampled_dataset(dataset, test_set, args.n_test_samples, balanced=True)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.train_batch, shuffle=shuffle_train, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers
    ) if val_set else None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, val_loader, test_loader, dim, num_channels, num_classes


def get_dataset(
        dataset: str, eval_set: str, augment: str = '', normalize: bool = False, root: str = '', root_prefix: str = '',
        indexed: bool = False, val_set_source: str = 'train', val_split: float = 0.0, n_train_samples: int = None,
        n_test_samples: int = None
    ) -> Tuple[
        torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset,
        List[int], List[int], int, int, int
    ]:
    """Get train-, validation- and testset for a specified dataset.

    Args:
        dataset (str): The name of the dataset to load.
        eval_set (str): Datasplit to use as test_set.
        augment (str, optional): If set, specified data augmentation transforms are used. Defaults to None.
        normalize (bool, optional): If set, data is normalized. Defaults to True.
        root (str, optional): Path to dataset root folder. Defaults to ''.
        root_prefix (str, optional): Prefix path to default dataset root folder. Defaults to ''.
        indexed (bool, optional): If set, datasets are indexed, meaning __getitem__() returns tuple
            (sample, target, index). Defaults to False.
        val_set_source (str, optional): Whether to take the val set from the train or test set.
        val_split (float, optional): The fraction of the trainset to use as validation set.
        n_train_samples (int, optional): Number of training samples. Defaults to None.
        n_test_samples (int, optional): Number of test samples. Defaults to None.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset,
            List[int], List[int], int, int, int]:
                Train dataset, validation dataset, test dataset, train sampling order,
                test sampling order, image dimension, image channels, number of classes.
    """
    dim, num_channels, num_classes = get_dataset_stats(dataset)
    transform_train, transform_test = get_transforms(dataset, augment, normalize)

    if not root:
        root = default_data_dir(dataset)
    root_train, root_test = root, root

    DatasetClass = DATASET_CLASSES[dataset]
    if indexed:
        DatasetClass = indexed_dataset(DatasetClass)

    if issubclass(DatasetClass, Synth):
        # default number of samples to generate for synthetic datasets
        if not n_train_samples:
            n_train_samples = 1000
        if not n_test_samples:
            n_test_samples = 1000

    logging.info(f'==> Using {eval_set}set as evaluation set.')
    if issubclass(DatasetClass, torchvision.datasets.ImageFolder):
        root = root[0]
        train_dirname = 'train'
        test_dirname = 'val' if eval_set == 'test' else eval_set
        root_train = os.path.join(root, train_dirname)
        root_test =  os.path.join(root, test_dirname)
        train_set = DatasetClass(root=root_train, transform=transform_train)
        test_set = DatasetClass(root=root_test, transform=transform_test)
    elif issubclass(DatasetClass, ImageFolderGenPatch):
        root = root[0]
        train_dirname = 'train'
        test_dirname = 'val' if eval_set == 'test' else eval_set
        root_train = os.path.join(root, train_dirname)
        root_test =  os.path.join(root, test_dirname)
        train_set = DatasetClass(root=root_train, transform=transform_train)
        test_set = DatasetClass(root=root_test, transform=transform_test)
    elif issubclass(DatasetClass, ImageFolderGenPatchConcat):
        train_dirname = 'train'
        test_dirname = 'val' if eval_set == 'test' else eval_set
        root_train = [os.path.join(r, train_dirname) for r in root]
        root_test =  [os.path.join(r, test_dirname) for r in root]
        train_set = DatasetClass(root=root_train, transform=transform_train)
        test_set = DatasetClass(root=root_test, transform=transform_test)
    elif issubclass(DatasetClass, Synth):
        train_set = DatasetClass(dataset, n_samples=n_train_samples)
        test_set = DatasetClass(dataset, n_samples=n_test_samples)
    else:
        # cifar10, cifar10_H3, cifar100 datasets have additional arguments
        root = root[0]
        root_train, root_test = root, root
        train_set = DatasetClass(root=root_train, train=True, download=True, transform=transform_train)
        test_set = DatasetClass(root=root_test, train=False, download=True, transform=transform_test)

    train_sampling_order = get_sampling_order(train_set, balanced=True)
    test_sampling_order = get_sampling_order(test_set, balanced=True)

    if val_set_source == 'train':
        # split off the validation set from the train set
        if issubclass(DatasetClass, torchvision.datasets.ImageFolder) or issubclass(DatasetClass, ImageFolderGenPatch):
            val_set = DatasetClass(root=root_train, transform=transform_test)
        elif issubclass(DatasetClass, ImageFolderGenPatchConcat):
            val_set = DatasetClass(root=root_train, transform=transform_test)
        elif issubclass(DatasetClass, Synth):
            val_set = DatasetClass(dataset, n_samples=n_test_samples)
        else:
            val_set = DatasetClass(root=root_train, train=True, download=True, transform=transform_test)

        train_set, val_set = get_val_split(train_set, val_set, val_split, train_sampling_order)
    else:
        # split off the validation set from the test set
        if issubclass(DatasetClass, torchvision.datasets.ImageFolder) or issubclass(DatasetClass, ImageFolderGenPatch):
            val_set = DatasetClass(root=root_test, transform=transform_test)
        elif issubclass(DatasetClass, ImageFolderGenPatchConcat):
            val_set = DatasetClass(root=root_test, transform=transform_test)
        elif issubclass(DatasetClass, Synth):
            val_set = DatasetClass(dataset, n_samples=n_test_samples)
        else:
            val_set = DatasetClass(root=root_test, train=False, download=True, transform=transform_test)

        test_set, val_set = get_val_split(test_set, val_set, val_split, test_sampling_order, overlap=True)

    return train_set, val_set, test_set, train_sampling_order, test_sampling_order, dim, num_channels, num_classes


def get_subsampled_dataset(
        dataset_name: str, dataset: torch.utils.data.Dataset, n_samples: int, balanced: bool = True
    ) -> torch.utils.data.Dataset:
    """Get subsampled dataset.

    Args:
        dataset_name (str): Name of the dataset.
        dataset (torch.utils.data.Dataset): The dataset object.
        n_samples (int): Number of subsamples.
        balanced (bool, optional): If set, examples are sampled weight by class bincount. Defaults to True.

    Returns:
        torch.utils.data.Dataset: Subsampled dataset.
    """
    if not n_samples or n_samples == len(dataset):
        return dataset
    assert n_samples <= len(dataset), f'Error: dataset only has {len(dataset)} samples'

    sub_dataset = None
    sampling_order = get_sampling_order(dataset, balanced=balanced)
    if type(dataset) == torch.utils.data.Subset:
        sub_dataset = torch.utils.data.Subset(dataset.dataset, sampling_order[:n_samples])
    else:
        sub_dataset = torch.utils.data.Subset(dataset, sampling_order[:n_samples])

    return sub_dataset


def get_indicator_subsample(
        dataloader: torch.utils.data.DataLoader, inputs: torch.tensor, targets: torch.tensor,
        sample_indices: torch.tensor, sample_indicators: Union[torch.tensor, np.ndarray] = None
    ) -> Tuple[torch.tensor, torch.tensor]:
    """Returns subsampled data batch that only contains samples which have indicator = 1
    in sample_indicators.

    Args:
        dataloader (torch.utils.data.DataLoader): the dataloader to subsample.
        inputs (torch.tensor): inputs batch.
        targets (torch.tensor): targets batch.
        sample_indices (torch.tensor): sample indices of the batch, respective to the original dataset.
        sample_indicators (Union[torch.tensor, np.ndarray]): sample weights of all sample in dataloader.

    Returns:
        Tuple[torch.tensor, torch.tensor]: subsampled inputs, subsampled targets.
    """
    if sample_indicators is None:
        return inputs, targets

    rel_sample_indices = get_rel_sample_indices(dataloader, sample_indices)
    if type(sample_indicators) == torch.Tensor:
        idx_to_pick = torch.nonzero(sample_indicators[rel_sample_indices], as_tuple=True)[0]
    else:
        idx_to_pick = np.nonzero(sample_indicators[rel_sample_indices])[0]

    return inputs[idx_to_pick], targets[idx_to_pick]


def get_rel_sample_indices(dataloader: torch.utils.data.DataLoader, sample_indices: torch.tensor) -> torch.tensor:
    """In subsampled datasets, sample_indices are relative to the original dataset.
    To get the relative sample indices towards the subsampled dataset itself, we
    need to locate sample_indices in the Subset and get indices relative to the Subset.

    Args:
        dataloader (torch.utils.data.DataLoader): the dataloader.
        sample_indices (torch.Tensor): sample indices of the batch, respective to the original dataset.

    Returns:
        torch.tensor: Tensor with sample indices relative to the dataloader.
    """
    if type(dataloader.dataset) == torch.utils.data.dataset.Subset:
        rel_sample_indices = np.nonzero(sample_indices.numpy()[:, None] == dataloader.dataset.indices)[1]
        rel_sample_indices = torch.from_numpy(rel_sample_indices)
    else:
        rel_sample_indices = sample_indices

    return rel_sample_indices


def get_targets(dataloader: torch.utils.data.DataLoader) -> np.ndarray:
    """Get targets of all samples present in the dataloader.
    Underlying dataset may be a Subset.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader to extract targets form.

    Returns:
        np.ndarray: List of targets.
    """
    if isinstance(dataloader.dataset, torch.utils.data.dataset.Subset):
        targets = np.array(
            dataloader.dataset.dataset.targets
        )[dataloader.dataset.indices]
    else:
        targets = np.array(dataloader.dataset.targets)

    return targets


def get_robustbench_models() -> Dict[str, Dict[str, Dict]]:
    """Get dict of robust models from https://github.com/RobustBench/robustbench

    Returns:
        Dict[str, Dict[str, Dict]]: dict mapping to robustbench models.
    """
    robustbench_models = {}
    for dataset_enum in rb_model_dict.keys():
        for threat_model in rb_model_dict[dataset_enum].keys():
            for model_name, model_dict in rb_model_dict[dataset_enum][threat_model].items():
                if model_name not in robustbench_models:
                    robustbench_models[model_name] = {}

                robustbench_models[model_name][dataset_enum.value] = {
                    'dataset': dataset_enum.value,
                    'model_dict': model_dict
                }

    return robustbench_models


def get_minibatches(
        batch: Tuple[torch.tensor, torch.tensor], num_batches: int
    ) -> Tuple[torch.tensor, torch.tensor]:
    """Split a data batch into minibatches.

    Args:
        batch (Tuple[torch.tensor, torch.tensor]): Inputs and targets.
        num_batches (int): Number of minibatches to split into.

    Yields:
        Tuple[torch.tensor, torch.tensor]: Minibatch iterator
    """
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]

