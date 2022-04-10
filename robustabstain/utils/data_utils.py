import os
import re
import warnings
import pandas as pd
from pathlib import Path
from typing import Tuple, List

from robustabstain.data.synth.datasets import SYNTH_DATASETS
from robustabstain.utils.paths import  get_root_package_dir, default_data_dir


# supported datasets
DATASETS = [
    'cifar10',
    'cifar10_h3',
    'cifar100',
    'mtsd',
    'mtsd_l',
    'sbb',
    'sbb_l',
    'sbbpred',
    'sbbpred_l',
    'sbbpredh',
    'sbbpredh_l',
    'sbbc',
    'sbbc_l',
    'sbbca',
    'sbbca_l',
    'sbbcsp',
    'sbbcsp_l',
    'sbbcsi',
    'sbbcsi_l',
    'sbbbcs',
    'sbbbcs_l'
] + SYNTH_DATASETS


# dataset splits
DATA_SPLITS = ['train', 'test', 'business_eval']


# dataset means/stds for normalization
DATASET_MEAN_STD = {
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010]
    },
    'cifar10_h3': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010]
    },
    'cifar100': {
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    },
    'gtsrb': {
        'mean': [0.3337, 0.3064, 0.3171],
        'std': [0.2672, 0.2564, 0.2629]
    },
    'mtsd': {
        'mean': [0.4082, 0.3888, 0.3554],
        'std': [0.2477, 0.2329, 0.2383]
    },
    'mtsd_l': {
        'mean': [0.3684, 0.3784, 0.3492],
        'std': [0.2411, 0.2381, 0.2485]
    },
    'sbb': {
        'mean': [0.5290, 0.5290, 0.5290],
        'std': [0.2555, 0.2555, 0.2555]
    },
    'sbb_l': {
        'mean': [0.5038, 0.5038, 0.5038],
        'std': [0.2611, 0.2611, 0.2611]
    },
    'sbbpred': {
        'mean': [0.5467, 0.5467, 0.5467],
        'std': [0.2460, 0.2460, 0.2460]
    },
    'sbbpred_l': {
        'mean': [0.5194, 0.5194, 0.5194],
        'std': [0.2536, 0.2536, 0.2536]
    },
    'sbbpredh': {
        'mean': [0.5449, 0.5449, 0.5449],
        'std': [0.2460, 0.2460, 0.2460]
    },
    'sbbpredh_l': {
        'mean': [0.5185, 0.5185, 0.5185],
        'std': [0.2533, 0.2533, 0.2533]
    },
    'sbbc': {
        'mean': [0.5737, 0.5737, 0.5737],
        'std': [0.2165, 0.2165, 0.2165]
    },
    'sbbc_l': {
        'mean': [0.5573, 0.5572, 0.5573],
        'std': [0.2218, 0.2218, 0.2219]
    },
    'sbbca': {
        'mean': [0.5737, 0.5737, 0.5737],
        'std': [0.2165, 0.2165, 0.2165]
    },
    'sbbca_l': {
        'mean': [0.5573, 0.5572, 0.5573],
        'std': [0.2218, 0.2218, 0.2219]
    },
    'sbbcsp': {
        'mean': [0.5737, 0.5737, 0.5737],
        'std': [0.2165, 0.2165, 0.2165]
    },
    'sbbcsp_l': {
        'mean': [0.5573, 0.5572, 0.5573],
        'std': [0.2218, 0.2218, 0.2219]
    },
    'sbbcsi': {
        'mean': [0.5737, 0.5737, 0.5737],
        'std': [0.2165, 0.2165, 0.2165]
    },
    'sbbcsi_l': {
        'mean': [0.5573, 0.5572, 0.5573],
        'std': [0.2218, 0.2218, 0.2219]
    },
    'sbbbcs': {
        'mean': [0.5723, 0.5721, 0.5723],
        'std': [0.2188, 0.2188, 0.2188]
    },
    'sbbbcs_l': {
        'mean': [0.5551, 0.5550, 0.5551],
        'std': [0.2244, 0.2244, 0.2245]
    },

}

DATASET_RESIZING = {
    'cifar10': 32,
    'cifar10_h3': 32,
    'cifar100': 32,
    'mtsd': 32,
    'mtsd_l': 64,
    'sbb': 64,
    'sbb_l': 64,
    'sbbpred': 64,
    'sbbpred_l': 64,
    'sbbpredh': 64,
    'sbbpredh_l': 64,
    'sbbc': 64,
    'sbbc_l': 64,
    'sbbca': 64,
    'sbbca_l': 64,
    'sbbcsp': 64,
    'sbbcsp_l': 64,
    'sbbcsi': 64,
    'sbbcsi_l': 64,
    'sbbbcs': 64,
    'sbbbcs_l': 64
}
for synth_dataset in SYNTH_DATASETS:
    DATASET_RESIZING[synth_dataset] = 2


def get_dataset_stats(dataset: str) -> Tuple[int, int, int]:
    """Stats about dataset.

    Arguments:
        dataset (str): Name of the dataset.

    Returns:
        Tuple[int, int, int]: Resized image dimension, number of channels, number of classes.
    """
    dim, num_channels, num_classes = None, None, None
    if dataset == 'cifar10':
        num_channels, num_classes = 3, 10
    elif dataset == 'cifar10_h3':
        num_channels, num_classes = 3, 3
    elif dataset == 'cifar100':
        num_channels, num_classes = 3, 100
    elif dataset == 'gtsrb':
        num_channels, num_classes = 3, 43
    elif dataset in ['mtsd', 'mtsd_l']:
        num_channels, num_classes = 3, 400
    elif 'sbb' in dataset:
        if 'predh' in dataset:
            num_channels, num_classes = 3, 6
        else:
            num_channels, num_classes = 3, 5
    elif dataset in SYNTH_DATASETS:
        num_channels, num_classes = 1, 3
    else:
        raise ValueError(f'Error: unknown dataset {dataset}.')

    dim = DATASET_RESIZING[dataset]

    return dim, num_channels, num_classes


def dataset_label_names(dataset: str) -> List[str]:
    """Get a mapping from label id to label name for the dataset

    Args:
        dataset (str): Name of the dataset to get the mapping for.

    Returns:
        List[str]: List mapping label id to label name.
    """
    data_dir = default_data_dir(dataset)[0]
    if dataset in ['cifar10', 'cifar100']:
        data_dir = os.path.join(data_dir, dataset)
    else:
        data_dir = Path(data_dir).parent.absolute() 

    labels_file = os.path.join(data_dir, 'labels.csv')
    if not os.path.isfile(labels_file):
        warnings.warn(f'Error: no labels file {labels_file} found.')
        return []

    labels_df = pd.read_csv(labels_file)
    label_names = [''] * len(labels_df)
    for _, row in labels_df.iterrows():
        label_names[int(row['id'])] = row['name']

    return label_names

