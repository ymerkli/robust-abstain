import os
from typing import List

from robustabstain.data.synth.datasets import SYNTH_DATASETS

# name of the root repository directory
ROOT_REPOSITORY_NAME = 'robust-abstain'
# name of the top level package directory
ROOT_PACKAGE_NAME = 'robustabstain'
# relative directory for exported figures
FIGURES_DIR = 'analysis/figures'


def splitall(path: str) -> List[str]:
    """Split a POSIX path into all of its parts.

    Args:
        path (str): Path to split

    Returns:
        List[str]: List of all path components.
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])

    return allparts


def get_root_package_dir() -> str:
    """Get path to root directory of the robustabstain package.
    Assumes that the name of the repository remains unchanged to
    be 'robust-abstain'.

    Returns:
        str: Absolute path to the root directory of the robustabstain package.
    """
    cwdparts = splitall(os.getcwd())
    root_dir = ''
    for part in cwdparts:
        if part == ROOT_REPOSITORY_NAME:
            root_dir = os.path.join(root_dir, part, ROOT_PACKAGE_NAME)
            return root_dir
        root_dir = os.path.join(root_dir, part)

    raise RuntimeError('Error: Run the script from within the robust-abstain repository.')


def default_data_dir(dataset: str) -> List[str]:
    """Returns the default data directory.

    Args:
        dataset (str): the dataset

    Returns:
        str: Default data directory path of the dataset.
    """
    data_dirs = None
    if dataset == 'cifar10':
        data_dirs = ['./data']
    elif dataset == 'cifar10_h3':
        data_dirs = ['./data']
    elif dataset == 'cifar100':
        data_dirs = ['./data']
    elif dataset == 'gtsrb':
        data_dirs = ['./data/gtsrb/']
    elif dataset == 'mtsd':
        data_dirs = ['./data/mtsd/images_crop/']
    elif dataset == 'mtsd_l':
        data_dirs = ['./data/mtsd/images_crop_loose/']
    elif dataset == 'sbb':
        data_dirs = ['./data/sbb/images_crop/']
    elif dataset == 'sbb_l':
        data_dirs = ['./data/sbb/images_crop_loose/']
    elif dataset == 'sbbpred':
        data_dirs = ['./data/sbbpred/images_crop/']
    elif dataset == 'sbbpred_l':
        data_dirs = ['./data/sbbpred/images_crop_loose/']
    elif dataset == 'sbbpredh':
        data_dirs = ['./data/sbbpredh/images_crop/']
    elif dataset == 'sbbpredh_l':
        data_dirs = ['./data/sbbpredh/images_crop_loose/']
    elif dataset in ['sbbc', 'sbbca', 'sbbcsp', 'sbbcsi']:
        data_dirs = ['./data/sbbgen/images_crop/']
    elif dataset in ['sbbc_l', 'sbbca_l', 'sbbcsp_l', 'sbbcsi_l']:
        data_dirs = ['./data/sbbgen/images_crop_loose/']
    elif dataset == 'sbbbcs':
        data_dirs = ['./data/sbb/images_crop/', './data/sbbgen/images_crop/']
    elif dataset == 'sbbbcs_l':
        data_dirs = ['./data/sbb/images_crop_loose/', './data/sbbgen/images_crop_loose/']
    elif dataset in SYNTH_DATASETS:
        data_dirs = ['./data/synth/']
    else:
        raise ValueError(f'Error: unknown dataset {dataset}.')

    root_package_dir = get_root_package_dir()
    return [os.path.join(root_package_dir, d) for d in data_dirs]


def model_out_dir(train_mode: str, dataset: str) -> str:
    """Returns parent model output directory.

    Args:
        train_mode (str): Employed training mode.
        dataset (str): Name of used dataset.

    Returns:
        str: Model output directory
    """
    root_package_dir = get_root_package_dir()
    rel_out_dir = f'./models/{train_mode}/{dataset}'

    return os.path.join(root_package_dir, rel_out_dir)


def eval_report_filename(dataset: str, eval_set: str) -> str:
    """Get filename of the evaluation report.

    Args:
        dataset (str): Name of the evaluated dataset.
        eval_set (str): Data split being evaluated.

    Returns:
        str: Report file name.
    """
    return f'{dataset}_{eval_set}set_report.json'


def eval_attack_log_filename(
        eval_set: str, dataset: str, adv_norm: str, adv_attack: str
    ) -> str:
    """Get filename of the attack log file.

    Args:
        eval_set (str): Dataset split that is evaluated ('train', 'val', 'test').
        dataset (str): Name of the evaluated dataset.
        adv_norm (str): Adverarsial norm (Linf, L2, etc.).
        adv_attack (str): Name of the adversarial attack being used.

    Returns:
        str: Attack log filename.
    """
    return f'{eval_set}set_{dataset}_{adv_norm}_{adv_attack}.csv'


def eval_smoothing_log_filename(
        eval_set: str, smoothing_sigma: str, smoothing_N0: int, smoothing_N: int
    ) -> str:
    """Get filename of the smoothing log file.

    Args:
        eval_set (str): Dataset split that is evaluated ('train', 'val', 'test').
        dataset (str): Name of the evaluated dataset.
        adv_norm (str): Adverarsial norm (Linf, L2, etc.).
        adv_attack (str): Name of the adversarial attack being used.

    Returns:
        str: Smoothing log filename.
    """
    return f'{eval_set}set_smooth{smoothing_sigma}_N0{smoothing_N0}_N{smoothing_N}.csv'
