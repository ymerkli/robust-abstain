import torch
import torchvision
from torch.utils.data import Sampler, Dataset
from torchvision.datasets.folder import (
    ImageFolder, default_loader,
    has_file_allowed_extension, IMG_EXTENSIONS)
from torchvision.datasets.vision import VisionDataset

import pandas as pd
import numpy as np
import os
import re
import pickle
import logging
import warnings
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Optional, Tuple, Dict, Optional, Union, List, cast

from robustabstain.data.synth.datasets import SYNTH_DATASETS, generate_synth
from robustabstain.utils.paths import get_root_package_dir
from robustabstain.utils.regex import GEN_PATCH_FNAME_RE


ArrayLike = Union[List[Any], np.ndarray, pd.Series]


class ObjectDetectionDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        label_names: List[str],
        split_name: Optional[str] = None,
    ):
        if "file_name" not in df.columns:
            raise ValueError("df does not have a 'file_name' column")

        if label_names is None:
            raise ValueError(f"label_names is expected to be a list of strings, got '{label_names}' instead")

        # The label names are provided. In this case, the labels will be the class
        # ids and we assume all labels map to a label name.
        self.label_to_class_id = {label: i for i, label in enumerate(label_names)}

        self.df = df
        self.dataset_name = dataset_name
        self.split_name = split_name

    def get_df(self) -> pd.DataFrame:
        return self.df

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"ObjectDetectionDataset(name='{self.dataset_name}', split='{self.split_name}')"

    def add_metadata(self, data: ArrayLike, col_name: str) -> None:
        if col_name in self.df.columns:
            raise ValueError(f"columns '{col_name}' already present in df")

        self.df = self.df.assign(**{col_name: data})


class SBBRailsDataset(ObjectDetectionDataset):
    LABEL_NAMES = ["joint", "surface_defect", "welding", "squat"]

    def __init__(self,
                 split: str,
                 dataset_dir: Union[str, Path] = "/data/sbb/original/cleaned_dataset_original_plus_spring20_rerescaled",
                 version: str = "v2"
                 ):

        versions = ["v1", "v2", "preprocessed"]
        if version not in versions:
            raise ValueError(
                f"Expected version to be one of '{versions}' but got '{version}'."
            )

        valid_splits = ["train", "val", "business_eval"]
        if split not in valid_splits:
            raise ValueError(
                f"Expected split to be one of '{valid_splits}' but got '{split}'."
            )
        if version == "v1" and split == "business_eval":
            raise ValueError("business_eval split is not available for v1 version of the dataset.")

        if isinstance(dataset_dir, str):
            dataset_dir = Path(dataset_dir)

        # Adjust the path for the dataset split
        if version == "preprocessed":
            dataset_dir = dataset_dir / "preprocessed"
        else:
            if split in ["train", "val"]:
                dataset_dir = dataset_dir / "cleaned_dataset_original_plus_spring20_rerescaled"
            if split == "business_eval":
                dataset_dir = dataset_dir / "quality_v2"

        if not dataset_dir.exists():
            raise ValueError(
                f"Dataset not found at directory '{str(dataset_dir)}'."
                f"This is a proprietary dataset and has to be downloaded manually."
            )

        data_path = Path(get_root_package_dir()) / 'data' / 'sbb' / 'splits' / version / split / "data.pkl"
        df: pd.DataFrame = pd.read_pickle(data_path)

        if version == "v1":
            # Resolve path to point to the absolute location on the disk
            df['file_name'] = df['file_name'].apply(lambda x: str((dataset_dir / split / x).resolve()))
        elif version == "v2" or version == "preprocessed":
            df['file_name'] = df['image'].apply(lambda x: str((dataset_dir / x).resolve()))

        super().__init__(df, "sbb_rails", SBBRailsDataset.LABEL_NAMES, split)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class DatasetFolderGen(VisionDataset):
    """A generic data loader for datasets where a single sample is
    replicated with augmentations. The __getitem__ function takes this into account
    and only randomly samples one of the generated variations per sample.
    This class is mostly copied and modified from torchvision.datasets.folder.DatasetFolder.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        orig_only (bool, optional): If set, the dataset consists of only the original
            samples, no generated ones. It is assumed that original samples have
            'original' in filename. Defaults to False.
        synth_only (bool, optional): If set, the dataset consists of only the generated 
            samples, no original ones. Defaults to False.
        sample_synth (float, optional): A random synthetic sample is sampled with probability sample_synth
            and the original version is sampled with probability 1-sample_synth. Defaults to 0.0.
        interp_synth (float, optional): Interpolate between a random synthetic sample and the original sample.
            Resulting sample is (1-interp_synth)*s_orig+interp_synth*s_synth. Defaults to 0.0.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            orig_only: Optional[bool] = False,
            synth_only: Optional[bool] = False,
            sample_synth: Optional[float] = 0.0,
            interp_synth: Optional[float] = 0.0
        ) -> None:
        super(DatasetFolderGen, self).__init__(root, transform=transform, target_transform=target_transform)
        self.orig_only = orig_only
        self.synth_only = synth_only
        self.sample_synth = sample_synth 
        self.interp_synth = interp_synth 
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[0][1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )

        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue

            class_instances = {}
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if is_valid_file(fname):
                        match = re.match(GEN_PATCH_FNAME_RE, fname)
                        if not match:
                            raise ValueError(f'Error: invalid filename {fname}')
                        source_img = match.group('source_img')
                        gen_id = match.group('gen_id')
                        patch_id = match.group('patch_id')
                        path = os.path.join(root, fname)
                        item = path, class_index
                        if source_img+patch_id not in class_instances:
                            class_instances[source_img+patch_id] = []

                        # the original image is first in list for easy access
                        if 'orig' in fname:
                            if gen_id is not None:
                                assert 'orig' in gen_id, "Error: gen_id is not equal to 'orig'"
                            class_instances[source_img+patch_id].insert(0, item)
                        else:
                            class_instances[source_img+patch_id].append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

                # extend class_instances dict into instances, sorted by fname
                for key in sorted(class_instances.keys()):
                    instances.append(class_instances[key])

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::
            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext
        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.
        Args:
            directory(str): Root directory path, corresponding to ``self.root``
        Raises:
            FileNotFoundError: If ``dir`` has no class folders.
        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        n_synth = len(self.samples[index][1:])
        # select the original sample 
        path_orig, target = self.samples[index][0]
        assert 'orig' in path_orig, f'Error: first sample version is not the original one {path_orig}'
        sample_orig = self.loader(path_orig)
        if self.transform is not None:
            sample_orig = self.transform(sample_orig)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if n_synth == 0:
            return sample_orig, target

        #select a random synthetic version of the original sample
        path_synth, target_synth = self.samples[index][np.random.randint(1, n_synth+1)]
        assert 'orig' not in path_synth, f'Error: sample version is not generated {path_synth}'
        sample_synth = self.loader(path_synth)
        if self.transform is not None:
            sample_synth = self.transform(sample_synth)
        if self.target_transform is not None:
            target_synth = self.target_transform(target_synth)
        assert target_synth == target, 'Error: target of original and synthetic sample dont match.'

        # apply sampling strategy
        if self.orig_only:
            sample = sample_orig
        elif self.synth_only:
            sample = sample_synth
        elif self.sample_synth > 0.0:
            if np.random.rand() <= self.sample_synth:
                sample = sample_synth
            else:
                sample = sample_orig
        elif self.interp_synth > 0.0:
            sample = (1 - self.interp_synth) * sample_orig + self.interp_synth * sample_synth
        else:
            if np.random.rand() <= 0.5:
                sample = sample_synth
            else:
                sample = sample_orig

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolderGenPatch(DatasetFolderGen):
    """
    A generic data loader where the images are arranged in this way by default:
        root/dog/xxx_0_0.png
        root/dog/xxy_1_0.png
        root/dog/[...]/xxz_original_1.png
        root/cat/123_1_1.png
        root/cat/nsdf3_0_9.png
        root/cat/[...]/asd932_0_2.png
    The filenames should follow this regex pattern:
        (?P<source_img>\S+)_(?P<gen_id>\d+|original)_(?P<patch_id>\d+)
    Where source_img is base image that was synthetically replicated via augmentations.
    The gen_id is the identifier of the augmentation and the patch_id is the identifier
    of the patch which was cropped from the source_img.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        orig_only (bool, optional): If set, the dataset consists of only the original
            samples, no generated ones. It is assumed that original samples have
            'original' in filename. Defaults to False.
        sample_synth (float, optional): A random synthetic sample is sampled with probability sample_synth
            and the original version is sampled with probability 1-sample_synth. Defaults to 0.0.
        interp_synth (float, optional): Interpolate between a random synthetic sample and the original sample.
            Resulting sample is (1-interp_synth)*s_orig+interp_synth*s_synth. Defaults to 0.0.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            orig_only: bool = False,
            synth_only: bool = False,
            sample_synth: Optional[float] = 0.0,
            interp_synth: Optional[float] = 0.0
        ):
        super(ImageFolderGenPatch, self).__init__(
            root, loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform, target_transform=target_transform,
            is_valid_file=is_valid_file, orig_only=orig_only,
            synth_only=synth_only, sample_synth=sample_synth,
            interp_synth=interp_synth
        )
        self.imgs = self.samples


class ImageFolderGenPatchConcat(VisionDataset):
    """
    Like ImageFolderGenPatch but for multiple data roots.

    Args:
        root (List[string]): List of root directory paths.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        loader (callable): A function to load a sample given its path.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        orig_only (bool, optional): If set, the dataset consists of only the original
            samples, no generated ones. It is assumed that original samples have
            'original' in filename. Defaults to False.
        synth_only (bool, optional): If set, the dataset consists of only the generated 
            samples, no original ones. Defaults to False.
        sample_synth (float, optional): A random synthetic sample is sampled with probability sample_synth
            and the original version is sampled with probability 1-sample_synth. Defaults to 0.0.
        interp_synth (float, optional): Interpolate between a random synthetic sample and the original sample.
            Resulting sample is (1-interp_synth)*s_orig+interp_synth*s_synth. Defaults to 0.0.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(
            self,
            root: List[str],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            orig_only: Optional[bool] = False,
            synth_only: Optional[bool] = False,
            sample_synth: Optional[float] = 0.0,
            interp_synth: Optional[float] = 0.0
        ):
        assert len(root) > 0
        super(ImageFolderGenPatchConcat, self).__init__(root=None, transform=transform, target_transform=target_transform)
        datasets = []
        for root_i in root:
            datasets.append(ImageFolderGenPatch(root_i, transform, target_transform, loader, is_valid_file))

        classes = datasets[0].classes
        class_to_idx = datasets[0].class_to_idx
        for dataset in datasets:
            assert dataset.classes == classes
            assert dataset.class_to_idx == class_to_idx

        self.loader = loader
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.orig_only = orig_only
        self.synth_only = synth_only
        self.sample_synth = sample_synth 
        self.interp_synth = interp_synth

        samples = []
        for class_id in range(len(classes)):
            for dataset in datasets:
                mask = list(np.array(dataset.targets) == class_id)
                samples += [sample for sample, select in zip(dataset.samples, mask) if select]

        self.samples = samples
        self.targets = [s[0][1] for s in samples]
        self.imgs = samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        n_synth = len(self.samples[index][1:])
        # select the original sample 
        path_orig, target = self.samples[index][0]
        assert 'orig' in os.path.basename(path_orig), f'Error: first sample version is not the original one {path_orig}'
        sample_orig = self.loader(path_orig)
        if self.transform is not None:
            sample_orig = self.transform(sample_orig)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if n_synth == 0:
            return sample_orig, target

        #select a random synthetic version of the original sample
        path_synth, target_synth = self.samples[index][np.random.randint(1, n_synth+1)]
        assert 'orig' not in os.path.basename(path_synth), f'Error: sample version is not synthetic {path_synth}'
        sample_synth = self.loader(path_synth)
        if self.transform is not None:
            sample_synth = self.transform(sample_synth)
        if self.target_transform is not None:
            target_synth = self.target_transform(target_synth)
        assert target_synth == target, 'Error: target of original and synthetic sample dont match.'

        # apply sampling strategy
        if self.orig_only:
            sample = sample_orig
        elif self.synth_only:
            sample = sample_synth
        elif self.sample_synth > 0.0:
            if np.random.rand() <= self.sample_synth:
                sample = sample_synth
            else:
                sample = sample_orig
        elif self.interp_synth > 0.0:
            sample = (1 - self.interp_synth) * sample_orig + self.interp_synth * sample_synth
        else:
            if np.random.rand() <= 0.5:
                sample = sample_synth
            else:
                sample = sample_orig

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class CIFAR10(torchvision.datasets.CIFAR10):
    """Subclassing the torchvision CIFAR10 dataset to properly name the base_folder.
    """
    base_folder = 'cifar10'


class CIFAR10_H3(CIFAR10):
    """CIFAR10 subclass conisting of the 3 difficult classes cat, deer, dog (3,4,5).
    Each class is further subsampled to create a small dataset to quickly iterate on.
    """

    base_folder = 'cifar10'

    def __init__(
            self,
            root: str,
            train: bool = True,
            subsample: float = 0.5,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False
        ) -> None:
        super(CIFAR10_H3, self).__init__(
            root, train=train, transform=transform,
            target_transform=target_transform, download=download
        )

        self.select_labels = [3,4,5]
        self.subsample = subsample

        self.targets = np.array(self.targets)
        indices = np.where((self.targets == 3) | (self.targets == 4) | (self.targets == 5))[0]
        indices = np.random.choice(indices, int(len(indices) * subsample), replace=False)
        self.targets = list(self.targets[indices])
        self.data = self.data[indices]


class CIFAR100(torchvision.datasets.CIFAR100):
    """Subclassing the torchvision CIFAR100 dataset to properly name the base_folder.
    """
    base_folder = 'cifar100'


class MTSD(ImageFolder):
    """Dataset class for the Mapilliary traffic sign dataset (MTSD).
    Traffic signs are exactly cropped to the ground truth bounding boxes.
    """


class MTSD_l(ImageFolder):
    """Dataset class for the loosly cropped Mapilliary traffic sign dataset (MTSD_l).
    Traffic signs are loosly cropped to the ground truth bounding boxes, giving extra
    background space around the traffic signs.
    """


class SBB(ImageFolder):
    """Dataset class for the SBB rails dataset (SBB).
    Rail defects are exactly cropped to the ground truth bounding boxes.
    """
    

class SBB_l(ImageFolder):
    """Dataset class for the loosly cropped SBB rails dataset (SBB_l).
    Rail defects are loosly cropped to the ground truth bounding boxes, giving extra
    background space around the defects.
    """


class SBBpred(ImageFolder):
    """Dataset class for the SBBpred rails dataset (SBBpred).
    Rail defects are exactly cropped to the predicted bounding boxes.
    """
    

class SBBpred_l(ImageFolder):
    """Dataset class for the loosly cropped SBB rails dataset (SBBpred_l).
    Rail defects are loosly cropped to the predicted bounding boxes, giving extra
    background space around the defects.
    """


class SBBpredh(ImageFolder):
    """Dataset class for the SBBpredh rails dataset (SBBpredh).
    Rail defects are exactly cropped to the predicted bounding boxes.
    Predicted bboxes without GT bbox match are labeled as 'hardnodefect'.
    """


class SBBpredh_l(ImageFolder):
    """Dataset class for the loosly cropped SBB rails dataset (SBBpredh_l).
    Rail defects are loosly cropped to the predicted bounding boxes, giving extra
    background space around the defects. Predicted bboxes without GT bbox match
    are labeled as 'hardnodefect'.
    """
    

class SBBc(ImageFolderGenPatch):
    """Dataset class for the extactly cropped cut SBB rails dataset (SBBc),
    i.e. the rail images have no background. Only the original samples are included.
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None
        ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file, orig_only=True
        )


class SBBc_l(ImageFolderGenPatch):
    """Dataset class for the loosly cropped cut SBB rails dataset (SBBc_l),
    i.e. the rail images have no background. Only the original samples are included. 
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None
        ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file, orig_only=True
        )


class SBBca(ImageFolder):
    """Dataset class for the exactly cropped cut SBB rails dataset (SBB).
    Dataset includes ALL samples, i.e. all original and all synthetic samples.
    """
    

class SBBca_l(ImageFolder):
    """Dataset class for the loosly cropped cut SBB rails dataset (SBB_l).
    Dataset includes ALL samples, i.e. all original and all synthetic samples.
    """


class SBBcsp(ImageFolderGenPatch):
    """Dataset class for the exactly cropped cut SBB rails dataset (SBBcsp),
    i.e. the rail images have no background. We sample between the original sample
    and a random synthetic sample.
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_synth: Optional[float] = 0.7,
        ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file, sample_synth=sample_synth 
        )


class SBBcsp_l(ImageFolderGenPatch):
    """Dataset class for the loosly cropped cut SBB rails dataset (SBBc_l),
    i.e. the rail images have no background. We sample between the original sample
    and a random synthetic sample.
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            sample_synth: Optional[float] = 0.7,
        ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file, sample_synth=sample_synth
        )


class SBBcsi(ImageFolderGenPatch):
    """Dataset class for the exactly cropped cut SBB rails dataset (SBBcsp),
    i.e. the rail images have no background. We interpolate between the original sample
    and a random synthetic sample.
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            interp_synth: Optional[float] = 0.5
        ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file, interp_synth=interp_synth 
        )


class SBBcsi_l(ImageFolderGenPatch):
    """Dataset class for the cropped/preprocessed SBB rails dataset (SBBc_l),
    i.e. the rail images have no background. Rail defects are loosly cropped
    to the ground truth bounding boxes, giving extra background space around the defects.
    We sample between the original sample and a random synthetic sample.
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            interp_synth: Optional[float] = 0.5
        ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file, interp_synth=interp_synth 
        )


class SBBbcs(ImageFolderGenPatchConcat):
    """Concatenated dataset class for the exact cropped base and cut SBB rails dataset (SBB_l).
    For cut samples, a random synthetic version is picked.
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None
        ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file, synth_only=True
        )


class SBBbcs_l(ImageFolderGenPatchConcat):
    """Concatenated dataset class for the loosly cropped generated and original SBB rails dataset (SBB_l).
    For cut samples, a random synthetic version is picked.
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None
        ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file, synth_only=True
        )


class Synth(Dataset):
    """Dataset class for synthetic datasets.

    Args:
        dataset (str): Name of the synthetic dataset to generate.
        n_samples (int, optional): Number of samples to generate.
    """
    def __init__(self, dataset: str, n_samples=1000) -> None:
        data, targets = generate_synth(dataset, n_samples=n_samples)
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.data[idx], self.targets[idx]


DATASET_CLASSES = {
    'cifar10': CIFAR10,
    'cifar10_h3': CIFAR10_H3,
    'cifar100': CIFAR100,
    'mtsd': MTSD,
    'mtsd_l': MTSD_l,
    'sbb': SBB,
    'sbb_l': SBB_l,
    'sbbpred': SBBpred,
    'sbbpred_l': SBBpred_l,
    'sbbpredh': SBBpredh,
    'sbbpredh_l': SBBpredh_l,
    'sbbc': SBBc,
    'sbbc_l': SBBc_l,
    'sbbca': SBBca,
    'sbbca_l': SBBca_l,
    'sbbcsp': SBBcsp,
    'sbbcsp_l': SBBcsp_l,
    'sbbcsi': SBBcsi,
    'sbbcsi_l': SBBcsi_l,
    'sbbbcs': SBBbcs,
    'sbbbcs_l': SBBbcs_l
}
for synth_dataset in SYNTH_DATASETS:
    DATASET_CLASSES[synth_dataset] = Synth
