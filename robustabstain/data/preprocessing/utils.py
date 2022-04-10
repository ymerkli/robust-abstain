import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import csv
import cv2
import numpy as np
import os
import random
import pathlib
from typing import Optional, Union, Any, List, Tuple

PathLike = Union[str, pathlib.Path]


def verify_npimg(img: np.ndarray) -> None:
    """Assert that the numpy array is a valid image.

    Args:
        img (np.ndarray): Image to verify.
    """
    assert img.ndim == 3
    assert img.shape[-1] == 3
    assert img.dtype == np.uint8
    assert 0 <= np.min(img) and np.max(img) <= 255


def load_image(image_path: PathLike) -> np.ndarray:
    """Load an image from path.

    Args:
        image_path (PathLike): Path to image.

    Returns:
        np.ndarray: Loaded image.
    """
    if isinstance(image_path, pathlib.Path):
        image_path = image_path.resolve()
    image: np.ndarray = cv2.imread(str(image_path))
    if image is None or image.size == 0:
        raise ValueError(f"Couldn't read image at path {image_path}")

    verify_npimg(image)

    return image


def crop_to_patch(
        img: np.ndarray, img_key: str, data_split: str,
        xmin: int, xmax: int, ymin: int, ymax: int, obj_key: str,
        obj_label_name: str, obj_label_id: int,
        patch_filepath: PathLike, gt_filepath: PathLike,
        loose: bool = False, loose_factor: float = 0.0, verbose: bool = True
    ) -> None:
    """Crops an image to a given bounding box and writes the resulting image
    to disk. Cropping is either precisely according to the given bounding box
    or loosly, adding random extra space around the ground truth bounding box.

    Args:
        img (np.ndarray): The source image to crop from.
        img_key (str): Key uniquely identifying the image.
        data_split (str): Data split to which the image belongs.
        xmin (int): Patch bounding box lower limit of x dim.
        xmax (int): Patch bounding box upper limit of x dim.
        ymin (int): Patch bounding box lower limit of y dim.
        ymax (int): Patch bounding box upper limit of y dim.
        obj_key (str): Key uniquely identifying the cropped object.
        patch_filepath (PathLike): File path to write cropped image to.
        gt_filepath (PathLike): Ground truth .csv file path.
        loose (bool, optional): If true, cropping is done loosly. Defaults to False.
        loose_factor (float, optional): Fraction of added pixels to loose
            cropbox relative to the ground truth bbox. Defaults to 0.0.
        verbose (bool): If set, verbose logging. Defaults to False.
    """
    img_y, img_x, _ = img.shape
    if loose:
        # loosly crop image to bbox
        dx, dy = xmax - xmin, ymax - ymin
        x_loose = loose_factor * dx
        y_loose = loose_factor * dy
        x_split = random.random()
        y_split = random.random()

        xmin -= int(x_split * x_loose)
        xmax += int((1-x_split) * x_loose)
        ymin -= int(y_split * y_loose)
        ymax += int((1-y_split) * y_loose)

    # ensure crop box does not leave image boundaries
    xmin, xmax = max(xmin, 0), min(xmax, img_x)
    ymin, ymax = max(ymin, 0), min(ymax, img_y)
    if xmin >= xmax or ymin >= ymax:
        print(f'[{data_split}] Invalid crop dimension [{xmin}, {ymin}, {xmax}, {ymax}] for object {obj_key}, skipping..')
        return

    crop_img = img[ymin:ymax, xmin:xmax, :]
    if verbose:
        print(f'[{data_split}] Cropping object {img_key}:{obj_key} of label {obj_label_name} ({obj_label_id}).')

    # store cropped image
    if not cv2.imwrite(patch_filepath, crop_img):
        print(f'[{data_split}] Could not write {patch_filepath}, skipping..')
        return

    # write to ground truth file
    with open(gt_filepath, 'a') as fid:
        writer = csv.writer(fid, delimiter=',', quotechar='"')
        writer.writerow([obj_key, xmax - xmin, ymax - ymin, obj_label_id, obj_label_name])


def setup(data_dir: str, data_split: str, label_names: List[str]) -> None:
    """Setup data directory for dataset split and create a subdirectory for
    each label.

    Args:
        data_dir (str): Path to data dir
        data_split (str): 'train', 'val', or 'test'
        label_names (List[str]): List mapping label_id to label_name.
    """
    split_dir = os.path.join(data_dir, data_split)
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    # create ground truth file
    gt_file = os.path.join(split_dir, f'GT_{data_split}.csv')
    with open(gt_file, 'w') as fid:
        writer = csv.writer(fid, delimiter=',', quotechar='"')
        writer.writerow(['key', 'xDim', 'yDim', 'LabelId', 'Label'])

    for label in label_names:
        class_dir = os.path.join(split_dir, label)
        if not os.path.isdir(class_dir):
            os.makedirs(class_dir)


def write_labels(label_names: List[str], out_dir: str) -> None:
    """Write labels to labels.csv

    Args:
        label_names (List[str]): List mapping label_id to label_name.
        out_dir (str): Directory to write the .csv file to.
    """
    # we index labels according to their sorted order
    labels = sorted(label_names)
    labels_file = os.path.join(out_dir, 'labels.csv')
    with open(labels_file, 'w') as fid:
        writer = csv.writer(fid, delimiter=',', quotechar='"')
        writer.writerow(['id', 'name'])

        for class_id, label in enumerate(labels):
            writer.writerow([class_id, label])


def calc_mean_std(data_dir: PathLike, img_size: Tuple[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and std
    """
    base = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    data = torch.utils.data.ConcatDataset([
        datasets.ImageFolder(root=f'{data_dir}/train/', transform=base),
        datasets.ImageFolder(root=f'{data_dir}/val/', transform=base)
    ])

    data_r = np.dstack([data[i][0][0,:,:] for i in range(len(data))])
    data_g = np.dstack([data[i][0][1,:,:] for i in range(len(data))])
    data_b = np.dstack([data[i][0][2,:,:] for i in range(len(data))])

    mean = (np.mean(data_r), np.mean(data_g), np.mean(data_b))
    std = (np.std(data_r), np.std(data_g), np.std(data_b))

    return mean, std
