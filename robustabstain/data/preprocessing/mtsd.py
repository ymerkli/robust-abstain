"""
Crop MTSD images to traffic sign bounding boxes
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import csv
import cv2
import json
import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List

from robustabstain.utils.paths import get_root_package_dir
from robustabstain.data.preprocessing.utils import calc_mean_std, load_image, \
    setup, write_labels, crop_to_patch


# data directories
MTSD_DIR = os.path.join(get_root_package_dir(), 'data', 'mtsd')
SOURCE_DATA_DIR = os.path.join(MTSD_DIR, 'images/')
IMAGES_CROP_DIR = os.path.join(MTSD_DIR, 'images_crop/')
IMAGES_CROP_LOOSE_DIR = os.path.join(MTSD_DIR, 'images_crop_loose/')
SPLIT_DIR = os.path.join(MTSD_DIR, 'splits/')


def move(source_data_dir: str, split_dir: str):
    """Move train, val, test images from images/ directory to train/, val, test/ directories.

    Args:
        source_data_dir (str): Path to directory containing source images (i.e. base dataset).
        split_dir (str): Path to directory containing data split files.
    """
    train_imgs, val_imgs, test_imgs = [], [], []
    with open(os.path.join(split_dir, 'train.txt')) as f:
        train_imgs = f.read().splitlines()

    with open(os.path.join(split_dir, 'val.txt')) as f:
        val_imgs = f.read().splitlines()

    with open(os.path.join(split_dir + 'test.txt')) as f:
        test_imgs = f.read().splitlines()

    for img in train_imgs:
        img_from = os.path.join(source_data_dir, img + '.jpg')
        img_to = os.path.join(source_data_dir, 'train/', img + '.jpg')
        if os.path.isfile(img_from):
            print(f'Moving {img_from} to {img_to}.')
            os.rename(img_from, img_to)

    for img in val_imgs:
        img_from = os.path.join(source_data_dir, img + '.jpg')
        img_to = os.path.join(source_data_dir, 'val/', img + '.jpg')
        if os.path.isfile(img_from):
            print(f'Moving {img_from} to {img_to}.')
            os.rename(img_from, img_to)

    for img in test_imgs:
        img_from = os.path.join(source_data_dir, img + '.jpg')
        img_to = os.path.join(source_data_dir, 'test/', img + '.jpg')
        if os.path.isfile(img_from):
            print(f'Moving {img_from} to {img_to}.')
            os.rename(img_from, img_to)


def get_labels() -> List[str]:
    """Read all MDTS class labels.

    Returns:
        List[str]: List mapping index to label name.
    """
    labels = set([])
    others_label = 'other-sign'
    for filename in os.listdir(os.path.join(MTSD_DIR, 'annotations')):
        if filename.endswith('.json'):
            image_key = os.path.splitext(filename)[0]
            anno = load_annotation(image_key)
            for obj in anno['objects']:
                labels.add(obj['label'])

    labels.remove(others_label)
    labels = sorted(labels)

    return labels


def load_annotation(image_key: str):
    with open(os.path.join(MTSD_DIR, 'annotations', f'{image_key}.json'), 'r') as fid:
        anno = json.load(fid)
    return anno


def crop_all(
        source_data_dir: str, out_data_dir: str, data_split: str, label_names: List[str],
        loose: bool = False, loose_factor: float = 0.0, verbose: bool = False
    ) -> None:
    """Crop all annotated bboxed traffic signs in dataset.

    Args:
        source_data_dir (str): Path to directory containing source images (i.e. base dataset).
        out_data_dir (str): Directory to store cropped images in
        data_split (str): 'train', 'val', or 'test'
        label_names (List[str]): List mapping label_id to label_name.
        loose (bool, optional): If set, bbox cropping is loose, meaning
            the cropbox is larger than the actual bbox. Defaults to False.
        loose_factor (float, optional): Fraction of added pixels to loose
            cropbox relative to the ground truth bbox.
        crop_filename = os.path.join(data_dir, data_split, label, obj_key + '.jpg')
    """
    gt_file = os.path.join(out_data_dir, data_split, f'GT_{data_split}.csv')
    obj_cnt = 0
    for img_filepath in Path(os.path.join(source_data_dir, data_split)).iterdir():
        if img_filepath.suffix == '.jpg':
            img_key = img_filepath.stem
            img = load_image(img_filepath)
            anno = load_annotation(img_key)

            for obj in anno['objects']:
                if (
                    obj['properties']['occluded'] or
                    obj['properties']['out-of-frame'] or
                    obj['properties']['exterior'] or
                    obj['properties']['ambiguous'] or
                    obj['properties']['included'] or
                    obj['properties']['dummy'] or
                    obj['label'] == 'other-sign'
                ):
                    # only crop on clean traffic signs
                    continue

                label_name = obj['label']
                label_id = label_names.index(label_name)
                obj_key = obj['key']
                crop_filepath = os.path.join(out_data_dir, data_split, label_name, obj_key + '.jpg')

                xmin, xmax = int(obj['bbox']['xmin']), int(obj['bbox']['xmax'])
                ymin, ymax = int(obj['bbox']['ymin']), int(obj['bbox']['ymax'])

                crop_to_patch(
                    img, img_key, data_split, xmin, xmax, ymin, ymax, obj_key,
                    label_name, label_id, crop_filepath, gt_file,
                    loose, loose_factor, verbose
                )
                obj_cnt += 1

    print(f'[{data_split}] Cropped {obj_cnt} objects to {data_split} set.')


def main():
    # prepare MTSD data
    move(SOURCE_DATA_DIR, SPLIT_DIR)
    label_names = get_labels()
    write_labels(label_names, MTSD_DIR)

    """
    Produce crop images
    """
    # setup training and validation data directories
    setup(IMAGES_CROP_DIR, 'train', label_names)
    setup(IMAGES_CROP_DIR, 'val', label_names)
    #setup(IMAGES_CROP_DIR, 'test', label_names) # test set has no annotations

    # crop images to bounding boxes and store into directories
    crop_all(SOURCE_DATA_DIR, IMAGES_CROP_DIR, 'train', label_names, verbose=True)
    crop_all(SOURCE_DATA_DIR, IMAGES_CROP_DIR, 'val', label_names, verbose=True)
    #crop_all(IMAGES_CROP_DIR, 'test', label_names) # test set has no annotations

    mean, std = calc_mean_std(IMAGES_CROP_DIR, img_size=(32,32))
    print(f'mean = {mean} (over all cropped traffic signs, resized to 32x32).')
    print(f'std = {std} (over all cropped traffic signs, resized to 32x32).')

    """
    Produce loose crop images
    """
    # setup training and validation data directories
    setup(IMAGES_CROP_LOOSE_DIR, 'train', label_names)
    setup(IMAGES_CROP_LOOSE_DIR, 'val', label_names)
    #setup('test', label_names) # test set has no annotations

    # crop images to bounding boxes and store into directories
    crop_all(
        SOURCE_DATA_DIR, IMAGES_CROP_LOOSE_DIR, 'train', label_names,
        loose=True, loose_factor=1.0, verbose=True
    )
    crop_all(
        SOURCE_DATA_DIR, IMAGES_CROP_LOOSE_DIR, 'val', label_names,
        loose=True, loose_factor=1.0, verbose=True
    )
    #crop_all('test', label_names) # test set has no annotations

    mean, std = calc_mean_std(IMAGES_CROP_LOOSE_DIR, img_size=(64,64))
    print(f'mean = {mean} (over all loose cropped traffic signs, resized to 64x64).')
    print(f'std = {std} (over all loose cropped traffic signs, resized to 64x64).')


if __name__ == '__main__':
    main()