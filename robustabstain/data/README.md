# data

This directory contains downloading and preprocessing scripts that handle the data needed for this repository and acts as a storage directory for the atual data.

## Download instructions

### CIFAR10, CIFAR100

Handled by PyTorch.

### MTSD

1) Request access to the [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign) and wait for Mapillary to send you the dataset download links via email.

2) Copy the download links of the given `.zip` files in the email and copy them into `robust-abstain/robustabstain/data/mtsd_get.sh`:

    - Assign the download link of `mtsd_v2_fully_annotated_annotation.zip` to the `anno` variable in `robust-abstain/robustabstain/data/mtsd_get.sh`
    - Assign the download link of `mtsd_v2_fully_annotated_images.test.zip` to the `test` variable in `robust-abstain/robustabstain/data/mtsd_get.sh`
    - Assign the download link of `mtsd_v2_fully_annotated_images.train.0.zip` to the `train0` variable in `robust-abstain/robustabstain/data/mtsd_get.sh`
    - Assign the download link of `mtsd_v2_fully_annotated_images.train.1.zip` to the `train1` variable in `robust-abstain/robustabstain/data/mtsd_get.sh`
    - Assign the download link of `mtsd_v2_fully_annotated_images.train.2.zip` to the `train2` variable in `robust-abstain/robustabstain/data/mtsd_get.sh`
    - Assign the download link of `mtsd_v2_fully_annotated_images.val.zip` to the `val` variable in `robust-abstain/robustabstain/data/mtsd_get.sh`

3) Run  `bash mtsd_get.sh`


### SBB

This is a proprietary dataset.


### Synth

Synthetic datasets are generated on the fly everytime loaded.

## Preprocessing

### CIFAR10, CIFAR100

Handled by PyTorch.

### MTSD

```bash
python preprocessing/mtsd.py
```

### SBB

The SBB dataset is proprietary.
