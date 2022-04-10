from robustness.datasets import CIFAR as CIFARSuper
from robustness.datasets import DataSet

from robustabstain.utils.transforms import get_transforms, get_mean_sigma
from robustabstain.utils.checkpointing import get_net


"""
Datasets for training with the robustness library (https://github.com/MadryLab/robustness)
"""

class CIFAR(CIFARSuper):
    """Wrapper around the cifar10 dataset for the robustness library (https://github.com/MadryLab/robustness).
    Replacing model loading with custom loader.
    """
    def __init__(self, data_path, **kwargs):
        super(CIFAR, self).__init__(data_path, **kwargs)

    def get_model(self, arch, pretrained):
        if pretrained:
            raise ValueError("Error: CIFAR does not support pretrained=True.")

        return get_net(arch, 'cifar10', self.num_classes, device='cpu', normalize=False, parallel=False)


class MTSD(DataSet):
    """Wrapper around the Mapillary Traffic Sign DataSet needed for the robustness library (https://github.com/MadryLab/robustness).
    """
    def __init__(self, data_path, **kwargs):
        mean, std = get_mean_sigma('cpu', 'mtsd')
        train_transforms, test_transforms = get_transforms('mtsd', augment=None, normalize=False)
        ds_kwargs = {
            'num_classes': 400,
            'mean': mean,
            'std': std,
            'custom_class': None,
            'label_mapping': None,
            'transform_train': train_transforms[0],
            'transform_test': test_transforms[0]
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(MTSD, self).__init__('mtsd', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        if pretrained:
            raise ValueError("Error: MTSD does not support pretrained=True.")

        return get_net(arch, 'mtsd', self.num_classes, device='cpu', normalize=False, parallel=False)
