from vilt.datasets import ImageNet1kDataset
from .datamodule_base import BaseDataModule
from torch.utils.data import DataLoader


class ImageNet1kDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ImageNet1kDataset

    @property
    def dataset_cls_no_false(self):
        return ImageNet1kDataset

    @property
    def dataset_name(self):
        return "imagenet1k"
