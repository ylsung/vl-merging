from vilt.datasets import ImageNetDataset
from .datamodule_base import BaseDataModule
from torch.utils.data import DataLoader


class ImageNetDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ImageNetDataset

    @property
    def dataset_cls_no_false(self):
        return ImageNetDataset

    @property
    def dataset_name(self):
        return "imagenet"
