from vilt.datasets import CCSVWDataset, CCSVDataset
from .datamodule_base import BaseDataModule
from torch.utils.data import DataLoader


class CCSVWDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CCSVWDataset

    @property
    def dataset_cls_no_false(self):
        return CCSVWDataset

    @property
    def dataset_name(self):
        return "ccsvw"


class CCSVDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CCSVDataset

    @property
    def dataset_cls_no_false(self):
        return CCSVDataset

    @property
    def dataset_name(self):
        return "ccsv"
