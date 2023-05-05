from vilt.datasets import CcDataset
from .datamodule_base import BaseDataModule


class CcDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CcDataset

    @property
    def dataset_name(self):
        return "cc"
