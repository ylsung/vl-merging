from vilt.datasets import YfccDataset
from .datamodule_base import BaseDataModule


class YfccDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return YfccDataset

    @property
    def dataset_name(self):
        return "yfcc"
