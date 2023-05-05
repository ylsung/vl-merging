from vilt.datasets import WebVIDDataset
from .datamodule_base import BaseDataModule
from torch.utils.data import DataLoader


class WebVIDDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return WebVIDDataset

    @property
    def dataset_cls_no_false(self):
        return WebVIDDataset

    @property
    def dataset_name(self):
        return "webvid"
