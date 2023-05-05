from vilt.datasets import WikipediaDataset
from .datamodule_base import BaseDataModule


class WikipediaDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return WikipediaDataset

    @property
    def dataset_cls_no_false(self):
        return WikipediaDataset

    @property
    def dataset_name(self):
        return "wikipedia"

