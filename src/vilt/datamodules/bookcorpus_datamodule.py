from vilt.datasets import BookCorpusDataset
from .datamodule_base import BaseDataModule


class BookCorpusDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return BookCorpusDataset

    @property
    def dataset_cls_no_false(self):
        return BookCorpusDataset

    @property
    def dataset_name(self):
        return "bookcorpus"

