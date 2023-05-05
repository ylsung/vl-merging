from vilt.datasets import MSRVTTDataset
from .datamodule_base import BaseDataModule
from torch.utils.data import DataLoader


class MSRVTTDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MSRVTTDataset

    @property
    def dataset_cls_no_false(self):
        return MSRVTTDataset

    @property
    def dataset_name(self):
        return "msrvtt"

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.train_dataset.read_input_tsv,
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.val_dataset.read_input_tsv,
            collate_fn=self.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.test_dataset.read_input_tsv,
            collate_fn=self.collate,
        )
        return loader