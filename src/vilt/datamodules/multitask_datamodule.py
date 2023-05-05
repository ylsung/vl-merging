import functools
from copy import deepcopy

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules


class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()
        self.prepare_data_per_node = False

        self.dm_keys = datamodule_keys

        if _config["data_roots"] is not None:
            assert len(_config["data_roots"]) == len(datamodule_keys), "length of data_roots doesn't match the length of datasets"

            self.dm_dicts = {}
            for data_root, datamodule_key in zip(_config["data_roots"], datamodule_keys):
                copied_config = deepcopy(_config)
                copied_config["data_root"] = data_root
                self.dm_dicts[datamodule_key] = _datamodules[datamodule_key](copied_config)
        else:
            self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}

        self.dms = [v for k, v in self.dm_dicts.items()]

        self.batch_size = self.dms[0].batch_size
        self.vocab_size = self.dms[0].vocab_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        for dm in self.dms:
            dm.setup(stage)

        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
        self.tokenizer = self.dms[0].tokenizer

        self.collate = functools.partial(
            self.dms[0].train_dataset.collate, mlm_collator=self.dms[0].mlm_collator,
        )

        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            worker_init_fn=self.dms[0].train_dataset.read_input_tsv if hasattr(self.dms[0].train_dataset, 'read_input_tsv') else None,
            collate_fn=self.collate,
            drop_last=True,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            worker_init_fn=self.dms[0].val_dataset.read_input_tsv if hasattr(self.dms[0].val_dataset, 'read_input_tsv') else None,
            collate_fn=self.collate,
        )

        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            worker_init_fn=self.dms[0].test_dataset.read_input_tsv if hasattr(self.dms[0].test_dataset, 'read_input_tsv') else None,
            collate_fn=self.collate,
        )
        return loader
