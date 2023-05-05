import functools
from copy import deepcopy

import torch
import random
import math
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from .multitask_datamodule import MTDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data.distributed import DistributedSampler


class MultiMTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()
        self.prepare_data_per_node = False
        
        self.dm_dicts = {}
        
        for task_type, datasets, data_roots in zip(_config["tasks"], _config["datasets"], _config["data_roots"]):
            copied_config = deepcopy(_config)
            copied_config["datasets"] = datasets
            copied_config["data_roots"] = data_roots

            if task_type == "vl":
                copied_config["mlm_prob"] = copied_config["vl_mlm_prob"]

            self.dm_dicts[task_type] = MTDataModule(copied_config, dist)

        self.dms = [v for k, v in self.dm_dicts.items()]

        self.dist = dist

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        for dm in self.dms:
            dm.setup(stage)

            print(len(dm.train_dataset))
            print(len(dm.val_dataset))
            print(len(dm.test_dataset))

    def train_dataloader(self):
        return CombinedLoader(
            {task_type: dm.train_dataloader() for task_type, dm in self.dm_dicts.items()},
            mode="min_size"
            )

    def val_dataloader(self, batch_size=None):
        return CombinedLoader(
            {task_type: dm.val_dataloader(batch_size) for task_type, dm in self.dm_dicts.items()},
            mode="min_size"
            )

    def test_dataloader(self):
        return CombinedLoader(
            {task_type: dm.test_dataloader() for task_type, dm in self.dm_dicts.items()},
            mode="min_size"
            )


# class MultiDataset(Dataset):
#     def __init__(self, datasets, shuffle=True):
#         self.datasets = datasets
#         self.shuffle = shuffle

#         self.map_indices = {k: [] for k, _ in self.datasets.items()}
#         self.min_length = min(len(d) for k, d in self.datasets.items())
#         self.max_length = max(len(d) for k, d in self.datasets.items())

#     def __getitem__(self, i):
#         # return tuple(d[i] for d in self.datasets)

#         # self.map_indices will reload when calling self.__len__()
#         return {k: d[self.map_indices[k][i]] for k, d in self.datasets.items()}
#         # return tuple(d[m[i]] for d, m in zip(self.datasets, self.map_indices))

#     def construct_map_index(self):
#         """
#         Construct the mapping indices for every data. Because the __len__ is larger than the size of some datset,
#         the map_index is use to map the parameter "index" in __getitem__ to a valid index of each dataset.
#         Because of the dataset has different length, we should maintain different indices for them.
#         """

#         def update_indices(original_indices, data_length, max_data_length):
#             # update the sampling indices for this dataset

#             # return: a list, which maps the range(max_data_length) to the val index in the dataset

#             original_indices = original_indices[max_data_length:]  # remove used indices in last epoch
#             fill_num = max_data_length - len(original_indices)
#             batch = math.ceil(fill_num / data_length)

#             additional_indices = list(range(data_length)) * batch

#             if self.shuffle:
#                 random.shuffle(additional_indices)

#             original_indices += additional_indices

#             assert (
#                 len(original_indices) >= max_data_length
#             ), "the length of matcing indices is too small"

#             return original_indices

#         for key in self.map_indices.keys():
#             map_index = self.map_indices[key]
#             dataset = self.datasets[key]

#             self.map_indices[key] = update_indices(map_index, len(dataset), self.max_length)

#     def __len__(self):
#         # will be called every epoch
#         return self.max_length


# class MultiMTDataModule(LightningDataModule):
#     def __init__(self, _config, dist=False):
#         super().__init__()
#         self.prepare_data_per_node = False
        
#         self.dm_dicts = {}
        
#         for task_type, datasets, data_roots in zip(_config["tasks"], _config["datasets"], _config["data_roots"]):
#             copied_config = deepcopy(_config)
#             copied_config["datasets"] = datasets
#             copied_config["data_roots"] = data_roots

#             self.dm_dicts[task_type] = MTDataModule(copied_config, dist)

#         self.dms = [v for k, v in self.dm_dicts.items()]

#         self.batch_size = self.dms[0].batch_size
#         self.vocab_size = self.dms[0].vocab_size
#         self.num_workers = self.dms[0].num_workers

#         self.dist = dist

#     def prepare_data(self):
#         for dm in self.dms:
#             dm.prepare_data()

#     def setup(self, stage):
#         for dm in self.dms:
#             dm.setup(stage)

#         self.train_dataset = MultiDataset({k: dm.train_dataset for k, dm in self.dm_dicts.items()}, shuffle=True)
#         self.val_dataset = MultiDataset({k: dm.val_dataset for k, dm in self.dm_dicts.items()}, shuffle=True)
#         self.test_dataset = MultiDataset({k: dm.test_dataset for k, dm in self.dm_dicts.items()}, shuffle=False)
#         self.tokenizer = self.dms[0].tokenizer

#         self.collate = self.dms[0].collate

#         if self.dist:
#             self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
#             self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
#             self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
#         else:
#             self.train_sampler = None
#             self.val_sampler = None
#             self.test_sampler = None

#     def train_dataloader(self):
#         self.train_dataset.construct_map_index()
#         loader = DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             sampler=self.train_sampler,
#             num_workers=self.num_workers,
#             worker_init_fn=self.dms[0].train_dataset.read_input_tsv if hasattr(self.dms[0].train_dataset, 'read_input_tsv') else None,
#             collate_fn=self.collate,
#         )
#         return loader

#     def val_dataloader(self, batch_size=None):
#         self.val_dataset.construct_map_index()
#         loader = DataLoader(
#             self.val_dataset,
#             batch_size=batch_size if batch_size is not None else self.batch_size,
#             sampler=self.val_sampler,
#             num_workers=self.num_workers,
#             worker_init_fn=self.dms[0].val_dataset.read_input_tsv if hasattr(self.dms[0].val_dataset, 'read_input_tsv') else None,
#             collate_fn=self.collate,
#         )

#         return loader

#     def test_dataloader(self):
#         self.test_dataset.construct_map_index()
#         loader = DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             sampler=self.test_sampler,
#             num_workers=self.num_workers,
#             worker_init_fn=self.dms[0].test_dataset.read_input_tsv if hasattr(self.dms[0].test_dataset, 'read_input_tsv') else None,
#             collate_fn=self.collate,
#         )
#         return loader