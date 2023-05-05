import torch
import os.path as op
from vilt.datasets.tsv_dataset import TSVCompositeDataset


class CCSVWDataset(TSVCompositeDataset):
    def __init__(self, data_dir, *args, split="", **kwargs):
        if split == "train":
            yaml_file = op.join(data_dir, "train_cc3m-coco-sbu-vg-webvid2.5m_10.yaml")
        elif split == "val":
            yaml_file = op.join(data_dir, "val_cc3m-coco-webvid2.5m.yaml")
        else:
            yaml_file = op.join(data_dir, "val_cc3m-coco-webvid2.5m.yaml")

        super().__init__(data_dir, *args, **kwargs, yaml_file=yaml_file, split=split)

        if kwargs.get("debug", False):
            import transformers

            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


class CCSVDataset(TSVCompositeDataset):
    def __init__(self, data_dir, *args, split="", **kwargs):
        if split == "train":
            yaml_file = op.join(data_dir, "train_cc3m-coco-sbu-vg.yaml")
        elif split == "val":
            yaml_file = op.join(data_dir, "val_cc3m-coco.yaml")
        else:
            yaml_file = op.join(data_dir, "val_cc3m-coco.yaml")

        super().__init__(data_dir, *args, **kwargs, yaml_file=yaml_file, split=split)

        if kwargs.get("debug", False):
            import transformers

            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")