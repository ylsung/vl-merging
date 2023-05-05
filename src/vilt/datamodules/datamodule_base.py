import torch
import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.prepare_data_per_node = False
        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.patch_size = _config["patch_size"]
        self.num_mask_patches = _config["num_mask_patches"]
        self.max_mask_patches_per_block = _config["max_mask_patches_per_block"]
        self.min_mask_patches_per_block = _config["min_mask_patches_per_block"]
        self.dvae_image_size = _config["dvae_image_size"]
        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.image_only = _config["image_only"]
        self.max_vl_text_len = _config["max_vl_text_len"]
        self.size_frame = _config["num_frames"]

        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_vl_text_len=self.max_vl_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            patch_size=self.patch_size,
            num_mask_patches=self.num_mask_patches,
            max_mask_patches_per_block=self.max_mask_patches_per_block,
            min_mask_patches_per_block=self.min_mask_patches_per_block,
            dvae_image_size=self.dvae_image_size,
            size_frame=self.size_frame,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_vl_text_len=self.max_vl_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            patch_size=self.patch_size,
            num_mask_patches=self.num_mask_patches,
            max_mask_patches_per_block=self.max_mask_patches_per_block,
            min_mask_patches_per_block=self.min_mask_patches_per_block,
            dvae_image_size=self.dvae_image_size,
            size_frame=self.size_frame,
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                max_vl_text_len=self.max_vl_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
                patch_size=self.patch_size,
                num_mask_patches=self.num_mask_patches,
                max_mask_patches_per_block=self.max_mask_patches_per_block,
                min_mask_patches_per_block=self.min_mask_patches_per_block,
                dvae_image_size=self.dvae_image_size,
                size_frame=self.size_frame,
            )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_vl_text_len=self.max_vl_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
            patch_size=self.patch_size,
            num_mask_patches=self.num_mask_patches,
            max_mask_patches_per_block=self.max_mask_patches_per_block,
            min_mask_patches_per_block=self.min_mask_patches_per_block,
            dvae_image_size=self.dvae_image_size,
            size_frame=self.size_frame,
        )

    def make_no_false_test_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_vl_text_len=self.max_vl_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
            patch_size=self.patch_size,
            num_mask_patches=self.num_mask_patches,
            max_mask_patches_per_block=self.max_mask_patches_per_block,
            min_mask_patches_per_block=self.min_mask_patches_per_block,
            dvae_image_size=self.dvae_image_size,
            size_frame=self.size_frame,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_vl_text_len=self.max_vl_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            patch_size=self.patch_size,
            num_mask_patches=self.num_mask_patches,
            max_mask_patches_per_block=self.max_mask_patches_per_block,
            min_mask_patches_per_block=self.min_mask_patches_per_block,
            dvae_image_size=self.dvae_image_size,
            size_frame=self.size_frame,
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.collate = functools.partial(
                self.train_dataset.collate, mlm_collator=self.mlm_collator,
            )

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
