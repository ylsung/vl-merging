from vilt.datasets.base_dataset import BaseDataset


class F30KCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        if split == "train":
            names = ["f30k_caption_karpathy_train", "f30k_caption_karpathy_val"]
        elif split == "val":
            # names = ["f30k_caption_karpathy_val"]
            names = ["f30k_caption_karpathy_test"]
        elif split == "test":
            names = ["f30k_caption_karpathy_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)


if __name__ == "__main__":
    dset = F30KCaptionKarpathyDataset(
        data_dir="/storage/v-yilinsung/arrow_files/",
        transform_keys=["square_transform_randaug_mim"],
        split="test",
        draw_false_image=2,
        draw_false_text=2,
        image_size=224,
        # debug=True,
        patch_size=16,
        num_mask_patches=75,
        max_mask_patches_per_block=None,
        min_mask_patches_per_block=16,
        dvae_image_size=112,
        size_frame=4,
    )

    index_list = []
    for i in range(len(dset)):

        # self.index_mapper[index][0]
        index = dset.index_mapper[i][0]

        index_list.append(index)
    print(len(set(index_list)))
    exit()

    # dset.run_through_dataset()

    from transformers import (
        DataCollatorForLanguageModeling,
        DataCollatorForWholeWordMask,
        BertTokenizer,
    )
    from functools import partial

    collator = DataCollatorForWholeWordMask

    mlm_collator = collator(
        tokenizer=dset.tokenizer, mlm=True, mlm_probability=0.15
    )

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=dset.read_input_tsv,
        collate_fn=partial(dset.collate, mlm_collator=mlm_collator),
    )

    for i, l in enumerate(loader):
        print(l["image"][0].shape, l["image_target"][0].shape, l["image_masked_pos"][0].shape)
        if i > 2:
            break