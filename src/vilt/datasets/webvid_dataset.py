import torch
import os.path as op
from vilt.datasets.tsv_dataset import TSVCompositeDataset


class WebVIDDataset(TSVCompositeDataset):
    def __init__(self, data_dir, *args, split="", **kwargs):
        if split == "train":
            yaml_file = op.join(data_dir, "train_webvid2.5m_10.yaml")
        elif split == "val":
            yaml_file = op.join(data_dir, "val_webvid2.5m.yaml")
        else:
            yaml_file = op.join(data_dir, "val_webvid2.5m.yaml")

        # df = pd.read_csv(image_path, sep="\t")

        super().__init__(data_dir, *args, **kwargs, yaml_file=yaml_file, split=split)

        if kwargs.get("debug", False):
            import transformers

            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


if __name__ == "__main__":
    dset = WebVIDDataset(
        # data_dir="/storage/linjli/data/mtp_vlp_ray/pretrain/composite/",
        data_dir="/storage/v-yilinsung/cc/composite/",
        transform_keys=["square_transform_randaug_mim"],
        split="val",
        draw_false_image=0,
        draw_false_text=0,
        image_size=224,
        patch_size=16,
        num_mask_patches=75,
        max_mask_patches_per_block=None,
        min_mask_patches_per_block=16,
        dvae_image_size=112,
        size_frame=1,
        debug=True,
        save_images=True,
    )
    
    # indices = dset.get_composite_source_idx()
    # print(indices)
    # print(len(indices), len(dset))
    # exit()
    
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
        num_workers=0,
        pin_memory=True,
        # worker_init_fn=dset.read_input_tsv,
        collate_fn=partial(dset.collate, mlm_collator=mlm_collator),
    )

    for i, l in enumerate(loader):
        for i, _l in enumerate(l["text"]):
            print(i + 1, _l)

        print(l["image"][0].shape, l["image_target"][0].shape, l["image_masked_pos"][0].shape)
        
        exit()
       