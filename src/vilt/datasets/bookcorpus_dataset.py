import torch
from vilt.datasets.huggingface_dataset import HuggingfaceDataset


class BookCorpusDataset(HuggingfaceDataset):
    def __init__(self, *args, split="", **kwargs):
        super().__init__(
            *args,
            **kwargs,
            split=split,
            text_column_name="text",
            remove_duplicate=False,
        )

        if kwargs.get("debug", False):
            import transformers

            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


if __name__ == "__main__":
    dset = BookCorpusDataset(
        "/storage/v-yilinsung/huggingface/bookcorpus",
        ["square_transform_randaug_mim"],
        split="val",
        draw_false_image=2,
        draw_false_text=2,
        image_size=224,
        max_text_len=1024,
        debug=True,
        patch_size=16,
        num_mask_patches=75,
        max_mask_patches_per_block=None,
        min_mask_patches_per_block=16,
        dvae_image_size=112,
    )
    
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
        collate_fn=partial(dset.collate, mlm_collator=mlm_collator),
    )

    for i, l in enumerate(loader):
        print(l.keys())
        if i > 2:
            break

