import torch
import os.path as op
from vilt.datasets.msrvtt_dataset import TCSVBaseDataset


class DIDEMODataset(TCSVBaseDataset):
    def __init__(self, data_dir, *args, split="", **kwargs):
        image_path = data_dir + "/img_didemo.tsv"

        annotations_paths = [data_dir + "/txt_didemo-retrieval.json"]

        idx2line_path = data_dir + "/img_didemo.id2lineidx.pkl"

        # df = pd.read_csv(image_path, sep="\t")

        super().__init__(data_dir, *args, **kwargs, image_path=image_path, annotations_paths=annotations_paths,
            idx2line_path=idx2line_path, split=split)

        flatten_annotations = []

        for annotation in self.annotations:
            for caption in annotation["caption"]:
                d = {"video": annotation["video"], "caption": caption}

                flatten_annotations.append(d)

        self.annotations = flatten_annotations

        # only for debugging

        if kwargs.get("debug", False):
            import transformers

            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    def get_video_id(self, index, key):
        video_id = self.annotations[index][key]
        video_id =  int(video_id.split("@")[0])

        return video_id


if __name__ == "__main__":
    dset = DIDEMODataset(
        # data_dir="/storage/linjli/data/mtp_vlp_ray/rebuttal_tsv/",
        data_dir="/storage/v-yilinsung/didemo/",
        transform_keys=["square_transform_randaug_mim"],
        split="train",
        draw_false_image=2,
        draw_false_text=2,
        image_size=224,
        debug=True,
        patch_size=16,
        num_mask_patches=75,
        max_mask_patches_per_block=None,
        min_mask_patches_per_block=16,
        dvae_image_size=112,
        size_frame=4,
    )

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
