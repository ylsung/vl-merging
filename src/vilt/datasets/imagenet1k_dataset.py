import json
import torch
import os.path as op
from vilt.datasets.tsv_dataset import TSVCompositeDataset


class ImageNet1kDataset(TSVCompositeDataset):
    def __init__(self, data_dir, *args, split="", **kwargs):
        if split == "train":
            yaml_file = op.join(data_dir, "train_imagenet-1k.yaml")
        elif split == "val":
            yaml_file = op.join(data_dir, "val_imagenet-1k.yaml")
        else:
            yaml_file = op.join(data_dir, "val_imagenet-1k.yaml")

        self.only_train_with_image = True

        super().__init__(data_dir, *args, **kwargs, yaml_file=yaml_file, split=split)

        if kwargs.get("debug", False):
            import transformers

            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    def get_text(self, raw_idx, img_idx, cap_idx):
        row = self.get_row_from_tsv(self.cap_tsv, img_idx)
        label = json.loads(row[1])
        # TODO: convert int label to text class label
        text = f"test"
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "class_label": label,
            "text": (text, encoding),
            "img_index": img_idx,
            "cap_index": cap_idx,
            "raw_index": raw_idx,
        }

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]

        for img_key in img_keys:
            new_imgs = [tmp_img[0] for tmp_img in dict_batch[img_key]]
            batch_new_imgs = torch.stack(new_imgs, dim=0)
            dict_batch[img_key] = [batch_new_imgs]

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        if "class_label" in keys:
            dict_batch["class_label"] = torch.LongTensor(dict_batch["class_label"])

        if self.only_train_with_image:
            dict_batch["only_train_with_image"] = True

        return dict_batch


if __name__ == "__main__":
    dset = ImageNet1kDataset(
        data_dir="/storage/v-yilinsung/imagenet-1k/composite/",
        transform_keys=["square_transform_randaug_mim"],
        split="val",
        draw_false_image=0,
        draw_false_text=0,
        image_size=224,
        debug=True,
        patch_size=16,
        num_mask_patches=75,
        max_mask_patches_per_block=None,
        min_mask_patches_per_block=16,
        dvae_image_size=112,
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
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        # worker_init_fn=dset.read_input_tsv,
        collate_fn=partial(dset.collate, mlm_collator=mlm_collator),
    )

    for i, l in enumerate(loader):
        
        print(l["class_label"])
        # print(l["image"][0].shape, l["image_target"][0].shape, l["class_label"].shape)

        exit()
        # print(l["class_label"])
        # exit()
       
