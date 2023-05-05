import io, base64
import os
import os.path as op
import yaml
import json
import torch
import cv2
import math
import pickle
import random
import errno
import pandas as pd
import numpy as np
from PIL import Image
from vilt.transforms import keys_to_transforms, keys_to_transforms_for_mim

from vilt.datasets.tsv_file import TSVFile, CompositeTSVFile, tsv_reader
from vilt.datasets.masking_generator import MaskingGenerator


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


class TSVCompositeDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_dir: str,
        transform_keys: list,
        image_size: int,
        split: str,
        yaml_file: str,
        patch_size: int,
        num_mask_patches: int,
        max_mask_patches_per_block: int,
        min_mask_patches_per_block: int,
        dvae_image_size: int,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        size_frame=1,
        use_asr=False,
        **kwargs
        ):

        self.text_column_name = text_column_name
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.split = split

        self.use_mim_transform = any(t.endswith("mim") for t in transform_keys)

        if self.use_mim_transform:
            window_size = image_size // patch_size
            self.masked_position_generator = MaskingGenerator(
                window_size, num_masking_patches=num_mask_patches,
                max_num_patches=max_mask_patches_per_block,
                min_num_patches=min_mask_patches_per_block,
            )

            self.transforms = keys_to_transforms_for_mim(transform_keys, size=image_size, second_size=dvae_image_size)
        else:
            self.transforms = keys_to_transforms(transform_keys, size=image_size)

        # label: trainval.label.tsv
        # label_linelist: trainval.label.lineidx
        # composite: false
        # img: trainval.tsv
        # img_linelist: trainval.lineidx

        self.is_composite = False

        self.visual_file = os.path.join(self.data_dir, f"{split}.tsv")
        self.visual_tsv = self.get_tsv_file(self.visual_file)

        self.cap_file = os.path.join(self.data_dir, f"{split}.label.tsv")
        self.cap_tsv = self.get_tsv_file(self.cap_file)

        # one caption per image/video
        self.img_line_list = [i for i in range(self.cap_tsv.num_rows())]
        self.cap_line_list = [0 for i in range(self.cap_tsv.num_rows())]

        # if split == "val" or split == "test":
        #     # self.img_line_list = self.img_line_list[:1000] # there is no validation data, so sample some training data as the dummy validation data.
        #     self.img_line_list = [i for i in range(1000)]
        #     self.cap_line_list = [0 for i in range(1000)]
        # else:
        #     self.img_line_list = [i for i in range(self.cap_tsv.num_rows())]
        #     self.cap_line_list = [0 for i in range(self.cap_tsv.num_rows())]
        self.is_train = split == "train" or split == "trainval"
        if self.is_train:
            assert self.cap_tsv is not None
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.img_res = image_size
        self.patch_size = patch_size
        self.size_frame = size_frame
        self.num_mask_patches = num_mask_patches
        self.max_mask_patches_per_block = max_mask_patches_per_block
        self.min_mask_patches_per_block = min_mask_patches_per_block

        self.use_asr = use_asr
        # for MERLOT/HT100M only
        self.append_pred_mf_cap = False
        self.pred_mf_cap_only = False
        self.alternate_asr_pred_cap = False
        self.alternate_asr_pred_cap = (
            self.alternate_asr_pred_cap and self.use_asr
            and self.pred_mf_cap_only)

        self.on_memory = False
        self.use_action_label = False # add labels to tags

        self.index = 0
    
    def __len__(self):
        return len(self.img_line_list)

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(
                    tsv_file, self.cap_linelist_file, root=self.root)
            return TSVFile(tsv_file)

    def get_valid_tsv(self):
        if self.is_train:
            return self.cap_tsv
        # sorted by file size
        if self.cap_tsv:
            return self.cap_tsv
        if self.visual_tsv:
            return self.visual_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i): i for i in range(tsv.num_rows())}

    def str2img(self, b):
        try:
            img = Image.fromarray(
                cv2.imdecode(
                    np.frombuffer(base64.b64decode(b), np.uint8),
                    cv2.IMREAD_COLOR)[:, :, ::-1]
                ).convert('RGB')
        except Exception:
            img = Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB')
        return img

    def get_row_from_tsv(self, tsv, img_idx):
        row = tsv[img_idx]
        if self.is_composite:
            assert self.image_keys[img_idx].endswith(row[0])
        else:
            assert row[0] == self.image_keys[img_idx]
        return row

    def sampling(self, start, end, n):
        if n == 1:
            return [int(round((start+end)/2.))]
        if n < 1:
            raise Exception("behaviour not defined for n<2")
        step = (end-start)/float(n-1)
        return [int(round(start+x*step)) for x in range(n)]

    def temporal_sample(self, list_of_b, random_sample=False):
        max_size_frame = len(list_of_b)
        if max_size_frame == 1 or self.size_frame == max_size_frame:
            return list_of_b
        if max_size_frame < self.size_frame:
            print(f"Error in size_frame",
                  f"\tasked for {size_frame} from {max_size_frame} frames")

        size_frame = min(self.size_frame, max_size_frame)
        size_clips = int(math.ceil(max_size_frame / size_frame))
        if random_sample:
            sampled_start = random.choice(range(size_clips))
            sampled_end = min(
                sampled_start + (size_frame - 1) * size_clips,
                max_size_frame - 1)
        else:
            sampled_start = 0
            sampled_end = max_size_frame - 1
        sampled_index = self.sampling(sampled_start, sampled_end, size_frame)
        sampled_video = [list_of_b[i] for i in sampled_index]
        return sampled_video

    def get_img_or_video(self, list_of_b):
        # bufs = self.temporal_sample(
        #     list_of_b, random_sample=(self.split == 'train'))

        bufs = self.temporal_sample(
            list_of_b, random_sample=True)
        
        imgs = []

        img_targets = []

        for i, b in enumerate(bufs):
            single_img = self.str2img(b)
            # single_img.save(self.data_dir + f"/{self.index}.png")
            self.index += 1
            if self.use_mim_transform:
                single_img, single_img_target = self.transforms[0](single_img) # only take one transformation
                img_targets.append(single_img_target.unsqueeze(0))
            else:
                single_img = self.transforms[0](single_img) # only take one transformation
    
            imgs.append(single_img.unsqueeze(0))

        imgs = torch.cat(imgs, dim=0)

        if self.use_mim_transform:
            img_targets = torch.cat(img_targets, dim=0)

        return imgs, img_targets

    def get_image_cap_index(self, idx):
        return self.img_line_list[idx], self.cap_line_list[idx]

    def get_input(self, raw_idx, img_idx, cap_idx):
        row = self.get_row_from_tsv(self.visual_tsv, img_idx)

        # sample from the video
        if len(row) >= self.size_frame + 2:
            _input, input_targets = self.get_img_or_video(row[2:]) # (L, C, H, W)
            is_video = True
        elif len(row) == self.size_frame + 1:
            _input, input_targets = self.get_img_or_video(row[1:]) # (L, C, H, W)
            is_video = True
        else:  # if the input is a single image
            _input, input_targets = self.get_img_or_video([row[-1]])
            is_video = False

        if self.size_frame == 1:
            # use single frame, make the _input be 3D tensor by removing the temporal dimension
            _input = _input[0] # (C, H, W)
            input_targets = input_targets[0] # (C, H, W)
            is_video = False

        image_tensor = [_input]

        ret = {
            "image": image_tensor,
            "img_index": img_idx,
            "cap_index": cap_idx,
            "raw_index": raw_idx,
            "is_video": is_video,
        }

        if self.use_mim_transform:
            ret["image_target"] = [input_targets]
            ret["image_masked_pos"] = [torch.LongTensor(self.masked_position_generator())]

        return ret

    def get_false_input(self, rep, key="video"):
        random_index = random.randint(0, self.__len__() - 1)
        img_idx, cap_idx = self.get_image_cap_index(random_index)
        image_tensor = self.get_input(random_index, img_idx, cap_idx)["image"]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_idx, img_idx, cap_idx):
        row = self.get_row_from_tsv(self.cap_tsv, img_idx)
        label = json.loads(row[1])
        # TODO: convert int label to text class label
        text = f"{label[0]['class']}"
        label = 0
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

    def get_false_text(self, rep):
        random_index = random.randint(0, self.__len__() - 1)

        img_idx, cap_idx = self.get_image_cap_index(random_index)

        text_and_encoding = self.get_text(random_index, img_idx, cap_idx)["text"]

        return {f"false_text_{rep}": text_and_encoding}

    def get_suite(self, idx):
        img_idx, cap_idx = self.get_image_cap_index(idx)

        ret = dict()
        ret.update(self.get_input(idx, img_idx, cap_idx))

        if not self.image_only:
            txt = self.get_text(idx, img_idx, cap_idx)
            # ret.update({"replica": True if txt["cap_index"] > 0 else False})
            ret.update(txt)

        for i in range(self.draw_false_image):
            ret.update(self.get_false_input(i))
        for i in range(self.draw_false_text):
            ret.update(self.get_false_text(i))

        return ret

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

    def __getitem__(self, index):
        return self.get_suite(index)


class ImageNetDataset(TSVCompositeDataset):
    def __init__(self, data_dir, *args, split="", **kwargs):
        # if split == "val":
        #     split = "test" # the validation set is named as test.tsv

        # if split == "train" or split == "val":
        
        if split == "train":
            split = "trainval"

        self.only_train_with_image = True

        super().__init__(data_dir, *args, **kwargs, yaml_file=None, split=split)

        if kwargs.get("debug", False):
            import transformers

            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


if __name__ == "__main__":
    dset = ImageNetDataset(
        data_dir="/storage/v-yilinsung/imagenet-22k/",
        transform_keys=["square_transform_randaug_mim"],
        split="val",
        draw_false_image=2,
        draw_false_text=2,
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
        num_workers=4,
        pin_memory=True,
        # worker_init_fn=dset.read_input_tsv,
        collate_fn=partial(dset.collate, mlm_collator=mlm_collator),
    )

    for i, l in enumerate(loader):
        pass
        # print(l["image"][0].shape, l["image_target"][0].shape, l["class_label"].shape)

        # print(l["class_label"])
        # exit()
       
