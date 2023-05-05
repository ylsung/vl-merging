import io, base64
import json
import math
import torch
import cv2
import pickle
import random
import pandas as pd
import numpy as np
from PIL import Image
from vilt.transforms import keys_to_transforms, keys_to_transforms_for_mim
from vilt.datasets.masking_generator import MaskingGenerator


class TCSVBaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        split: str,
        image_path: str,
        annotations_paths: list,
        idx2line_path: str,
        patch_size: int,
        num_mask_patches: int,
        max_mask_patches_per_block: int,
        min_mask_patches_per_block: int,
        dvae_image_size: int,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        max_vl_text_len=None,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        size_frame=1,
        **kwargs,
    ):
        assert len(transform_keys) >= 1
        super().__init__()

        self.text_column_name = text_column_name
        self.max_text_len = max_text_len
        self.max_vl_text_len = max_vl_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.image_path = image_path
        self.idx2line_path = idx2line_path
        self.annotations_paths = annotations_paths
        self.data_dir = data_dir
        self.split = split
        self.size_frame = size_frame
        self.patch_size = patch_size
        self.num_mask_patches = num_mask_patches
        self.max_mask_patches_per_block = max_mask_patches_per_block
        self.min_mask_patches_per_block = min_mask_patches_per_block

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

        annotations = []

        for annotations_path in annotations_paths:
            with open(annotations_path, "r") as f:
                # annotations are list of dictionary like [{'video': 'video0', 'caption': 'a car is shown'}, ...]
                annotations += json.load(f)[split] # load the annotation of specific split

        self.annotations = annotations

        self.imgs = open(image_path, "r")
        self.id2lineidx = pickle.load(open(idx2line_path, 'rb'))

    def read_input_tsv(self, worker_id):
        self.imgs = open(self.image_path, "r")
        print(f"Launch for {worker_id}")

    def seek_img_tsv(self, pos):
        self.imgs.seek(pos)
        return [s.strip() for s in self.imgs.readline().split('\t')]

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

    def run_through_dataset(self):
        for vid, lineidx in self.id2lineidx.items():
            img_str_list = self.seek_img_tsv(lineidx)[2:]

            img_list = []
            for i, img_str in enumerate(img_str_list):
                try:
                    img = self.str2img(img_str)
                    img_list.append(img)
                except:
                    print(f"vid: {vid}, {i}th frame")

            try:
                assert len(img_list) == 32
            except:
                print(f"{vid} has {len(img_list)}")

    def sampling(self, start, end, n):
        if n == 1:
            return [int(round((start+end)/2.))]
        if n < 1:
            raise Exception("behaviour not defined for n<2")
        step = (end-start)/float(n-1)
        return [int(round(start+x*step)) for x in range(n)]

    def temporal_sample(self, list_of_b, random_sample=False, center_frame=False):
        max_size_frame = len(list_of_b)
        if max_size_frame == 1 or self.size_frame == max_size_frame:
            return list_of_b
        if max_size_frame < self.size_frame:
            print(f"Error in size_frame",
                  f"\tasked for {size_frame} from {max_size_frame} frames")

        size_frame = min(self.size_frame, max_size_frame)
        size_clips = int(math.ceil(max_size_frame / size_frame))
        if center_frame:
            # sample the middle frame.
            sampled_start = max_size_frame // 2
            sampled_end = sampled_start
        elif random_sample:
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

    def get_video_id(self, index, key):
        video_id = self.annotations[index][key]
        video_id =  int(video_id[5:])

        return video_id

    def get_raw_input(self, index, key="video"):
        raw_video_id = self.annotations[index][key]

        idx = self.id2lineidx[raw_video_id]

        img_str_list = self.seek_img_tsv(idx)[2:]

        random_sample = self.split == 'train'
        img_str_list = self.temporal_sample(
            img_str_list, 
            random_sample=random_sample,
            center_frame=(not random_sample and self.size_frame == 1)
        )

        img_list = []
        for i, img_str in enumerate(img_str_list):
            img = self.str2img(img_str)
            img_list.append(img)

        return img_list

    def get_input(self, index, key="video"):
        _input = self.get_raw_input(index, key=key)
        video_id = self.get_video_id(index, key)
        # image_tensor = [tr(_input[0]) for tr in self.transforms] # take the central image for now

        image_tensor = [self.transforms[0](img) for img in _input]

        ret = {
            "img_index": video_id,
            "cap_index": index,
            "raw_index": index,
        }

        if self.use_mim_transform:
            # overwrite image_tensor
            image_tensor, image_tensor_target = list(zip(*image_tensor))
            ret["image_target"] = [torch.stack(image_tensor_target, dim=0)]
            ret["image_masked_pos"] = [torch.LongTensor(self.masked_position_generator())]

        ret["image"] = [torch.stack(image_tensor, dim=0)]

        return ret

    def get_false_input(self, rep, key="video"):
        random_index = random.randint(0, len(self.annotations) - 1)
        _input = self.get_raw_input(random_index, key=key)
        # image_tensor = [tr(_input[0]) for tr in self.transforms]

        image_tensor = [self.transforms[0](img) for img in _input]

        # don't retreive image label (for dvae) for false images
        if self.use_mim_transform:
            image_tensor = [_image_tensor[0] for _image_tensor in image_tensor]

        image_tensor = [torch.stack(image_tensor, dim=0)]

        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        text = self.annotations[raw_index]["caption"]

        video_id = self.get_video_id(raw_index, "video")
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len if self.max_vl_text_len is None else self.max_vl_text_len,
            return_special_tokens_mask=True,
        )

        return {
            "text": (text, encoding),
            "img_index": video_id,
            "cap_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.annotations) - 1)

        text = self.annotations[random_index]["caption"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len if self.max_vl_text_len is None else self.max_vl_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):

        ret = dict()
        ret.update(self.get_input(index))
        if not self.image_only:
            txt = self.get_text(index)
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
            

            flatten_text = [t for text in texts for t in text]
            # print(len(flatten_encodings))
            # for i, f in enumerate(flatten_encodings):
                # print(flatten_text[i])
                # print(i, len(f["input_ids"]))

            flatten_mlms = mlm_collator(flatten_encodings)

            # print(flatten_mlms["input_ids"].shape)

            # print(flatten_mlms)

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

        return dict_batch

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return self.get_suite(index)


class MSRVTTDataset(TCSVBaseDataset):
    def __init__(self, data_dir, *args, split="", **kwargs):
        image_path = data_dir + "/img_msrvtt.tsv"

        annotations_paths = [data_dir + "/txt_msrvtt-retrieval.json"]

        idx2line_path = data_dir + "/img_msrvtt.id2lineidx.pkl"

        # df = pd.read_csv(image_path, sep="\t")

        super().__init__(data_dir, *args, **kwargs, image_path=image_path, annotations_paths=annotations_paths,
            idx2line_path=idx2line_path, split=split)

        # caption = self.annotations[100]["caption"]
        # video_id = self.annotations[100]["video"]

        # idx = self.id2lineidx[video_id]

        # print(caption, video_id, idx)

        # b = self.seek_img_tsv(idx)
        # print(len(b))
        # b = b[2:]

        # for i, _b in enumerate(b):
        #     img = self.str2img(_b)

        #     img.save(kwargs.get("data_dir") + f"/{i}.png")

        # only for debugging

        if kwargs.get("debug", False):
            import transformers

            self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


if __name__ == "__main__":
    dset = MSRVTTDataset(
        data_dir="/storage/v-yilinsung/vqa/",
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
