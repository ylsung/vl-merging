import transformers

import random
import torch
import io
import pyarrow as pa
import os

from datasets import load_dataset, load_from_disk


class HuggingfaceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list, # dummy, to accompany with the other datasets and the general datamodule creation call
        split: str,
        text_column_name: str = "",
        max_text_len=40,
        draw_false_text=0,
        **kwargs):
        super().__init__()

        self.text_column_name = text_column_name
        self.max_text_len = max_text_len
        self.draw_false_text = draw_false_text

        dataset = load_from_disk(data_dir) # Dataset Dict
        
        if split in dataset:
            self.data = dataset[split] # Dataset object
        else:
            self.data = dataset["train"].select(range(0, 1000), keep_in_memory=True)  # testing use. Use for running through pytorch-lighting validation

            # self.data = []
            # self.data = [{"text": "test"}] # testing use. Use for running through pytorch-lighting validation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_text(self, raw_index):
        text = self.data[raw_index][self.text_column_name]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "cap_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, self.__len__() - 1)
        text = self.get_text(random_index)["text"]

        return {f"false_text_{rep}": text}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)

                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, self.__len__() - 1)

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

        return dict_batch


if __name__ == "__main__":
    d = load_from_disk("/storage/v-yilinsung/huggingface/wikipedia_20200501_en")

    # print(d)

    # d_train = d["train"]

    # print(d_train)

    # print(len(d_train))

    # print(type(d_train["text"]), len(d_train["text"]))

    # print(d_train["text"][0])

    # # for i, _d in enumerate(d_train):
    # #     print(_d)
    # #     if i > 3:
    # #         break
    