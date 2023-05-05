import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
# import vilt.modules.vision_transformer as vit
from vilt.modules import vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
from pytorch_lightning.utilities.distributed import rank_zero_info
from scipy import interpolate

from vilt.modules.modeling_discrete_vae import create_d_vae 


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config, ufo_config, ln_config, moe_config):
        super().__init__()
        self.prepare_data_per_node = False
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
            position_embedding_type="rel_pos",
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        self.moe_config = moe_config

        if self.hparams.config["load_path"] == "" and not self.hparams.config["random_initialization"]:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config, ufo_config=ufo_config, ln_config=ln_config, moe_config=moe_config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config, ufo_config=ufo_config, ln_config=ln_config, moe_config=moe_config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["text_only_mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)
        
        if config["loss_names"]["ifm"] > 0:
            self.ifm_text_proj = heads.IFMHead(config["hidden_size"])
            self.ifm_image_proj = heads.IFMHead(config["hidden_size"])
            self.ifm_text_proj.apply(objectives.init_weights)
            self.ifm_image_proj.apply(objectives.init_weights)

            self.ifm_vl_text_proj = heads.IFMHead(config["hidden_size"])
            self.ifm_vl_image_proj = heads.IFMHead(config["hidden_size"])
            self.ifm_vl_text_proj.apply(objectives.init_weights)
            self.ifm_vl_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if config["loss_names"]["irtr"] > 0:
            self.ifm_text_proj = heads.IFMHead(config["hidden_size"])
            self.ifm_image_proj = heads.IFMHead(config["hidden_size"])
            self.ifm_text_proj.apply(objectives.init_weights)
            self.ifm_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if config["loss_names"]["mim"] > 0 or config["loss_names"]["image_only_mim"] > 0:
            self.d_vae = create_d_vae(
                weight_path=config["discrete_vae_weight_path"], 
                d_vae_type="dall-e",
                device="cuda", 
                image_size=config["dvae_image_size"]
            )
            self.mim_score = nn.Linear(config["hidden_size"], self.d_vae.decoder.vocab_size)
            self.mim_score.apply(objectives.init_weights)

        window_size = (int(config["image_size"]/config["patch_size"]), int(config["image_size"]/config["patch_size"])) #(14, 14)
        rank_zero_info("window_size: {}".format(window_size))
        num_heads = config["num_heads"] #16
        num_layers = config["num_layers"] #24
        max_text_len_of_initckpt = config["max_text_len_of_initckpt"] #196
        max_text_len = config["max_text_len"] #40
        self.max_text_len = max_text_len
        self.max_vl_text_len = config["max_vl_text_len"]
        max_imag_len = window_size[0]*window_size[1]+1 #197
        self.max_imag_len = max_imag_len
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # self.text_num_relative_distance = 2 * max_text_len
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        self.all_num_relative_distance = self.num_relative_distance + self.text_num_relative_distance + 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance + self.text_num_relative_distance + 2, num_heads * num_layers))  # 2*Wh-1 * 2*Ww-1, nH
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

        # print(relative_coords)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)

        # cls to token & token 2 cls & cls to cls
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        # print(relative_position_index)
        self.register_buffer("relative_position_index", relative_position_index)
        
        text_position_ids = torch.arange(max_text_len-1)
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2-max_text_len_of_initckpt) #-194
        rank_zero_info("min_distance: {}".format(min_distance))
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len, ) * 2, dtype=relative_coords.dtype)
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.register_buffer("text_relative_position_index", text_relative_position_index)
        
        text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (self.num_relative_distance)
        imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (self.num_relative_distance + 1)

        text_row_relative_position_index = torch.cat((self.text_relative_position_index, text2imag_relative_position_index), 1)
        imag_row_relative_position_index = torch.cat((imag2text_relative_position_index, self.relative_position_index), 1)
        text_imag_relative_position_index = torch.cat((text_row_relative_position_index, imag_row_relative_position_index), 0)
        self.register_buffer("text_imag_relative_position_index", text_imag_relative_position_index)


        ###
        # [Temporal embedding starts]
        # This is code for temporal modeling, which makes codes a bit unreadable.
        # Please ignore all of them as they do not effect the results of training.
        # But I still keep them because the trained checkpoints have them.
        ###
        if self.max_vl_text_len is not None:
            vl_text_row_relative_position_index = torch.cat(
                (self.text_relative_position_index[:self.max_vl_text_len, :self.max_vl_text_len], text2imag_relative_position_index[:self.max_vl_text_len]), 
                1
            )
            vl_imag_row_relative_position_index = torch.cat(
                (imag2text_relative_position_index[:, :self.max_vl_text_len], self.relative_position_index), 
                1
            )
            vl_text_imag_relative_position_index = torch.cat((vl_text_row_relative_position_index, vl_imag_row_relative_position_index), 0)

            self.register_buffer("vl_text_imag_relative_position_index", vl_text_imag_relative_position_index)

            print("vl index shape: ", self.vl_text_imag_relative_position_index.shape)

        print(relative_position_index.max(), relative_position_index.min())
        print(text_relative_position_index.max(), text_relative_position_index.min())
        print(text_imag_relative_position_index.max(), text_imag_relative_position_index.min())

        print(relative_position_index.shape, text_relative_position_index.shape, text_imag_relative_position_index.shape)

        self.num_frames = self.hparams.config["num_frames"]

        if self.num_frames >= 1:
            text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len * self.num_frames) * (self.num_relative_distance)
            imag2text_relative_position_index = torch.ones(max_imag_len * self.num_frames, max_text_len) * (self.num_relative_distance + 1)

            video_relative_position_index = self.relative_position_index.repeat(self.num_frames, self.num_frames)

            text_row_relative_position_index = torch.cat((self.text_relative_position_index, text2imag_relative_position_index), 1) # (max_text_len, max_text_len + max_imag_len)
            imag_row_relative_position_index = torch.cat((imag2text_relative_position_index, video_relative_position_index), 1) # (max_text_len, max_text_len + max_imag_len)
            text_video_relative_position_index = torch.cat((text_row_relative_position_index, imag_row_relative_position_index), 0) # (max_text_len + max_imag_len, max_text_len + max_imag_len)

            self.register_buffer("video_relative_position_index", video_relative_position_index)
            self.register_buffer("text_video_relative_position_index", text_video_relative_position_index)
            
            self.temporal_relative_position_bias_table = nn.Parameter(
                torch.zeros(2 * self.num_frames, num_heads * num_layers))
            temporal_position_ids = torch.arange(self.num_frames)
            temporal_rel_pos_mat = temporal_position_ids.unsqueeze(-2) - temporal_position_ids.unsqueeze(-1)
            min_distance = temporal_rel_pos_mat.min()
            temporal_relative_position_index = temporal_rel_pos_mat - min_distance

            temporal_relative_position_index = temporal_relative_position_index.repeat(max_imag_len, max_imag_len)

            self.register_buffer("temporal_relative_position_index", temporal_relative_position_index)

            print(temporal_relative_position_index.shape)

            mask_for_combining_temporal = torch.eye(self.num_frames)
            mask_for_combining_temporal = mask_for_combining_temporal.repeat_interleave(max_imag_len, dim=1).repeat_interleave(max_imag_len, dim=0)
            mask_for_combining_temporal = mask_for_combining_temporal.unsqueeze(0)

            self.register_buffer("mask_for_combining_temporal", mask_for_combining_temporal)

            if self.max_vl_text_len is not None:
                vl_text_row_relative_position_index = torch.cat(
                    (self.text_relative_position_index[:self.max_vl_text_len, :self.max_vl_text_len], text2imag_relative_position_index[:self.max_vl_text_len]), 
                    1
                )
                vl_video_row_relative_position_index = torch.cat(
                    (imag2text_relative_position_index[:, :self.max_vl_text_len], self.video_relative_position_index), 
                    1
                )
                vl_text_video_relative_position_index = torch.cat((vl_text_row_relative_position_index, vl_video_row_relative_position_index), 0)

                self.register_buffer("vl_text_video_relative_position_index", vl_text_video_relative_position_index)

        ###
        # [Temporal embedding ends]
        ###

        # ===================== Downstream ===================== #

        # Load or merge weights for fine-tuning
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["validation_only"]
        ):  
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))
            if self.hparams.config["use_beit_weight"]:
                state_dict = self.modify_checkpoint_beit(ckpt)
            elif self.hparams.config["use_self_weight"]:
                state_dict = self.modify_checkpoint_self(ckpt)
            else:
                state_dict = self.modify_checkpoint_vlmo(ckpt)
  
            if self.hparams.config["merge_weights"]:
                state_dict = self.merge_weights(state_dict)
            
            elif self.hparams.config["sum_task_vectors"]:
                state_dict = self.sum_task_vectors(state_dict)

            elif self.hparams.config["regmean"]:
                state_dict = self.regmean(state_dict)

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            rank_zero_info("missing_keys: {}".format(missing_keys))
            rank_zero_info("unexpected_keys: {}".format(unexpected_keys))

        hs = self.hparams.config["hidden_size"]
        self.vlffn_start_layer_index = config["vlffn_start_layer_index"]
        self.num_layers = config["num_layers"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["img_cls"] > 0:
            vs = self.hparams.config["img_cls_label_size"]
            # self.img_cls_classifier = nn.Sequential(
            #     nn.Linear(hs, hs * 2),
            #     nn.LayerNorm(hs * 2),
            #     nn.GELU(),
            #     nn.Linear(hs * 2, vs),
            # )

            self.img_cls_classifier = nn.Linear(hs, vs)

            self.img_cls_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        # load or merge weight for evaluation
        if self.hparams.config["load_path"] != "" and (self.hparams.config["test_only"] or self.hparams.config["validation_only"]):
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")

            if self.hparams.config["use_beit_weight"]:
                state_dict = self.modify_checkpoint_beit(ckpt)
            elif self.hparams.config["use_self_weight"]:
                state_dict = self.modify_checkpoint_self(ckpt)
            else:
                state_dict = ckpt["state_dict"]

            if self.hparams.config["merge_weights"]:
                state_dict = self.merge_weights(state_dict)
            
            elif self.hparams.config["sum_task_vectors"]:
                state_dict = self.sum_task_vectors(state_dict)

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            rank_zero_info("missing_keys: {}".format(missing_keys))
            rank_zero_info("unexpected_keys: {}".format(unexpected_keys))

    def regmean(self, state_dict):
        # RegMean merging
        new_state_dict = {}

        for k, v in state_dict.items():
            # add weights that are not in transformer blocks
            if "transformer.blocks." not in k or "gamma" in k:
                new_state_dict[k] = v
                # print(k)

        layer_orders = [
            ["transformer.blocks.{}.attn.{}.qkv.weight", "transformer.blocks.{}.attn.qkv.weight"],
            ["transformer.blocks.{}.attn.{}.proj.{}", "transformer.blocks.{}.attn.proj.{}"],
            ["transformer.blocks.{}.attn.{}.{}", "transformer.blocks.{}.attn.{}"],
            ["transformer.blocks.{}.mlp.{}.fc1.{}", "transformer.blocks.{}.mlp.fc1.{}"],
            ["transformer.blocks.{}.mlp.{}.fc2.{}", "transformer.blocks.{}.mlp.fc2.{}"],
            ["transformer.blocks.{}.norm1.{}.{}", "transformer.blocks.{}.norm1.{}"],
            ["transformer.blocks.{}.norm2.{}.{}", "transformer.blocks.{}.norm2.{}"],
        ]

        gram_matrices = torch.load(self.hparams.config["gram_matrices"], map_location="cpu")

        def scale_G(G):
            scaling_for_non_diag = self.hparams.config["scaling_for_non_diag"]
            diag = torch.diag_embed(torch.diag(G))

            return scaling_for_non_diag * G + (1 - scaling_for_non_diag) * diag


        for i in range(12):

            if i < self.hparams.config["vlffn_start_layer_index"]:
                modalities = ["v", "l"]
            elif self.hparams.config["loss_names"]["irtr"] > 0:
                modalities = ["v", "l"]
            elif self.hparams.config["loss_names"]["vqa"] > 0:
                modalities = ["vl"]
            else:
                modalities = ["v", "l", "vl"]

            for layer in layer_orders:
                if "qkv" in layer[0]:
                    later_name = layer[1].format(i)

                    summed_gram = 0

                    later_weight = 0

                    for modality in modalities:
                        name = layer[0].format(i, modality)
                        gram_name = name.replace(".qkv.weight", "")

                        if name in state_dict:
                            if gram_name not in gram_matrices:
                                continue

                            G = scale_G(gram_matrices[gram_name])
                            summed_gram += G
                            later_weight += torch.matmul(state_dict[name].to(torch.float64), G)
                        else:
                            later_weight = state_dict[later_name]
                            break

                    if isinstance(summed_gram, int):
                        new_state_dict[later_name] = later_weight
                    else:
                        normalization_factor = torch.inverse(summed_gram)

                        new_state_dict[later_name] = torch.matmul(later_weight, normalization_factor)

                elif "attn" in layer[0] and "proj" not in layer[0]:
                    for n in ["q_bias", "v_bias"]:
                        later_name = layer[1].format(i, n)

                        later_weight = 0

                        num_valid_modality = 0

                        for modality in modalities:
                            name = layer[0].format(i, modality, n)
                            
                            if name in state_dict:
                                later_weight += state_dict[name]
                                num_valid_modality += 1
                            else:
                                later_weight = state_dict[later_name]
                                break

                        if num_valid_modality == 0:
                            new_state_dict[later_name] = later_weight
                        else:
                            new_state_dict[later_name] = later_weight / num_valid_modality

                elif "mlp" in layer[0] or "proj" in layer[0]:
                    n = "weight"
                    later_name = layer[1].format(i, n)
                    summed_gram = 0
                    later_weight = 0

                    for modality in modalities:
                        name = layer[0].format(i, modality, n)
                        gram_name = name.replace(".weight", "")

                        if name in state_dict:
                            if gram_name not in gram_matrices:
                                continue

                            G = scale_G(gram_matrices[gram_name])
                            summed_gram += G
                            later_weight += torch.matmul(state_dict[name].to(torch.float64), G)
                        else:
                            later_weight = state_dict[later_name]
                            break

                    if isinstance(summed_gram, int):
                        new_state_dict[later_name] = later_weight
                    else:
                        normalization_factor = torch.inverse(summed_gram)
                        new_state_dict[later_name] = torch.matmul(later_weight, normalization_factor)

                    # simple avg
                    n = "bias"
                    later_name = layer[1].format(i, n)
                    later_weight = 0

                    num_valid_modality = 0

                    for modality in modalities:
                        name = layer[0].format(i, modality, n)
                        
                        if name in state_dict:
                            later_weight += state_dict[name]
                            num_valid_modality += 1
                        else:
                            later_weight = state_dict[later_name]
                            break

                    if num_valid_modality == 0:
                        new_state_dict[later_name] = later_weight
                    else:
                        new_state_dict[later_name] = later_weight / num_valid_modality

                else:
                    # norm
                    for n in ["weight", "bias"]:
                        later_name = layer[1].format(i, n)
                        later_weight = 0

                        num_valid_modality = 0

                        for modality in modalities:
                            name = layer[0].format(i, modality, n)
                            
                            if name in state_dict:
                                later_weight += state_dict[name]
                                num_valid_modality += 1
                            else:
                                later_weight = state_dict[later_name]
                                break

                        if num_valid_modality == 0:
                            new_state_dict[later_name] = later_weight
                        else:
                            new_state_dict[later_name] = later_weight / num_valid_modality  

        return new_state_dict

    def merge_weights(self, state_dict):
        # interpolation merging
        new_state_dict = {}

        for k, v in state_dict.items():
            # add weights that are not in transformer blocks
            if "transformer.blocks." not in k or "gamma" in k:
                new_state_dict[k] = v
                # print(k)

        layer_orders = [
            ["transformer.blocks.{}.attn.{}.qkv.weight", "transformer.blocks.{}.attn.qkv.weight"],
            ["transformer.blocks.{}.attn.{}.proj.{}", "transformer.blocks.{}.attn.proj.{}"],
            ["transformer.blocks.{}.attn.{}.{}", "transformer.blocks.{}.attn.{}"],
            ["transformer.blocks.{}.mlp.{}.fc1.{}", "transformer.blocks.{}.mlp.fc1.{}"],
            ["transformer.blocks.{}.mlp.{}.fc2.{}", "transformer.blocks.{}.mlp.fc2.{}"],
            ["transformer.blocks.{}.norm1.{}.{}", "transformer.blocks.{}.norm1.{}"],
            ["transformer.blocks.{}.norm2.{}.{}", "transformer.blocks.{}.norm2.{}"],
        ]

        for i in range(12):

            modalities = None

            if i < self.hparams.config["vlffn_start_layer_index"]:
                modalities = ["v", "l"]
            elif self.hparams.config["only_activate_used_experts"]:
                if self.hparams.config["loss_names"]["irtr"] > 0:
                    modalities = ["v", "l"]
                elif self.hparams.config["loss_names"]["vqa"] > 0:
                    modalities = ["vl"]
            else:
                modalities = ["v", "l", "vl"]

            if len(modalities) == 1:
                ratios = {
                    modalities[0]: 1,
                }

            elif len(modalities) == 3:
                ratios = {
                    "v": (2 / 3) * self.hparams.config["merge_ratio"],
                    "l": (2 / 3) * (1 - self.hparams.config["merge_ratio"]),
                    "vl": 1 / 3
                }
            else:
                ratios = {
                    "v": self.hparams.config["merge_ratio"],
                    "l": (1 - self.hparams.config["merge_ratio"]),
                }

            for layer in layer_orders:
                if "qkv" in layer[0]:
                    later_name = layer[1].format(i)

                    later_weight = 0

                    for modality in modalities:
                        name = layer[0].format(i, modality)

                        if name in state_dict:
                            later_weight += ratios[modality] * state_dict[name]
                        else:
                            later_weight = state_dict[later_name]
                            break

                    new_state_dict[later_name] = later_weight
                    
                elif "attn" in layer[0] and "proj" not in layer[0]:
                    for n in ["q_bias", "v_bias"]:
                        later_name = layer[1].format(i, n)

                        later_weight = 0

                        for modality in modalities:
                            name = layer[0].format(i, modality, n)
                            
                            if name in state_dict:
                                later_weight += ratios[modality] * state_dict[name]
                            else:
                                later_weight = state_dict[later_name]
                                break

                        new_state_dict[later_name] = later_weight
                        
                else:
                    for n in ["weight", "bias"]:
                        later_name = layer[1].format(i, n)

                        later_weight = 0

                        for modality in modalities:
                            name = layer[0].format(i, modality, n)
                            
                            if name in state_dict:
                                later_weight += ratios[modality] * state_dict[name]
                            else:
                                later_weight = state_dict[later_name]
                                break

                        new_state_dict[later_name] = later_weight
                        

        return new_state_dict

    def sum_task_vectors(self, state_dict):
        # modality arithmetic
        new_state_dict = {}

        for k, v in state_dict.items():
            # add weights that are not in transformer blocks
            if "transformer.blocks." not in k or "gamma" in k:
                new_state_dict[k] = v
                # print(k)

        layer_orders = [
            ["transformer.blocks.{}.attn.{}.qkv.weight", "transformer.blocks.{}.attn.qkv.weight"],
            ["transformer.blocks.{}.attn.{}.proj.{}", "transformer.blocks.{}.attn.proj.{}"],
            ["transformer.blocks.{}.attn.{}.{}", "transformer.blocks.{}.attn.{}"],
            ["transformer.blocks.{}.mlp.{}.fc1.{}", "transformer.blocks.{}.mlp.fc1.{}"],
            ["transformer.blocks.{}.mlp.{}.fc2.{}", "transformer.blocks.{}.mlp.fc2.{}"],
            ["transformer.blocks.{}.norm1.{}.{}", "transformer.blocks.{}.norm1.{}"],
            ["transformer.blocks.{}.norm2.{}.{}", "transformer.blocks.{}.norm2.{}"],
        ]

        central_weight = torch.load(self.hparams.config["central_weight"], map_location="cpu")

        if "state_dict" in central_weight:
            central_weight = central_weight["state_dict"]

        for i in range(12):
            modalities = None

            if i < self.hparams.config["vlffn_start_layer_index"]:
                modalities = ["v", "l"]
            elif self.hparams.config["only_activate_used_experts"]:
                if self.hparams.config["loss_names"]["irtr"] > 0:
                    modalities = ["v", "l"]
                elif self.hparams.config["loss_names"]["vqa"] > 0:
                    modalities = ["vl"]
            else:
                modalities = ["v", "l", "vl"]

            if len(modalities) == 1:
                ratios = {
                    modalities[0]: 1,
                }
            elif len(modalities) == 3:
                ratios = {
                    "v": self.hparams.config["sum_lambda"],
                    "l": self.hparams.config["sum_lambda"],
                    "vl": self.hparams.config["sum_lambda"]
                }
            else:
                ratios = {
                    "v": self.hparams.config["sum_lambda"],
                    "l": self.hparams.config["sum_lambda"],
                }

            for layer in layer_orders:
                if "qkv" in layer[0]:
                    later_name = layer[1].format(i)

                    later_weight = central_weight[later_name]

                    for modality in modalities:
                        name = layer[0].format(i, modality)

                        if name in state_dict:
                            later_weight += ratios[modality] * (state_dict[name] - central_weight[later_name])
                        else:
                            later_weight = state_dict[later_name]
                            break

                    new_state_dict[later_name] = later_weight

                elif "attn" in layer[0] and "proj" not in layer[0]:
                    for n in ["q_bias", "v_bias"]:
                        later_name = layer[1].format(i, n)

                        later_weight = central_weight[later_name]

                        for modality in modalities:
                            name = layer[0].format(i, modality, n)
                            
                            if name in state_dict:
                                later_weight += ratios[modality] * (state_dict[name] - central_weight[later_name])
                            else:
                                later_weight = state_dict[later_name]
                                break

                        new_state_dict[later_name] = later_weight
                else:
                    for n in ["weight", "bias"]:
                        later_name = layer[1].format(i, n)

                        later_weight = central_weight[later_name]

                        for modality in modalities:
                            name = layer[0].format(i, modality, n)
                            
                            if name in state_dict:
                                later_weight += ratios[modality] * (state_dict[name] - central_weight[later_name])
                            else:
                                later_weight = state_dict[later_name]
                                break

                        new_state_dict[later_name] = later_weight

        return new_state_dict


    def modify_checkpoint_vlmo(self, ckpt):
        print("VL Checkpoint Init")
        if "state_dict" in ckpt:
            rank_zero_info("state_dict in ckpt")
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if state_dict["text_embeddings.position_embeddings.weight"].size(0) != self.max_text_len:
            state_dict["text_embeddings.position_embeddings.weight"].data = state_dict["text_embeddings.position_embeddings.weight"].data[:self.max_text_len, :]
            state_dict["text_embeddings.position_ids"].data = state_dict["text_embeddings.position_ids"].data[:, :self.max_text_len]
            rank_zero_info("text position_embeddings size: {}".format(state_dict["text_embeddings.position_embeddings.weight"].size()))
            state_dict.pop("relative_position_index")
            state_dict.pop("text_relative_position_index")
            state_dict.pop("text_imag_relative_position_index")

        rel_pos_bias = state_dict["relative_position_bias_table"]
        src_num_pos, num_attn_heads = rel_pos_bias.size()
        dst_num_pos, _ = self.relative_position_bias_table.size()
        dst_patch_shape = self.transformer.patch_embed.patch_shape

        print(rel_pos_bias.shape, self.relative_position_bias_table.shape, dst_patch_shape)
        if dst_patch_shape[0] != dst_patch_shape[1]:
            raise NotImplementedError()

        non_image_tokens_num = self.text_num_relative_distance + 2 + 3 # num of text tokens + extra_tokens for text + extra_tokens for image

        src_size = int((src_num_pos - non_image_tokens_num) ** 0.5) # get the (2 * Wh - 1)
        dst_size = int((dst_num_pos - non_image_tokens_num) ** 0.5)

        pop_keys = ["relative_position_index", "text_relative_position_index", "text_imag_relative_position_index",
            "video_relative_position_index", "text_video_relative_position_index", "temporal_relative_position_index", 
            "mask_for_combining_temporal"]

        for p in pop_keys:
            if p in state_dict:
                state_dict.pop(p)

        ## Assume the max_text_len of the checkpoint is the same as we are using
        if src_size != dst_size:
            
            rank_zero_info("Position interpolate from %dx%d to %dx%d" % (
                src_size, src_size, dst_size, dst_size))
            extra_tokens = rel_pos_bias[-non_image_tokens_num:, :]
            rel_pos_bias = rel_pos_bias[:-non_image_tokens_num, :]

            all_rel_pos_bias = []

            embed = rel_pos_bias.transpose(0, 1).view(-1, src_size, src_size)
            embed = torch.nn.functional.interpolate(embed.unsqueeze(0), size=(dst_size, dst_size), mode='bicubic')
            embed = embed.squeeze(0).permute((1, 2, 0))
            embed = embed.contiguous().view(-1, embed.size(-1))

            new_rel_pos_bias = torch.cat((embed, extra_tokens), dim=0)
            print(new_rel_pos_bias.shape)
            state_dict["relative_position_bias_table"] = new_rel_pos_bias

        return state_dict

    def modify_checkpoint_beit(self, ckpt):
        print("BEIT Checkpoint Init")
        if "state_dict" in ckpt:
            rank_zero_info("state_dict in ckpt")
            state_dict = ckpt["state_dict"]

            # print(state_dict.keys())

            is_beit_pt_weight = "transformer.rel_pos_bias.relative_position_bias_table" in state_dict # beit_base_patch16_224_pt22k_ft22k.ckpt

            is_beit_pt_ft_weight = "transformer.blocks.0.attn.relative_position_bias_table" in state_dict # beit_base_patch16_224_pt22k.ckpt

            if is_beit_pt_weight or is_beit_pt_ft_weight:

                if is_beit_pt_weight: 
                    # rel_pos_bias.shape is (768, 12)
                    rel_pos_bias = state_dict.pop("transformer.rel_pos_bias.relative_position_bias_table")
                    state_dict.pop("transformer.rel_pos_bias.relative_position_index")
                else:
                    rel_pos_bias_list = []
                    # each bias in rel_pos_bias_list has shape (768, 12)
                    for i in range(self.hparams.config["num_layers"]):
                        rel_pos_bias_list.append(state_dict.pop(f"transformer.blocks.{i}.attn.relative_position_bias_table"))
                        state_dict.pop(f"transformer.blocks.{i}.attn.relative_position_index")

                    rel_pos_bias = torch.cat(rel_pos_bias_list, dim=-1)
                    print(rel_pos_bias_list[0].shape, "rel")
                    print(self.hparams.config["num_layers"])
                    # rel_pos_bias.shape is (768, 12 * 12)

                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, num_layers_heads = self.relative_position_bias_table.size()
                dst_patch_shape = self.transformer.patch_embed.patch_shape

                print(rel_pos_bias.shape, self.relative_position_bias_table.shape, dst_patch_shape)
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()

                non_image_tokens_num = self.text_num_relative_distance + 2 + 3 # num of text tokens + extra_tokens for text + extra_tokens for image

                src_size = int((src_num_pos - 3) ** 0.5) # get the (2 * Wh - 1), but no text tokens and extra_tokens for text in beit checkpoint
                dst_size = int((dst_num_pos - non_image_tokens_num) ** 0.5)

                ## Assume the max_text_len of the checkpoint is the same as we are using
                
                rank_zero_info("Position interpolate from %dx%d to %dx%d" % (
                    src_size, src_size, dst_size, dst_size))
                extra_tokens = self.relative_position_bias_table[-(non_image_tokens_num - 3):, :] # text and extra tokens for text don't exist in the beit checkpoint, so use random initialized parameters.

                extra_tokens_for_img = rel_pos_bias[-3:, :]
                rel_pos_bias = rel_pos_bias[:-3, :] # image part

                # rel_pos_bias in beit checkpoint shares the values across layer
                # but in this code, the relative_position_bias_table independent values for each layer, 
                # so we use initialize same value for the heads in the same layer with beit checckpoint
                embed = rel_pos_bias.transpose(0, 1).view(-1, src_size, src_size)
                embed = torch.nn.functional.interpolate(embed.unsqueeze(0), size=(dst_size, dst_size), mode='bicubic')
                embed = embed.squeeze(0).permute((1, 2, 0))
                embed = embed.contiguous().view(-1, embed.size(-1))

                num_layers = num_layers_heads // num_attn_heads

                print(num_layers)

                print(embed.shape, extra_tokens_for_img.shape, extra_tokens.shape)

                if is_beit_pt_weight: 
                    embed = embed.repeat((1, num_layers))
                    extra_tokens_for_img = extra_tokens_for_img.repeat((1, num_layers))

                # bias for (resized image patches, extra token for img, text tokens, extra_token for text)
                new_rel_pos_bias = torch.cat((embed, extra_tokens_for_img, extra_tokens), dim=0)
                print(new_rel_pos_bias.shape)
                state_dict["relative_position_bias_table"] = new_rel_pos_bias

            # modify the name of some weight for loading checkpoint correctly
            if self.hparams.config["use_moe"]:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if self.moe_config.in_ffn and "mlp" in k:
                        k_list = k.split(".")
                        k_list.insert(-2, "v")
                        new_k = ".".join(k_list)
                        # print(new_k)
                    elif self.moe_config.in_attn and "attn" in k:

                        k_list = k.split(".")

                        if "attn.q_bias" in k or "attn.v_bias" in k:
                            k_list.insert(-1, "v")
                        else:
                            k_list.insert(-2, "v")
                        new_k = ".".join(k_list)
                    else:
                        new_k = k

                    new_state_dict[new_k] = v

                state_dict = new_state_dict

            if self.hparams.config["use_custom_ln_attn"]:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if ".norm1" in k:
                        # avoid to overwrite name by two times for moe cases
                        k_list = k.split(".")
                        k_list.insert(-1, "v")
                        new_k = ".".join(k_list)
                        # print(new_k)
                    else:
                        new_k = k

                    new_state_dict[new_k] = v

                state_dict = new_state_dict

            if self.hparams.config["use_custom_ln_ffn"]:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if ".norm2" in k:
                        # avoid to overwrite name by two times for moe cases
                        k_list = k.split(".")
                        k_list.insert(-1, "v")
                        new_k = ".".join(k_list)
                        # print(new_k)
                    else:
                        new_k = k

                    new_state_dict[new_k] = v

                state_dict = new_state_dict

            if self.hparams.config["use_vision_weights_for_other_modalities"]:
                new_state_dict = {}

                for k, v in state_dict.items():
                    if ".v." in k:
                        # avoid to overwrite name by two times for moe cases
                        
                        # add language modality
                        l_v = k.replace(".v.", ".l.")
                        new_state_dict[l_v] = v

                        # add vl modality

                        layer_idx = int(k.split(".")[2])

                        if layer_idx >= self.hparams.config["vlffn_start_layer_index"]:
                            l_v = k.replace(".v.", ".vl.")
                            new_state_dict[l_v] = v

                    new_state_dict[k] = v

                state_dict = new_state_dict

            if "transformer.fc_norm.weight" in state_dict:
                state_dict["transformer.norm.weight"] = state_dict["transformer.fc_norm.weight"]
                state_dict["transformer.norm.bias"] = state_dict["transformer.fc_norm.bias"]

                del state_dict["transformer.fc_norm.weight"]
                del state_dict["transformer.fc_norm.bias"]

            return state_dict

        return None

    def modify_checkpoint_self(self, ckpt):
        print("Self Checkpoint Init")
        rank_zero_info("state_dict in ckpt")
        state_dict = ckpt

        if state_dict["text_embeddings.position_embeddings.weight"].size(0) != self.max_text_len:
            state_dict["text_embeddings.position_embeddings.weight"].data = state_dict["text_embeddings.position_embeddings.weight"].data[:self.max_text_len, :]
            state_dict["text_embeddings.position_ids"].data = state_dict["text_embeddings.position_ids"].data[:, :self.max_text_len]
            rank_zero_info("text position_embeddings size: {}".format(state_dict["text_embeddings.position_embeddings.weight"].size()))


        is_beit_pt_weight = "transformer.rel_pos_bias.relative_position_bias_table" in state_dict # beit_base_patch16_224_pt22k_ft22k.ckpt

        is_beit_pt_ft_weight = "transformer.blocks.0.attn.relative_position_bias_table" in state_dict # beit_base_patch16_224_pt22k.ckpt

        if is_beit_pt_weight or is_beit_pt_ft_weight:

            if is_beit_pt_weight: 
                # rel_pos_bias.shape is (768, 12)
                rel_pos_bias = state_dict.pop("transformer.rel_pos_bias.relative_position_bias_table")
                state_dict.pop("transformer.rel_pos_bias.relative_position_index")
            else:
                rel_pos_bias_list = []
                # each bias in rel_pos_bias_list has shape (768, 12)
                for i in range(self.hparams.config["num_layers"]):
                    rel_pos_bias_list.append(state_dict.pop(f"transformer.blocks.{i}.attn.relative_position_bias_table"))
                    state_dict.pop(f"transformer.blocks.{i}.attn.relative_position_index")

                rel_pos_bias = torch.cat(rel_pos_bias_list, dim=-1)
                print(rel_pos_bias_list[0].shape, "rel")
                print(self.hparams.config["num_layers"])
                # rel_pos_bias.shape is (768, 12 * 12)

            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, num_layers_heads = self.relative_position_bias_table.size()
            dst_patch_shape = self.transformer.patch_embed.patch_shape

            print(rel_pos_bias.shape, self.relative_position_bias_table.shape, dst_patch_shape)
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()

            non_image_tokens_num = self.text_num_relative_distance + 2 + 3 # num of text tokens + extra_tokens for text + extra_tokens for image

            src_size = int((src_num_pos - 3) ** 0.5) # get the (2 * Wh - 1), but no text tokens and extra_tokens for text in beit checkpoint
            dst_size = int((dst_num_pos - non_image_tokens_num) ** 0.5)

            ## Assume the max_text_len of the checkpoint is the same as we are using
            
            rank_zero_info("Position interpolate from %dx%d to %dx%d" % (
                src_size, src_size, dst_size, dst_size))
            extra_tokens = self.relative_position_bias_table[-(non_image_tokens_num - 3):, :] # text and extra tokens for text don't exist in the beit checkpoint, so use random initialized parameters.

            extra_tokens_for_img = rel_pos_bias[-3:, :]
            rel_pos_bias = rel_pos_bias[:-3, :] # image part

            # rel_pos_bias in beit checkpoint shares the values across layer
            # but in this code, the relative_position_bias_table independent values for each layer, 
            # so we use initialize same value for the heads in the same layer with beit checckpoint
            embed = rel_pos_bias.transpose(0, 1).view(-1, src_size, src_size)
            embed = torch.nn.functional.interpolate(embed.unsqueeze(0), size=(dst_size, dst_size), mode='bicubic')
            embed = embed.squeeze(0).permute((1, 2, 0))
            embed = embed.contiguous().view(-1, embed.size(-1))

            num_layers = num_layers_heads // num_attn_heads

            print(num_layers)

            print(embed.shape, extra_tokens_for_img.shape, extra_tokens.shape)

            if is_beit_pt_weight: 
                embed = embed.repeat((1, num_layers))
                extra_tokens_for_img = extra_tokens_for_img.repeat((1, num_layers))

            # bias for (resized image patches, extra token for img, text tokens, extra_token for text)
            new_rel_pos_bias = torch.cat((embed, extra_tokens_for_img, extra_tokens), dim=0)
            print(new_rel_pos_bias.shape)
            state_dict["relative_position_bias_table"] = new_rel_pos_bias

        if "transformer.fc_norm.weight" in state_dict:
            state_dict["transformer.norm.weight"] = state_dict["transformer.fc_norm.weight"]
            state_dict["transformer.norm.bias"] = state_dict["transformer.fc_norm.bias"]

            del state_dict["transformer.fc_norm.weight"]
            del state_dict["transformer.fc_norm.bias"]

        return state_dict

    def get_rel_pos_bias(self, relative_position_index):
        relative_position_bias = F.embedding(relative_position_index.long(), self.relative_position_bias_table)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, x, y
        return relative_position_bias

    def get_temporal_rel_pos_bias(self, temporal_relative_position_index):
        temporal_relative_position_bias = F.embedding(temporal_relative_position_index.long(), self.temporal_relative_position_bias_table)
        temporal_relative_position_bias = temporal_relative_position_bias.permute(2, 0, 1).contiguous() # nH, x, y
        return temporal_relative_position_bias

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        bool_masked_pos=None,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
                bool_masked_pos=bool_masked_pos,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        image_masks = image_masks.type_as(text_masks)
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        # handle the text length in VL tasks is shorter than which in the NLP tasks when doing mixed single and multi-task training
        position_index = self.vl_text_imag_relative_position_index if self.max_vl_text_len is not None else self.text_imag_relative_position_index

        all_relative_position_bias = self.get_rel_pos_bias(position_index)
        relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks, type_id=2, relative_position_bias=relative_position_bias_list[i])

        x = self.transformer.norm(x)

        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )

        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "image": img,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret


    def infer_text(
        self,
        batch,
        mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []

        if self.max_vl_text_len is not None:
            # handle the text length in VL tasks is shorter than which in the NLP tasks when doing mixed single and multi-task training
            true_length = text_ids.shape[1] # sometimes the input is pure language, sometimes is VL, so the true length depends on the input
            all_relative_position_bias = self.get_rel_pos_bias(self.text_relative_position_index[:true_length, :true_length])
        else:
            all_relative_position_bias = self.get_rel_pos_bias(self.text_relative_position_index)

        relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks, type_id=1, relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)
        
        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index-1]
        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens, _ = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, type_id=2, relative_position_bias=relative_position_bias_list[vlffn_index])

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        cls_feats = self.ifm_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
        cls_vlffn_feats = self.ifm_vl_text_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": None,
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": None,
        }

        return ret


    def infer_text_ft(
        self,
        batch,
        mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []

        if self.max_vl_text_len is not None:
            # handle the text length in VL tasks is shorter than which in the NLP tasks when doing mixed single and multi-task training
            true_length = text_ids.shape[1] # sometimes the input is pure language, sometimes is VL, so the true length depends on the input
            all_relative_position_bias = self.get_rel_pos_bias(self.text_relative_position_index[:true_length, :true_length])
        else:
            all_relative_position_bias = self.get_rel_pos_bias(self.text_relative_position_index)

        relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks, type_id=1, relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        if getattr(self, "ifm_text_proj", None) is not None:
            cls_feats = self.ifm_text_proj(lffn_hiddens[:, 0])
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
        else:
            cls_feats = None

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_labels": None,
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": None,
        }

        return ret

    def infer_image(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        bool_masked_pos=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        text_masks = batch[f"text_masks"]

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
                bool_masked_pos=bool_masked_pos,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        image_masks = image_masks.type_as(text_masks)
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        all_hidden_states = []

        all_relative_position_bias = self.get_rel_pos_bias(self.relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks, type_id=0, relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)
        
        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index-1]
        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens, _ = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, type_id=2, relative_position_bias=relative_position_bias_list[vlffn_index])
        
        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)

        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )
        # cls_feats = self.pooler(x)
        cls_feats = self.ifm_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
        cls_vlffn_feats = self.ifm_vl_image_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret


    def infer_image_ft(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        bool_masked_pos=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        text_masks = batch[f"text_masks"]

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
                bool_masked_pos=bool_masked_pos,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        image_masks = image_masks.type_as(text_masks)
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        all_hidden_states = []

        all_relative_position_bias = self.get_rel_pos_bias(self.relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks, type_id=0, relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)

        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )
        # cls_feats = self.pooler(x)

        # For image only MIM
        if getattr(self, "ifm_image_proj", None) is not None:
            cls_feats = self.ifm_image_proj(vffn_hiddens[:, 0])
            cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
        else:
            # cls_feats = None
            cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret


    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # mixed of single-modal and multi-modal training
        if self.hparams.config["tasks"] is not None:
            if "v" in batch:
                # Image only
                if "image_only_mim" in self.current_tasks:
                    ret.update(objectives.compute_mim_image_only(self, batch["v"]))

            if "l" in batch:
                # Language only
                if "text_only_mlm" in self.current_tasks:
                    ret.update(objectives.compute_mlm_text_only(self, batch["l"]))

            if "vl" in batch:
                batch = batch["vl"]
            else:
                # don't do the VL task loss
                return ret

        ## For VL tasks
        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Image Modeling
        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        # Contrastive loss
        if "ifm" in self.current_tasks:
            ret.update(objectives.compute_ifm(self, batch))

        # Contrastive loss during ft
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_hardneg(self, batch, ret["ifm_i2t_logits"], ret["ifm_t2i_logits"]))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        if "img_cls" in self.current_tasks:
            ret.update(objectives.compute_img_cls(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
        os.environ["WDS_EPOCH"] = str(self.current_epoch+1)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, self.hparams.config["log_dir"])
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
