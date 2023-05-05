import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from pytorch_lightning.utilities.distributed import rank_zero_info

from vilt.modules.dist_utils import all_gather


def compute_mim(pl_module, batch):
    bool_masked_pos = batch["image_masked_pos"][0].flatten(1).to(torch.bool)
    image_target = batch["image_target"][0]
    infer = pl_module.infer(batch, mask_text=False, mask_image=True, bool_masked_pos=bool_masked_pos)
    image_feats = infer["image_feats"][:, 1:] # get the image tokens without cls token
    image_feats = image_feats[bool_masked_pos]
    mim_logits = pl_module.mim_score(image_feats) # only return the logits of masked tokens
    
    # get the labels from dVAE
    with torch.no_grad():
        input_ids = pl_module.d_vae.get_codebook_indices(image_target).flatten(1)
        mim_labels = input_ids[bool_masked_pos]

    mim_loss = F.cross_entropy(
        mim_logits.view(-1, pl_module.d_vae.decoder.vocab_size),
        mim_labels.view(-1),
    )

    ret = {
        "mim_loss": mim_loss,
        "mim_logits": mim_logits,
        "mim_labels": mim_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mim_loss")(ret["mim_loss"])
    acc = getattr(pl_module, f"{phase}_mim_accuracy")(
        ret["mim_logits"], ret["mim_labels"]
    )
    pl_module.log(f"mim/{phase}/loss", loss)
    pl_module.log(f"mim/{phase}/accuracy", acc)

    return ret


def compute_mim_image_only(pl_module, batch):
    bool_masked_pos = batch["image_masked_pos"][0].flatten(1).to(torch.bool)
    image_target = batch["image_target"][0]
    infer = pl_module.infer_image_ft(batch, mask_image=True, bool_masked_pos=bool_masked_pos)
    image_feats = infer["image_feats"][:, 1:] # get the image tokens without cls token
    image_feats = image_feats[bool_masked_pos]
    mim_logits = pl_module.mim_score(image_feats) # only return the logits of masked tokens
    
    # get the labels from dVAE
    with torch.no_grad():
        input_ids = pl_module.d_vae.get_codebook_indices(image_target).flatten(1)
        mim_labels = input_ids[bool_masked_pos]

    mim_loss = F.cross_entropy(
        mim_logits.view(-1, pl_module.d_vae.decoder.vocab_size),
        mim_labels.view(-1),
    )

    ret = {
        "image_only_mim_loss": mim_loss,
        "image_only_mim_logits": mim_logits,
        "image_only_mim_labels": mim_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_image_only_mim_loss")(ret["image_only_mim_loss"])
    acc = getattr(pl_module, f"{phase}_image_only_mim_accuracy")(
        ret["image_only_mim_logits"], ret["image_only_mim_labels"]
    )
    pl_module.log(f"image_only_mim/{phase}/loss", loss)
    pl_module.log(f"image_only_mim/{phase}/accuracy", acc)

    return ret


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss * pl_module.hparams.config["vl_mlm_weight"],
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_mlm_text_only(pl_module, batch):
    infer = pl_module.infer_text_ft(batch, mask_text=True)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "text_only_mlm_loss": mlm_loss,
        "text_only_mlm_logits": mlm_logits,
        "text_only_mlm_labels": mlm_labels,
        "text_only_mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_text_only_mlm_loss")(ret["text_only_mlm_loss"])
    acc = getattr(pl_module, f"{phase}_text_only_mlm_accuracy")(
        ret["text_only_mlm_logits"], ret["text_only_mlm_labels"]
    )
    pl_module.log(f"text_only_mlm/{phase}/loss", loss)
    pl_module.log(f"text_only_mlm/{phase}/accuracy", acc)

    return ret


def compute_itm_hardneg(pl_module, batch, sim_i2t, sim_t2i):
    pos_len = batch["text_ids_mlm"].size(0) #len(batch["text"])
    neg_len = batch["text_ids_mlm"].size(0) #len(batch["text"])
    bsz = batch["text_ids_mlm"].size(0) #len(batch["text"])
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )

    batch = {k: v for k, v in batch.items()}
    infer_pos = pl_module.infer(batch, mask_text=False, mask_image=False)

    batch_text_ids = infer_pos["text_ids"]
    batch_text_masks = infer_pos["text_masks"]
    batch_image = infer_pos["image"]

    with torch.no_grad():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more hard negative candidates.
        gathered_text_ids = [
            torch.zeros_like(batch_text_ids) for _ in range(world_size)
        ]
        gathered_text_masks = [
            torch.zeros_like(batch_text_masks) for _ in range(world_size)
        ]
        gathered_image = [
            torch.zeros_like(batch_image) for _ in range(world_size)
        ]

        dist.all_gather(gathered_text_ids, batch_text_ids)
        dist.all_gather(gathered_text_masks, batch_text_masks)
        dist.all_gather(gathered_image, batch_image)

        all_text_ids = torch.cat(
            [batch_text_ids]
            + gathered_text_ids[:rank]
            + gathered_text_ids[rank + 1 :]
        )
        all_text_masks = torch.cat(
            [batch_text_masks]
            + gathered_text_masks[:rank]
            + gathered_text_masks[rank + 1 :]
        )
        all_image = torch.cat(
            [batch_image]
            + gathered_image[:rank]
            + gathered_image[rank + 1 :]
        )

    with torch.no_grad():       
        weights_i2t = F.softmax(sim_i2t[:bsz, :], dim=1)
        weights_t2i = F.softmax(sim_t2i[:bsz, :], dim=1)

        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)
    
    images_neg = []    
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        images_neg.append(all_image[neg_idx])
    images_neg = torch.stack(images_neg, dim=0)   

    # select a negative text for each image
    text_ids_neg = []
    text_masks_neg = []
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_ids_neg.append(all_text_ids[neg_idx])
        text_masks_neg.append(all_text_masks[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)     
    text_masks_neg = torch.stack(text_masks_neg, dim=0)      

    batch_imgs_neg = {"image":[images_neg], "text_ids":batch["text_ids"], "text_labels":batch["text_labels"], "text_masks":batch["text_masks"]}
    infer_imags_neg = pl_module.infer(batch_imgs_neg, mask_text=False, mask_image=False)
    
    batch_text_neg = {"image":batch["image"], "text_ids":text_ids_neg, "text_labels":batch["text_labels"], "text_masks":text_masks_neg}
    infer_text_neg = pl_module.infer(batch_text_neg, mask_text=False, mask_image=False)

    all_cls_feats = torch.cat([infer_pos["cls_feats"], infer_imags_neg["cls_feats"], infer_text_neg["cls_feats"]], dim=0)

    itm_logits = pl_module.itm_score(all_cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_ifm(pl_module, batch, aggregate=True):
    # pl_module.logit_scale.data = torch.clamp(pl_module.logit_scale.data, 0, 4.6052)
    # pl_module.logit_vl_scale.data = torch.clamp(pl_module.logit_vl_scale.data, 0, 4.6052)

    infer_imag = pl_module.infer_image(batch, mask_image=False)
    infer_text = pl_module.infer_text(batch, mask_text=False)

    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    logit_scale = pl_module.logit_scale.exp().mean()

    image_vlffn_features = infer_imag["cls_vlffn_feats"]
    text_vlffn_features = infer_text["cls_vlffn_feats"]
    logit_vl_scale = pl_module.logit_vl_scale.exp().mean()

    if aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_vlffn_features = [
            torch.zeros_like(image_vlffn_features) for _ in range(world_size)
        ]
        gathered_text_vlffn_features = [
            torch.zeros_like(text_vlffn_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_vlffn_features, image_vlffn_features)
        dist.all_gather(gathered_text_vlffn_features, text_vlffn_features)

        all_image_vlffn_features = torch.cat(
            [image_vlffn_features]
            + gathered_image_vlffn_features[:rank]
            + gathered_image_vlffn_features[rank + 1 :]
        )
        all_text_vlffn_features = torch.cat(
            [text_vlffn_features]
            + gathered_text_vlffn_features[:rank]
            + gathered_text_vlffn_features[rank + 1 :]
        )
        # this is needed to send gradients back everywhere.
        logits_per_vlffn_image = logit_vl_scale * all_image_vlffn_features @ all_text_vlffn_features.t()
        logits_per_vlffn_text = logits_per_vlffn_image.t()

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())
    
    ifm_loss = (
        F.cross_entropy(logits_per_image.float(), ground_truth)
        + F.cross_entropy(logits_per_text.float(), ground_truth)
    ) / 2

    ifm_vlffn_loss = (
        F.cross_entropy(logits_per_vlffn_image.float(), ground_truth)
        + F.cross_entropy(logits_per_vlffn_text.float(), ground_truth)
    ) / 2

    ifm_total_loss = (pl_module.hparams.config["ifm_weight"] * ifm_loss + ifm_vlffn_loss) * 0.5

    ret = {
        "ifm_loss": ifm_total_loss,
        "ifm_i2t_logits": logits_per_image,
        "ifm_t2i_logits": logits_per_text,
        "ifm_labels": ground_truth,
        "ifm_logit_scale": logit_scale,
        "ifm_logit_vl_scale": logit_vl_scale,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ifm_loss")(ret["ifm_loss"])
    scale = getattr(pl_module, f"{phase}_ifm_logit_scale")(ret["ifm_logit_scale"])
    i2t_acc = getattr(pl_module, f"{phase}_ifm_i2t_accuracy")(
        ret["ifm_i2t_logits"], ret["ifm_labels"]
    )
    t2i_acc = getattr(pl_module, f"{phase}_ifm_t2i_accuracy")(
        ret["ifm_t2i_logits"], ret["ifm_labels"]
    )
    pl_module.log(f"ifm/{phase}/loss", loss)
    pl_module.log(f"ifm/{phase}/logit_scale", scale)
    pl_module.log(f"ifm/{phase}/i2t_accuracy", i2t_acc)
    pl_module.log(f"ifm/{phase}/t2i_accuracy", t2i_acc)

    # vl_loss = getattr(pl_module, f"{phase}_ifm_vl_loss")(ret["ifm_loss"])
    vl_scale = getattr(pl_module, f"{phase}_ifm_vl_logit_scale")(ret["ifm_logit_vl_scale"])
    vl_i2t_acc = getattr(pl_module, f"{phase}_ifm_vl_i2t_accuracy")(
        logits_per_vlffn_image, ret["ifm_labels"]
    )
    vl_t2i_acc = getattr(pl_module, f"{phase}_ifm_vl_t2i_accuracy")(
        logits_per_vlffn_text, ret["ifm_labels"]
    )
    pl_module.log(f"ifm/{phase}/vl_logit_scale", vl_scale)
    pl_module.log(f"ifm/{phase}/vl_i2t_accuracy", vl_i2t_acc)
    pl_module.log(f"ifm/{phase}/vl_t2i_accuracy", vl_t2i_acc)

    return ret


def compute_irtr(pl_module, batch, aggregate=True):
    # pl_module.logit_scale.data = torch.clamp(pl_module.logit_scale.data, 0, 4.6052)

    infer_imag = pl_module.infer_image_ft(batch, mask_image=False)
    infer_text = pl_module.infer_text_ft(batch, mask_text=False)

    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    logit_scale = pl_module.logit_scale.exp().mean()

    if aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())

    irtr_loss = (
        F.cross_entropy(logits_per_image.float(), ground_truth)
        + F.cross_entropy(logits_per_text.float(), ground_truth)
    ) / 2

    ret = {
        "irtr_loss": irtr_loss,
        "irtr_i2t_logits": logits_per_image,
        "irtr_t2i_logits": logits_per_text,
        "irtr_labels": ground_truth,
        "irtr_logit_scale": logit_scale,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    scale = getattr(pl_module, f"{phase}_irtr_logit_scale")(ret["irtr_logit_scale"])
    i2t_acc = getattr(pl_module, f"{phase}_irtr_i2t_accuracy")(
        ret["irtr_i2t_logits"], ret["irtr_labels"]
    )
    t2i_acc = getattr(pl_module, f"{phase}_irtr_t2i_accuracy")(
        ret["irtr_t2i_logits"], ret["irtr_labels"]
    )
    pl_module.log(f"irtr/{phase}/loss", loss)
    pl_module.log(f"irtr/{phase}/logit_scale", scale)
    pl_module.log(f"irtr/{phase}/i2t_accuracy", i2t_acc)
    pl_module.log(f"irtr/{phase}/t2i_accuracy", t2i_acc)

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    # print("vqa_logits", vqa_logits.max(), vqa_logits.min(), vqa_logits.mean(), vqa_logits.std())
    # print("cls_feats", infer['cls_feats'].max(), infer['cls_feats'].min(), infer['cls_feats'].mean(), infer['cls_feats'].std())

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_img_cls(pl_module, batch):
    infer = pl_module.infer_image_ft(batch, mask_image=False)
    cls_logits = pl_module.img_cls_classifier(infer["image_feats"].mean(1))

    cls_labels = batch["class_label"]

    cls_loss = F.cross_entropy(cls_logits, cls_labels)

    ret = {
        "cls_loss": cls_loss,
        "cls_logits": cls_logits,
        "cls_labels": cls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_img_cls_loss")(ret["cls_loss"])
    score = getattr(pl_module, f"{phase}_img_cls_accuracy")(
        ret["cls_logits"], ret["cls_labels"]
    )
    pl_module.log(f"img_cls/{phase}/loss", loss)
    pl_module.log(f"img_cls/{phase}/accuracy", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module, split="test"):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if split == "val":
        print("Use val set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    else:
        print("Use test set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset()

    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        worker_init_fn=text_dset.read_input_tsv if hasattr(text_dset, 'read_input_tsv') else None,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    if split == "val":
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
            image_only=True
        )
    else:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(
            image_only=True
        )

    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        worker_init_fn=image_dset.read_input_tsv if hasattr(image_dset, 'read_input_tsv') else None,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    tmp_text_masks = None
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        tmp_text_masks = _b["text_masks"]
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    # rank_zero_info("len(tiids): {}".format(len(tiids)))

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append(
            {
                "image": [_b["image"][0].to(pl_module.device)],
                "img_index": _b["img_index"],
                "text_masks": tmp_text_masks.to(pl_module.device),
            }
        )
    iids = list()
    for pre in image_preload:
        iids += pre["img_index"]
    iids = torch.tensor(iids)

    # rank_zero_info("len(iids): {}".format(len(iids)))

    txt_cls_feats = list()
    for txt_batch in text_preload:
        with torch.cuda.amp.autocast():
            cls_feats = pl_module.infer_text_ft(
                    {
                        "text_ids": txt_batch["text_ids"],
                        "text_masks": txt_batch["text_masks"],
                        "text_labels": txt_batch["text_labels"],
                    }
                )["cls_feats"]
        txt_cls_feats.append(cls_feats)

    img_cls_feats = list()
    for img_batch in image_preload:
        with torch.cuda.amp.autocast():
            cls_feats = pl_module.infer_image_ft(
                    {
                        "image": img_batch["image"],
                        "text_masks": img_batch["text_masks"],
                    }
                )["cls_feats"]
        img_cls_feats.append(cls_feats)

    txt_cls_feats = torch.cat(txt_cls_feats)
    img_cls_feats = torch.cat(img_cls_feats)

    # rank_zero_info("txt_cls_feats.size(): {}".format(txt_cls_feats.size()))
    # rank_zero_info("img_cls_feats.size(): {}".format(img_cls_feats.size()))

    scores = img_cls_feats @ txt_cls_feats.t()

    # rank_zero_info("scores.size(): {}".format(scores.size()))

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name, output_dir):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        # os.makedirs("result", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
