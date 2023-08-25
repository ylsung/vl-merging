from sacred import Experiment

ex = Experiment("VLMo")


def _loss_names(d):
    ret = {
        "itm": 0, # image-text matching loss
        "ifm": 0, # image-text contrastive loss
        "mlm": 0, # masked language modeling loss
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0, # retrieval task ft
        "mim": 0, # masked image modeling loss
        "image_only_mim": 0,
        "text_only_mlm": 0,
        "img_cls": 0, # image classification loss
        "mnc": 0, # modality negative consine similarity loss
        "mld": 0, # modality gap loss
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vlmo"
    seed = 1
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "ifm": 1, "mlm": 1})
    batch_size = 1024  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    image_size = 224
    max_image_len = -1
    patch_size = 32
    draw_false_image = 0
    image_only = False
    img_cls_label_size = 1000

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    max_text_len_of_initckpt = 196
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0
    vl_mlm_weight = 1
    ifm_weight = 1

    # For Video
    num_frames = 1

    # VL Setting
    max_vl_text_len = None # used to separate from pure NLP text length
    use_temporal_roll_module = False
    vl_mlm_prob = 0.15 # used to separate from pure NLP mlm_prob

    # Transformer Setting
    vit = "vit_base_patch16_224"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    vlffn_start_layer_index = -1 #start from 0

    # Optimizer Setting
    optim_type = "adamw"
    beta_2 = 0.98
    learning_rate = 1e-4
    weight_decay = 0.01
    weight_decay_custom_modules = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 200000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    use_cpu = False

    all_mlp_mult = False
    all_vl_mult = False
    all_v_mult = False
    all_l_mult = False

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    validation_only = False
    use_sharded_training = False
    use_webdataset = False
    resume_during_pretraining = False
    limit_val_batches = 1.0
    limit_train_batches = 1.0

    # below params varies with the environment
    data_root = ""
    data_roots = None # a list of data root, meaning to use different roots for different datasets.
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16
    compute_memory = False

    # for extracting middle representations from the model
    get_middle_representation = False
    get_block_representation = False
    get_finegrained_representation = False
    representation_name = "tmp"
    
    # whether use beit weights
    use_beit_weight = False

    # whether use combined weights I do myself
    use_self_weight = False

    # ufo
    use_ufo = False # the model would use UFO as default
    separate_inference = True
    # moe
    use_moe = False
    self_attn_for_single_mode = False
    use_vision_weights_for_other_modalities = False
    in_attn = False
    in_ffn = True

    # merge
    merge_weights = False
    merge_ratio = 0.5

    sum_task_vectors = False
    central_weight = None
    sum_lambda = 1

    only_activate_used_experts = False

    regmean = False
    gram_matrices = None
    scaling_for_non_diag = 1

    # custom layer norm
    use_custom_ln_attn = False
    use_custom_ln_ffn = False

    # for Masked Image Modeling (MIM) loss
    discrete_vae_weight_path = ""
    num_mask_patches = 75 # number of the visual tokens/patches need be masked
    max_mask_patches_per_block = None
    min_mask_patches_per_block = 16
    dvae_image_size = 112

    # for combining the single and multi modal training
    tasks = None
    random_initialization = False

# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm_ifm_square_randaug_base():
    exp_name = "mlm_itm_ifm_square_randaug_base"
    datasets = ["coco", "vg", "sbu", "gcc"]
    # datasets = ["coco"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "ifm": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = 196
    max_text_len_of_initckpt = 196
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    vit = "vit_base_patch16_224"


@ex.named_config
def task_finetune_nlvr2_square_randaug_base():
    exp_name = "finetune_nlvr2_square_randaug_base"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_transform_keys = ["square_transform"]
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_224"


@ex.named_config
def task_finetune_nlvr2_square_randaug_base_image384():
    exp_name = "finetune_nlvr2_square_randaug_base_image384"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-5
    val_transform_keys = ["square_transform"]
    image_size = 384
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_384"


@ex.named_config
def task_finetune_vqa_square_randaug_base_image384():
    exp_name = "finetune_vqa_square_randaug_base_image384"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_transform_keys = ["square_transform"]
    val_check_interval = 1.0
    lr_mult = 10
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_384"
    use_moe = False


@ex.named_config
def task_all_in_one_pretraining():
    exp_name = "all_in_one_pretraining"
    train_transform_keys = ["square_transform_randaug_mim"]
    tasks = [
        "v", 
        "l", 
        "vl",
    ]

    datasets = [
        ["imagenet"],
        ["bookcorpus", "wikipedia"],
        ["webvid", "sbu", "gcc", "coco", "vg"],
    ]
    data_roots = [
        ["/storage/v-yilinsung/imagenet-22k/"],
        ["/storage/v-yilinsung/huggingface/bookcorpus/", "/storage/v-yilinsung/huggingface/wikipedia_20200501_en/"],
        ["/storage/linjli/data/mtp_vlp_ray/pretrain/composite/", "/storage/linjli/data/vilt/pretrain_arrows_code224/", "/storage/linjli/data/vilt/pretrain_arrows_code224/", "/storage/linjli/data/vilt/pretrain_arrows_code224/", "/storage/linjli/data/vilt/pretrain_arrows_code224/"],
    ]

    discrete_vae_weight_path = "/storage/v-yilinsung/dall_e_tokenizer_weight"

    loss_names = _loss_names({"image_only_mim": 1, "text_only_mlm": 1, "mim": 1, "itm": 1, "mlm": 1, "ifm": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_transform_keys = ["square_transform_mim"]
    val_check_interval = 1.0
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_224"
    use_moe = False
    random_initialization = True
    max_vl_text_len = 40


@ex.named_config
def task_finetune_vqa_square_randaug_base_image384_ufo():
    exp_name = "finetune_vqa_square_randaug_base_image384_ufo"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 3e-5
    val_transform_keys = ["square_transform"]
    val_check_interval = 1.0
    lr_mult = 10
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_384"
    use_moe = False


@ex.named_config
def task_finetune_vqa_square_randaug_large_image384_ufo():
    exp_name = "finetune_vqa_square_randaug_large_image384_ufo"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 3e-5
    val_transform_keys = ["square_transform"]
    val_check_interval = 1.0
    lr_mult = 10
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 21
    use_sharded_training = False
    vit = "vit_large_patch16_384"
    hidden_size = 1024
    num_heads = 16
    num_layers = 24 
    use_moe = False


@ex.named_config
def task_finetune_imagenet_square_randaug_base_image384():
    exp_name = "finetune_imagenet_square_randaug_base_image384_ufo"
    datasets = ["imagenet1k"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"img_cls": 1})
    batch_size = 512
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-3
    val_transform_keys = ["square_transform"]
    val_check_interval = 1.0
    lr_mult = 10
    image_size = 384
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training = False
    vit = "vit_base_patch16_384"
    use_moe = False


@ex.named_config
def task_finetune_imagenet_square_randaug_base_image224():
    exp_name = "finetune_imagenet_square_randaug_base_image224_ufo"
    datasets = ["imagenet1k"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"img_cls": 1})
    batch_size = 512
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.2
    draw_false_image = 0
    weight_decay = 0.05
    learning_rate = 3e-3
    val_transform_keys = ["square_transform"]
    val_check_interval = 1.0
    lr_mult = 1
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training = False
    vit = "vit_base_patch16_384"
    use_moe = False


@ex.named_config
def task_finetune_irtr_f30k_square_randaug_base():
    exp_name = "finetune_irtr_f30k_square_randaug_base"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 1024
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 0
    learning_rate = 5e-5
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_224"


@ex.named_config
def task_finetune_irtr_msrvtt_frame_square_randaug_base():
    exp_name = "finetune_irtr_msrvtt_frame_square_randaug_base"
    datasets = ["msrvtt"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0, "ifm": 1.0, "itm": 1.0})
    batch_size = 1024
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 0
    learning_rate = 5e-5
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_224"
    use_moe = False


@ex.named_config
def task_finetune_irtr_f30k_square_randaug_base_image384():
    exp_name = "finetune_irtr_f30k_square_randaug_base_image384"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 1024
    max_epoch = 40
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 0
    learning_rate = 5e-5
    image_size = 384
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_384"


@ex.named_config
def task_finetune_irtr_f30k_square_randaug_large_image384():
    exp_name = "finetune_irtr_f30k_square_randaug_large_image384"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 1024
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 0
    learning_rate = 5e-5
    image_size = 384
    patch_size = 16
    vlffn_start_layer_index = 21
    use_sharded_training=False
    vit = "vit_large_patch16_384"
    hidden_size = 1024
    num_heads = 16
    num_layers = 24 


@ex.named_config
def task_finetune_irtr_coco_square_randaug_base_image384():
    exp_name = "finetune_irtr_coco_square_randaug_base_image384"
    datasets = ["coco"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 1024
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 0
    learning_rate = 2e-5
    image_size = 384
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_384"


@ex.named_config
def task_mlm_itm_ifm_square_randaug_base_vl():
    exp_name = "mlm_itm_ifm_square_randaug_base_vl"
    train_transform_keys = ["square_transform_randaug"]
    tasks = [
        "vl",
    ]

    datasets = [
        ["sbu", "gcc", "coco", "vg"],
    ]
    data_roots = [
        ["/storage/linjli/data/vilt/pretrain_arrows_code224/", "/storage/linjli/data/vilt/pretrain_arrows_code224/", "/storage/linjli/data/vilt/pretrain_arrows_code224/", "/storage/linjli/data/vilt/pretrain_arrows_code224/"],
    ]

    discrete_vae_weight_path = "/storage/v-yilinsung/dall_e_tokenizer_weight"

    # check on the weight of different tasks
    loss_names = _loss_names({"itm": 1, "mlm": 1, "ifm": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-4
    val_transform_keys = ["square_transform"]
    val_check_interval = 1.0
    image_size = 224
    patch_size = 16
    vlffn_start_layer_index = 10
    use_sharded_training=False
    vit = "vit_base_patch16_224"
    max_vl_text_len = 40
    max_text_len = 40


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end
@ex.named_config
def step10k():
    max_epoch = 100
    max_steps = 10000


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    warmup_steps = 625
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    warmup_steps = 1250
    max_steps = 100000


@ex.named_config
def step150k():
    max_epoch = 150
    warmup_steps = 1875
    max_steps = 150000


@ex.named_config
def step200k():
    max_epoch = 200
    warmup_steps = 2500
    max_steps = 200000


@ex.named_config
def step400k():
    max_epoch = 300
    warmup_steps = 5000
    max_steps = 400000


@ex.named_config
def epoch100():
    max_epoch = 100
    warmup_steps = 10000

@ex.named_config
def ufo():
    use_ufo = True
    separate_inference = True


@ex.named_config
def ln_moe():
    use_moe = False
    in_attn = False
    in_ffn = False

    use_custom_ln_attn = True
    use_custom_ln_ffn = True
    separate_inference = True


@ex.named_config
def attn_moe():
    use_moe = True
    in_attn = True
    in_ffn = False

    use_custom_ln_attn = True
    use_custom_ln_ffn = False
    self_attn_for_single_mode = True

@ex.named_config
def ffn_moe():
    use_moe = True
    in_attn = False
    in_ffn = True

    use_custom_ln_attn = False
    use_custom_ln_ffn = True

    separate_inference = True


@ex.named_config
def all_moe():
    use_moe = True
    in_attn = True
    in_ffn = True

    use_custom_ln_ffn = True
    use_custom_ln_attn = True
    self_attn_for_single_mode = True
