import os
import copy
import torch
import pytorch_lightning as pl

from collections import defaultdict

import sys
root_dir = "/workspace/ort"
sys.path.append(root_dir)

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule
from vilt.datamodules.multi_multitask_datamodule import MultiMTDataModule
from vilt.datamodules import MSRVTTDataModule
from vilt.custom_ln import LNConfig
from vilt.moe import MOEConfig
from vilt.ufo.config import UFOConfig

import os
import re
from pytorch_lightning.plugins import environments as pl_env
from pytorch_lightning.utilities.distributed import rank_zero_info


# class OMPIClusterEnvironment(pl_env.ClusterEnvironment):
#     def __init__(self):
#         super().__init__()

#     def creates_children(self) -> bool:
#         # return True if the cluster is managed (you don't launch processes yourself)
#         assert (
#             "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
#         )  # this cluster is managed
#         return True

#     def world_size(self) -> int:
#         return int(os.environ["OMPI_COMM_WORLD_SIZE"])

#     def set_world_size(self, size: int):
#         pass
#         # raise RuntimeError("this cluster is managed")

#     def global_rank(self) -> int:
#         return int(os.environ["OMPI_COMM_WORLD_RANK"])

#     def set_global_rank(self, rank: int):
#         pass
#         # raise RuntimeError("this cluster is managed")

#     def local_rank(self) -> int:
#         return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

#     def node_rank(self) -> int:
#         # mpi doesn't set node rank and it cannot be deduced
#         # global_rank = local_rank + node_rank * numprocesses
#         if "NODE_RANK" in os.environ:
#             return int(os.environ["NODE_RANK"])
#         else:
#             return 0

#     def master_address(self) -> str:
#         return os.environ["MASTER_ADDR"]

#     def master_port(self) -> int:
#         return int(os.environ["MASTER_PORT"])


class OMPIClusterEnvironment(pl_env.ClusterEnvironment):
    def __init__(self):
        super().__init__()

    @property
    def creates_processes_externally(self) -> bool:
        return (
            "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
        )  # this cluster is managed

    def creates_children(self) -> bool:
        # return True if the cluster is managed (you don't launch processes yourself)
        assert (
            "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
        )  # this cluster is managed
        return True

    def world_size(self) -> int:
        ws = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
        os.environ['WORLD_SIZE'] = str(ws)
        return ws

    def local_size(self):
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 1))

    def set_world_size(self, size: int):
        pass
        # raise RuntimeError("this cluster is managed")

    def global_rank(self) -> int:
        return int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))

    def set_global_rank(self, rank: int):
        pass
        # raise RuntimeError("this cluster is managed")

    def local_rank(self) -> int:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))

    def node_rank(self) -> int:
        # mpi doesn't set node rank and it cannot be deduced
        # global_rank = local_rank + node_rank * numprocesses
        if "NODE_RANK" in os.environ:
            return int(os.environ["NODE_RANK"])
        else:
            nr = int(os.environ.get("OMPI_COMM_WORLD_NODE_RANK", 0))
            os.environ['NODE_RANK'] = str(nr)  # '''
            return nr

    def master_address(self) -> str:
        return os.environ.get("MASTER_ADDR", 'localhost')

    def master_port(self) -> int:
        return int(os.environ.get("MASTER_PORT", 12345))


def get_cluster_plugin(num_gpus=1, num_nodes=1):
    if num_nodes > 1 or (
        num_nodes == 1 and "OMPI_COMM_WORLD_SIZE" in os.environ
    ):
        rank_zero_info("ClusterPlugin: using OMPI Cluster Environment")
        return OMPIClusterEnvironment()
    # ITP also allows MPI jobs on one node
    # either way, the azure_runner will set the environment variables accordingly
    # WORLD_SIZE, RANK, LOCAL_RANK
    if num_gpus >= 1:
        rank_zero_info("ClusterPlugin: using Lightning Cluster Environment")
        return pl_env.LightningEnvironment()
    return None


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    pl.seed_everything(_config["seed"])

    print(_config)

    import subprocess

    subprocess.call("nvidia-smi", shell=True)
    # subprocess.call("ls /workspace", shell=True)
    # subprocess.call("export PYTHONPATH=$PYTHONPATH:/workspace/ort", shell=True)
    # subprocess.call("echo $PYTHONPATH", shell=True)
    
    # if _config["datasets"][0] == "msrvtt":
    #     dm = MSRVTTDataModule(_config)
    # else:

    if _config["tasks"] is not None:
        dm = MultiMTDataModule(_config, dist=True)
    else:
        dm = MTDataModule(_config, dist=True)

    ln_config = None
    moe_config = None
    ufo_config = None

    if _config["use_ufo"]:
        ufo_config = UFOConfig()
        ufo_config.separate_inference = _config["separate_inference"]

    if _config["use_custom_ln_attn"] or _config["use_custom_ln_ffn"]:
        ln_config = LNConfig()
        ln_config.use_custom_ln_attn = _config["use_custom_ln_attn"]
        ln_config.use_custom_ln_ffn = _config["use_custom_ln_ffn"]

    if _config["use_moe"]:
        moe_config = MOEConfig()
        moe_config.in_attn = _config["in_attn"]
        moe_config.in_ffn = _config["in_ffn"]
        moe_config.self_attn_for_single_mode = _config["self_attn_for_single_mode"]
        moe_config.separate_inference = _config["separate_inference"]

    model = ViLTransformerSS(_config, ufo_config, ln_config, moe_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    rank_zero_info("grad_steps: {}".format(grad_steps))

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    resume_ckpt = None
    if _config["resume_during_pretraining"]:
        for index in range(100):
            ckpt_path = os.path.join(_config["log_dir"], f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}', "version_{}/checkpoints/last.ckpt".format(index))
            if os.path.exists(ckpt_path):
                resume_ckpt = ckpt_path
    
    rank_zero_info("resume_ckpt: {}".format(resume_ckpt))

    cluster_plugin = get_cluster_plugin(
        _config["num_gpus"], _config["num_nodes"]
    )
    plugin_list = [cluster_plugin]
    if _config["use_sharded_training"]:
        plugin_list.append("ddp_sharded")
    
    rank_zero_info("plugin_list: {}".format(plugin_list))
    
    middle_representations = defaultdict(float)
    def hook_mlp(module, input, output):
        middle_representations[module.module_name].append(output.detach().cpu())

    def hook_attn(module, input, output):
        middle_representations[module.module_name].append(output[0].detach().cpu())

    def hook_input(module, input, output):
        middle_representations[module.module_name].append(input.detach().cpu())

    def hook_gram_input(module, input, output):

        if isinstance(input, tuple):
            input = input[0]

        flatten_input = input.reshape(-1, input.shape[-1]).to(torch.float64) # (B * L, D)
        gram = torch.matmul(flatten_input.T, flatten_input)

        middle_representations[module.module_name] += gram.detach().cpu()

    def hook_input_output(module, input, output):
        # module_name should be a dictionary
        middle_representations[module.module_name["input"]].append(input[0].detach().cpu())
        middle_representations[module.module_name["output"]].append(output.detach().cpu())

    def hook_output(module, input, output):
        middle_representations[module.module_name].append(output.detach().cpu())
    
    if _config["use_moe"]: 
        all_keys = [
            "mlp.fc1", "mlp.fc1",
            "mlp.v.fc1", "mlp.l.fc1", "mlp.vl.fc1", "mlp.v.fc2", "mlp.l.fc2", "mlp.vl.fc2",
            # "norm1.v", "norm1.l", "norm1.vl",
            # "norm2.v", "norm2.l", "norm2.vl",
            "attn",
            "attn.v", "attn.l", "attn.vl",
            "attn.proj",
            "attn.v.proj", "attn.l.proj", "attn.vl.proj",
            ]
    else: # ufo
        all_keys = ["mlp.fc1", "mlp.fc2", "attn.proj", "norm1", "norm2"]

    for name, module in model.named_modules():
        if any([name.endswith(n) for n in all_keys]) and ".bias" not in name:
            module.module_name = name
            module.register_forward_hook(hook_gram_input)

    if _config["use_cpu"]:
        trainer = pl.Trainer(
            devices=_config["num_gpus"] * _config["num_nodes"],
            precision=_config["precision"],
            accelerator="cpu",
            strategy="ddp",
            benchmark=True,
            deterministic=False,
            max_epochs=_config["max_epoch"] if max_steps is None else 1000,
            max_steps=max_steps,
            callbacks=callbacks,
            logger=logger,
            prepare_data_per_node=False,
            replace_sampler_ddp=False,
            accumulate_grad_batches=grad_steps,
            log_every_n_steps=10,
            flush_logs_every_n_steps=10,
            resume_from_checkpoint=resume_ckpt,
            reload_dataloaders_every_epoch=False,
            weights_summary="top",
            fast_dev_run=_config["fast_dev_run"],
            val_check_interval=_config["val_check_interval"],
            limit_train_batches=_config["limit_train_batches"],
            limit_val_batches=_config["limit_val_batches"],
            # plugins=plugin_list,
        )
    else:
        trainer = pl.Trainer(
            gpus=_config["num_gpus"],
            num_nodes=_config["num_nodes"],
            # num_processes=2,
            precision=_config["precision"],
            accelerator="ddp",
            benchmark=True,
            deterministic=False,
            max_epochs=_config["max_epoch"] if max_steps is None else 1000,
            max_steps=max_steps,
            callbacks=callbacks,
            logger=logger,
            prepare_data_per_node=False,
            replace_sampler_ddp=False,
            accumulate_grad_batches=grad_steps,
            log_every_n_steps=10,
            flush_logs_every_n_steps=10,
            resume_from_checkpoint=resume_ckpt,
            reload_dataloaders_every_epoch=False,
            weights_summary="top",
            fast_dev_run=_config["fast_dev_run"],
            val_check_interval=_config["val_check_interval"],
            limit_train_batches=_config["limit_train_batches"],
            limit_val_batches=_config["limit_val_batches"],
            plugins=plugin_list,
        )

    # dm.prepare_data()

    trainer.validate(model, datamodule=dm)

    for k, v in middle_representations.items():
        if not isinstance(middle_representations[k], dict):
            print(k, middle_representations[k].shape)

            print(middle_representations[k].min(), middle_representations[k].max())
        else:
            print(k, middle_representations[k]["l_embed"].shape, middle_representations[k]["v_embed"].shape)

    torch.save(middle_representations, os.path.join(_config["log_dir"], f"{_config['representation_name']}.pth"))

    if torch.cuda.is_available() and _config["compute_memory"]:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
