from dataclasses import dataclass


@dataclass
class MOEConfig(object):
    tasks = ["vl"]
    tasks_for_shallow_layers = ["v", "l"]
    tasks_for_deep_layers = ["v", "l", "vl"]

    in_attn = False
    in_ffn = True

    self_attn_for_single_mode = False
