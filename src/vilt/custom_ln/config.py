from dataclasses import dataclass


@dataclass
class LNConfig(object):
    tasks = ["vl"]
    tasks_for_shallow_layers = ["v", "l"]
    tasks_for_deep_layers = ["v", "l", "vl"]

    use_custom_ln_attn = False
    use_custom_ln_ffn = False
  