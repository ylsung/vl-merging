from dataclasses import dataclass


@dataclass
class UFOConfig(object):
    tasks = ["vl"]
    tasks_for_shallow_layers = ["v", "l"]
    tasks_for_deep_layers = ["v", "l", "vl"]

    separate_inference = False