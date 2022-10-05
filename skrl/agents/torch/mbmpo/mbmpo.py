from typing import Union, Tuple, Dict, Any

import gym
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters

from ....memories.torch import Memory
from ....models.torch import Model

from .. import Agent


MBMPO_DEFAULT_CONFIG = {
    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately
    }
}


class MBMPO(Agent):
    pass
