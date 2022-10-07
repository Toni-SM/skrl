from typing import Union, Tuple, Dict, Any, Type, Optional

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
    "ensemble_size": 5,             # number of dynamics models
    "horizon": 100,                 # horizon length of the dynamics models

    "mini_batches": 500,            # number of mini batches during each learning epoch

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately
    }
}


class MBMPO(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 agent_cls: Type[Agent],
                 agent_kwargs: Dict[str, Any],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
                 device: Union[str, torch.device] = "cuda:0",
                 cfg: Optional[dict] = None) -> None:
        """Model-Based Meta-Policy-Optimization (MB-MPO)

        https://arxiv.org/abs/1809.05214

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param agent_cls: Agent class to be used for the (meta-)policy
        :type agent_cls: Type[Agent]
        :param agent_kwargs: Keyword arguments for the agent
        :type agent_kwargs: dict
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Computing device (default: "cuda:0")
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(MBMPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # configuration
        self._ensemble_size = self.cfg["ensemble_size"]
        self._horizon = self.cfg["horizon"]
        self._mini_batches = self.cfg["mini_batches"]

        # models
        self.dynamics_model = self.models.get("dynamics", None)
        self.dynamics_models = [copy.deepcopy(self.dynamics_model) for _ in range(self._ensemble_size)]
        # Reinitialize all models to guarantee different parameters
        for dm in self.dynamics_models:
            dm.init_parameters(method_name="normal_", mean=0.0, std=0.1)

        # checkpoint models
        for idx, dm in enumerate(self.dynamics_models):
            self.checkpoint_modules[f"dynamics_{idx}"] = dm

        # agents
        self.agent = agent_cls(observation_space=observation_space,
                               action_space=action_space,
                               **{agent_kwargs})
        self.meta_agents = [agent_cls(observation_space=observation_space,
                                      action_space=action_space,
                                      **{agent_kwargs}
                                     ) for _ in range(self._ensemble_size)]