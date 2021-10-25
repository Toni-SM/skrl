from __future__ import annotations
from typing import Union, Tuple

import gym
import torch

from . import Model


class DeterministicModel(Model):
    def __init__(self, observation_space: Union[int, tuple[int], gym.Space, None] = None, action_space: Union[int, tuple[int], gym.Space, None] = None, device: str = "cuda:0") -> None:
        """
        Deterministic model (Deterministic)

        # TODO: describe internal properties
        """
        super().__init__(observation_space=observation_space, action_space=action_space, device=device)
        
    def act(self, states, taken_actions=None, inference=False):
        # map from states/observations to actions
        actions = self.compute(states, taken_actions)

        return actions, torch.Tensor(), torch.Tensor()
        