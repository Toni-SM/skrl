# [start-model]
from typing import Optional, Union, Sequence

import gym

import torch

from skrl.models.torch import Model     # from . import Model


class CustomModel(Model):
    def __init__(self,
                 observation_space: Union[int, Sequence[int], gym.Space],
                 action_space: Union[int, Sequence[int], gym.Space],
                 device: Union[str, torch.device] = "cuda:0") -> None:
        """
        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gym.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gym.Space
        :param device: Device on which a torch tensor is or will be allocated (default: ``"cuda:0"``)
        :type device: str or torch.device, optional
        """
        super().__init__(observation_space, action_space, device)

    def act(self,
            states: torch.Tensor,
            taken_actions: Optional[torch.Tensor] = None,
            role: str = "") -> Sequence[torch.Tensor]:
        """Act according to the specified behavior

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: ``None``).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Child class must implement this method

        :return: Action to be taken by the agent given the state of the environment.
                 The typical sequence's components are the actions, the log of the probability density function and mean actions.
                 Deterministic agents must ignore the last two components and return empty tensors or None for them
        :rtype: sequence of torch.Tensor
        """
        # ==============================
        # - act in response to the state
        # ==============================
# [end-model]

# =============================================================================

# [start-mixin]
from typing import Optional, Sequence

import gym

import torch


class CustomMixin:
    def __init__(self, clip_actions: bool = False, role: str = "") -> None:
        """
        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: ``False``)
        :type clip_actions: bool, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        """
        # e.g. storage custom parameter
        if not hasattr(self, "_custom_clip_actions"):
            self._custom_clip_actions = {}
        self._custom_clip_actions[role]

    def act(self,
            states: torch.Tensor,
            taken_actions: Optional[torch.Tensor] = None,
            role: str = "") -> Sequence[torch.Tensor]:
        """Act according to the specified behavior

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: ``None``).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Child class must implement this method

        :return: Action to be taken by the agent given the state of the environment.
                 The typical sequence's components are the actions, the log of the probability density function and mean actions.
                 Deterministic agents must ignore the last two components and return empty tensors or None for them
        :rtype: sequence of torch.Tensor
        """
        # ==============================
        # - act in response to the state
        # ==============================

        # e.g. retrieve clip actions according to role
        clip_actions = self._custom_clip_actions[role] if role in self._custom_clip_actions else self._custom_clip_actions[""]
# [end-mixin]
