# [start-model]
from typing import Union, Mapping, Sequence, Tuple, Any
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
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act according to the specified behavior

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dictionary
        """
        # ==============================
        # - act in response to the state
        # ==============================
# [end-model]

# =============================================================================

# [start-mixin]
from typing import Union, Mapping, Sequence, Tuple, Any

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
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act according to the specified behavior

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dictionary
        """
        # ==============================
        # - act in response to the state
        # ==============================

        # e.g. retrieve clip actions according to role
        clip_actions = self._custom_clip_actions[role] if role in self._custom_clip_actions else self._custom_clip_actions[""]
# [end-mixin]
