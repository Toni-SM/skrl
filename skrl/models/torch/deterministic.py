from typing import Any, Mapping, Tuple, Union

import gym
import gymnasium

import torch


class DeterministicMixin:
    def __init__(self, clip_actions: bool = False, role: str = "") -> None:
        """Deterministic mixin model (deterministic model)

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: ``False``)
        :type clip_actions: bool, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            # define the model
            >>> import torch
            >>> import torch.nn as nn
            >>> from skrl.models.torch import Model, DeterministicMixin
            >>>
            >>> class Value(DeterministicMixin, Model):
            ...     def __init__(self, observation_space, action_space, device="cuda:0", clip_actions=False):
            ...         Model.__init__(self, observation_space, action_space, device)
            ...         DeterministicMixin.__init__(self, clip_actions)
            ...
            ...         self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, 32),
            ...                                  nn.ELU(),
            ...                                  nn.Linear(32, 1))
            ...
            ...     def compute(self, inputs, role):
            ...         return self.net(inputs["states"]), {}
            ...
            >>> # given an observation_space: gym.spaces.Box with shape (60,)
            >>> # and an action_space: gym.spaces.Box with shape (8,)
            >>> model = Value(observation_space, action_space)
            >>>
            >>> print(model)
            Value(
              (net): Sequential(
                (0): Linear(in_features=60, out_features=32, bias=True)
                (1): ELU(alpha=1.0)
                (2): Linear(in_features=32, out_features=32, bias=True)
                (3): ELU(alpha=1.0)
                (4): Linear(in_features=32, out_features=1, bias=True)
              )
            )
        """
        if not hasattr(self, "_d_clip_actions"):
            self._d_clip_actions = {}
        self._d_clip_actions[role] = clip_actions and (issubclass(type(self.action_space), gym.Space) or \
            issubclass(type(self.action_space), gymnasium.Space))

        if self._d_clip_actions[role]:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act deterministically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is ``None``. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, _, outputs = model.act({"states": states})
            >>> print(actions.shape, outputs)
            torch.Size([4096, 1]) {}
        """
        # map from observations/states to actions
        actions, outputs = self.compute(inputs, role)

        # clip actions
        if self._d_clip_actions[role] if role in self._d_clip_actions else self._d_clip_actions[""]:
            actions = torch.clamp(actions, min=self.clip_actions_min, max=self.clip_actions_max)

        return actions, None, outputs
