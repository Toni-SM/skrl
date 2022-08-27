from typing import Optional, Sequence

import gym

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
            ...     def compute(self, states, taken_actions, role):
            ...         return self.net(states)
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
        self._d_clip_actions[role] = clip_actions and issubclass(type(self.action_space), gym.Space)

        if self._d_clip_actions[role]:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

            # backward compatibility: torch < 1.9 clamp method does not support tensors
            self._backward_compatibility = tuple(map(int, (torch.__version__.split(".")[:2]))) < (1, 9)
        
    def act(self, 
            states: torch.Tensor, 
            taken_actions: Optional[torch.Tensor] = None, 
            inference: bool = False,
            role: str = "") -> Sequence[torch.Tensor]:
        """Act deterministically in response to the state of the environment

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: ``None``).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor, optional
        :param inference: Flag to indicate whether the model is making inference (default: ``False``)
        :type inference: bool, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Action to be taken by the agent given the state of the environment.
                 The sequence's components are the computed actions and None for the last two components
        :rtype: sequence of torch.Tensor

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> output = model.act(states)
            >>> print(output[0].shape, output[1], output[2])
            torch.Size([4096, 1]) None None
        """
        # map from observations/states to actions
        if self._instantiator_net is None:
            actions = self.compute(states.to(self.device), 
                                   taken_actions.to(self.device) if taken_actions is not None else taken_actions, role)
        else:
            actions = self._get_instantiator_output(states.to(self.device), \
                taken_actions.to(self.device) if taken_actions is not None else taken_actions)

        # clip actions 
        if self._d_clip_actions[role] if role in self._d_clip_actions else self._d_clip_actions[""]:
            if self._backward_compatibility:
                actions = torch.max(torch.min(actions, self.clip_actions_max), self.clip_actions_min)
            else:
                actions = torch.clamp(actions, min=self.clip_actions_min, max=self.clip_actions_max)

        if inference:
            return actions.detach(), None, None
        return actions, None, None
        