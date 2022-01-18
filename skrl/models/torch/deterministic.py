from typing import Union, Tuple

import gym

import torch

from . import Model


class DeterministicModel(Model):
    def __init__(self, observation_space: Union[int, Tuple[int], gym.Space, None] = None, action_space: Union[int, Tuple[int], gym.Space, None] = None, device: Union[str, torch.device] = "cuda:0", clip_actions: bool = False) -> None:
        """Deterministic model (deterministic model)

        :param observation_space: Observation/state space or shape (default: None).
                                  If it is not None, the num_observations property will contain the size of that space (number of elements)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None).
                             If it is not None, the num_actions property will contain the size of that space (number of elements)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: False)
        :type clip_actions: bool, optional
        """
        super(DeterministicModel, self).__init__(observation_space, action_space, device)

        self.clip_actions = clip_actions and issubclass(type(self.action_space), gym.Space)

        if self.clip_actions:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)
        
    def act(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None, inference=False) -> Tuple[torch.Tensor]:
        """Act deterministically in response to the state of the environment

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: None).
                              The use of these actions only makes sense in critical networks, e.g.
        :type taken_actions: torch.Tensor or None, optional
        :param inference: Flag to indicate whether the network is making inference (default: False).
                          If True, the returned tensors will be detached from the current graph
        :type inference: bool, optional

        :return: Action to be taken by the agent given the state of the environment.
                 The tuple's components are the computed actions and None for the last two components
        :rtype: tuple of torch.Tensor
        """
        # map from observations/states to actions
        actions = self.compute(states.to(self.device), 
                               taken_actions.to(self.device) if taken_actions is not None else taken_actions)

        # clip actions 
        if self.clip_actions:
            actions = torch.clamp(actions, min=self.clip_actions_min, max=self.clip_actions_max)

        if inference:
            return actions.detach(), None, None
        return actions, None, None
        