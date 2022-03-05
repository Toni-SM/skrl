from typing import Union, Tuple

import gym

import torch

from . import Model


class TabularModel(Model):
    def __init__(self, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0",
                 num_envs: int = 1) -> None:
        """Tabular model

        :param observation_space: Observation/state space or shape (default: None).
                                  If it is not None, the num_observations property will contain the size of that space
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None).
                             If it is not None, the num_actions property will contain the size of that space
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        :param num_envs: Number of environments (default: 1)
        :type num_envs: int, optional
        """
        super(TabularModel, self).__init__(observation_space, action_space, device)

        self.num_envs = num_envs
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions), dtype=torch.float32, device=self.device)

    def act(self, 
            states: torch.Tensor, 
            taken_actions: Union[torch.Tensor, None] = None, 
            inference=False) -> Tuple[torch.Tensor]:
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
        actions = torch.argmax(self.q_table[torch.arange(self.num_envs).view(-1, 1), states], 
                               dim=-1, keepdim=True).view(-1,1)
        return actions, None, None
        
    def table(self) -> torch.Tensor:
        """Return the Q-table

        :return: Q-table
        :rtype: torch.Tensor
        """
        return self.q_table
