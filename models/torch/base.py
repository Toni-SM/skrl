from typing import Union, Tuple
from __future__ import annotations

import gym
import numpy as np
import torch
import torch.nn as nn

from ...env import Environment


class Model(nn.Module):
    def __init__(self, env: Union[Environment, gym.Env], device: str) -> None:
        """
        Base class that represent a neural network model

        # TODO: describe internal properties and methods

        Parameters
        ----------
        env: skrl.env.Environment or gym.Env
            RL environment
        device: str
            Device on which a torch tensor is or will be allocated
        """
        super(Model, self).__init__()

        self.env = env
        self.device = device

        self.num_observation = np.prod(self.env.observation_space.shape)
        self.num_action = np.prod(self.env.action_space.shape)
        
    def forward(self):
        raise NotImplementedError("Implement .act() and .compute() methods instead of this")

    def compute(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor]:
        """
        Defines the computation performed by all involved networks

        Parameters
        ----------
        states: torch.Tensor
            States/observations of the environment used to make the decision
        taken_actions: torch.Tensor or None
            Actions performed by a policy.
            Using these actions only makes sense in critic networks

        Returns
        -------
        torch.Tensor or tuple
            Computation performed by all involved networks
        """
        raise NotImplementedError("The computation performed by all involved networks (.compute()) is not implemented")

    def act(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None, inference=False) -> Tuple[torch.Tensor]:
        """
        Act according to the specified behavior

        Parameters
        ----------
        states: torch.Tensor
            States/observations of the environment used to make the decision
        taken_actions: torch.Tensor or None
            Actions performed by a policy.
            Using these actions only makes sense in critic networks
        inference: bool
            Flag to indicate whether the network is making inference
        
        Returns
        -------
        tuple of torch.Tensor
            Action performed by the agent.
            The typical tuple's components are the actions, the log of the probability density function and mean actions.
            Deterministic agents must ignore the last two components and return empty tensors for them
        """
        raise NotImplementedError("The action performed by the agent (.act()) is not implemented")
        
    def set_mode(self, mode: str) -> None:
        """
        Set the network mode (training or evaluation)

        Parameters
        ----------
        mode: str
            Mode: "train" for training or "eval" for evaluation.
            https://pytorch.org/docs/1.8.1/generated/torch.nn.Module.html#torch.nn.Module.train
        """
        # TODO: set mode for registered networks
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        else:
            raise ValueError("Invalid mode. Use 'train' for training or 'eval' for evaluation")

    def save(self, path):
        # TODO: implement load method according to involved networks
        # torch.save(self.network.state_dict(), path)
        pass

    def load(self, path):
        # TODO: implement load method according to involved networks
        # self.network.load_state_dict(torch.load(path))
        pass
    
    def update_parameters(self, network: Model, polyak: float = 0) -> None:
        """
        Update internal parameters by hard or soft (polyak averaging) update

        - Hard update: `parameters = network.parameters`
        - Soft (polyak averaging) update: `parameters = polyak * parameters + (1 - polyak) * network.parameters`

        Parameters
        ----------
        network: skrl.models.torch.Model
            Network used to update the internal parameters
        polyak: float
            Polyak hyperparameter between 0 and 1 (usually close to 1).
            A hard update is performed when its value is 0
        """
        # hard update
        if not polyak:
            for parameters, network_parameters in zip(self.parameters(), network.parameters()):
                parameters.data.copy_(network_parameters.data)
        # soft update (use in-place operations to avoid creating new parameters)
        else:
            for parameters, network_parameters in zip(self.parameters(), network.parameters()):
                parameters.data.mul_(polyak)
                parameters.data.add_((1 - polyak) * network_parameters.data)
