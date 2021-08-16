from typing import Union, Tuple

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
            Device on which a PyTorch tensor is or will be allocated
        """
        super(Model, self).__init__()
        self.env = env
        self.device = device

        self.num_observation = np.prod(self.env.observation_space.shape)
        self.num_action = np.prod(self.env.action_space.shape)
        
    def forward(self):
        raise NotImplementedError("Implement and call .act() and .compute() methods instead of this")

    def compute(self, states: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Defines the computation performed by all involved networks

        Parameters
        ----------
        states: torch.Tensor
            States/observations of the environment used to make the decision

        Returns
        -------
        torch.Tensor or tuple
            Computation performed by all involved networks
        """
        raise NotImplementedError("The computation performed by all involved networks (.compute()) is not implemented")

    def act(self, states: torch.Tensor, inference=False) -> Tuple[torch.Tensor]:
        """
        Act according to the specified behavior

        Parameters
        ----------
        states: torch.Tensor
            States/observations of the environment used to make the decision
        inference: bool
            Flag to indicate whether the network is making inference
        
        Returns
        -------
        tuple of torch.Tensor
            Action performed by the agent.
            The tuple's components are the actions, the log of the probability density function and mean actions.
            Deterministic agents must ignore the last two components and must return empty tensors for them
        """
        raise NotImplementedError("The action performed by the agent (.act()) is not implemented")

    def to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            return torch.FloatTensor(data).to(self.device)
        return data

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
    