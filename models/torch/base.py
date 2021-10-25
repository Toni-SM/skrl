from __future__ import annotations
from typing import Union, Tuple

import gym
import math
import torch


class Model(torch.nn.Module):
    def __init__(self, observation_space: Union[int, tuple[int], gym.Space, None] = None, action_space: Union[int, tuple[int], gym.Space, None] = None, device: str = "cuda:0") -> None:
        """
        Base class that represent a neural network model

        # TODO: describe internal properties and methods

        Parameters
        ----------
        observation_space: int, tuple, list, gym.Space or None, optional
            Observation/state space or shape (default: None).
            If it is not None, the num_observations property will contain the size of that space (number of elements)
        action_space: int, tuple, list, gym.Space or None, optional
            Action space or shape (default: None).
            If it is not None, the num_actions property will contain the size of that space (number of elements)
        device: str, optional
            Device on which a torch tensor is or will be allocated (default: "cuda:0")
        """
        super(Model, self).__init__()

        self.device = device

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_observations = None if observation_space is None else self._get_space_size(observation_space)
        self.num_actions = None if action_space is None else self._get_space_size(action_space)
        
    def _get_space_size(self, space: Union[int, tuple[int], gym.Space]) -> int:
        """
        Get the size (number of elements) of a space

        Parameters
        ----------
        space: int, tuple, list, gym.Space or None, optional
           Space or form from which to obtain the number of elements
        
        Returns
        -------
        int
            Space size (number of elements)
        """
        if type(space) in [tuple, list]:
            return math.prod(space)
        elif issubclass(type(space), gym.Space):
            return math.prod(space.shape)
        return space

    def forward(self):
        raise NotImplementedError("Implement .act() and .compute() methods instead of this")

    def init_parameters(self, method_name: str = "uniform_", *args, **kwargs) -> None:
        """
        Initializes the parameters of the module according to the specified method

        Method names are from the [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) module. 
        Allowed method names are "uniform_", "normal_", "constant_", etc.

        Parameters
        ----------
        method_name: str, optional
            Name of the [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) method (default: "uniform_")
        args: tuple, optional
            Positional arguments of the method to be called
        kwargs: dict, optional
            Key-value arguments of the method to be called
        """
        for parameters in self.parameters():
            exec("torch.nn.init.{}(parameters, *args, **kwargs)".format(method_name))

    def compute(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor]:
        """
        Defines the computation performed by all involved networks

        Parameters
        ----------
        states: torch.Tensor
            States/observations of the environment used to make the decision
        taken_actions: torch.Tensor or None, optional
            Actions performed by a policy (default: None).
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
        taken_actions: torch.Tensor or None, optional
            Actions performed by a policy (default: None).
            Using these actions only makes sense in critic networks
        inference: bool, optional
            Flag to indicate whether the network is making inference (default: False)
        
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
        polyak: float, optional
            Polyak hyperparameter between 0 and 1 (usually close to 1).
            A hard update is performed when its value is 0 (default)
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
