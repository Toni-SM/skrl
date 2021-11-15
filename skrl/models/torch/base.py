from typing import Union, Tuple

import gym
import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, observation_space: Union[int, Tuple[int], gym.Space, None] = None, action_space: Union[int, Tuple[int], gym.Space, None] = None, device: str = "cuda:0") -> None:
        """
        Base class representing a neural network model

        # TODO: describe internal properties and methods

        Parameters
        ----------
        observation_space: int, tuple or list of integers, gym.Space or None, optional
            Observation/state space or shape (default: None).
            If it is not None, the num_observations property will contain the size of that space (number of elements)
        action_space: int, tuple or list of integers, gym.Space or None, optional
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
        
    def _get_space_size(self, space: Union[int, Tuple[int], gym.Space]) -> int:
        """
        Get the size (number of elements) of a space

        Parameters
        ----------
        space: int, tuple or list of integers, gym.Space or None
           Space or shape from which to obtain the number of elements
        
        Returns
        -------
        int
            Space size (number of elements)
        """
        if type(space) in [tuple, list]:
            return np.prod(space)
        elif issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                return space.n
            return np.prod(space.shape)
        return space

    def random_act(self, states: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Act randomly

        Parameters
        ----------
        states: torch.Tensor
            Observation/state of the environment used to get the shape of the action space
            
        Returns
        -------
        tuple of torch.Tensor
            Random action to be taken by the agent
        """
        # TODO: sample taking into account states' shape
        # TODO: sample taking into account bounds
        distribution = torch.distributions.uniform.Uniform(low=self.action_space.low[0], high=self.action_space.high[0])
        actions = distribution.sample(sample_shape=(self.num_actions, ))
        return actions.to(self.device), None, None

    def init_parameters(self, method_name: str = "normal_", *args, **kwargs) -> None:
        """
        Initialize the model parameters according to the specified method name

        Method names are from the [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) module. 
        Allowed method names are "uniform_", "normal_", "constant_", etc.

        Parameters
        ----------
        method_name: str, optional
            Name of the [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) method (default: "normal_")
        args: tuple, optional
            Positional arguments of the method to be called
        kwargs: dict, optional
            Key-value arguments of the method to be called
        """
        for parameters in self.parameters():
            exec("torch.nn.init.{}(parameters, *args, **kwargs)".format(method_name))

    def forward(self):
        raise NotImplementedError("Implement .act() and .compute() methods instead of this")

    def compute(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Define the computation performed (to be implemented by the inheriting classes) by all involved networks

        Parameters
        ----------
        states: torch.Tensor
            Observation/state of the environment used to make the decision
        taken_actions: torch.Tensor or None, optional
            Actions taken by a policy to the given states (default: None).
            The use of these actions only makes sense in critical networks, e.g.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Computation performed by all involved networks
        """
        raise NotImplementedError("The computation performed by all involved networks (.compute()) is not implemented")

    def act(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None, inference=False) -> Tuple[torch.Tensor]:
        """
        Act according to the specified behavior (to be implemented by the inheriting classes)

        Agents will call this method, during training and evaluation, to obtain the decision to be taken given the state of the environment.
        This method is currently implemented in the predefined models (GaussianModel, DeterministicModel, etc.).
        The classes that inherit from the latter must only implement the .compute() method
        
        Parameters
        ----------
        states: torch.Tensor
            Observation/state of the environment used to make the decision
        taken_actions: torch.Tensor or None, optional
            Actions taken by a policy to the given states (default: None).
            The use of these actions only makes sense in critical networks, e.g.
        inference: bool, optional
            Flag to indicate whether the network is making inference (default: False)
        
        Returns
        -------
        tuple of torch.Tensor
            Action to be taken by the agent given the state of the environment.
            The typical tuple's components are the actions, the log of the probability density function and mean actions.
            Deterministic agents must ignore the last two components and return empty tensors or None for them
        """
        raise NotImplementedError("The action to be taken by the agent (.act()) is not implemented")
        
    def set_mode(self, mode: str) -> None:
        """
        Set the network mode (training or evaluation)

        Parameters
        ----------
        mode: str
            Mode: "train" for training or "eval" for evaluation.
            https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch%20nn%20module%20train#torch.nn.Module.train
        """
        if mode == "train":
            self.train(True)
        elif mode == "eval":
            self.train(False)
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
    
    def freeze_parameters(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze internal parameters

        - Freeze: disable gradient computation (`parameters.requires_grad = False`)
        - Unfreeze: enable gradient computation (`parameters.requires_grad = True`) 
        
        Parameters
        ----------
        freeze: bool
            Freeze the internal parameters if True, otherwise unfreeze them
        """
        for parameters in self.parameters():
            parameters.requires_grad = not freeze

    def update_parameters(self, network: torch.nn.Module, polyak: float = 0) -> None:
        """
        Update internal parameters by hard or soft (polyak averaging) update

        - Hard update: `parameters = network.parameters`
        - Soft (polyak averaging) update: `parameters = polyak * parameters + (1 - polyak) * network.parameters`

        Parameters
        ----------
        network: torch.nn.Module (skrl.models.torch.Model)
            Network used to update the internal parameters
        polyak: float, optional
            Polyak hyperparameter between 0 and 1 (usually close to 1).
            A hard update is performed when its value is 0 (default)
        """
        with torch.no_grad():
            # hard update
            if not polyak:
                for parameters, network_parameters in zip(self.parameters(), network.parameters()):
                    parameters.data.copy_(network_parameters.data)
            # soft update (use in-place operations to avoid creating new parameters)
            else:
                for parameters, network_parameters in zip(self.parameters(), network.parameters()):
                    parameters.data.mul_(polyak)
                    parameters.data.add_((1 - polyak) * network_parameters.data)
