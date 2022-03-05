from typing import Union, Tuple

import gym
import numpy as np

import torch


class Model(torch.nn.Module):
    def __init__(self, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0") -> None:
        """Base class representing a function approximator

        The following properties are defined:

        - ``device`` (torch.device): Device to be used for the computations
        - ``observation_space`` (int, tuple or list of integers, gym.Space or None): Observation/state space
        - ``action_space`` (int, tuple or list of integers, gym.Space or None): Action space
        - ``num_observations`` (int or None): Number of elements in the observation/state space
        - ``num_actions`` (int or None): Number of elements in the action space
        
        :param observation_space: Observation/state space or shape (default: None).
                                  If it is not None, the num_observations property will contain the size of that space
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None).
                             If it is not None, the num_actions property will contain the size of that space
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        """
        # TODO: export to onnx (https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
        super(Model, self).__init__()

        self.device = torch.device(device)

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_observations = None if observation_space is None else self._get_space_size(observation_space)
        self.num_actions = None if action_space is None else self._get_space_size(action_space)

        self._random_distribution = None

        # internal variables to be used by the model instantiators
        self._instantiator_net = None
        self._instantiator_input_type = 0
        self._instantiator_parameter = None
        self._instantiator_output_scale = 1.0
        
    def _get_instantiator_output(self, 
                                 states: torch.Tensor, 
                                 taken_actions: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor]:
        """Get the output of the instantiator model
        
        Input shape depends on the instantiator (see skrl.utils.model_instantiator.Shape) as follows:

        - STATES / OBSERVATIONS = 0
        - ACTIONS = -1
        - STATES_ACTIONS = -2

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: None)
        :type taken_actions: torch.Tensor, optional

        :return: Output of the instantiator model
        :rtype: tuple of torch.Tensor
        """
        if self._instantiator_input_type == 0:
            output = self._instantiator_net(states)
        elif self._instantiator_input_type == -1:
            output = self._instantiator_net(taken_actions)
        elif self._instantiator_input_type == -2:
            output = self._instantiator_net(torch.cat((states, taken_actions), dim=1))
        
        # deterministic and categorical output
        if self._instantiator_parameter is None:
            return output * self._instantiator_output_scale
        # gaussian output
        else:
            return output * self._instantiator_output_scale, self._instantiator_parameter

    def _get_space_size(self, space: Union[int, Tuple[int], gym.Space]) -> int:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, tuple or list of integers, or gym.Space

        :return: [description]
        :rtype: Space size (number of elements)
        """
        if type(space) in [tuple, list]:
            return np.prod(space)
        elif issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                return space.n
            return np.prod(space.shape)
        return space

    def random_act(self, 
                   states: torch.Tensor, 
                   taken_actions: Union[torch.Tensor, None] = None, 
                   inference=False) -> Tuple[torch.Tensor]:
        """Act randomly according to the action space

        :param states: Observation/state of the environment used to get the shape of the action space
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: None).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor or None, optional
        :param inference: Flag to indicate whether the model is making inference (default: False)
        :type inference: bool, optional

        :raises NotImplementedError: Unsupported action space

        :return: Random actions to be taken by the agent
        :rtype: tuple of torch.Tensor
        """
        # discrete action space (Discrete)
        if issubclass(type(self.action_space), gym.spaces.Discrete):
             return torch.randint(self.action_space.n, (states.shape[0], 1), device=self.device), None, None
        # continuous action space (Box)
        elif issubclass(type(self.action_space), gym.spaces.Box):
            if self._random_distribution is None:
                self._random_distribution = torch.distributions.uniform.Uniform(
                    low=torch.tensor(self.action_space.low[0], device=self.device, dtype=torch.float32),
                    high=torch.tensor(self.action_space.high[0], device=self.device, dtype=torch.float32))
            
            return self._random_distribution.sample(sample_shape=(states.shape[0], self.num_actions)), None, None
        else:
            raise NotImplementedError("Action space type ({}) not supported".format(type(self.action_space)))

    def init_parameters(self, method_name: str = "normal_", *args, **kwargs) -> None:
        """Initialize the model parameters according to the specified method name

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module. 
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name (default: "normal\_")
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional
        """
        for parameters in self.parameters():
            exec("torch.nn.init.{}(parameters, *args, **kwargs)".format(method_name))

    def init_weights(self, method_name: str = "orthogonal_", *args, **kwargs) -> None:
        """Initialize the model weights according to the specified method name
        
        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module. 
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        The following layers will be initialized:
        - torch.nn.Linear
        
        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name (default: "orthogonal\_")
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional
        """
        def _update_weights(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_weights(layer)
                elif isinstance(layer, torch.nn.Linear):
                    exec("torch.nn.init.{}(layer.weight, *args, **kwargs)".format(method_name))
        
        _update_weights(self.children(), method_name, args, kwargs)

    def forward(self):
        """Forward pass of the model

        :raises NotImplementedError: Child class must ``.act()`` and ``.compute()`` methods
        """
        raise NotImplementedError("Implement .act() and .compute() methods instead of this")

    def compute(self, 
                states: torch.Tensor, 
                taken_actions: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Define the computation performed (to be implemented by the inheriting classes) by the models

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: None).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor or None, optional

        :raises NotImplementedError: Child class must implement this method
        
        :return: Computation performed by the models
        :rtype: torch.Tensor or tuple of torch.Tensor
        """
        raise NotImplementedError("The computation performed by the models (.compute()) is not implemented")

    def act(self, 
            states: torch.Tensor, 
            taken_actions: Union[torch.Tensor, None] = None, 
            inference=False) -> Tuple[torch.Tensor]:
        """Act according to the specified behavior (to be implemented by the inheriting classes)

        Agents will call this method to obtain the decision to be taken given the state of the environment.
        This method is currently implemented by the helper models (**GaussianModel**, etc.).
        The classes that inherit from the latter must only implement the ``.compute()`` method

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: None).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor or None, optional
        :param inference: Flag to indicate whether the model is making inference (default: False)
        :type inference: bool, optional

        :raises NotImplementedError: Child class must implement this method
        
        :return: Action to be taken by the agent given the state of the environment.
                 The typical tuple's components are the actions, the log of the probability density function and mean actions.
                 Deterministic agents must ignore the last two components and return empty tensors or None for them
        :rtype: tuple of torch.Tensor
        """
        raise NotImplementedError("The action to be taken by the agent (.act()) is not implemented")
        
    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: "train" for training or "eval" for evaluation. 
            See `torch.nn.Module.train <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train>`_
        :type mode: str

        :raises ValueError: Mode must be ``"train"`` or ``"eval"``
        """
        if mode == "train":
            self.train(True)
        elif mode == "eval":
            self.train(False)
        else:
            raise ValueError("Invalid mode. Use 'train' for training or 'eval' for evaluation")

    def save(self, path):
        """Save the model to the specified path
            
        :param path: Path to save the model to
        :type path: str
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load the model from the specified path
                
        :param path: Path to load the model from
        :type path: str
        """
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze internal parameters

        - Freeze: disable gradient computation (``parameters.requires_grad = False``)
        - Unfreeze: enable gradient computation (``parameters.requires_grad = True``) 
        
        :param freeze: Freeze the internal parameters if True, otherwise unfreeze them
        :type freeze: bool, optional
        """
        for parameters in self.parameters():
            parameters.requires_grad = not freeze

    def update_parameters(self, model: torch.nn.Module, polyak: float = 1) -> None:
        """Update internal parameters by hard or soft (polyak averaging) update

        - Hard update: :math:`\\theta = \\theta_{net}`
        - Soft (polyak averaging) update: :math:`\\theta = (1 - \\rho) \\theta + \\rho \\theta_{net}`
        
        :param model: Model used to update the internal parameters
        :type model: torch.nn.Module (skrl.models.torch.Model)
        :param polyak: Polyak hyperparameter between 0 and 1 (usually close to 0).
                       A hard update is performed when its value is 1 (default)
        :type polyak: float, optional
        """
        with torch.no_grad():
            # hard update
            if polyak == 1:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.copy_(model_parameters.data)
            # soft update (use in-place operations to avoid creating new parameters)
            else:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.mul_(1 - polyak)
                    parameters.data.add_(polyak * model_parameters.data)
