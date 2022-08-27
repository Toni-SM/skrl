from typing import Optional, Union, Mapping, Sequence

import gym
import collections
import numpy as np

import torch

from skrl import logger


class Model(torch.nn.Module):
    def __init__(self, 
                 observation_space: Union[int, Sequence[int], gym.Space], 
                 action_space: Union[int, Sequence[int], gym.Space], 
                 device: Union[str, torch.device] = "cuda:0") -> None:
        """Base class representing a function approximator

        The following properties are defined:

        - ``device`` (torch.device): Device to be used for the computations
        - ``observation_space`` (int, sequence of int, gym.Space): Observation/state space
        - ``action_space`` (int, sequence of int, gym.Space): Action space
        - ``num_observations`` (int): Number of elements in the observation/state space
        - ``num_actions`` (int): Number of elements in the action space
        
        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gym.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gym.Space
        :param device: Device on which a torch tensor is or will be allocated (default: ``"cuda:0"``)
        :type device: str or torch.device, optional

        Custom models should override the ``act`` method::

            import torch
            from skrl.models.torch import Model

            class CustomModel(Model):
                def __init__(self, observation_space, action_space, device="cuda:0"):
                    Model.__init__(self, observation_space, action_space, device)

                    self.layer_1 = nn.Linear(self.num_observations, 64)
                    self.layer_2 = nn.Linear(64, self.num_actions)

                def act(self, states, taken_actions=None, inference=False, role=""):
                    x = F.relu(self.layer_1(states))
                    x = F.relu(self.layer_2(x))
                    return x
        """
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
                                 taken_actions: Optional[torch.Tensor] = None) -> Sequence[torch.Tensor]:
        """Get the output of the instantiator model
        
        Input shape depends on the instantiator (see skrl.utils.model_instantiator.Shape) as follows:

        - STATES / OBSERVATIONS = 0
        - ACTIONS = -1
        - STATES_ACTIONS = -2

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: ``None``)
        :type taken_actions: torch.Tensor, optional

        :return: Output of the instantiator model
        :rtype: sequence of torch.Tensor
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

    def _get_space_size(self, 
                        space: Union[int, Sequence[int], gym.Space],
                        number_of_elements: bool = True) -> int:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, sequence of int, or gym.Space
        :param number_of_elements: Whether the number of elements occupied by the space is returned (default: ``True``). 
                                   If ``False``, the shape of the space is returned. It only affects Discrete spaces
        :type number_of_elements: bool, optional

        :raises ValueError: If the space is not supported

        :return: Size of the space (number of elements)
        :rtype: int

        Example::

            # from int
            >>> model._get_space_size(2)
            2

            # from sequence of int
            >>> model._get_space_size([2, 3])
            6

            # Box space
            >>> space = gym.spaces.Box(low=-1, high=1, shape=(2, 3))
            >>> model._get_space_size(space)
            6

            # Discrete space
            >>> space = gym.spaces.Discrete(4)
            >>> model._get_space_size(space)
            4
            >>> model._get_space_size(space, number_of_elements=False)
            1

            # Dict space
            >>> space = gym.spaces.Dict({'a': gym.spaces.Box(low=-1, high=1, shape=(2, 3)), 
            ...                          'b': gym.spaces.Discrete(4)})
            >>> model._get_space_size(space)
            10
            >>> model._get_space_size(space, number_of_elements=False)
            7
        """
        size = None
        if type(space) in [int, float]:
            size = space
        elif type(space) in [tuple, list]:
            size = np.prod(space)
        elif issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                if number_of_elements:
                    size = space.n
                else:
                    size = 1
            elif issubclass(type(space), gym.spaces.Box):
                size = np.prod(space.shape)
            elif issubclass(type(space), gym.spaces.Dict):
                size = sum([self._get_space_size(space.spaces[key], number_of_elements) for key in space.spaces])
        if size is None:
            raise ValueError("Space type {} not supported".format(type(space)))
        return int(size)

    def tensor_to_space(self, 
                        tensor: torch.Tensor, 
                        space: gym.Space, 
                        start: int = 0) -> Union[torch.Tensor, dict]:
        """Map a flat tensor to a Gym space

        The mapping is done in the following way:

        - Tensors belonging to Discrete spaces are returned without modification
        - Tensors belonging to Box spaces are reshaped to the corresponding space shape 
          keeping the first dimension (number of samples) as they are
        - Tensors belonging to Dict spaces are mapped into a dictionary with the same keys as the original space

        :param tensor: Tensor to map from
        :type tensor: torch.Tensor
        :param space: Space to map the tensor to
        :type space: gym.Space
        :param start: Index of the first element of the tensor to map (default: ``0``)
        :type start: int, optional

        :raises ValueError: If the space is not supported

        :return: Mapped tensor or dictionary
        :rtype: torch.Tensor or dict

        Example::

            >>> space = gym.spaces.Dict({'a': gym.spaces.Box(low=-1, high=1, shape=(2, 3)), 
            ...                          'b': gym.spaces.Discrete(4)})
            >>> tensor = torch.tensor([[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 2]])
            >>>
            >>> model.tensor_to_space(tensor, space)
            {'a': tensor([[[-0.3000, -0.2000, -0.1000],
                           [ 0.1000,  0.2000,  0.3000]]]),
             'b': tensor([[2.]])}
        """
        if issubclass(type(space), gym.spaces.Discrete):
            return tensor
        elif issubclass(type(space), gym.spaces.Box):
            return tensor.view(tensor.shape[0], *space.shape)
        elif issubclass(type(space), gym.spaces.Dict):
            output = {}
            for k in sorted(space.keys()):
                end = start + self._get_space_size(space[k], number_of_elements=False)
                output[k] = self.tensor_to_space(tensor[:, start:end], space[k], end)
                start = end
            return output
        raise ValueError("Space type {} not supported".format(type(space)))

    def random_act(self, 
                   states: torch.Tensor, 
                   taken_actions: Optional[torch.Tensor] = None, 
                   inference: bool = False,
                   role: str = "") -> Sequence[torch.Tensor]:
        """Act randomly according to the action space

        :param states: Observation/state of the environment used to get the shape of the action space
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: ``None``).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor, optional
        :param inference: Flag to indicate whether the model is making inference (default: ``False``)
        :type inference: bool, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Unsupported action space

        :return: Random actions to be taken by the agent
        :rtype: sequence of torch.Tensor
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

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name (default: ``"normal_"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all parameters with an orthogonal distribution with a gain of 0.5
            >>> model.init_parameters("orthogonal_", gain=0.5)

            # initialize all parameters as a sparse matrix with a sparsity of 0.1
            >>> model.init_parameters("sparse_", sparsity=0.1)
        """
        for parameters in self.parameters():
            exec("torch.nn.init.{}(parameters, *args, **kwargs)".format(method_name))

    def init_weights(self, method_name: str = "orthogonal_", *args, **kwargs) -> None:
        """Initialize the model weights according to the specified method name
        
        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module. 
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        The following layers will be initialized:
        - torch.nn.Linear
        
        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name (default: ``"orthogonal_"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all weights with uniform distribution in range [-0.1, 0.1]
            >>> model.init_weights(method_name="uniform_", a=-0.1, b=0.1)

            # initialize all weights with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_weights(method_name="normal_", mean=0.0, std=0.25)
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
                taken_actions: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """Define the computation performed (to be implemented by the inheriting classes) by the models

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: ``None``).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor, optional

        :raises NotImplementedError: Child class must implement this method
        
        :return: Computation performed by the models
        :rtype: torch.Tensor or sequence of torch.Tensor
        """
        raise NotImplementedError("The computation performed by the models (.compute()) is not implemented")

    def act(self, 
            states: torch.Tensor, 
            taken_actions: Optional[torch.Tensor] = None, 
            inference: bool = False,
            role: str = "") -> Sequence[torch.Tensor]:
        """Act according to the specified behavior (to be implemented by the inheriting classes)

        Agents will call this method to obtain the decision to be taken given the state of the environment.
        This method is currently implemented by the helper models (**GaussianModel**, etc.).
        The classes that inherit from the latter must only implement the ``.compute()`` method

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: ``None``).
                              The use of these actions only makes sense in critical models, e.g.
        :type taken_actions: torch.Tensor, optional
        :param inference: Flag to indicate whether the model is making inference (default: ``False``)
        :type inference: bool, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Child class must implement this method
        
        :return: Action to be taken by the agent given the state of the environment.
                 The typical sequence's components are the actions, the log of the probability density function and mean actions.
                 Deterministic agents must ignore the last two components and return empty tensors or None for them
        :rtype: sequence of torch.Tensor
        """
        logger.warn("Make sure to place Mixins before Model during model definition")
        raise NotImplementedError("The action to be taken by the agent (.act()) is not implemented")
        
    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: ``"train"`` for training or ``"eval"`` for evaluation. 
            See `torch.nn.Module.train <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train>`_
        :type mode: str

        :raises ValueError: If the mode is not ``"train"`` or ``"eval"``
        """
        if mode == "train":
            self.train(True)
        elif mode == "eval":
            self.train(False)
        else:
            raise ValueError("Invalid mode. Use 'train' for training or 'eval' for evaluation")

    def save(self, path: str, state_dict: Optional[dict] = None) -> None:
        """Save the model to the specified path
            
        :param path: Path to save the model to
        :type path: str
        :param state_dict: State dictionary to save (default: ``None``).
                           If None, the model's state_dict will be saved
        :type state_dict: dict, optional

        Example::

            # save the current model to the specified path
            >>> model.save("/tmp/model.pt")

            # save an older version of the model to the specified path
            >>> old_state_dict = copy.deepcopy(model.state_dict())
            >>> # ...
            >>> model.save("/tmp/model.pt", old_state_dict)

        """
        torch.save(self.state_dict() if state_dict is None else state_dict, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str

        Example::

            # load the model onto the CPU
            >>> model = Model(observation_space, action_space, device="cpu")
            >>> model.load("model.pt")

            # load the model onto the GPU 1
            >>> model = Model(observation_space, action_space, device="cuda:1")
            >>> model.load("model.pt")
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()

    def migrate(self,
                state_dict: Mapping[str, torch.Tensor],
                name_map: Mapping[str, str] = {},
                auto_mapping: bool = True,
                show_names: bool = False) -> bool:
        """Migrate the specified extrernal model's state dict to the current model

        :param state_dict: External model's state dict to migrate from
        :type state_dict: Mapping[str, torch.Tensor]
        :param name_map: Name map to use for the migration (default: ``{}``).
                         Keys are the current parameter names and values are the external parameter names
        :type name_map: Mapping[str, str], optional
        :param auto_mapping: Automatically map the external state dict to the current state dict (default: ``True``)
        :type auto_mapping: bool, optional
        :param show_names: Show the names of both, current and external state dicts parameters (default: ``False``)
        :type show_names: bool, optional

        :return: True if the migration was successful, False otherwise.
                 Migration is successful if all parameters of the current model are found in the external model
        :rtype: bool
        """
        # Show state_dict
        if show_names:
            print("Model migration")
            print("Current state_dict:")
            for name, tensor in self.state_dict().items():
                print("  |-- {} : {}".format(name, tensor.shape))
            print("Source state_dict:")
            for name, tensor in state_dict.items():
                print("  |-- {} : {}".format(name, tensor.shape))

        # migrate the state dict to current model
        new_state_dict = collections.OrderedDict()
        match_counter = collections.defaultdict(list)
        used_counter = collections.defaultdict(list)
        for name, tensor in self.state_dict().items():
            for external_name, external_tensor in state_dict.items():
                # mapped names
                if name_map.get(name, "") == external_name:
                    if tensor.shape == external_tensor.shape:
                        new_state_dict[name] = external_tensor
                        match_counter[name].append(external_name)
                        used_counter[external_name].append(name)
                        break
                    else:
                        print("Shape mismatch for {} <- {} : {} != {}".format(name, external_name, tensor.shape, external_tensor.shape))
                # auto-mapped names
                if auto_mapping:
                    if tensor.shape == external_tensor.shape:
                        if name.endswith(".weight"):
                            if external_name.endswith(".weight"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                        elif name.endswith(".bias"):
                            if external_name.endswith(".bias"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                        else:
                            if not external_name.endswith(".weight") and not external_name.endswith(".bias"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)

        # show ambiguous matches
        status = True
        for name, tensor in self.state_dict().items():
            if len(match_counter.get(name, [])) > 1:
                print("Ambiguous match for {} <- {}".format(name, match_counter.get(name, [])))
                status = False
        # show missing matches
        for name, tensor in self.state_dict().items():
            if not match_counter.get(name, []):
                print("Missing match for {}".format(name))
                status = False
        # show duplicated uses
        for name, tensor in state_dict.items():
            if len(used_counter.get(name, [])) > 1:
                print("Duplicated use of {} -> {}".format(name, used_counter.get(name, [])))
                status = False

        # load new state dict
        self.load_state_dict(new_state_dict, strict=False)
        self.eval()

        return status
    
    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze internal parameters

        - Freeze: disable gradient computation (``parameters.requires_grad = False``)
        - Unfreeze: enable gradient computation (``parameters.requires_grad = True``) 
        
        :param freeze: Freeze the internal parameters if True, otherwise unfreeze them (default: ``True``)
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
        :param polyak: Polyak hyperparameter between 0 and 1 (default: ``1``).
                       A hard update is performed when its value is 1
        :type polyak: float, optional

        Example::

            # hard update (from source model)
            >>> model.update_parameters(source_model)

            # soft update (from source model)
            >>> model.update_parameters(source_model, polyak=0.005)
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
