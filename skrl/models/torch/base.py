from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import collections
from abc import ABC, abstractmethod
import gymnasium
from packaging import version

import torch

from skrl import config, logger
from skrl.utils.spaces.torch import compute_space_size, flatten_tensorized_space, sample_space


class Model(torch.nn.Module, ABC):
    def __init__(
        self,
        *,
        observation_space: Optional[gymnasium.Space] = None,
        state_space: Optional[gymnasium.Space] = None,
        action_space: Optional[gymnasium.Space] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Base model class for implementing custom models.

        :param observation_space: Observation space. The ``num_observations`` property will contain the size of the space.
        :param state_space: State space. The ``num_states`` property will contain the size of the space.
        :param action_space: Action space. The ``num_actions`` property will contain the size of the space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        """
        super(Model, self).__init__()

        self.device = config.torch.parse_device(device)

        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space
        self.num_observations = compute_space_size(observation_space)
        self.num_states = compute_space_size(state_space)
        self.num_actions = compute_space_size(action_space)

    def init_state_dict(self, inputs: Mapping[str, Union[torch.Tensor, Any]] = {}, *, role: str = "") -> None:
        """Initialize lazy PyTorch modules' parameters.

        .. hint::

            Calling this method only makes sense when using models that contain lazy PyTorch modules
            (e.g. model instantiators), and always before performing any operation on model parameters.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.

            If not specified, ``inputs`` will have random samples from the observation, state and action spaces.
        :param role: Role played by the model.
        """
        if not inputs:
            inputs = {
                "observations": flatten_tensorized_space(
                    sample_space(self.observation_space, backend="native", device=self.device)
                ),
                "states": flatten_tensorized_space(
                    sample_space(self.state_space, backend="native", device=self.device)
                ),
                "taken_actions": flatten_tensorized_space(
                    sample_space(self.action_space, backend="native", device=self.device)
                ),
            }
        # init parameters
        self.to(device=self.device)
        self.compute(inputs=inputs, role=role)

    def random_act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], *, role: str = ""
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Act randomly according to the action space.

        .. warning::

            Sampling from unbounded action spaces may lead to numerical instabilities.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.

        :param role: Role played by the model.

        :return: Randomly sampled actions with the same batch size as the given observation (``"observations"``)
            in the ``inputs`` as the first component. The second component is an empty dictionary.

        :raises ValueError: Unsupported action space.
        """
        sample = sample_space(
            self.action_space, batch_size=inputs["observations"].shape[0], backend="native", device=self.device
        )
        return flatten_tensorized_space(sample), {}

    def init_parameters(self, method_name: str = "normal_", *args, **kwargs) -> None:
        """Initialize the model parameters according to the specified method name.

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are ``"uniform_"``, ``"normal_"``, ``"constant_"``, etc.

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name.
        :param args: Positional arguments of the method to be called.
        :param kwargs: Key-value arguments of the method to be called.

        Example::

            # initialize all parameters with an orthogonal distribution with a gain of 0.5
            >>> model.init_parameters("orthogonal_", gain=0.5)

            # initialize all parameters as a sparse matrix with a sparsity of 0.1
            >>> model.init_parameters("sparse_", sparsity=0.1)
        """
        for parameters in self.parameters():
            exec(f"torch.nn.init.{method_name}(parameters, *args, **kwargs)")

    def init_weights(self, method_name: str = "orthogonal_", *args, **kwargs) -> None:
        """Initialize the model weights according to the specified method name.

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are ``"uniform_"``, ``"normal_"``, ``"constant_"``, etc.

        The following layers will be initialized:

        - torch.nn.Linear

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name.
        :param args: Positional arguments of the method to be called.
        :param kwargs: Key-value arguments of the method to be called.

        Example::

            # initialize all weights with uniform distribution in range [-0.1, 0.1]
            >>> model.init_weights(method_name="uniform_", a=-0.1, b=0.1)

            # initialize all weights with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_weights(method_name="normal_", mean=0.0, std=0.25)
        """

        def _update_weights(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_weights(layer, method_name, args, kwargs)
                elif isinstance(layer, torch.nn.Linear):
                    exec(f"torch.nn.init.{method_name}(layer.weight, *args, **kwargs)")

        _update_weights(self.children(), method_name, args, kwargs)

    def init_biases(self, method_name: str = "constant_", *args, **kwargs) -> None:
        """Initialize the model biases according to the specified method name.

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are ``"uniform_"``, ``"normal_"``, ``"constant_"``, etc.

        The following layers will be initialized:

        - torch.nn.Linear

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name.
        :param args: Positional arguments of the method to be called.
        :param kwargs: Key-value arguments of the method to be called.

        Example::

            # initialize all biases with a constant value (0)
            >>> model.init_biases(method_name="constant_", val=0)

            # initialize all biases with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_biases(method_name="normal_", mean=0.0, std=0.25)
        """

        def _update_biases(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_biases(layer, method_name, args, kwargs)
                elif isinstance(layer, torch.nn.Linear):
                    exec(f"torch.nn.init.{method_name}(layer.bias, *args, **kwargs)")

        _update_biases(self.children(), method_name, args, kwargs)

    def get_specification(self) -> Mapping[str, Any]:
        """Returns the specification of the model.

        The following keys are used by the agents for initialization:

        - ``"rnn"``: Recurrent Neural Network (RNN) specification for RNN, LSTM and GRU layers/cells.
        - ``"sizes"``: List of RNN shapes (number of layers, number of environments, number of features in the RNN state).
          There must be as many tuples as there are states in the recurrent layer/cell.
          E.g.: LSTM has 2 states (hidden and cell).

        :return: Dictionary containing advanced specification of the model.

        Example::

            # model with a LSTM layer
            # - number of layers: 1
            # - number of environments: 4
            # - number of features in the RNN state: 64
            >>> model.get_specification()
            {'rnn': {'sizes': [(1, 4, 64), (1, 4, 64)]}}
        """
        return {}

    def forward(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], *, role: str = ""
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Forward pass of the model.

        .. note::

            This method calls the :py:meth:`act` method and returns its outputs.
            It exists for compatibility with the :py:class:`torch.nn.Module` class.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing extra output values according to the model.
        """
        return self.act(inputs, role=role)

    @abstractmethod
    def compute(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], *, role: str = ""
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Define the computation performed by the model.

        .. warning::

            This method is abstract and must be implemented by subclasses.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Computation performed by the model.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("The computation performed by the model (.compute()) is not implemented")

    @abstractmethod
    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], *, role: str = ""
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Act according to the specified behavior.

        Agents will call this method to get the expected action/value based on the observations/states.

        .. warning::

            This method is currently implemented by the helper models (e.g.: :py:class:`~skrl.models.torch.gaussian.GaussianMixin`).
            The classes that inherit from the latter must only implement the :py:meth:`compute` method.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing extra output values according to the model.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        logger.warning("Make sure to place Mixins classes before the Model class during model definition")
        raise NotImplementedError("The action to be taken by the agent (.act()) is not implemented")

    def enable_training_mode(self, enabled: bool = True) -> None:
        """Set the training mode of the model: enabled (training) or disabled (evaluation).

        :param enabled: True to enable the training mode, False to enable the evaluation mode.
            See :py:meth:`torch.nn.Module.train` for more details.
        """
        self.train(enabled)

    def save(self, path: str, state_dict: Optional[dict] = None) -> None:
        """Save the model to the specified path.

        :param path: Path to save the model to.
        :param state_dict: State dictionary to save. If ``None``, the model's ``state_dict`` will be saved.

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
        """Load the model from the specified path.

        .. note::

            The final storage device is determined by the constructor of the model.

        :param path: Path to load the model from.

        Example::

            # load the model onto the CPU
            >>> model = Model(device="cpu")
            >>> model.load("model.pt")

            # load the model onto the GPU 1
            >>> model = Model(device="cuda:1")
            >>> model.load("model.pt")
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            state_dict = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        else:
            state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()

    def migrate(
        self,
        *,
        state_dict: Optional[Mapping[str, torch.Tensor]] = None,
        path: Optional[str] = None,
        name_map: Mapping[str, str] = {},
        auto_mapping: bool = True,
        verbose: bool = False,
    ) -> bool:
        """Migrate the specified external model's ``state_dict`` to the current model.

        .. note::

            The final storage device is determined by the constructor of the model.

        Only one of ``state_dict`` or ``path`` can be specified.
        The ``path`` parameter allows automatic loading the ``state_dict`` only from files generated
        by the *rl_games* and *stable-baselines3* libraries at the moment.

        For ambiguous models (where 2 or more parameters, for source or current model, have equal shape)
        it is necessary to define the ``name_map``, at least for those parameters, to perform the migration successfully.

        :param state_dict: External model's ``state_dict`` to migrate from.
        :param path: Path to the external checkpoint to migrate from.
        :param name_map: Name map to use for the migration.
            Keys are the current parameter names and values are the external parameter names.
        :param auto_mapping: Automatically map the external ``state_dict`` to the current ``state_dict``.
        :param verbose: Show model names and migration.

        :return: True if the migration was successful, False otherwise.
            Migration is successful if all parameters of the current model are found in the external model.

        :raises ValueError: If neither or both of ``state_dict`` and ``path`` parameters have been set.
        :raises ValueError: If the correct file type cannot be identified from the ``path`` parameter.

        Example::

            # migrate a rl_games checkpoint with unambiguous state_dict
            >>> model.migrate(path="./runs/Ant/nn/Ant.pth")
            True

            # migrate a rl_games checkpoint with ambiguous state_dict
            >>> model.migrate(path="./runs/Cartpole/nn/Cartpole.pth", verbose=False)
            [skrl:WARNING] Ambiguous match for log_std_parameter <- [value_mean_std.running_mean, value_mean_std.running_var, a2c_network.sigma]
            [skrl:WARNING] Ambiguous match for net.0.bias <- [a2c_network.actor_mlp.0.bias, a2c_network.actor_mlp.2.bias]
            [skrl:WARNING] Ambiguous match for net.2.bias <- [a2c_network.actor_mlp.0.bias, a2c_network.actor_mlp.2.bias]
            [skrl:WARNING] Ambiguous match for net.4.weight <- [a2c_network.value.weight, a2c_network.mu.weight]
            [skrl:WARNING] Ambiguous match for net.4.bias <- [a2c_network.value.bias, a2c_network.mu.bias]
            [skrl:WARNING] Multiple use of a2c_network.actor_mlp.0.bias -> [net.0.bias, net.2.bias]
            [skrl:WARNING] Multiple use of a2c_network.actor_mlp.2.bias -> [net.0.bias, net.2.bias]
            False
            >>> name_map = {"log_std_parameter": "a2c_network.sigma",
            ...             "net.0.bias": "a2c_network.actor_mlp.0.bias",
            ...             "net.2.bias": "a2c_network.actor_mlp.2.bias",
            ...             "net.4.weight": "a2c_network.mu.weight",
            ...             "net.4.bias": "a2c_network.mu.bias"}
            >>> model.migrate(path="./runs/Cartpole/nn/Cartpole.pth", name_map=name_map, verbose=True)
            [skrl:INFO] Models
            [skrl:INFO]   |-- current: 7 items
            [skrl:INFO]   |    |-- log_std_parameter : torch.Size([1])
            [skrl:INFO]   |    |-- net.0.weight : torch.Size([32, 4])
            [skrl:INFO]   |    |-- net.0.bias : torch.Size([32])
            [skrl:INFO]   |    |-- net.2.weight : torch.Size([32, 32])
            [skrl:INFO]   |    |-- net.2.bias : torch.Size([32])
            [skrl:INFO]   |    |-- net.4.weight : torch.Size([1, 32])
            [skrl:INFO]   |    |-- net.4.bias : torch.Size([1])
            [skrl:INFO]   |-- source: 15 items
            [skrl:INFO]   |    |-- value_mean_std.running_mean : torch.Size([1])
            [skrl:INFO]   |    |-- value_mean_std.running_var : torch.Size([1])
            [skrl:INFO]   |    |-- value_mean_std.count : torch.Size([])
            [skrl:INFO]   |    |-- running_mean_std.running_mean : torch.Size([4])
            [skrl:INFO]   |    |-- running_mean_std.running_var : torch.Size([4])
            [skrl:INFO]   |    |-- running_mean_std.count : torch.Size([])
            [skrl:INFO]   |    |-- a2c_network.sigma : torch.Size([1])
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.0.weight : torch.Size([32, 4])
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.0.bias : torch.Size([32])
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.2.weight : torch.Size([32, 32])
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.2.bias : torch.Size([32])
            [skrl:INFO]   |    |-- a2c_network.value.weight : torch.Size([1, 32])
            [skrl:INFO]   |    |-- a2c_network.value.bias : torch.Size([1])
            [skrl:INFO]   |    |-- a2c_network.mu.weight : torch.Size([1, 32])
            [skrl:INFO]   |    |-- a2c_network.mu.bias : torch.Size([1])
            [skrl:INFO] Migration
            [skrl:INFO]   |-- map:  log_std_parameter <- a2c_network.sigma
            [skrl:INFO]   |-- auto: net.0.weight <- a2c_network.actor_mlp.0.weight
            [skrl:INFO]   |-- map:  net.0.bias <- a2c_network.actor_mlp.0.bias
            [skrl:INFO]   |-- auto: net.2.weight <- a2c_network.actor_mlp.2.weight
            [skrl:INFO]   |-- map:  net.2.bias <- a2c_network.actor_mlp.2.bias
            [skrl:INFO]   |-- map:  net.4.weight <- a2c_network.mu.weight
            [skrl:INFO]   |-- map:  net.4.bias <- a2c_network.mu.bias
            False

            # migrate a stable-baselines3 checkpoint with unambiguous state_dict
            >>> model.migrate(path="./ddpg_pendulum.zip")
            True

            # migrate from any exported model by loading its state_dict (unambiguous state_dict)
            >>> state_dict = torch.load("./external_model.pt")
            >>> model.migrate(state_dict=state_dict)
            True
        """
        if (state_dict is not None) + (path is not None) != 1:
            raise ValueError("Exactly one of state_dict or path may be specified")

        # load state_dict from path
        if path is not None:
            state_dict = {}
            # rl_games checkpoint
            if path.endswith(".pt") or path.endswith(".pth"):
                checkpoint = torch.load(path, map_location=self.device)
                if type(checkpoint) is dict:
                    state_dict = checkpoint.get("model", {})
            # stable-baselines3
            elif path.endswith(".zip"):
                import zipfile

                try:
                    archive = zipfile.ZipFile(path, "r")
                    with archive.open("policy.pth", mode="r") as file:
                        state_dict = torch.load(file, map_location=self.device)
                except KeyError as e:
                    logger.warning(str(e))
                    state_dict = {}
            else:
                raise ValueError("Cannot identify file type")

        # show state_dict
        if verbose:
            logger.info("Models")
            logger.info(f"  |-- current: {len(self.state_dict().keys())} items")
            for name, tensor in self.state_dict().items():
                logger.info(f"  |    |-- {name} : {list(tensor.shape)}")
            logger.info(f"  |-- source: {len(state_dict.keys())} items")
            for name, tensor in state_dict.items():
                logger.info(f"  |    |-- {name} : {list(tensor.shape)}")
            logger.info("Migration")

        # migrate the state_dict to current model
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
                        if verbose:
                            logger.info(f"  |-- map:  {name} <- {external_name}")
                        break
                    else:
                        logger.warning(
                            f"Shape mismatch for {name} <- {external_name} : {tensor.shape} != {external_tensor.shape}"
                        )
                # auto-mapped names
                if auto_mapping and name not in name_map:
                    if tensor.shape == external_tensor.shape:
                        if name.endswith(".weight"):
                            if external_name.endswith(".weight"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")
                        elif name.endswith(".bias"):
                            if external_name.endswith(".bias"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")
                        else:
                            if not external_name.endswith(".weight") and not external_name.endswith(".bias"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")

        # show ambiguous matches
        status = True
        for name, tensor in self.state_dict().items():
            if len(match_counter.get(name, [])) > 1:
                logger.warning("Ambiguous match for {} <- [{}]".format(name, ", ".join(match_counter.get(name, []))))
                status = False
        # show missing matches
        for name, tensor in self.state_dict().items():
            if not match_counter.get(name, []):
                logger.warning(f"Missing match for {name}")
                status = False
        # show multiple uses
        for name, tensor in state_dict.items():
            if len(used_counter.get(name, [])) > 1:
                logger.warning("Multiple use of {} -> [{}]".format(name, ", ".join(used_counter.get(name, []))))
                status = False

        # load new state dict
        self.load_state_dict(new_state_dict, strict=False)
        self.eval()

        return status

    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze internal parameters.

        - Freeze: disable gradient computation (``parameters.requires_grad = False``).
        - Unfreeze: enable gradient computation (``parameters.requires_grad = True``).

        :param freeze: Whether to freeze or unfreeze the internal parameters.

        Example::

            # freeze model parameters
            >>> model.freeze_parameters(True)

            # unfreeze model parameters
            >>> model.freeze_parameters(False)
        """
        for parameters in self.parameters():
            parameters.requires_grad = not freeze

    def update_parameters(self, model: torch.nn.Module, *, polyak: float = 1.0) -> None:
        """Update internal parameters by hard or soft (polyak averaging) update.

        - Hard update: :math:`\\theta = \\theta_{net}`
        - Soft (polyak averaging) update: :math:`\\theta = (1 - \\rho) \\theta + \\rho \\theta_{net}`

        :param model: Model used to update the internal parameters.
        :param polyak: Polyak hyperparameter between 0 and 1. A hard update is performed when its value is 1.

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

    def broadcast_parameters(self, rank: int = 0):
        """Broadcast model parameters to the whole group (e.g.: across all nodes) in distributed runs.

        After calling this method, the distributed model will contain the broadcasted parameters from ``rank``.

        :param rank: Worker/process rank from which to broadcast model parameters.

        Example::

            # broadcast model parameter from worker/process with rank 1
            >>> if config.torch.is_distributed:
            ...     model.broadcast_parameters(rank=1)
        """
        object_list = [self.state_dict()]
        torch.distributed.broadcast_object_list(object_list, rank)
        self.load_state_dict(object_list[0])

    def reduce_parameters(self):
        """Reduce model parameters across all workers/processes in the whole group (e.g.: across all nodes).

        After calling this method, the distributed model parameters will be bitwise identical for all workers/processes.

        Example::

            # reduce model parameter across all workers/processes
            >>> if config.torch.is_distributed:
            ...     model.reduce_parameters()
        """
        # batch all_reduce ops: https://github.com/entity-neural-network/incubator/pull/220
        gradients = []
        for parameters in self.parameters():
            if parameters.grad is not None:
                gradients.append(parameters.grad.view(-1))
        gradients = torch.cat(gradients)

        torch.distributed.all_reduce(gradients, op=torch.distributed.ReduceOp.SUM)

        offset = 0
        for parameters in self.parameters():
            if parameters.grad is not None:
                parameters.grad.data.copy_(
                    gradients[offset : offset + parameters.numel()].view_as(parameters.grad.data)
                    / config.torch.world_size
                )
                offset += parameters.numel()
