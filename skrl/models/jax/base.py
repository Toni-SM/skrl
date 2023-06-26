from typing import Optional, Union, Mapping, Sequence, Tuple, Any, Callable

import gym
import gymnasium
import collections
import numpy as np

import jax
import jaxlib
import jax.numpy as jnp
import flax

from skrl import logger
from skrl import config


class StateDict(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, **kwargs):
        return cls(apply_fn=apply_fn, params=params, **kwargs)


class Model(flax.linen.Module):
    observation_space: Union[int, Sequence[int], gym.Space, gymnasium.Space]
    action_space: Union[int, Sequence[int], gym.Space, gymnasium.Space]
    device: Optional[Union[str, jaxlib.xla_extension.Device]] = None

    def __init__(self,
                 observation_space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                 action_space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                 device: Optional[Union[str, jaxlib.xla_extension.Device]] = None,
                 parent: Optional[Any] = None,
                 name: Optional[str] = None) -> None:
        """Base class representing a function approximator

        The following properties are defined:

        - ``device`` (jaxlib.xla_extension.Device): Device to be used for the computations
        - ``observation_space`` (int, sequence of int, gym.Space, gymnasium.Space): Observation/state space
        - ``action_space`` (int, sequence of int, gym.Space, gymnasium.Space): Action space
        - ``num_observations`` (int): Number of elements in the observation/state space
        - ``num_actions`` (int): Number of elements in the action space

        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gym.Space, gymnasium.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gym.Space, gymnasium.Space
        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or jaxlib.xla_extension.Device, optional
        :param parent: The parent Module of this Module (default: ``None``).
                       It is a Flax reserved attribute
        :type parent: str, optional
        :param name: The name of this Module (default: ``None``).
                     It is a Flax reserved attribute
        :type name: str, optional

        Custom models should override the ``act`` method::

            import flax.linen as nn
            from skrl.models.jax import Model

            class CustomModel(Model):
                def __init__(self, observation_space, action_space, device=None, **kwargs):
                    Model.__init__(self, observation_space, action_space, device, **kwargs)

                    # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.IncorrectPostInitOverrideError
                    flax.linen.Module.__post_init__(self)

                @nn.compact
                def __call__(self, inputs, role):
                    x = nn.relu(nn.Dense(64)(inputs["states"]))
                    x = nn.relu(nn.Dense(self.num_actions)(x))
                    return x, None, {}
        """
        if device is None:
            self.device = jax.devices()[0]
        else:
            self.device = device if isinstance(device, jaxlib.xla_extension.Device) else jax.devices(device)[0]

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_observations = None if observation_space is None else self._get_space_size(observation_space)
        self.num_actions = None if action_space is None else self._get_space_size(action_space)

        self.state_dict: StateDict
        self.training = False

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ReservedModuleAttributeError
        self.parent = parent
        self.name = name

    def init_state_dict(self, key, inputs, role):
        if isinstance(inputs["states"], (int, np.int32, np.int64)):
            inputs["states"] = np.array(inputs["states"]).reshape(-1,1)
        self.state_dict = StateDict.create(apply_fn=self.apply,
                                           params=self.init(key, inputs, role),)

    def _get_space_size(self,
                        space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                        number_of_elements: bool = True) -> int:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, sequence of int, gym.Space, or gymnasium.Space
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
        elif issubclass(type(space), gymnasium.Space):
            if issubclass(type(space), gymnasium.spaces.Discrete):
                if number_of_elements:
                    size = space.n
                else:
                    size = 1
            elif issubclass(type(space), gymnasium.spaces.Box):
                size = np.prod(space.shape)
            elif issubclass(type(space), gymnasium.spaces.Dict):
                size = sum([self._get_space_size(space.spaces[key], number_of_elements) for key in space.spaces])
        if size is None:
            raise ValueError("Space type {} not supported".format(type(space)))
        return int(size)

    def tensor_to_space(self,
                        tensor: jnp.ndarray,
                        space: Union[gym.Space, gymnasium.Space],
                        start: int = 0) -> Union[jnp.ndarray, dict]:
        """Map a flat tensor to a Gym/Gymnasium space

        The mapping is done in the following way:

        - Tensors belonging to Discrete spaces are returned without modification
        - Tensors belonging to Box spaces are reshaped to the corresponding space shape
          keeping the first dimension (number of samples) as they are
        - Tensors belonging to Dict spaces are mapped into a dictionary with the same keys as the original space

        :param tensor: Tensor to map from
        :type tensor: jnp.ndarray
        :param space: Space to map the tensor to
        :type space: gym.Space or gymnasium.Space
        :param start: Index of the first element of the tensor to map (default: ``0``)
        :type start: int, optional

        :raises ValueError: If the space is not supported

        :return: Mapped tensor or dictionary
        :rtype: jnp.ndarray or dict

        Example::

            >>> space = gym.spaces.Dict({'a': gym.spaces.Box(low=-1, high=1, shape=(2, 3)),
            ...                          'b': gym.spaces.Discrete(4)})
            >>> tensor = jnp.array([[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 2]])
            >>>
            >>> model.tensor_to_space(tensor, space)
            {'a': Array([[[-0.3, -0.2, -0.1],
                          [ 0.1,  0.2,  0.3]]], dtype=float32),
             'b': Array([[2.]], dtype=float32)}
        """
        if issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                return tensor
            elif issubclass(type(space), gym.spaces.Box):
                return tensor.reshape(tensor.shape[0], *space.shape)
            elif issubclass(type(space), gym.spaces.Dict):
                output = {}
                for k in sorted(space.keys()):
                    end = start + self._get_space_size(space[k], number_of_elements=False)
                    output[k] = self.tensor_to_space(tensor[:, start:end], space[k], end)
                    start = end
                return output
        else:
            if issubclass(type(space), gymnasium.spaces.Discrete):
                return tensor
            elif issubclass(type(space), gymnasium.spaces.Box):
                return tensor.reshape(tensor.shape[0], *space.shape)
            elif issubclass(type(space), gymnasium.spaces.Dict):
                output = {}
                for k in sorted(space.keys()):
                    end = start + self._get_space_size(space[k], number_of_elements=False)
                    output[k] = self.tensor_to_space(tensor[:, start:end], space[k], end)
                    start = end
                return output
        raise ValueError("Space type {} not supported".format(type(space)))

    def random_act(self,
                   inputs: Mapping[str, Union[jnp.ndarray, Any]],
                   role: str = "",
                   params: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None], Mapping[str, Union[jnp.ndarray, Any]]]:
        """Act randomly according to the action space

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically jnp.ndarray
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        :param params: Parameters used to compute the output (default: ``None``).
                       If ``None``, internal parameters will be used
        :type params: jnp.array

        :raises NotImplementedError: Unsupported action space

        :return: Model output. The first component is the action to be taken by the agent
        :rtype: tuple of jnp.ndarray, None, and dictionary
        """
        # discrete action space (Discrete)
        if issubclass(type(self.action_space), gym.spaces.Discrete) or issubclass(type(self.action_space), gymnasium.spaces.Discrete):
             return np.random.randint(self.action_space.n, size=(inputs["states"].shape[0], 1)), None, {}
        # continuous action space (Box)
        elif issubclass(type(self.action_space), gym.spaces.Box) or issubclass(type(self.action_space), gymnasium.spaces.Box):
            return np.random.uniform(low=self.action_space.low[0], high=self.action_space.high[0], size=(inputs["states"].shape[0], self.num_actions)), None, {}
        else:
            raise NotImplementedError("Action space type ({}) not supported".format(type(self.action_space)))

    def init_parameters(self, method_name: str = "normal", *args, **kwargs) -> None:
        """Initialize the model parameters according to the specified method name

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.initializers>`_ module.
        Allowed method names are *uniform*, *normal*, *constant*, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.initializers>`_ method name (default: ``"normal"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all parameters with an orthogonal distribution with a scale of 0.5
            >>> model.init_parameters("orthogonal", scale=0.5)

            # initialize all parameters as a normal distribution with a standard deviation of 0.1
            >>> model.init_parameters("normal", stddev=0.1)
        """
        if method_name in ["ones", "zeros"]:
            method = eval(f"flax.linen.initializers.{method_name}")
        else:
            method = eval(f"flax.linen.initializers.{method_name}(*args, **kwargs)")
        params = jax.tree_util.tree_map(lambda param: method(config.jax.key, param.shape), self.state_dict.params)
        self.state_dict = self.state_dict.replace(params=params)

    def init_weights(self, method_name: str = "normal", *args, **kwargs) -> None:
        """Initialize the model weights according to the specified method name

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.initializers>`_ module.
        Allowed method names are *uniform*, *normal*, *constant*, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.initializers>`_ method name (default: ``"normal"``)
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
        if method_name in ["ones", "zeros"]:
            method = eval(f"flax.linen.initializers.{method_name}")
        else:
            method = eval(f"flax.linen.initializers.{method_name}(*args, **kwargs)")
        params = jax.tree_util.tree_map_with_path(lambda path, param: method(config.jax.key, param.shape) if path[-1].key == "kernel" else param,
                                                  self.state_dict.params)
        self.state_dict = self.state_dict.replace(params=params)

    def init_biases(self, method_name: str = "constant_", *args, **kwargs) -> None:
        """Initialize the model biases according to the specified method name

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.initializers>`_ module.
        Allowed method names are *uniform*, *normal*, *constant*, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module-flax.linen.initializers>`_ method name (default: ``"normal"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all biases with a constant value (0)
            >>> model.init_biases(method_name="constant_", val=0)

            # initialize all biases with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_biases(method_name="normal_", mean=0.0, std=0.25)
        """
        if method_name in ["ones", "zeros"]:
            method = eval(f"flax.linen.initializers.{method_name}")
        else:
            method = eval(f"flax.linen.initializers.{method_name}(*args, **kwargs)")
        params = jax.tree_util.tree_map_with_path(lambda path, param: method(config.jax.key, param.shape) if path[-1].key == "bias" else param,
                                                  self.state_dict.params)
        self.state_dict = self.state_dict.replace(params=params)

    def get_specification(self) -> Mapping[str, Any]:
        """Returns the specification of the model

        The following keys are used by the agents for initialization:

        - ``"rnn"``: Recurrent Neural Network (RNN) specification for RNN, LSTM and GRU layers/cells

          - ``"sizes"``: List of RNN shapes (number of layers, number of environments, number of features in the RNN state).
            There must be as many tuples as there are states in the recurrent layer/cell. E.g., LSTM has 2 states (hidden and cell).

        :return: Dictionary containing advanced specification of the model
        :rtype: dict

        Example::

            # model with a LSTM layer.
            # - number of layers: 1
            # - number of environments: 4
            # - number of features in the RNN state: 64
            >>> model.get_specification()
            {'rnn': {'sizes': [(1, 4, 64), (1, 4, 64)]}}
        """
        return {}

    def act(self,
            inputs: Mapping[str, Union[jnp.ndarray, Any]],
            role: str = "",
            params: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None], Mapping[str, Union[jnp.ndarray, Any]]]:
        """Act according to the specified behavior (to be implemented by the inheriting classes)

        Agents will call this method to obtain the decision to be taken given the state of the environment.
        The classes that inherit from the latter must only implement the ``.__call__()`` method

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically jnp.ndarray
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        :param params: Parameters used to compute the output (default: ``None``).
                       If ``None``, internal parameters will be used
        :type params: jnp.array

        :raises NotImplementedError: Child class must implement this method

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of jnp.ndarray, jnp.ndarray or None, and dictionary
        """
        raise NotImplementedError

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: ``"train"`` for training or ``"eval"`` for evaluation
        :type mode: str

        :raises ValueError: If the mode is not ``"train"`` or ``"eval"``
        """
        if mode == "train":
            self.training = True
        elif mode == "eval":
            self.training = False
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
            >>> model.save("/tmp/model.flax")

            # TODO: save an older version of the model to the specified path
        """
        # HACK: Does it make sense to use https://github.com/google/orbax
        with open(path, "wb") as file:
            file.write(flax.serialization.to_bytes(self.state_dict.params if state_dict is None else state_dict.params))

    def load(self, path: str) -> None:
        """Load the model from the specified path

        :param path: Path to load the model from
        :type path: str

        Example::

            # load the model
            >>> model = Model(observation_space, action_space)
            >>> model.load("model.flax")
        """
        # HACK: Does it make sense to use https://github.com/google/orbax
        with open(path, "rb") as file:
            params = flax.serialization.from_bytes(self.state_dict.params, file.read())
        self.state_dict = self.state_dict.replace(params=params)
        self.set_mode("eval")

    def migrate(self,
                state_dict: Optional[Mapping[str, Any]] = None,
                path: Optional[str] = None,
                name_map: Mapping[str, str] = {},
                auto_mapping: bool = True,
                verbose: bool = False) -> bool:
        """Migrate the specified extrernal model's state dict to the current model

        .. warning::

            This method is not implemented yet, just maintains compatibility with other ML frameworks

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze internal parameters

        .. note::

            This method does nothing, just maintains compatibility with other ML frameworks

        :param freeze: Freeze the internal parameters if True, otherwise unfreeze them (default: ``True``)
        :type freeze: bool, optional

        Example::

            # freeze model parameters
            >>> model.freeze_parameters(True)

            # unfreeze model parameters
            >>> model.freeze_parameters(False)
        """
        pass

    def update_parameters(self, model: flax.linen.Module, polyak: float = 1) -> None:
        """Update internal parameters by hard or soft (polyak averaging) update

        - Hard update: :math:`\\theta = \\theta_{net}`
        - Soft (polyak averaging) update: :math:`\\theta = (1 - \\rho) \\theta + \\rho \\theta_{net}`

        :param model: Model used to update the internal parameters
        :type model: flax.linen.Module (skrl.models.jax.Model)
        :param polyak: Polyak hyperparameter between 0 and 1 (default: ``1``).
                       A hard update is performed when its value is 1
        :type polyak: float, optional

        Example::

            # hard update (from source model)
            >>> model.update_parameters(source_model)

            # soft update (from source model)
            >>> model.update_parameters(source_model, polyak=0.005)
        """
        # hard update
        if polyak == 1:
            self.state_dict = self.state_dict.replace(params=model.state_dict.params)
        # soft update
        else:
            # HACK: Does it make sense to use https://optax.readthedocs.io/en/latest/api.html?#optax.incremental_update
            params = jax.tree_util.tree_map(lambda params, model_params: polyak * model_params + (1 - polyak) * params,
                                            self.state_dict.params, model.state_dict.params)
            self.state_dict = self.state_dict.replace(params=params)
