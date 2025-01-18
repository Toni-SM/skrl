from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.utils.spaces.jax import compute_space_size, flatten_tensorized_space, sample_space, unflatten_tensorized_space


@jax.jit
def _vectorize_leaves(leaves: Sequence[jax.Array]) -> jax.Array:
    return jnp.expand_dims(jnp.concatenate(list(map(jnp.ravel, leaves)), axis=-1), 0)


@jax.jit
def _unvectorize_leaves(leaves: Sequence[jax.Array], vector: jax.Array) -> Sequence[jax.Array]:
    offset = 0
    for i, leaf in enumerate(leaves):
        leaves[i] = leaves[i].at[:].set(vector.at[0, offset : offset + leaf.size].get().reshape(leaf.shape))
        offset += leaf.size
    return leaves


class StateDict(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, **kwargs):
        return cls(apply_fn=apply_fn, params=params, **kwargs)


class Model(flax.linen.Module):
    observation_space: Union[int, Sequence[int], gymnasium.Space]
    action_space: Union[int, Sequence[int], gymnasium.Space]
    device: Optional[Union[str, jax.Device]] = None

    def __init__(
        self,
        observation_space: Union[int, Sequence[int], gymnasium.Space],
        action_space: Union[int, Sequence[int], gymnasium.Space],
        device: Optional[Union[str, jax.Device]] = None,
        parent: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:
        """Base class representing a function approximator

        The following properties are defined:

        - ``device`` (jax.Device): Device to be used for the computations
        - ``observation_space`` (int, sequence of int, gymnasium.Space): Observation/state space
        - ``action_space`` (int, sequence of int, gymnasium.Space): Action space
        - ``num_observations`` (int): Number of elements in the observation/state space
        - ``num_actions`` (int): Number of elements in the action space

        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gymnasium.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gymnasium.Space
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional
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
        self._jax = config.jax.backend == "jax"

        self.device = config.jax.parse_device(device)

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_observations = None if observation_space is None else compute_space_size(observation_space)
        self.num_actions = None if action_space is None else compute_space_size(action_space)

        self.state_dict: StateDict
        self.training = False

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ReservedModuleAttributeError
        self.parent = parent
        self.name = name

    def init_state_dict(
        self, role: str, inputs: Mapping[str, Union[np.ndarray, jax.Array]] = {}, key: Optional[jax.Array] = None
    ) -> None:
        """Initialize state dictionary

        :param role: Role play by the model
        :type role: str
        :param inputs: Model inputs. The most common keys are:

                        - ``"states"``: state of the environment used to make the decision
                        - ``"taken_actions"``: actions taken by the policy for the given states

                       If not specified, the keys will be populated with observation and action space samples
        :type inputs: dict of np.ndarray or jax.Array, optional
        :param key: Pseudo-random number generator (PRNG) key (default: ``None``).
                    If not provided, the skrl's PRNG key (``config.jax.key``) will be used
        :type key: jax.Array, optional
        """
        if not inputs:
            inputs = {
                "states": flatten_tensorized_space(
                    sample_space(self.observation_space, backend="jax", device=self.device), self._jax
                ),
                "taken_actions": flatten_tensorized_space(
                    sample_space(self.action_space, backend="jax", device=self.device), self._jax
                ),
            }
        if key is None:
            key = config.jax.key
        if isinstance(inputs["states"], (int, np.int32, np.int64)):
            inputs["states"] = np.array(inputs["states"]).reshape(-1, 1)
        # init internal state dict
        with jax.default_device(self.device):
            self.state_dict = StateDict.create(apply_fn=self.apply, params=self.init(key, inputs, role))

    def tensor_to_space(
        self, tensor: Union[np.ndarray, jax.Array], space: gymnasium.Space, start: int = 0
    ) -> Union[Union[np.ndarray, jax.Array], dict]:
        """Map a flat tensor to a Gym/Gymnasium space

        .. warning::

            This method is deprecated in favor of the :py:func:`skrl.utils.spaces.jax.unflatten_tensorized_space`

        :param tensor: Tensor to map from
        :type tensor: np.ndarray or jax.Array
        :param space: Space to map the tensor to
        :type space: gymnasium.Space
        :param start: Index of the first element of the tensor to map (default: ``0``)
        :type start: int, optional

        :raises ValueError: If the space is not supported

        :return: Mapped tensor or dictionary
        :rtype: np.ndarray or jax.Array, or dict

        Example::

            >>> space = gymnasium.spaces.Dict({'a': gymnasium.spaces.Box(low=-1, high=1, shape=(2, 3)),
            ...                                'b': gymnasium.spaces.Discrete(4)})
            >>> tensor = jnp.array([[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 2]])
            >>>
            >>> model.tensor_to_space(tensor, space)
            {'a': Array([[[-0.3, -0.2, -0.1],
                          [ 0.1,  0.2,  0.3]]], dtype=float32),
             'b': Array([[2.]], dtype=float32)}
        """
        return unflatten_tensorized_space(space, tensor)

    def random_act(
        self,
        inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[Union[np.ndarray, jax.Array], None],
        Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
    ]:
        """Act randomly according to the action space

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically np.ndarray or jax.Array
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        :param params: Parameters used to compute the output (default: ``None``).
                       If ``None``, internal parameters will be used
        :type params: jnp.array

        :raises NotImplementedError: Unsupported action space

        :return: Model output. The first component is the action to be taken by the agent
        :rtype: tuple of np.ndarray or jax.Array, None, and dict
        """
        # discrete action space (Discrete)
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            actions = np.random.randint(self.action_space.n, size=(inputs["states"].shape[0], 1))
        # continuous action space (Box)
        elif isinstance(self.action_space, gymnasium.spaces.Box):
            actions = np.random.uniform(
                low=self.action_space.low[0],
                high=self.action_space.high[0],
                size=(inputs["states"].shape[0], self.num_actions),
            )
        else:
            raise NotImplementedError(f"Action space type ({type(self.action_space)}) not supported")

        if self._jax:
            return jax.device_put(actions), None, {}
        return actions, None, {}

    def init_parameters(self, method_name: str = "normal", *args, **kwargs) -> None:
        """Initialize the model parameters according to the specified method name

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ module.
        Allowed method names are *uniform*, *normal*, *constant*, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ method name (default: ``"normal"``)
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

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ module.
        Allowed method names are *uniform*, *normal*, *constant*, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ method name (default: ``"normal"``)
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
        params = jax.tree_util.tree_map_with_path(
            lambda path, param: method(config.jax.key, param.shape) if path[-1].key == "kernel" else param,
            self.state_dict.params,
        )
        self.state_dict = self.state_dict.replace(params=params)

    def init_biases(self, method_name: str = "constant_", *args, **kwargs) -> None:
        """Initialize the model biases according to the specified method name

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ module.
        Allowed method names are *uniform*, *normal*, *constant*, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ method name (default: ``"normal"``)
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
        params = jax.tree_util.tree_map_with_path(
            lambda path, param: method(config.jax.key, param.shape) if path[-1].key == "bias" else param,
            self.state_dict.params,
        )
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

    def act(
        self,
        inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Union[jax.Array, None], Mapping[str, Union[jax.Array, Any]]]:
        """Act according to the specified behavior (to be implemented by the inheriting classes)

        Agents will call this method to obtain the decision to be taken given the state of the environment.
        The classes that inherit from the latter must only implement the ``.__call__()`` method

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically np.ndarray or jax.Array
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        :param params: Parameters used to compute the output (default: ``None``).
                       If ``None``, internal parameters will be used
        :type params: jnp.array

        :raises NotImplementedError: Child class must implement this method

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of jax.Array, jax.Array or None, and dict
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

    def migrate(
        self,
        state_dict: Optional[Mapping[str, Any]] = None,
        path: Optional[str] = None,
        name_map: Mapping[str, str] = {},
        auto_mapping: bool = True,
        verbose: bool = False,
    ) -> bool:
        """Migrate the specified external model's state dict to the current model

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
            # HACK: Does it make sense to use https://optax.readthedocs.io/en/latest/api/apply_updates.html#optax.incremental_update
            params = jax.tree_util.tree_map(
                lambda params, model_params: polyak * model_params + (1 - polyak) * params,
                self.state_dict.params,
                model.state_dict.params,
            )
            self.state_dict = self.state_dict.replace(params=params)

    def broadcast_parameters(self, rank: int = 0):
        """Broadcast model parameters to the whole group (e.g.: across all nodes) in distributed runs

        After calling this method, the distributed model will contain the broadcasted parameters from ``rank``

        :param rank: Worker/process rank from which to broadcast model parameters (default: ``0``)
        :type rank: int

        Example::

            # broadcast model parameter from worker/process with rank 1
            >>> if config.jax.is_distributed:
            ...     model.broadcast_parameters(rank=1)
        """
        is_source = jax.process_index() == rank
        params = jax.experimental.multihost_utils.broadcast_one_to_all(self.state_dict.params, is_source=is_source)
        self.state_dict = self.state_dict.replace(params=params)

    def reduce_parameters(self, tree: Any) -> Any:
        """Reduce model parameters across all workers/processes in the whole group (e.g.: across all nodes)

        After calling this method, the distributed model parameters will be bitwise identical for all workers/processes

        :param tree: pytree to apply collective reduction
        :type tree: Any

        :return: all-reduced pytree
        :rtype: Any

        Example::

            # reduce model parameter across all workers/processes
            >>> if config.jax.is_distributed:
            ...     model.reduce_parameters(grad)
        """
        # # collective all-reduce mean for each pytree leaves
        # return jax.tree_util.tree_map(lambda g: jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')
        #                               (jnp.expand_dims(g, 0)).squeeze(0) / config.jax.world_size, tree)

        # # using https://jax.readthedocs.io/en/latest/_autosummary/jax.flatten_util.ravel_pytree.html
        # vector, unflatten = jax.flatten_util.ravel_pytree(tree)
        # vector = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(jnp.expand_dims(vector, 0)) / config.jax.world_size
        # return unflatten(jnp.squeeze(vector, 0))

        leaves, treedef = jax.tree.flatten(tree)
        vector = (
            jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(_vectorize_leaves(leaves)) / config.jax.world_size
        )
        return jax.tree.unflatten(treedef, _unvectorize_leaves(leaves, vector))
