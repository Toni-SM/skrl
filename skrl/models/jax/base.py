from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config, logger
from skrl.utils.spaces.jax import compute_space_size, flatten_tensorized_space, sample_space


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
    observation_space: Optional[gymnasium.Space] = None
    state_space: Optional[gymnasium.Space] = None
    action_space: Optional[gymnasium.Space] = None
    device: Optional[Union[str, jax.Device]] = None

    def __init__(
        self,
        *,
        observation_space: Optional[gymnasium.Space] = None,
        state_space: Optional[gymnasium.Space] = None,
        action_space: Optional[gymnasium.Space] = None,
        device: Optional[Union[str, jax.Device]] = None,
        parent: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:
        """Base model class for implementing custom models.

        :param observation_space: Observation space. The ``num_observations`` property will contain the size of the space.
        :param state_space: State space. The ``num_states`` property will contain the size of the space.
        :param action_space: Action space. The ``num_actions`` property will contain the size of the space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param parent: The parent Module of this Module. It is a Flax reserved attribute.
        :param name: The name of this Module. It is a Flax reserved attribute.
        """
        self._jax = config.jax.backend == "jax"

        self.device = config.jax.parse_device(device)

        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space
        self.num_observations = compute_space_size(observation_space)
        self.num_states = compute_space_size(state_space)
        self.num_actions = compute_space_size(action_space)

        self.state_dict: StateDict
        self.training = False

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ReservedModuleAttributeError
        self.parent = parent
        self.name = name

    def init_state_dict(
        self,
        inputs: Mapping[str, Union[np.ndarray, jax.Array, Any]] = {},
        *,
        role: str = "",
        key: Optional[jax.Array] = None,
    ) -> None:
        """Initialize state dictionary.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.

            If not specified, ``inputs`` will have random samples from the observation, state and action spaces.
        :param role: Role played by the model.
        :param key: Pseudo-random number generator (PRNG) key. If not provided, ``config.jax.key`` will be used.
        """
        if not inputs:
            inputs = {
                "observations": flatten_tensorized_space(
                    sample_space(self.observation_space, backend="native", device=self.device), _jax=self._jax
                ),
                "states": flatten_tensorized_space(
                    sample_space(self.state_space, backend="native", device=self.device), _jax=self._jax
                ),
                "taken_actions": flatten_tensorized_space(
                    sample_space(self.action_space, backend="native", device=self.device), _jax=self._jax
                ),
            }
        if key is None:
            key = config.jax.key
        # init internal state dict
        with jax.default_device(self.device):
            self.state_dict = StateDict.create(apply_fn=self.apply, params=self.init(key, inputs, role))

    def random_act(
        self,
        inputs: Mapping[str, Union[np.ndarray, jax.Array, Any]],
        *,
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[Union[np.ndarray, jax.Array], Mapping[str, Union[np.ndarray, jax.Array, Any]]]:
        """Act randomly according to the action space.

        .. warning::

            Sampling from unbounded action spaces may lead to numerical instabilities.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.

        :param role: Role played by the model.
        :param params: Parameters used to compute the output. If not provided, internal parameters will be used.

        :return: Randomly sampled actions with the same batch size as the given observation (``"observations"``)
            in the ``inputs`` as the first component. The second component is an empty dictionary.

        :raises ValueError: Unsupported action space.
        """
        sample = sample_space(
            self.action_space, batch_size=inputs["observations"].shape[0], backend="native", device=self.device
        )
        return flatten_tensorized_space(sample), {}

    def init_parameters(self, method_name: str = "normal", *args, **kwargs) -> None:
        """Initialize the model parameters according to the specified method name.

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ module.
        Allowed method names are ``"uniform"``, ``"normal"``, ``"constant"``, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ method name.
        :param args: Positional arguments of the method to be called.
        :param kwargs: Key-value arguments of the method to be called.

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
        """Initialize the model weights according to the specified method name.

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ module.
        Allowed method names are ``"uniform"``, ``"normal"``, ``"constant"``, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ method name.
        :param args: Positional arguments of the method to be called.
        :param kwargs: Key-value arguments of the method to be called.

        Example::

            # initialize all weights with uniform distribution in range [-0.1, 0.1]
            >>> model.init_weights(method_name="uniform_", a=-0.1, b=0.1)

            # initialize all weights with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_weights(method_name="normal", stddev=0.25)
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
        """Initialize the model biases according to the specified method name.

        Method names are from the `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ module.
        Allowed method names are ``"uniform"``, ``"normal"``, ``"constant"``, etc.

        :param method_name: `flax.linen.initializers <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/initializers.html>`_ method name.
        :param args: Positional arguments of the method to be called.
        :param kwargs: Key-value arguments of the method to be called.

        Example::

            # initialize all biases with a constant value (0)
            >>> model.init_biases(method_name="constant_", val=0)

            # initialize all biases with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_biases(method_name="normal", stddev=0.25)
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

    def act(
        self,
        inputs: Mapping[str, Union[np.ndarray, jax.Array, Any]],
        *,
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Mapping[str, Union[jax.Array, Any]]]:
        """Act according to the specified behavior.

        Agents will call this method to get the expected action/value based on the observations/states.

        .. warning::

            This method is currently implemented by the helper models (e.g.: :py:class:`~skrl.models.torch.gaussian.GaussianMixin`).
            The classes that inherit from the latter must only implement the ``.__call__()`` method.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.
        :param params: Parameters used to compute the output. If not provided, internal parameters will be used.


        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing extra output values according to the model.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        logger.warning("Make sure to place Mixins classes before the Model class during model definition")
        raise NotImplementedError("The action to be taken by the agent (.act()) is not implemented")

    def enable_training_mode(self, enabled: bool = True) -> None:
        """Set the training mode of the model: enabled (training) or disabled (evaluation).

        :param enabled: True to enable the training mode, False to enable the evaluation mode.
            The specific behavior can be accessed via the ``training`` property.
        """
        self.training = enabled

    def save(self, path: str, state_dict: Optional[dict] = None) -> None:
        """Save the model to the specified path.

        :param path: Path to save the model to.
        :param state_dict: State dictionary to save. If ``None``, the model's ``state_dict`` will be saved.

        Example::

            # save the current model to the specified path
            >>> model.save("/tmp/model.flax")
        """
        # HACK: Does it make sense to use https://github.com/google/orbax
        # TODO: save an older version of the model to the specified path
        with open(path, "wb") as file:
            file.write(flax.serialization.to_bytes(self.state_dict.params if state_dict is None else state_dict.params))

    def load(self, path: str) -> None:
        """Load the model from the specified path.

        :param path: Path to load the model from.

        Example::

            # load the model
            >>> model = Model(observation_space, action_space)
            >>> model.load("model.flax")
        """
        # HACK: Does it make sense to use https://github.com/google/orbax
        with open(path, "rb") as file:
            params = flax.serialization.from_bytes(self.state_dict.params, file.read())
        self.state_dict = self.state_dict.replace(params=params)
        self.enable_training_mode(False)

    def migrate(
        self,
        *,
        state_dict: Optional[Mapping[str, Any]] = None,
        path: Optional[str] = None,
        name_map: Mapping[str, str] = {},
        auto_mapping: bool = True,
        verbose: bool = False,
    ) -> bool:
        """Migrate the specified external model's ``state dict`` to the current model.

        .. warning::

            This method is not implemented yet, just maintains compatibility with other ML frameworks.

        :raises NotImplementedError: Not implemented.
        """
        raise NotImplementedError

    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze internal parameters.

        .. note::

            This method does nothing, just maintains compatibility with other ML frameworks.

        :param freeze: Whether to freeze or unfreeze the internal parameters.

        Example::

            # freeze model parameters
            >>> model.freeze_parameters(True)

            # unfreeze model parameters
            >>> model.freeze_parameters(False)
        """
        pass

    def update_parameters(self, model: flax.linen.Module, *, polyak: float = 1.0) -> None:
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
        """Broadcast model parameters to the whole group (e.g.: across all nodes) in distributed runs.

        After calling this method, the distributed model will contain the broadcasted parameters from ``rank``.

        :param rank: Worker/process rank from which to broadcast model parameters.

        Example::

            # broadcast model parameter from worker/process with rank 1
            >>> if config.jax.is_distributed:
            ...     model.broadcast_parameters(rank=1)
        """
        is_source = jax.process_index() == rank
        params = jax.experimental.multihost_utils.broadcast_one_to_all(self.state_dict.params, is_source=is_source)
        self.state_dict = self.state_dict.replace(params=params)

    def reduce_parameters(self, tree: Any) -> Any:
        """Reduce model parameters across all workers/processes in the whole group (e.g.: across all nodes).

        After calling this method, the distributed model parameters will be bitwise identical for all workers/processes.

        :param tree: Pytree to apply collective reduction.

        :return: All-reduced pytree.

        Example::

            # reduce model parameter across all workers/processes
            >>> if config.jax.is_distributed:
            ...     model.reduce_parameters(grad)
        """
        # # collective all-reduce mean for each pytree leaves
        # return jax.tree_util.tree_map(
        #     lambda g: jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(jnp.expand_dims(g, 0)).squeeze(0)
        #     / config.jax.world_size,
        #     tree,
        # )

        # # using https://jax.readthedocs.io/en/latest/_autosummary/jax.flatten_util.ravel_pytree.html
        # vector, unflatten = jax.flatten_util.ravel_pytree(tree)
        # vector = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(jnp.expand_dims(vector, 0)) / config.jax.world_size
        # return unflatten(jnp.squeeze(vector, 0))

        leaves, treedef = jax.tree.flatten(tree)
        vector = (
            jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(_vectorize_leaves(leaves)) / config.jax.world_size
        )
        return jax.tree.unflatten(treedef, _unvectorize_leaves(leaves, vector))
