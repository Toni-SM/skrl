from typing import Any, Mapping, Optional, Tuple, Union

from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("unnormalized_log_prob"))
def _categorical(net_output,
                 unnormalized_log_prob,
                 taken_actions,
                 key):
    # normalize
    if unnormalized_log_prob:
        logits = net_output - jax.scipy.special.logsumexp(net_output, axis=-1, keepdims=True)
        # probs = jax.nn.softmax(logits)
    else:
        probs = net_output / net_output.sum(-1, keepdims=True)
        eps = jnp.finfo(probs.dtype).eps
        logits = jnp.log(probs.clip(min=eps, max=1 - eps))

    # sample actions
    actions = jax.random.categorical(key, logits, axis=-1, shape=None)

    # log of the probability density function
    taken_actions = actions if taken_actions is None else taken_actions.astype(jnp.int32).reshape(-1)
    log_prob = jax.nn.log_softmax(logits)[jnp.arange(taken_actions.shape[0]), taken_actions]

    return actions.reshape(-1, 1), log_prob.reshape(-1, 1)

@jax.jit
def _entropy(logits):
    logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logits = logits.clip(min=jnp.finfo(logits.dtype).min)
    p_log_p = logits * jax.nn.softmax(logits)
    return -p_log_p.sum(-1)


class MultiCategoricalMixin:
    def __init__(self, unnormalized_log_prob: bool = True, reduction: str = "sum", role: str = "") -> None:
        """MultiCategorical mixin model (stochastic model)

        :param unnormalized_log_prob: Flag to indicate how to be interpreted the model's output (default: ``True``).
                                      If True, the model's output is interpreted as unnormalized log probabilities
                                      (it can be any real number), otherwise as normalized probabilities
                                      (the output must be non-negative, finite and have a non-zero sum)
        :type unnormalized_log_prob: bool, optional
        :param reduction: Reduction method for returning the log probability density function: (default: ``"sum"``).
                          Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``, the log probability density
                          function is returned as a tensor of shape ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``
        :type reduction: str, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises ValueError: If the reduction method is not valid

        Example::

            # define the model
            >>> import flax.linen as nn
            >>> from skrl.models.jax import Model, MultiCategoricalMixin
            >>>
            >>> class Policy(MultiCategoricalMixin, Model):
            ...     def __init__(self, observation_space, action_space, device=None, unnormalized_log_prob=True, reduction="sum", **kwargs):
            ...         Model.__init__(self, observation_space, action_space, device, **kwargs)
            ...         MultiCategoricalMixin.__init__(self, unnormalized_log_prob, reduction)
            ...
            ...     @nn.compact  # marks the given module method allowing inlined submodules
            ...     def __call__(self, inputs, role):
            ...         x = nn.elu(nn.Dense(32)(inputs["states"]))
            ...         x = nn.elu(nn.Dense(32)(x))
            ...         x = nn.Dense(self.num_actions)(x)
            ...         return x, {}
            ...
            >>> # given an observation_space: gym.spaces.Box with shape (4,)
            >>> # and an action_space: gym.spaces.MultiDiscrete with nvec = [3, 2]
            >>> model = Policy(observation_space, action_space)
            >>>
            >>> print(model)
            Policy(
                # attributes
                observation_space = Box(-1.0, 1.0, (4,), float32)
                action_space = MultiDiscrete([3 2])
                device = StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0)
            )
        """
        self._unnormalized_log_prob = unnormalized_log_prob

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._reduction = jnp.mean if reduction == "mean" else jnp.sum if reduction == "sum" \
            else jnp.prod if reduction == "prod" else None

        self._i = 0
        self._key = config.jax.key

        self._action_space_nvec = np.cumsum(self.action_space.nvec).tolist()
        self._action_space_shape = self._get_space_size(self.action_space, number_of_elements=False)

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.IncorrectPostInitOverrideError
        flax.linen.Module.__post_init__(self)

    def act(self,
            inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
            role: str = "",
            params: Optional[jax.Array] = None) -> Tuple[jax.Array, Union[jax.Array, None], Mapping[str, Union[jax.Array, Any]]]:
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically np.ndarray or jax.Array
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        :param params: Parameters used to compute the output (default: ``None``).
                       If ``None``, internal parameters will be used
        :type params: jnp.array

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the network output ``"net_output"``
                 and extra output values
        :rtype: tuple of jax.Array, jax.Array or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 4)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["net_output"].shape)
            (4096, 2) (4096, 1) (4096, 5)
        """
        with jax.default_device(self.device):
            self._i += 1
            subkey = jax.random.fold_in(self._key, self._i)
            inputs["key"] = subkey

        # map from states/observations to normalized probabilities or unnormalized log probabilities
        net_output, outputs = self.apply(self.state_dict.params if params is None else params, inputs, role)

        # split inputs
        net_outputs = jnp.split(net_output, self._action_space_nvec, axis=-1)
        if "taken_actions" in inputs:
            taken_actions = jnp.split(inputs["taken_actions"], self._action_space_shape, axis=-1)
        else:
            taken_actions = [None] * self._action_space_shape

        # compute actions and log_prob
        actions, log_prob = [], []
        for _net_output, _taken_actions in zip(net_outputs, taken_actions):
            _actions, _log_prob = _categorical(_net_output,
                                               self._unnormalized_log_prob,
                                               _taken_actions,
                                               subkey)
            actions.append(_actions)
            log_prob.append(_log_prob)

        actions = jnp.concatenate(actions, axis=-1)
        log_prob = jnp.concatenate(log_prob, axis=-1)

        if self._reduction is not None:
            log_prob = self._reduction(log_prob, axis=-1)
        if log_prob.ndim != actions.ndim:
            log_prob = jnp.expand_dims(log_prob, -1)

        outputs["net_output"] = net_output
        # avoid jax.errors.UnexpectedTracerError
        outputs["stddev"] = jnp.full_like(log_prob, jnp.nan)
        return actions, log_prob, outputs

    def get_entropy(self, logits: jax.Array, role: str = "") -> jax.Array:
        """Compute and return the entropy of the model

        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Entropy of the model
        :rtype: jax.Array

        Example::

            # given a standard deviation array: stddev
            >>> entropy = model.get_entropy(stddev)
            >>> print(entropy.shape)
            (4096, 8)
        """
        return _entropy(logits)
