from typing import Any, Mapping, Optional, Tuple, Union

from functools import partial
import gym
import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("reduction"))
def _gaussian(loc,
              log_std,
              log_std_min,
              log_std_max,
              clip_actions_min,
              clip_actions_max,
              taken_actions,
              key,
              reduction):
    # clamp log standard deviations
    log_std = jnp.clip(log_std, a_min=log_std_min, a_max=log_std_max)

    # distribution
    scale = jnp.exp(log_std)

    # sample actions
    actions = jax.random.normal(key, loc.shape) * scale + loc

    # clip actions
    actions = jnp.clip(actions, a_min=clip_actions_min, a_max=clip_actions_max)

    # log of the probability density function
    taken_actions = actions if taken_actions is None else taken_actions
    log_prob = -jnp.square(taken_actions - loc) / (2 * jnp.square(scale)) - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)

    if reduction is not None:
        log_prob = reduction(log_prob, axis=-1)
    if log_prob.ndim != actions.ndim:
        log_prob = jnp.expand_dims(log_prob, -1)

    return actions, log_prob, log_std, scale

@jax.jit
def _entropy(scale):
    return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(scale)


class GaussianMixin:
    def __init__(self,
                 clip_actions: bool = False,
                 clip_log_std: bool = True,
                 min_log_std: float = -20,
                 max_log_std: float = 2,
                 reduction: str = "sum",
                 role: str = "") -> None:
        """Gaussian mixin model (stochastic model)

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: ``False``)
        :type clip_actions: bool, optional
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: ``True``)
        :type clip_log_std: bool, optional
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True (default: ``-20``)
        :type min_log_std: float, optional
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True (default: ``2``)
        :type max_log_std: float, optional
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
            >>> from skrl.models.jax import Model, GaussianMixin
            >>>
            >>> class Policy(GaussianMixin, Model):
            ...     def __init__(self, observation_space, action_space, device=None,
            ...                  clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
            ...         Model.__init__(self, observation_space, action_space, device, **kwargs)
            ...         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
            ...
            ...     def setup(self):
            ...         self.layer_1 = nn.Dense(32)
            ...         self.layer_2 = nn.Dense(32)
            ...         self.layer_3 = nn.Dense(self.num_actions)
            ...
            ...         self.log_std_parameter = self.param("log_std_parameter", lambda _: jnp.zeros(self.num_actions))
            ...
            ...     def __call__(self, inputs, role):
            ...         x = nn.elu(self.layer_1(inputs["states"]))
            ...         x = nn.elu(self.layer_2(x))
            ...         return self.layer_3(x), self.log_std_parameter, {}
            ...
            >>> # given an observation_space: gym.spaces.Box with shape (60,)
            >>> # and an action_space: gym.spaces.Box with shape (8,)
            >>> model = Policy(observation_space, action_space)
            >>>
            >>> print(model)
            Policy(
                # attributes
                observation_space = Box(-1.0, 1.0, (60,), float32)
                action_space = Box(-1.0, 1.0, (8,), float32)
                device = StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0)
            )
        """
        self._clip_actions = clip_actions and (issubclass(type(self.action_space), gym.Space) or \
            issubclass(type(self.action_space), gymnasium.Space))

        if self._clip_actions:
            self.clip_actions_min = jnp.array(self.action_space.low, dtype=jnp.float32)
            self.clip_actions_max = jnp.array(self.action_space.high, dtype=jnp.float32)
        else:
            self.clip_actions_min = -jnp.inf
            self.clip_actions_max = jnp.inf

        self._clip_log_std = clip_log_std
        if self._clip_log_std:
            self._log_std_min = min_log_std
            self._log_std_max = max_log_std
        else:
            self._log_std_min = -jnp.inf
            self._log_std_max = jnp.inf

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._reduction = jnp.mean if reduction == "mean" else jnp.sum if reduction == "sum" \
            else jnp.prod if reduction == "prod" else None

        self._i = 0
        self._key = config.jax.key

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
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of jax.Array, jax.Array or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            (4096, 8) (4096, 1) (4096, 8)
        """
        self._i += 1
        subkey = jax.random.fold_in(self._key, self._i)
        inputs["key"] = subkey

        # map from states/observations to mean actions and log standard deviations
        mean_actions, log_std, outputs = self.apply(self.state_dict.params if params is None else params, inputs, role)

        actions, log_prob, log_std, stddev = _gaussian(mean_actions,
                                                       log_std,
                                                       self._log_std_min,
                                                       self._log_std_max,
                                                       self.clip_actions_min,
                                                       self.clip_actions_max,
                                                       inputs.get("taken_actions", None),
                                                       subkey,
                                                       self._reduction)

        outputs["mean_actions"] = mean_actions
        # avoid jax.errors.UnexpectedTracerError
        outputs["log_std"] = log_std
        outputs["stddev"] = stddev

        return actions, log_prob, outputs

    def get_entropy(self, stddev: jax.Array, role: str = "") -> jax.Array:
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
        return _entropy(stddev)
