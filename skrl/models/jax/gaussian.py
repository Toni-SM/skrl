from typing import Any, Mapping, Optional, Tuple, Union

from functools import partial
import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("reduction"))
def _gaussian(
    loc, log_std, log_std_min, log_std_max, clip_actions_min, clip_actions_max, taken_actions, key, reduction
):
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
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        role: str = "",
    ) -> None:
        """Gaussian mixin model (stochastic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
        :param reduction: Reduction method for returning the log probability density function.
            Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``.
            If ``"none"``, the log probability density function is returned as a tensor of shape
            ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
        :param role: Role played by the model.

        :raises ValueError: If the reduction method is not valid.
        """
        self._clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        self._clip_actions_min = jnp.array(self.action_space.low, dtype=jnp.float32) if self._clip_actions else -jnp.inf
        self._clip_actions_max = jnp.array(self.action_space.high, dtype=jnp.float32) if self._clip_actions else jnp.inf

        self._clip_log_std = clip_log_std
        self._log_std_min = min_log_std if self._clip_log_std else -jnp.inf
        self._log_std_max = max_log_std if self._clip_log_std else jnp.inf

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._reduction = (
            jnp.mean
            if reduction == "mean"
            else jnp.sum if reduction == "sum" else jnp.prod if reduction == "prod" else None
        )

        self._i = 0
        self._key = config.jax.key

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.IncorrectPostInitOverrideError
        flax.linen.Module.__post_init__(self)

    def act(
        self,
        inputs: Mapping[str, Union[np.ndarray, jax.Array, Any]],
        *,
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Mapping[str, Union[jax.Array, Any]]]:
        """Act stochastically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.
        :param params: Parameters used to compute the output. If not provided, internal parameters will be used.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing the following extra output values:

            - ``"log_std"``: log of the standard deviation.
            - ``"log_prob"``: log of the probability density function.
            - ``"mean_actions"``: mean actions (network output).
        """
        with jax.default_device(self.device):
            self._i += 1
            subkey = jax.random.fold_in(self._key, self._i)
            inputs["key"] = subkey

        # map from observations/states to mean actions and log standard deviations
        mean_actions, outputs = self.apply(self.state_dict.params if params is None else params, inputs, role)

        actions, log_prob, log_std, stddev = _gaussian(
            mean_actions,
            outputs["log_std"],
            self._log_std_min,
            self._log_std_max,
            self._clip_actions_min,
            self._clip_actions_max,
            inputs.get("taken_actions", None),
            subkey,
            self._reduction,
        )

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        # avoid jax.errors.UnexpectedTracerError
        outputs["log_std"] = log_std
        outputs["stddev"] = stddev

        return actions, outputs

    def get_entropy(self, stddev: jax.Array, *, role: str = "") -> jax.Array:
        """Compute and return the entropy of the model.

        :param stddev: Model standard deviation.
        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        return _entropy(stddev)
