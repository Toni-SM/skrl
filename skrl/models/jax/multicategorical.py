from typing import Any, Mapping, Optional, Tuple, Union

from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.utils.spaces.jax import compute_space_size


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("unnormalized_log_prob"))
def _categorical(net_output, unnormalized_log_prob, taken_actions, key):
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
    def __init__(self, *, unnormalized_log_prob: bool = True, reduction: str = "sum", role: str = "") -> None:
        """MultiCategorical mixin model (stochastic model).

        :param unnormalized_log_prob: Flag to indicate how to the model's output will be interpreted.
            If True, the model's output is interpreted as unnormalized log probabilities (it can be any real number),
            otherwise as normalized probabilities (the output must be non-negative, finite and have a non-zero sum).
        :param reduction: Reduction method for returning the log probability density function.
            Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``.
            If ``"none"``, the log probability density function is returned as a tensor of shape
            ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
        :param role: Role played by the model.

        :raises ValueError: If the reduction method is not valid.
        """
        self._mc_unnormalized_log_prob = unnormalized_log_prob

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._mc_reduction = (
            jnp.mean
            if reduction == "mean"
            else jnp.sum if reduction == "sum" else jnp.prod if reduction == "prod" else None
        )

        self._mc_i = 0
        self._mc_key = config.jax.key

        self._mc_action_space_nvec = np.cumsum(self.action_space.nvec).tolist()
        self._mc_action_space_shape = compute_space_size(self.action_space, occupied_size=True)

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

            - ``"log_prob"``: log of the probability density function.
            - ``"net_output"``: network output.
        """
        with jax.default_device(self.device):
            self._mc_i += 1
            subkey = jax.random.fold_in(self._mc_key, self._mc_i)
            inputs["key"] = subkey

        # map from observations/states to normalized probabilities or unnormalized log probabilities
        net_output, outputs = self.apply(self.state_dict.params if params is None else params, inputs, role)

        # split inputs
        net_outputs = jnp.split(net_output, self._mc_action_space_nvec, axis=-1)
        if "taken_actions" in inputs:
            taken_actions = jnp.split(inputs["taken_actions"], self._mc_action_space_shape, axis=-1)
        else:
            taken_actions = [None] * self._mc_action_space_shape

        # compute actions and log_prob
        actions, log_prob = [], []
        for _net_output, _taken_actions in zip(net_outputs, taken_actions):
            _actions, _log_prob = _categorical(_net_output, self._mc_unnormalized_log_prob, _taken_actions, subkey)
            actions.append(_actions)
            log_prob.append(_log_prob)

        actions = jnp.concatenate(actions, axis=-1)
        log_prob = jnp.concatenate(log_prob, axis=-1)

        if self._mc_reduction is not None:
            log_prob = self._mc_reduction(log_prob, axis=-1)
        if log_prob.ndim != actions.ndim:
            log_prob = jnp.expand_dims(log_prob, -1)

        outputs["log_prob"] = log_prob
        outputs["net_output"] = net_output
        # avoid jax.errors.UnexpectedTracerError
        outputs["stddev"] = jnp.full_like(log_prob, jnp.nan)
        return actions, outputs

    def get_entropy(self, stddev: jax.Array, *, role: str = "") -> jax.Array:
        """Compute and return the entropy of the model.

        :param stddev: Model standard deviation.
        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        return _entropy(stddev)
