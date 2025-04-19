from typing import Any, Mapping, Optional, Tuple, Union

from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config


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


class CategoricalMixin:
    def __init__(self, *, unnormalized_log_prob: bool = True, role: str = "") -> None:
        """Categorical mixin model (stochastic model).

        :param unnormalized_log_prob: Flag to indicate how to the model's output will be interpreted.
            If True, the model's output is interpreted as unnormalized log probabilities (it can be any real number),
            otherwise as normalized probabilities (the output must be non-negative, finite and have a non-zero sum).
        :param role: Role played by the model.
        """
        self._c_unnormalized_log_prob = unnormalized_log_prob

        self._c_i = 0
        self._c_key = config.jax.key

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
            self._c_i += 1
            subkey = jax.random.fold_in(self._c_key, self._c_i)
            inputs["key"] = subkey

        # map from observations/states to normalized probabilities or unnormalized log probabilities
        net_output, outputs = self.apply(self.state_dict.params if params is None else params, inputs, role)

        actions, log_prob = _categorical(
            net_output, self._c_unnormalized_log_prob, inputs.get("taken_actions", None), subkey
        )

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
