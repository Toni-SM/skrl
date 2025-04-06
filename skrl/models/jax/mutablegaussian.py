from typing import Any, Mapping, Optional, Tuple, Union

import jax
import numpy as np

from skrl.models.jax.gaussian import GaussianMixin, _gaussian


class MutableGaussianMixin(GaussianMixin):

    def act(
        self,
        inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
        role: str = "",
        train: bool = False,
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Union[jax.Array, None], Mapping[str, Union[jax.Array, Any]]]:
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
        with jax.default_device(self.device):
            self._i += 1
            subkey = jax.random.fold_in(self._key, self._i)
            inputs["key"] = subkey

        # map from states/observations to mean actions and log standard deviations
        params = (
            {"params": self.state_dict.params, "batch_stats": self.state_dict.batch_stats} if params is None else params
        )
        mutable = inputs.get("mutable", [])
        out = self.apply(params, inputs, train=train, mutable=mutable, role=role)
        mean_actions, log_std, outputs = out[0]

        actions, log_prob, log_std, stddev = _gaussian(
            mean_actions,
            log_std,
            self._log_std_min,
            self._log_std_max,
            self.clip_actions_min,
            self.clip_actions_max,
            inputs.get("taken_actions", None),
            subkey,
            self._reduction,
        )

        outputs["mean_actions"] = mean_actions
        # avoid jax.errors.UnexpectedTracerError
        outputs["log_std"] = log_std
        outputs["stddev"] = stddev

        return actions, log_prob, outputs
