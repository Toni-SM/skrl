from typing import Any, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from skrl.models.jax.deterministic import DeterministicMixin


class MutableDeterministicMixin(DeterministicMixin):

    def act(
        self,
        inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
        role: str = "",
        train: bool = False,
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Union[jax.Array, None], Mapping[str, Union[jax.Array, Any]]]:
        """Act deterministically in response to the state of the environment

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
                 The second component is ``None``. The third component is a dictionary containing extra output values
        :rtype: tuple of jax.Array, jax.Array or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, _, outputs = model.act({"states": states})
            >>> print(actions.shape, outputs)
            (4096, 1) {}
        """
        # map from observations/states to actions
        params = {"params" : self.state_dict.params, "batch_stats" : self.state_dict.batch_stats} if params is None else params
        mutable = inputs.get("mutable", [])
        actions, outputs = self.apply(params, inputs, mutable=mutable, train=train, role=role)
            
        # clip actions
        if self._d_clip_actions[role] if role in self._d_clip_actions else self._d_clip_actions[""]:
            actions = jnp.clip(actions, a_min=self.clip_actions_min, a_max=self.clip_actions_max)

        return actions, None, outputs

