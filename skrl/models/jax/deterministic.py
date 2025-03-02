from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np


class DeterministicMixin:
    def __init__(self, *, clip_actions: bool = False, role: str = "") -> None:
        """Deterministic mixin model (deterministic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param role: Role played by the model.
        """
        self._clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        if self._clip_actions:
            self._clip_actions_min = jnp.array(self.action_space.low, dtype=jnp.float32)
            self._clip_actions_max = jnp.array(self.action_space.high, dtype=jnp.float32)

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.IncorrectPostInitOverrideError
        flax.linen.Module.__post_init__(self)

    def act(
        self,
        inputs: Mapping[str, Union[np.ndarray, jax.Array, Any]],
        *,
        role: str = "",
        params: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Mapping[str, Union[jax.Array, Any]]]:
        """Act deterministically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.
        :param params: Parameters used to compute the output. If not provided, internal parameters will be used.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing extra output values according to the model.
        """
        # map from observations/states to actions
        actions, outputs = self.apply(self.state_dict.params if params is None else params, inputs, role)

        # clip actions
        if self._clip_actions:
            actions = jnp.clip(actions, a_min=self._clip_actions_min, a_max=self._clip_actions_max)

        return actions, outputs
