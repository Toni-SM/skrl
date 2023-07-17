from typing import Any, Mapping, Optional, Tuple, Union

import gym
import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np


class DeterministicMixin:
    def __init__(self, clip_actions: bool = False, role: str = "") -> None:
        """Deterministic mixin model (deterministic model)

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: ``False``)
        :type clip_actions: bool, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            # define the model
            >>> import flax.linen as nn
            >>> from skrl.models.jax import Model, DeterministicMixin
            >>>
            >>> class Value(DeterministicMixin, Model):
            ...     def __init__(self, observation_space, action_space, device=None, clip_actions=False, **kwargs):
            ...         Model.__init__(self, observation_space, action_space, device, **kwargs)
            ...         DeterministicMixin.__init__(self, clip_actions)
            ...
            ...     @nn.compact  # marks the given module method allowing inlined submodules
            ...     def __call__(self, inputs, role):
            ...         x = nn.elu(nn.Dense(32)([inputs["states"]))
            ...         x = nn.elu(nn.Dense(32)(x))
            ...         x = nn.Dense(1)(x)
            ...         return x, {}
            ...
            >>> # given an observation_space: gym.spaces.Box with shape (60,)
            >>> # and an action_space: gym.spaces.Box with shape (8,)
            >>> model = Value(observation_space, action_space)
            >>>
            >>> print(model)
            Value(
                # attributes
                observation_space = Box(-1.0, 1.0, (60,), float32)
                action_space = Box(-1.0, 1.0, (8,), float32)
                device = StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0)
            )
        """
        if not hasattr(self, "_d_clip_actions"):
            self._d_clip_actions = {}
        self._d_clip_actions[role] = clip_actions and (issubclass(type(self.action_space), gym.Space) or \
            issubclass(type(self.action_space), gymnasium.Space))

        if self._d_clip_actions[role]:
            self.clip_actions_min = jnp.array(self.action_space.low, dtype=jnp.float32)
            self.clip_actions_max = jnp.array(self.action_space.high, dtype=jnp.float32)

        # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.IncorrectPostInitOverrideError
        flax.linen.Module.__post_init__(self)

    def act(self,
            inputs: Mapping[str, Union[Union[np.ndarray, jax.Array], Any]],
            role: str = "",
            params: Optional[jax.Array] = None) -> Tuple[jax.Array, Union[jax.Array, None], Mapping[str, Union[jax.Array, Any]]]:
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
        actions, outputs = self.apply(self.state_dict.params if params is None else params, inputs, role)

        # clip actions
        if self._d_clip_actions[role] if role in self._d_clip_actions else self._d_clip_actions[""]:
            actions = jnp.clip(actions, a_min=self.clip_actions_min, a_max=self.clip_actions_max)

        return actions, None, outputs
