from __future__ import annotations

from typing import Any

import numpy as np

from skrl.utils.framework.jax.types import Tensor


# hack to fix: module 'numpy' has no attribute 'bool8'
try:
    np.bool8
except AttributeError:
    np.bool8 = np.bool

import gym
import gymnasium
from packaging import version

import jax
import jax.numpy as jnp

from skrl import config, logger
from skrl.envs.wrappers.jax.base import Wrapper
from skrl.utils.spaces.jax import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)


class GymWrapper(Wrapper):
    # info keys (checked in order) under which the environment may provide the final observations
    # - "terminal_observation": deprecated API (e.g.: OpenAI Gym < 0.25 vectorized environments)
    _final_observation_keys = (*Wrapper._final_observation_keys, "terminal_observation")

    def __init__(self, env: Any) -> None:
        """OpenAI Gym environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = np.asarray(jax.device_get(config.jax.key)).sum().item()
        self._vectorized = False
        try:
            if isinstance(env, gym.vector.VectorEnv):
                self._vectorized = True
                self._reset_once = True
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None
        self._info = {}

        self._deprecated_api = version.parse(gym.__version__) < version.parse("0.25.0")
        if self._deprecated_api:
            logger.warning(f"Using a deprecated version of OpenAI Gym's API: {gym.__version__}")

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        if self._vectorized:
            return convert_gym_space(self._env.single_observation_space)
        return convert_gym_space(self._env.observation_space)

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space.

        If the wrapped environment does not have the ``state_space`` property, ``None`` will be returned.
        """
        space = super().state_space
        return convert_gym_space(space) if isinstance(space, gym.Space) else space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        if self._vectorized:
            return convert_gym_space(self._env.single_action_space)
        return convert_gym_space(self._env.action_space)

    def _get_state(self) -> Any | None:
        """Get the (raw) state from the environment.

        :return: State, or ``None`` if the environment does not provide it.
        """
        state_fn = getattr(self._unwrapped, "state", None)
        if self.state_space is None or not callable(state_fn):
            return None
        try:
            return state_fn()
        except NotImplementedError:  # environment without state
            return None

    def _parse_group(self, *, observations: Any, states: Any | None) -> tuple[Tensor, Tensor | None]:
        """Parse the observations and states from the environment's output.

        :param observations: Observations returned by the environment.
        :param states: States returned by the environment (see :py:meth:`_get_state`).

        :return: Observation, state (``None`` if the environment does not provide it).
        """
        observations = flatten_tensorized_space(
            tensorize_space(self.observation_space, observations, device=self.device)
        )
        if states is not None and self.state_space is not None:
            states = flatten_tensorized_space(tensorize_space(self.state_space, states, device=self.device))
        else:
            states = None
        return observations, states

    def _parse_final_group(
        self, *, observations: Tensor, states: Tensor | None, terminated: Tensor, truncated: Tensor, info: Any
    ) -> tuple[Tensor, Tensor | None]:
        """Parse the final observations and states from the environment's output.

        The final observation/state of the (same-step) auto-reset environments is read from the info (see
        :py:meth:`_get_final_info`), and it is used for the (sub-)environments whose episodes ended,
        while the current value of the rest of them is kept.

        :param observations: Current observations (already parsed, see :py:meth:`_parse_group`).
        :param states: Current states (already parsed, see :py:meth:`_parse_group`).
        :param terminated: Terminated flags (already converted).
        :param truncated: Truncated flags (already converted).
        :param info: Info returned by the environment.

        :return: Final observation, state.
        """
        dones = jnp.logical_or(terminated, truncated)
        if not dones.any():
            return observations, states
        final_observation, final_state = self._get_final_info(info)
        # tensorize and flatten the final values, and use them for the environments whose episodes ended
        if final_observation is not None:
            final_observation = flatten_tensorized_space(
                tensorize_space(self.observation_space, final_observation, device=self.device)
            )
            observations = jnp.where(dones, final_observation, observations)
        if final_state is not None and states is not None:
            final_state = flatten_tensorized_space(tensorize_space(self.state_space, final_state, device=self.device))
            states = jnp.where(dones, final_state, states)
        return observations, states

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, np.asarray(jax.device_get(actions))),
            squeeze_batch_dimension=not self._vectorized,
        )
        if self._vectorized and isinstance(self.action_space, gymnasium.spaces.Discrete):
            actions = actions.flatten()

        if self._deprecated_api:
            observation, reward, terminated, info = self._env.step(actions)
            # truncated: https://gymnasium.farama.org/tutorials/handling_time_limits
            if isinstance(info, (tuple, list)):
                truncated = np.array([d.get("TimeLimit.truncated", False) for d in info], dtype=terminated.dtype)
                terminated *= np.logical_not(truncated)
            else:
                truncated = info.get("TimeLimit.truncated", False)
                if truncated:
                    terminated = False
        else:
            observation, reward, terminated, truncated, info = self._env.step(actions)

        reward = jax.device_put(np.array(reward, dtype=np.float32).reshape(self.num_envs, -1), device=self.device)
        terminated = jax.device_put(np.array(terminated, dtype=np.int8).reshape(self.num_envs, -1), device=self.device)
        truncated = jax.device_put(np.array(truncated, dtype=np.int8).reshape(self.num_envs, -1), device=self.device)

        self._observations, self._states = self._parse_group(observations=observation, states=self._get_state())
        self._final_observations, self._final_states = self._parse_final_group(
            observations=self._observations, states=self._states, terminated=terminated, truncated=truncated, info=info
        )
        # list of per-environment info (deprecated API) is not forwarded
        self._info = {} if self._deprecated_api and isinstance(info, (tuple, list)) else info

        return self._observations, reward, terminated, truncated, self._info

    def state(self) -> Tensor | None:
        """Get the environment state.

        The state is computed at the last :py:meth:`reset`/:py:meth:`step` call.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                if self._deprecated_api:
                    self._env.seed(self._seed)
                    observation = self._env.reset()
                    self._info = {}
                else:
                    observation, self._info = self._env.reset(seed=self._seed)
                self._observations, self._states = self._parse_group(observations=observation, states=self._get_state())
                self._reset_once = False
                self._seed = None
            self._final_observations = self._observations
            self._final_states = self._states
            return self._observations, self._info

        if self._deprecated_api:
            self._env.seed(self._seed)
            observation = self._env.reset()
            self._info = {}
        else:
            observation, self._info = self._env.reset(seed=self._seed)
        self._observations, self._states = self._parse_group(observations=observation, states=self._get_state())
        self._seed = None
        self._final_observations = self._observations
        self._final_states = self._states
        return self._observations, self._info

    def final(self) -> tuple[Tensor, Tensor | None]:
        """Get the observations and states with which the last transitions ended.

        The final observation/state of (same-step) auto-reset environments is read from the info under the first
        available info key: ``"final_obs"``, ``"final_observation"`` or ``"final_observations"`` for observations, and
        ``"final_state"`` or ``"final_states"`` for states. In addition, the ``"terminal_observation"`` info key
        (deprecated API) is checked last for the final observation.

        See :py:meth:`~skrl.envs.wrappers.jax.base.Wrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        if self._vectorized:
            return None
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
