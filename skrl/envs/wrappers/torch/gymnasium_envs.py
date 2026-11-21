from __future__ import annotations

from typing import Any

import numpy as np

from skrl.utils.framework.torch.types import Tensor


# hack to fix: module 'numpy' has no attribute 'bool8'
try:
    np.bool8
except AttributeError:
    np.bool8 = np.bool

import gymnasium

import torch

from skrl import config, logger
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)


class GymnasiumWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Gymnasium environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = config.torch.key
        self._vectorized = False
        try:
            self._vectorized = self._vectorized or isinstance(env, gymnasium.vector.VectorEnv)
        except Exception as e:
            pass
        try:
            self._vectorized = self._vectorized or isinstance(env, gymnasium.experimental.vector.VectorEnv)
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")
        if self._vectorized:
            self._reset_once = True
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None
        self._info = {}

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        if self._vectorized:
            return self._env.single_action_space
        return self._env.action_space

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
        :py:meth:`_get_final_info`), or from each item of the per-environment info list for vectorized environments,
        and it is used for the (sub-)environments whose episodes ended, while the current value of the rest of them
        is kept (see :py:meth:`_parse_final_value`).

        :param observations: Current observations (already parsed, see :py:meth:`_parse_group`).
        :param states: Current states (already parsed, see :py:meth:`_parse_group`).
        :param terminated: Terminated flags (already converted).
        :param truncated: Truncated flags (already converted).
        :param info: Info returned by the environment.

        :return: Final observation, state.
        """
        dones = terminated | truncated
        if not dones.any():
            return observations, states
        if self._vectorized and isinstance(info, (tuple, list)):
            # list of per-environment info (e.g.: deprecated API or info-to-list wrapper)
            final_observation, final_state = map(list, zip(*(self._get_final_info(item) for item in info)))
        else:
            final_observation, final_state = self._get_final_info(info)
        # tensorize and flatten the final values, and use them for the environments whose episodes ended
        observations = self._parse_final_value(self.observation_space, final_observation, observations, dones)
        if states is not None:
            states = self._parse_final_value(self.state_space, final_state, states, dones)
        return observations, states

    def _parse_final_value(self, space: gymnasium.Space, final_value: Any, value: Tensor, dones: Tensor) -> Tensor:
        """Use the final value (observation or state) for the (sub-)environments whose episodes ended.

        Vectorized environments may provide per-environment final values (list or object array) with ``None``
        for the sub-environments whose final value is not available (e.g.: their episodes did not end).
        Such values are batched, and the current value is kept for the unavailable ones.

        :param space: Single (non-batched) space of the value.
        :param final_value: Final value provided by the environment (``None`` if not available).
        :param value: Current value (already parsed).
        :param dones: Episode termination or truncation flags.

        :return: Value, with the final value used for the (sub-)environments whose episodes ended (if available).
        """
        if final_value is None:
            return value
        if self._vectorized and (
            isinstance(final_value, list) or (isinstance(final_value, np.ndarray) and final_value.dtype == object)
        ):
            available = [item is not None for item in final_value]
            if not any(available):
                return value
            # fill in the unavailable values (discarded below) with an available one, to batch them
            filler = final_value[available.index(True)]
            final_value = gymnasium.vector.utils.concatenate(
                space,
                [filler if item is None else item for item in final_value],
                gymnasium.vector.utils.create_empty_array(space, self.num_envs),
            )
            dones = dones & torch.tensor(available, device=self.device).view(-1, 1)
        final_value = flatten_tensorized_space(tensorize_space(space, final_value, device=self.device))
        return torch.where(dones, final_value, value)

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )
        if self._vectorized and isinstance(self.action_space, gymnasium.spaces.Discrete):
            actions = actions.flatten()

        observation, reward, terminated, truncated, info = self._env.step(actions)

        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        self._observations, self._states = self._parse_group(observations=observation, states=self._get_state())
        self._final_observations, self._final_states = self._parse_final_group(
            observations=self._observations, states=self._states, terminated=terminated, truncated=truncated, info=info
        )
        self._info = info

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
                observation, self._info = self._env.reset(seed=self._seed)
                self._observations, self._states = self._parse_group(observations=observation, states=self._get_state())
                self._reset_once = False
                self._seed = None
            self._final_observations = self._observations
            self._final_states = self._states
            return self._observations, self._info

        observation, self._info = self._env.reset(seed=self._seed)
        self._observations, self._states = self._parse_group(observations=observation, states=self._get_state())
        self._seed = None
        self._final_observations = self._observations
        self._final_states = self._states
        return self._observations, self._info

    def final(self) -> tuple[Tensor, Tensor | None]:
        """Get the observations and states with which the last transitions ended.

        The final observation/state of (same-step) auto-reset environments is read from the info under
        the first available info key: ``"final_obs"``, ``"final_observation"`` or ``"final_observations"``
        for observations, and ``"final_state"`` or ``"final_states"`` for states.

        See :py:meth:`~skrl.envs.wrappers.torch.base.Wrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        if self._vectorized:
            return self._env.call("render", *args, **kwargs)
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
