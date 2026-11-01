from __future__ import annotations

from typing import Any

import gymnasium

import numpy as np
import warp as wp

from skrl import config, logger
from skrl.envs.wrappers.warp.base import Wrapper
from skrl.utils.spaces.warp import (
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

        self._seed = config.warp.key
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

    def _get_final_observation(self, observation: Any, info: Any) -> Any | None:
        """Get the observation with which the episodes of the (same-step) auto-reset environments ended.

        The info keys are checked in the following order: ``"final_obs"`` (Gymnasium >= 1.0) and
        ``"final_observation"`` (Gymnasium < 1.0).

        For vectorized environments, the observation of the environments whose episodes ended is read from the info
        (or from each environment's info, if the info is a list) under the first available key, for the environments
        flagged by the ``"_<key>"`` mask (or, if the mask is not provided, whose final observation is not ``None``).
        The observation of the rest of the environments is kept. For non-vectorized environments, it is read from
        the info under the first available key (e.g.: Gymnasium < 1.0 auto-reset wrapper).

        Only NumPy-based vectorized infos are supported (e.g.: infos with tensors are ignored).

        :param observation: Observation returned by the environment.
        :param info: Info returned by the environment.

        :return: Final observation, or ``None`` if no episode ended or the final observation is not available.
        """
        keys = ("final_obs", "final_observation")
        if not self._vectorized:
            return next((info[key] for key in keys if key in info), None)
        # list of per-environment info (e.g.: deprecated API or info-to-list wrappers)
        if isinstance(info, (tuple, list)):
            final_observations = [next((item[key] for key in keys if key in item), None) for item in info]
            mask = None
        else:
            key = next((key for key in keys if key in info), None)
            if key is None:
                return None
            final_observations = info[key]
            mask = info.get(f"_{key}")
        if mask is None and isinstance(final_observations, (np.ndarray, list, tuple)):
            # mask not provided: infer it from the available final observations
            mask = [item is not None for item in final_observations]
        if isinstance(mask, (list, tuple)):
            mask = np.asarray(mask)
        if not isinstance(mask, np.ndarray):
            return None
        indexes = np.flatnonzero(mask)
        if not indexes.size:
            return None
        # replace the observation of the environments whose episodes ended
        # - fundamental spaces (e.g.: Box, Discrete): the observation is a batched array
        if isinstance(observation, np.ndarray):
            observation = observation.copy()
            observation[indexes] = np.stack([final_observations[i] for i in indexes])
            return observation
        # - composite spaces (e.g.: Dict, Tuple)
        space = self.observation_space
        num_envs = len(final_observations)
        items = list(gymnasium.vector.utils.iterate(gymnasium.vector.utils.batch_space(space, num_envs), observation))
        for i in indexes:
            items[i] = final_observations[i]
        return gymnasium.vector.utils.concatenate(
            space, items, gymnasium.vector.utils.create_empty_array(space, num_envs)
        )

    def _parse_group(self, observation: Any) -> tuple[wp.array, wp.array | None]:
        """Parse the observations and states from the environment's output.

        :param observation: Observation returned by the environment.

        :return: Observation, state (``None`` if the environment does not provide it).
        """
        observations = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation, device=self.device)
        )
        # state (``None`` if the environment does not provide it)
        states = None
        space = self.state_space
        state_fn = getattr(self._unwrapped, "state", None)
        if space is not None and callable(state_fn):
            state = None
            try:
                state = state_fn()
            except NotImplementedError:  # environment without state
                pass
            if state is not None:
                states = flatten_tensorized_space(tensorize_space(space, state, device=self.device))
        return observations, states

    def _parse_final_group(self, observation: Any, info: Any) -> tuple[wp.array, wp.array | None]:
        """Parse the final observations and states from the environment's output.

        The final observation of the (same-step) auto-reset environments is read from the info
        (see :py:meth:`_get_final_observation`). The current observations are used if no episode ended
        or the final observation is not available. The final state is not provided by the environment,
        so the current state is used.

        :param observation: Observation returned by the environment.
        :param info: Info returned by the environment.

        :return: Final observation, state.
        """
        final_observation = self._get_final_observation(observation, info)
        if final_observation is None:
            return self._observations, self._states
        final_observations = flatten_tensorized_space(
            tensorize_space(self.observation_space, final_observation, device=self.device)
        )
        return final_observations, self._states

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array, wp.array, Any]:
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

        self._observations, self._states = self._parse_group(observation)
        self._final_observations, self._final_states = self._parse_final_group(observation, info)
        self._info = info

        # convert response to warp
        reward = wp.array(np.array(reward), dtype=wp.float32).reshape((self.num_envs, -1))
        terminated = wp.array(np.array(terminated), dtype=wp.int8).reshape((self.num_envs, -1))
        truncated = wp.array(np.array(truncated), dtype=wp.int8).reshape((self.num_envs, -1))

        return self._observations, reward, terminated, truncated, self._info

    def state(self) -> wp.array | None:
        """Get the environment state.

        The state is computed at the last :py:meth:`reset`/:py:meth:`step` call.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[wp.array, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        # handle vectorized environments (vector environments are autoreset)
        if self._vectorized:
            if self._reset_once:
                observation, self._info = self._env.reset(seed=self._seed)
                self._observations, self._states = self._parse_group(observation)
                self._reset_once = False
                self._seed = None
            self._final_observations = self._observations
            self._final_states = self._states
            return self._observations, self._info

        observation, info = self._env.reset(seed=self._seed)
        self._seed = None
        self._observations, self._states = self._parse_group(observation)
        self._final_observations = self._observations
        self._final_states = self._states
        self._info = info
        return self._observations, self._info

    def final(self) -> tuple[wp.array, wp.array | None]:
        """Get the observations and states with which the last transitions ended.

        The final observation of (same-step) auto-reset environments is read from the info.
        The final state is not provided by the environment, so it is the same as the one returned by :py:meth:`state`.

        See :py:meth:`~skrl.envs.wrappers.warp.base.Wrapper.final` for more details.

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
