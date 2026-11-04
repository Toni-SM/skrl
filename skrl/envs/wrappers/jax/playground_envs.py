from __future__ import annotations

from typing import Any

import gymnasium
import mujoco

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config, logger
from skrl.envs.wrappers.jax.base import Wrapper
from skrl.utils.framework.jax.types import Tensor
from skrl.utils.spaces.jax import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """MuJoCo Playground environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None
        self._info = {}

        # build spaces
        observation_size = self._unwrapped.observation_size
        action_size = self._unwrapped.action_size
        if isinstance(observation_size, dict):
            self._observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=observation_size["state"])
            self._state_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=observation_size["privileged_state"]
            )
        else:
            self._observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,))
            self._state_space = None
        self._action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(action_size,))

        # set wrapped environment for just-in-time compilation with XLA
        self._env_state = None
        self._env_step = jax.jit(env.step)
        self._env_reset = jax.jit(env.reset)
        self._env_reset_key = jax.random.split(config.jax.key, self.num_envs)

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return self._action_space

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        return self._state_space

    def _parse_group(self, group: Any) -> tuple[Tensor, Tensor | None]:
        """Parse the observations and states from the environment's observation group.

        :param group: Observation group. If the environment provides states, it is a dictionary with the
            ``"state"`` (observation) and ``"privileged_state"`` (state) keys.

        :return: Observation, state (``None`` if the environment does not provide it).
        """
        if self._state_space is None:
            return flatten_tensorized_space(tensorize_space(self.observation_space, group)), None
        observations = flatten_tensorized_space(tensorize_space(self.observation_space, group["state"]))
        states = flatten_tensorized_space(tensorize_space(self.state_space, group["privileged_state"]))
        return observations, states

    def _parse_final_group(self, info: Any, terminated: Tensor, truncated: Tensor) -> tuple[Tensor, Tensor | None]:
        """Parse the final observations and states from the environment's info.

        The final observation group (see :py:meth:`_get_final_info`) is used for the environments whose episodes
        ended. The current observations/states are used for the rest of the environments, or for all of them
        if no episode ended or the environment does not provide the final observation group.

        :param info: Info returned by the environment.
        :param terminated: Terminated flags (already converted).
        :param truncated: Truncated flags (already converted).

        :return: Final observation, state.
        """
        final_observations = self._observations
        final_states = self._states
        final_group = self._get_final_info(info)[0]
        if final_group is not None:
            dones = jnp.logical_or(terminated, truncated)
            if dones.any():
                observations, states = self._parse_group(final_group)
                final_observations = jnp.where(dones, observations, final_observations)
                if states is not None:
                    final_states = jnp.where(dones, states, final_states)
        return final_observations, final_states

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        self._env_state = self._env_step(self._env_state, actions)
        self._info = self._env_state.info
        self._observations, self._states = self._parse_group(self._env_state.obs)

        reward = self._env_state.reward.reshape(-1, 1)
        terminated = self._env_state.done.reshape(-1, 1)
        truncated = self._info.get("truncation")
        if truncated is None:
            truncated = jnp.zeros_like(self._env_state.done)
        truncated = truncated.reshape(-1, 1)
        self._final_observations, self._final_states = self._parse_final_group(self._info, terminated, truncated)
        return self._observations, reward, terminated, truncated, self._info

    def state(self) -> Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            self._env_state = self._env_reset(self._env_reset_key)
            self._observations, self._states = self._parse_group(self._env_state.obs)
            self._info = self._env_state.info
            self._reset_once = False
        self._final_observations, self._final_states = self._observations, self._states
        return self._observations, self._info

    def final(self) -> tuple[Tensor, Tensor | None]:
        """Get the observations and states with which the last transitions ended.

        The final observations/states of the auto-reset environments are read from the final observation group,
        provided in the info under the first available key: ``"final_obs"``, ``"final_observation"`` or
        ``"final_observations"``. The final observation group must have the same structure as the environment's
        observation group (i.e.: including the ``"privileged_state"`` key, if the environment provides states).

        .. note::

            MuJoCo Playground's auto-reset wrapper (``wrap_for_brax_training``) does not provide the final
            observation group. In that case, the final observations/states are the ones returned by
            :py:meth:`step`/:py:meth:`state` (i.e.: the initial ones of the new episodes for the reset environments).

        See :py:meth:`~skrl.envs.wrappers.jax.base.Wrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        if self.num_envs > 1:
            logger.warning("Rendering is not supported for parallel environments. Rendering will be skipped")
            return

        # render frame
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        frame = self._unwrapped.render(self._env_state, width=640, height=480, scene_option=scene_option)
        # show rendered frame using OpenCV
        try:
            import cv2

            cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        except ImportError as e:
            logger.warning(f"Unable to import opencv-python: {e}. Frame will not be rendered.")
        return frame

    def close(self) -> None:
        """Close the environment."""
        pass
