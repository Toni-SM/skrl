from __future__ import annotations

from typing import Any

import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config, logger
from skrl.envs.wrappers.jax.base import Wrapper
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

    def step(self, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        self._env_state = self._env_step(self._env_state, actions)

        if self._state_space is None:
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, self._env_state.obs))
        else:
            self._observations = flatten_tensorized_space(
                tensorize_space(self.observation_space, self._env_state.obs["state"])
            )
            self._states = flatten_tensorized_space(
                tensorize_space(self.state_space, self._env_state.obs["privileged_state"])
            )
        reward = self._env_state.reward
        terminated = self._env_state.done
        truncated = self._env_state.info.get("truncation")
        if truncated is None:
            truncated = jnp.zeros_like(terminated)
        info = self._env_state.info

        return self._observations, reward.reshape(-1, 1), terminated.reshape(-1, 1), truncated.reshape(-1, 1), info

    def state(self) -> jax.Array | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[jax.Array, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            self._env_state = self._env_reset(self._env_reset_key)
            if self._state_space is None:
                self._observations = flatten_tensorized_space(
                    tensorize_space(self.observation_space, self._env_state.obs)
                )
            else:
                self._observations = flatten_tensorized_space(
                    tensorize_space(self.observation_space, self._env_state.obs["state"])
                )
                self._states = flatten_tensorized_space(
                    tensorize_space(self.state_space, self._env_state.obs["privileged_state"])
                )
            self._info = self._env_state.info
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        if self.num_envs > 1:
            logger.warning("Rendering is not supported for parallel environments. Rendering will be skipped")
            return

        import mujoco

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
