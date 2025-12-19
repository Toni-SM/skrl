from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import jax
import numpy as np

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
    def __init__(self, env: Any) -> None:
        """OpenAI Gym environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        # hack to fix: module 'numpy' has no attribute 'bool8'
        try:
            np.bool8
        except AttributeError:
            np.bool8 = np.bool

        import gym

        self._seed = np.asarray(jax.device_get(config.jax.key)).sum().item()
        self._vectorized = False
        try:
            if isinstance(env, gym.vector.VectorEnv):
                self._vectorized = True
                self._reset_once = True
                self._observation = None
                self._info = None
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")

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
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        if self._vectorized:
            return convert_gym_space(self._env.single_action_space)
        return convert_gym_space(self._env.action_space)

    def step(self, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Any]:
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
                info = {}
            else:
                truncated = info.get("TimeLimit.truncated", False)
                if truncated:
                    terminated = False
        else:
            observation, reward, terminated, truncated, info = self._env.step(actions)

        # convert response to numpy or jax
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))
        reward = jax.device_put(np.array(reward, dtype=np.float32).reshape(self.num_envs, -1), device=self.device)
        terminated = jax.device_put(np.array(terminated, dtype=np.int8).reshape(self.num_envs, -1), device=self.device)
        truncated = jax.device_put(np.array(truncated, dtype=np.int8).reshape(self.num_envs, -1), device=self.device)
        # save observation and info for vectorized envs
        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info

    def state(self) -> jax.Array | None:
        """Get the environment state.

        :return: State.
        """
        try:
            return flatten_tensorized_space(
                tensorize_space(self.state_space, self._unwrapped.state(), device=self.device)
            )
        except:
            return None

    def reset(self) -> tuple[jax.Array, dict[str, Any]]:
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
                self._observation = flatten_tensorized_space(
                    tensorize_space(self.observation_space, observation, device=self.device)
                )
                self._reset_once = False
                self._seed = None
            return self._observation, self._info

        if self._deprecated_api:
            self._env.seed(self._seed)
            observation = self._env.reset()
            info = {}
        else:
            observation, info = self._env.reset(seed=self._seed)
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))
        self._seed = None
        return observation, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        if self._vectorized:
            return None
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
