from typing import Tuple, Any, Optional

import gym
import numpy as np
from packaging import version

import jax
import jaxlib
import jax.numpy as jnp

from skrl.envs.jax.wrappers.base import Wrapper

from skrl import logger


class GymWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """OpenAI Gym environment wrapper

        :param env: The environment to wrap
        :type env: Any supported OpenAI Gym environment
        """
        super().__init__(env)

        self._vectorized = False
        try:
            if isinstance(env, gym.vector.SyncVectorEnv) or isinstance(env, gym.vector.AsyncVectorEnv):
                self._vectorized = True
                self._reset_once = True
                self._obs_tensor = None
                self._info_dict = None
        except Exception as e:
            print("[WARNING] Failed to check for a vectorized environment: {}".format(e))

        self._deprecated_api = version.parse(gym.__version__) < version.parse("0.25.0")
        if self._deprecated_api:
            logger.warning("Using a deprecated version of OpenAI Gym's API: {}".format(gym.__version__))

    @property
    def state_space(self) -> gym.Space:
        """State space

        An alias for the ``observation_space`` property
        """
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        if self._vectorized:
            return self._env.single_action_space
        return self._env.action_space

    def _observation_to_tensor(self, observation: Any, space: Optional[gym.Space] = None) -> jnp.ndarray:
        """Convert the OpenAI Gym observation to a flat tensor

        :param observation: The OpenAI Gym observation to convert to a tensor
        :type observation: Any supported OpenAI Gym observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: jnp.ndarray
        """
        observation_space = self._env.observation_space if self._vectorized else self.observation_space
        space = space if space is not None else observation_space

        if self._vectorized and isinstance(space, gym.spaces.MultiDiscrete):
            return jnp.array(observation, dtype=jnp.int64).reshape(self.num_envs, -1)
        elif isinstance(observation, int):
            return jnp.array(observation, dtype=jnp.int64).reshape(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return jnp.array(observation, dtype=jnp.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Discrete):
            return jnp.array(observation, dtype=jnp.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Box):
            return jnp.array(observation, dtype=jnp.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Dict):
            tmp = jnp.concatenate([self._observation_to_tensor(observation[k], space[k]) \
                for k in sorted(space.keys())], axis=-1).reshape(self.num_envs, -1)
            return tmp
        else:
            raise ValueError("Observation space type {} not supported. Please report this issue".format(type(space)))

    def _tensor_to_action(self, actions: jnp.ndarray) -> Any:
        """Convert the action to the OpenAI Gym expected format

        :param actions: The actions to perform
        :type actions: jnp.ndarray

        :raise ValueError: If the action space type is not supported

        :return: The action in the OpenAI Gym format
        :rtype: Any supported OpenAI Gym action space
        """
        space = self._env.action_space if self._vectorized else self.action_space

        if self._vectorized:
            if isinstance(space, gym.spaces.MultiDiscrete):
                return actions.astype(space.dtype).reshape(space.shape)
            elif isinstance(space, gym.spaces.Tuple):
                if isinstance(space[0], gym.spaces.Box):
                    return actions.astype(space[0].dtype).reshape(space.shape)
                elif isinstance(space[0], gym.spaces.Discrete):
                    return actions.astype(space[0].dtype).reshape(-1)
        elif isinstance(space, gym.spaces.Discrete):
            return actions.item()
        elif isinstance(space, gym.spaces.Box):
            return actions.astype(space.dtype).reshape(space.shape)
        raise ValueError("Action space type {} not supported. Please report this issue".format(type(space)))

    def step(self, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: jnp.ndarray

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of jnp.ndarray and any other info
        """
        if self._deprecated_api:
            observation, reward, terminated, info = self._env.step(self._tensor_to_action(actions))
            # truncated: https://gymnasium.farama.org/tutorials/handling_time_limits
            if type(info) is list:
                truncated = np.array([d.get("TimeLimit.truncated", False) for d in info], dtype=terminated.dtype)
                terminated *= np.logical_not(truncated)
            else:
                truncated = info.get("TimeLimit.truncated", False)
                if truncated:
                    terminated = False
        else:
            observation, reward, terminated, truncated, info = self._env.step(self._tensor_to_action(actions))

        # convert response to jax
        observation = self._observation_to_tensor(observation)
        reward = jnp.array(reward, dtype=jnp.float32).reshape(self.num_envs, -1)
        terminated = jnp.array(terminated, dtype=jnp.bool_).reshape(self.num_envs, -1)
        truncated = jnp.array(truncated, dtype=jnp.bool_).reshape(self.num_envs, -1)

        # save observation and info for vectorized envs
        if self._vectorized:
            self._obs_tensor = observation
            self._info_dict = info

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[jnp.ndarray, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: jnp.ndarray and any other info
        """
        # handle vectorized envs
        if self._vectorized:
            if not self._reset_once:
                return self._obs_tensor, self._info_dict
            self._reset_once = False

        # reset the env/envs
        if self._deprecated_api:
            observation = self._env.reset()
            info = {}
        else:
            observation, info = self._env.reset()
        return self._observation_to_tensor(observation), info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()