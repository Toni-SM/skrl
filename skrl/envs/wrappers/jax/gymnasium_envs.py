from typing import Any, Optional, Tuple, Union

import gymnasium

import jax
import numpy as np

from skrl import logger
from skrl.envs.wrappers.jax.base import Wrapper


class GymnasiumWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Gymnasium environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Gymnasium environment
        """
        super().__init__(env)

        self._vectorized = False
        try:
            if isinstance(env, gymnasium.vector.SyncVectorEnv) or isinstance(env, gymnasium.vector.AsyncVectorEnv):
                self._vectorized = True
                self._reset_once = True
                self._obs_tensor = None
                self._info_dict = None
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")

    @property
    def state_space(self) -> gymnasium.Space:
        """State space

        An alias for the ``observation_space`` property
        """
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space
        """
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space
        """
        if self._vectorized:
            return self._env.single_action_space
        return self._env.action_space

    def _observation_to_tensor(self, observation: Any, space: Optional[gymnasium.Space] = None) -> np.ndarray:
        """Convert the Gymnasium observation to a flat tensor

        :param observation: The Gymnasium observation to convert to a tensor
        :type observation: Any supported Gymnasium observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: np.ndarray
        """
        observation_space = self._env.observation_space if self._vectorized else self.observation_space
        space = space if space is not None else observation_space

        if self._vectorized and isinstance(space, gymnasium.spaces.MultiDiscrete):
            return observation.reshape(self.num_envs, -1).astype(np.int32)
        elif isinstance(observation, int):
            return np.array(observation, dtype=np.int32).reshape(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return observation.reshape(self.num_envs, -1).astype(np.float32)
        elif isinstance(space, gymnasium.spaces.Discrete):
            return np.array(observation, dtype=np.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Box):
            return observation.reshape(self.num_envs, -1).astype(np.float32)
        elif isinstance(space, gymnasium.spaces.Dict):
            tmp = np.concatenate([self._observation_to_tensor(observation[k], space[k]) \
                for k in sorted(space.keys())], axis=-1).reshape(self.num_envs, -1)
            return tmp
        else:
            raise ValueError(f"Observation space type {type(space)} not supported. Please report this issue")

    def _tensor_to_action(self, actions: np.ndarray) -> Any:
        """Convert the action to the Gymnasium expected format

        :param actions: The actions to perform
        :type actions: np.ndarray

        :raise ValueError: If the action space type is not supported

        :return: The action in the Gymnasium format
        :rtype: Any supported Gymnasium action space
        """
        space = self._env.action_space if self._vectorized else self.action_space

        if self._vectorized:
            if isinstance(space, gymnasium.spaces.MultiDiscrete):
                return actions.astype(space.dtype).reshape(space.shape)
            elif isinstance(space, gymnasium.spaces.Tuple):
                if isinstance(space[0], gymnasium.spaces.Box):
                    return actions.astype(space[0].dtype).reshape(space.shape)
                elif isinstance(space[0], gymnasium.spaces.Discrete):
                    return actions.astype(space[0].dtype).reshape(-1)
        if isinstance(space, gymnasium.spaces.Discrete):
            return actions.item()
        elif isinstance(space, gymnasium.spaces.Box):
            return actions.astype(space.dtype).reshape(space.shape)
        raise ValueError(f"Action space type {type(space)} not supported. Please report this issue")

    def step(self, actions: Union[np.ndarray, jax.Array]) -> \
        Tuple[Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array],
              Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array], Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        if self._jax:
            actions = jax.device_get(actions)
        observation, reward, terminated, truncated, info = self._env.step(self._tensor_to_action(actions))

        # convert response to numpy or jax
        observation = self._observation_to_tensor(observation)
        reward = np.array(reward, dtype=np.float32).reshape(self.num_envs, -1)
        terminated = np.array(terminated, dtype=np.int8).reshape(self.num_envs, -1)
        truncated = np.array(truncated, dtype=np.int8).reshape(self.num_envs, -1)
        # if self._jax:  # HACK: jax.device_put(...).block_until_ready()
        #     observation = jax.device_put(observation)
        #     reward = jax.device_put(reward)
        #     terminated = jax.device_put(terminated)
        #     truncated = jax.device_put(truncated)

        # save observation and info for vectorized envs
        if self._vectorized:
            self._obs_tensor = observation
            self._info_dict = info

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        # handle vectorized envs
        if self._vectorized:
            if not self._reset_once:
                return self._obs_tensor, self._info_dict
            self._reset_once = False

        # reset the env/envs
        observation, info = self._env.reset()

        # convert response to numpy or jax
        observation = self._observation_to_tensor(observation)
        # if self._jax:  # HACK: jax.device_put(...).block_until_ready()
        #     observation = jax.device_put(observation)

        return observation, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()
