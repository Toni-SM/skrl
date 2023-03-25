from typing import Tuple, Any, Optional

import gymnasium
import numpy as np

import jax
import jaxlib
import jax.numpy as jnp

from skrl.envs.jax.wrappers.base import Wrapper


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
            print("[WARNING] Failed to check for a vectorized environment: {}".format(e))

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

    def _observation_to_tensor(self, observation: Any, space: Optional[gymnasium.Space] = None) -> jnp.ndarray:
        """Convert the Gymnasium observation to a flat tensor

        :param observation: The Gymnasium observation to convert to a tensor
        :type observation: Any supported Gymnasium observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: jnp.ndarray
        """
        observation_space = self._env.observation_space if self._vectorized else self.observation_space
        space = space if space is not None else observation_space

        if self._vectorized and isinstance(space, gymnasium.spaces.MultiDiscrete):
            return jnp.array(observation, dtype=jnp.int64).reshape(self.num_envs, -1)
        elif isinstance(observation, int):
            return jnp.array(observation, dtype=jnp.int64).reshape(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return jnp.array(observation, dtype=jnp.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Discrete):
            return jnp.array(observation, dtype=jnp.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Box):
            return jnp.array(observation, dtype=jnp.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Dict):
            tmp = jnp.concatenate([self._observation_to_tensor(observation[k], space[k]) \
                for k in sorted(space.keys())], axis=-1).reshape(self.num_envs, -1)
            return tmp
        else:
            raise ValueError("Observation space type {} not supported. Please report this issue".format(type(space)))

    def _tensor_to_action(self, actions: jnp.ndarray) -> Any:
        """Convert the action to the Gymnasium expected format

        :param actions: The actions to perform
        :type actions: jnp.ndarray

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
        raise ValueError("Action space type {} not supported. Please report this issue".format(type(space)))

    def step(self, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: jnp.ndarray

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of jnp.ndarray and any other info
        """
        observation, reward, terminated, truncated, info = self._env.step(self._tensor_to_action(actions))

        # convert response to jnp
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
