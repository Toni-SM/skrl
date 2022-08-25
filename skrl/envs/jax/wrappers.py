from typing import Union, Tuple, Any, Optional

import gym
import numpy as np

import jax
import jaxlib
import jax.numpy as jnp

__all__ = ["wrap_env"]


class Wrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments

        :param env: The environment to wrap
        :type env: Any supported RL environment
        """
        self._env = env

        # device (faster than @property)
        if False and hasattr(self._env, "device"):
            pass  # TODO: get device from environment: jaxlib.xla_extension.Device
        else:
            self.device = jax.devices()[0] if len(jax.devices()) else jax.devices("cpu")[0]

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment

        :param key: The attribute name
        :type key: str

        :raises AttributeError: If the attribute does not exist

        :return: The attribute value
        :rtype: Any
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        raise AttributeError("Wrapped environment ({}) does not have attribute '{}'" \
            .format(self._env.__class__.__name__, key))

    def reset(self) -> jaxlib.xla_extension.DeviceArray:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: The state of the environment
        :rtype: jaxlib.xla_extension.DeviceArray
        """
        raise NotImplementedError

    def step(self, actions: jaxlib.xla_extension.DeviceArray) -> Tuple[jaxlib.xla_extension.DeviceArray, 
             jaxlib.xla_extension.DeviceArray, jaxlib.xla_extension.DeviceArray, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: jaxlib.xla_extension.DeviceArray

        :raises NotImplementedError: Not implemented

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of jaxlib.xla_extension.DeviceArray and any other info
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> None:
        """Render the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._env.num_envs if hasattr(self._env, "num_envs") else 1

    @property
    def state_space(self) -> gym.Space:
        """State space

        If the wrapped environment does not have the ``state_space`` property,
        the value of the ``observation_space`` property will be used
        """
        return self._env.state_space if hasattr(self._env, "state_space") else self._env.observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        return self._env.action_space


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
        except Exception as e:
            print("[WARNING] Failed to check for a vectorized environment: {}".format(e))

        if hasattr(self, "new_step_api"):
            self._new_step_api = self._env.new_step_api
        else:
            self._new_step_api = False

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

    def _observation_to_tensor(self, observation: Any, space: Union[gym.Space, None] = None) -> jaxlib.xla_extension.DeviceArray:
        """Convert the OpenAI Gym observation to a flat tensor

        :param observation: The OpenAI Gym observation to convert to a tensor
        :type observation: Any supported OpenAI Gym observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: jaxlib.xla_extension.DeviceArray
        """
        # TODO: move to device
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
        # elif isinstance(space, gym.spaces.Dict): # TODO: jax cat
        #     tmp = torch.cat([self._observation_to_tensor(observation[k], space[k]) \
        #         for k in sorted(space.keys())], dim=-1).reshape(self.num_envs, -1)
        #     return tmp
        else:
            raise ValueError("Observation space type {} not supported. Please report this issue".format(type(space)))

    def _tensor_to_action(self, actions: jaxlib.xla_extension.DeviceArray) -> Any:
        """Convert the action to the OpenAI Gym expected format

        :param actions: The actions to perform
        :type actions: jaxlib.xla_extension.DeviceArray

        :raise ValueError: If the action space type is not supported

        :return: The action in the OpenAI Gym format
        :rtype: Any supported OpenAI Gym action space
        """
        space = self._env.action_space if self._vectorized else self.action_space

        if self._vectorized and isinstance(space, gym.spaces.MultiDiscrete):
            return actions.astype(space.dtype).reshape(space.shape)
        elif isinstance(space, gym.spaces.Discrete):
            return actions.item()
        elif isinstance(space, gym.spaces.Box):
            return actions.astype(space.dtype).reshape(space.shape)
        else:
            raise ValueError("Action space type {} not supported. Please report this issue".format(type(space)))

    def step(self, actions: jaxlib.xla_extension.DeviceArray) -> Tuple[jaxlib.xla_extension.DeviceArray, 
             jaxlib.xla_extension.DeviceArray, jaxlib.xla_extension.DeviceArray, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: jaxlib.xla_extension.DeviceArray

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of jaxlib.xla_extension.DeviceArray and any other info
        """
        if self._new_step_api:
            observation, reward, termination, truncation, info = self._env.step(self._tensor_to_action(actions))
            done = termination or truncation
        else:
            observation, reward, done, info = self._env.step(self._tensor_to_action(actions))
        # convert response to jax
        return self._observation_to_tensor(observation), \
               jnp.array(reward, dtype=jnp.float32).reshape(self.num_envs, -1), \
               jnp.array(done, dtype=bool).reshape(self.num_envs, -1), \
               info

    def reset(self) -> jaxlib.xla_extension.DeviceArray:
        """Reset the environment

        :return: The state of the environment
        :rtype: jaxlib.xla_extension.DeviceArray
        """
        observation = self._env.reset()
        return self._observation_to_tensor(observation)

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()


def wrap_env(env: Any, wrapper: str = "auto", verbose: bool = True) -> Wrapper:
    """Wrap an environment to use a common interface

    Example::

        >>> from skrl.envs.jax import wrap_env
        >>>
        >>> # assuming that there is an environment called "env"
        >>> env = wrap_env(env)

    :param env: The environment to be wrapped
    :type env: Any supported environment
    :param wrapper: The type of wrapper to use (default: "auto").
                    If ``"auto"``, the wrapper will be automatically selected based on the environment class.
                    The supported wrappers are described in the following table:

                    .. raw:: html

                        <br>

                    +--------------------+-------------------------+
                    |Environment         |Wrapper tag              |
                    +====================+=========================+
                    |OpenAI Gym          |``"gym"``                |
                    +--------------------+-------------------------+
    :type wrapper: str, optional
    :param verbose: Whether to print the wrapper type (default: True)
    :type verbose: bool, optional

    :raises ValueError: Unknow wrapper type

    :return: Wrapped environment
    :rtype: Wrapper
    """
    if verbose:
        print("[INFO] Environment:", [str(base).replace("<class '", "").replace("'>", "") \
            for base in env.__class__.__bases__])
    if wrapper == "auto":
        if isinstance(env, gym.core.Env) or isinstance(env, gym.core.Wrapper):
            if verbose:
                print("[INFO] Wrapper: Gym")
            return GymWrapper(env)
        if verbose:
            print("[INFO] Wrapper: Gym")
        return GymWrapper(env)
    elif wrapper == "gym":
        if verbose:
            print("[INFO] Wrapper: Gym")
        return GymWrapper(env)
    else:
        raise ValueError("Unknown {} wrapper type".format(wrapper))
