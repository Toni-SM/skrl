from typing import Tuple, Any

import torch
import torch.utils.dlpack
import jax
import jax.dlpack

import jax
import jaxlib
import jax.numpy as jnp

from skrl.envs.jax.wrappers.base import Wrapper


def _jax2torch(array, device, from_jax=True):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(array)) if from_jax else torch.tensor(array, device=device)

def _torch2jax(tensor, to_jax=True):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous())) if to_jax else tensor.cpu().numpy()


class IsaacGymPreview2Wrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 2) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 2) environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_buf = None

    def step(self, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: jnp.ndarray

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of jnp.ndarray and any other info
        """
        actions = _jax2torch(actions, self._env.device, self._jax)

        self._obs_buf, reward, terminated, info = self._env.step(actions)
        truncated = torch.zeros_like(terminated)

        return _torch2jax(self._obs_buf, self._jax), \
               _torch2jax(reward.view(-1, 1), self._jax), \
               _torch2jax(terminated.view(-1, 1), self._jax), \
               _torch2jax(truncated.view(-1, 1), self._jax), \
               info

    def reset(self) -> Tuple[jnp.ndarray, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: jnp.ndarray and any other info
        """
        if self._reset_once:
            self._obs_buf = self._env.reset()
            self._reset_once = False
        return _torch2jax(self._obs_buf, self._jax), {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        pass


class IsaacGymPreview3Wrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None

    def step(self, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: jnp.ndarray

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of jnp.ndarray and any other info
        """
        actions = _jax2torch(actions, self._env.device, self._jax)

        self._obs_dict, reward, terminated, info = self._env.step(actions)
        truncated = torch.zeros_like(terminated)

        return _torch2jax(self._obs_dict["obs"], self._jax), \
               _torch2jax(reward.view(-1, 1), self._jax), \
               _torch2jax(terminated.view(-1, 1), self._jax), \
               _torch2jax(truncated.view(-1, 1), self._jax), \
               info

    def reset(self) -> Tuple[jnp.ndarray, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: jnp.ndarray and any other info
        """
        if self._reset_once:
            self._obs_dict = self._env.reset()
            self._reset_once = False
        return _torch2jax(self._obs_dict["obs"], self._jax), {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        pass
