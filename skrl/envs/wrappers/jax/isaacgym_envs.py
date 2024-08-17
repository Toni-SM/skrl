from typing import Any, Tuple, Union

import gym

import jax
import jax.dlpack as jax_dlpack
import numpy as np


try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except:
    pass  # TODO: show warning message

from skrl import logger
from skrl.envs.wrappers.jax.base import Wrapper


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"
if _CPU:
    logger.warning("IsaacGymEnvs runs on GPU, but there is no GPU backend for JAX. JAX operations will run on CPU.")

def _jax2torch(array, device, from_jax=True):
    if from_jax:
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)
    return torch.tensor(array, device=device)

def _torch2jax(tensor, to_jax=True):
    if to_jax:
        return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))
    return tensor.cpu().numpy()


class IsaacGymPreview2Wrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 2) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 2) environment
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}

    def step(self, actions: Union[np.ndarray, jax.Array]) -> \
        Tuple[Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array],
              Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array], Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        actions = _jax2torch(actions, self._env.device, self._jax)

        with torch.no_grad():
            self._observations, reward, terminated, self._info = self._env.step(actions)

        terminated = terminated.to(dtype=torch.int8)
        truncated = self._info["time_outs"].to(dtype=torch.int8) if "time_outs" in self._info else torch.zeros_like(terminated)

        return _torch2jax(self._observations, self._jax), \
               _torch2jax(reward.view(-1, 1), self._jax), \
               _torch2jax(terminated.view(-1, 1), self._jax), \
               _torch2jax(truncated.view(-1, 1), self._jax), \
               self._info

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        if self._reset_once:
            self._observations = self._env.reset()
            self._reset_once = False
        return _torch2jax(self._observations, self._jax), self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        return None

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
        self._observations = None
        self._info = {}

    @property
    def state_space(self) -> Union[gym.Space, None]:
        """State space
        """
        try:
            if self.num_states:
                return self._unwrapped.state_space
        except:
            pass
        return None

    def step(self, actions: Union[np.ndarray, jax.Array]) ->\
        Tuple[Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array],
              Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array], Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        actions = _jax2torch(actions, self._env.device, self._jax)

        with torch.no_grad():
            self._observations, reward, terminated, self._info = self._env.step(actions)

        terminated = terminated.to(dtype=torch.int8)
        truncated = self._info["time_outs"].to(dtype=torch.int8) if "time_outs" in self._info else torch.zeros_like(terminated)

        return _torch2jax(self._observations["obs"], self._jax), \
               _torch2jax(reward.view(-1, 1), self._jax), \
               _torch2jax(terminated.view(-1, 1), self._jax), \
               _torch2jax(truncated.view(-1, 1), self._jax), \
               self._info

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        if self._reset_once:
            self._observations = self._env.reset()
            self._reset_once = False
        return _torch2jax(self._observations["obs"], self._jax), self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        return None

    def close(self) -> None:
        """Close the environment
        """
        pass
