from typing import Any, Tuple, Union

import jax
import jax.dlpack as jax_dlpack
import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack

from skrl import logger
from skrl.envs.wrappers.jax.base import Wrapper


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"
if _CPU:
    logger.warning("Isaac Orbit runs on GPU, but there is no GPU backend for JAX. JAX operations will run on CPU.")

def _jax2torch(array, device, from_jax=True):
    if from_jax:
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)
    return torch.tensor(array, device=device)

def _torch2jax(tensor, to_jax=True):
    if to_jax:
        return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))
    return tensor.cpu().numpy()


class IsaacOrbitWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Orbit environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Orbit environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None

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
            self._obs_dict, reward, terminated, info = self._env.step(actions)

        terminated = terminated.to(dtype=torch.int8)
        truncated = info["time_outs"].to(dtype=torch.int8) if "time_outs" in info else torch.zeros_like(terminated)

        return _torch2jax(self._obs_dict["policy"], self._jax), \
               _torch2jax(reward.view(-1, 1), self._jax), \
               _torch2jax(terminated.view(-1, 1), self._jax), \
               _torch2jax(truncated.view(-1, 1), self._jax), \
               info

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        if self._reset_once:
            self._obs_dict = self._env.reset()
            self._reset_once = False
        return _torch2jax(self._obs_dict["policy"], self._jax), {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()
