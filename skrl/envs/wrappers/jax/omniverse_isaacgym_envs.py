from typing import Any, Optional, Tuple, Union

import jax
import jax.dlpack as jax_dlpack
import numpy as np


try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except:
    pass  # TODO: show warning message
else:
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space

from skrl import logger
from skrl.envs.wrappers.jax.base import Wrapper


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"
if _CPU:
    logger.warning("OmniIsaacGymEnvs runs on GPU, but there is no GPU backend for JAX. JAX operations will run on CPU.")


def _jax2torch(array, device, from_jax=True):
    if from_jax:
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)
    return torch.tensor(array, device=device)


def _torch2jax(tensor, to_jax=True):
    if to_jax:
        return jax_dlpack.from_dlpack(
            torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous())
        )
    return tensor.cpu().numpy()


class OmniverseIsaacGymWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Omniverse Isaac Gym environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Omniverse Isaac Gym environment
        """
        super().__init__(env)

        self._env_device = torch.device(self._unwrapped.device)
        self._reset_once = True
        self._observations = None
        self._info = {}

    def run(self, trainer: Optional["omni.isaac.gym.vec_env.vec_env_mt.TrainerMT"] = None) -> None:
        """Run the simulation in the main thread

        This method is valid only for the Omniverse Isaac Gym multi-threaded environments

        :param trainer: Trainer which should implement a ``run`` method that initiates the RL loop on a new thread
        :type trainer: omni.isaac.gym.vec_env.vec_env_mt.TrainerMT, optional
        """
        self._env.run(trainer)

    def step(self, actions: Union[np.ndarray, jax.Array]) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        actions = _jax2torch(actions, self._env_device, self._jax)

        with torch.no_grad():
            observations, reward, terminated, self._info = self._env.step(
                unflatten_tensorized_space(self.action_space, actions)
            )

        observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["obs"]))
        terminated = terminated.to(dtype=torch.int8)
        truncated = (
            self._info["time_outs"].to(dtype=torch.int8) if "time_outs" in self._info else torch.zeros_like(terminated)
        )

        self._observations = _torch2jax(observations, self._jax)
        return (
            self._observations,
            _torch2jax(reward.view(-1, 1), self._jax),
            _torch2jax(terminated.view(-1, 1), self._jax),
            _torch2jax(truncated.view(-1, 1), self._jax),
            self._info,
        )

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        if self._reset_once:
            observations = self._env.reset()
            observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["obs"]))
            self._observations = _torch2jax(observations, self._jax)
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
