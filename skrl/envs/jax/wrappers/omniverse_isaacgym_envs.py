from typing import Tuple, Any, Optional

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


class OmniverseIsaacGymWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Omniverse Isaac Gym environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Omniverse Isaac Gym environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None

    def run(self, trainer: Optional["omni.isaac.gym.vec_env.vec_env_mt.TrainerMT"] = None) -> None:
        """Run the simulation in the main thread

        This method is valid only for the Omniverse Isaac Gym multi-threaded environments

        :param trainer: Trainer which should implement a ``run`` method that initiates the RL loop on a new thread
        :type trainer: omni.isaac.gym.vec_env.vec_env_mt.TrainerMT, optional
        """
        self._env.run(trainer)

    def step(self, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: jnp.ndarray

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of jnp.ndarray and any other info
        """
        actions = _jax2torch(actions, self._env._task.device, self._jax)

        with torch.no_grad():
            self._obs_dict, reward, terminated, info = self._env.step(actions)
            terminated = terminated.to(dtype=torch.int8).view(-1, 1)

        return _torch2jax(self._obs_dict["obs"], self._jax), \
               _torch2jax(reward.view(-1, 1), self._jax), \
               _torch2jax(terminated, self._jax), \
               _torch2jax(terminated, self._jax), \
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
        self._env.close()
