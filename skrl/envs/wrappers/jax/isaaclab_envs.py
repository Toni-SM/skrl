from typing import Any, Mapping, Tuple, Union

import gymnasium

import jax
import jax.dlpack as jax_dlpack
import numpy as np


try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except:
    pass  # TODO: show warning message

from skrl import logger
from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper, Wrapper


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"
if _CPU:
    logger.warning("Isaac Lab runs on GPU, but there is no GPU backend for JAX. JAX operations will run on CPU.")

def _jax2torch(array, device, from_jax=True):
    if from_jax:
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)
    return torch.tensor(array, device=device)

def _torch2jax(tensor, to_jax=True):
    if to_jax:
        return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))
    return tensor.cpu().numpy()


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._env_device = torch.device(self._unwrapped.device)
        self._reset_once = True
        self._observations = None
        self._info = {}

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space
        """
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space
        """
        try:
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space["policy"]

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space
        """
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: Union[np.ndarray, jax.Array]) -> \
        Tuple[Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array],
              Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array], Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        actions = _jax2torch(actions, self._env_device, self._jax)

        with torch.no_grad():
            self._observations, reward, terminated, truncated, self._info = self._env.step(actions)

        terminated = terminated.to(dtype=torch.int8)
        truncated = truncated.to(dtype=torch.int8)

        return _torch2jax(self._observations["policy"], self._jax), \
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
            self._observations, self._info = self._env.reset()
            self._reset_once = False
        return _torch2jax(self._observations["policy"], self._jax), self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        return None

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()


class IsaacLabMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper for multi-agent implementation

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._env_device = torch.device(self._unwrapped.device)
        self._reset_once = True
        self._observations = None
        self._info = {}

    def step(self, actions: Mapping[str, Union[np.ndarray, jax.Array]]) -> \
        Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Any]]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dictionary of np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dictionaries of np.ndarray or jax.Array and any other info
        """
        actions = {uid: _jax2torch(value, self._env_device, self._jax) for uid, value in actions.items()}

        with torch.no_grad():
            observations, rewards, terminated, truncated, self._info = self._env.step(actions)

        self._observations = {uid: _torch2jax(value, self._jax) for uid, value in observations.items()}
        return self._observations, \
               {uid: _torch2jax(value.view(-1, 1), self._jax) for uid, value in rewards.items()}, \
               {uid: _torch2jax(value.to(dtype=torch.int8).view(-1, 1), self._jax) for uid, value in terminated.items()}, \
               {uid: _torch2jax(value.to(dtype=torch.int8).view(-1, 1), self._jax) for uid, value in truncated.items()}, \
               self._info

    def reset(self) -> Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = {uid: _torch2jax(value, self._jax) for uid, value in observations.items()}
            self._reset_once = False
        return self._observations, self._info

    def state(self) -> Union[np.ndarray, jax.Array, None]:
        """Get the environment state

        :return: State
        :rtype: np.ndarray, jax.Array or None
        """
        state = self._env.state()
        return None if state is None else _torch2jax(state, self._jax)

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        return None

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()
