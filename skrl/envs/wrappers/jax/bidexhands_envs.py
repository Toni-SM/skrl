from typing import Any, Mapping, Sequence, Tuple, Union

import gym

import jax
import jax.dlpack as jax_dlpack
import numpy as np


try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except:
    pass  # TODO: show warning message

from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"

def _jax2torch(array, device, from_jax=True):
    if from_jax:
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)
    return torch.tensor(array, device=device)

def _torch2jax(tensor, to_jax=True):
    if to_jax:
        return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))
    return tensor.cpu().numpy()


class BiDexHandsWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Bi-DexHands wrapper

        :param env: The environment to wrap
        :type env: Any supported Bi-DexHands environment
        """
        super().__init__(env)

        self._reset_once = True
        self._states = None
        self._observations = None
        self._info = {}

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        return self.possible_agents

    @property
    def possible_agents(self) -> Sequence[str]:
        """Names of all possible agents the environment could generate

        These can not be changed as an environment progresses
        """
        return [f"agent_{i}" for i in range(self.num_agents)]

    @property
    def state_spaces(self) -> Mapping[str, gym.Space]:
        """State spaces

        Since the state space is a global view of the environment (and therefore the same for all the agents),
        this property returns a dictionary (for consistency with the other space-related properties) with the same
        space for all the agents
        """
        return {uid: space for uid, space in zip(self.possible_agents, self._env.share_observation_space)}

    @property
    def observation_spaces(self) -> Mapping[str, gym.Space]:
        """Observation spaces
        """
        return {uid: space for uid, space in zip(self.possible_agents, self._env.observation_space)}

    @property
    def action_spaces(self) -> Mapping[str, gym.Space]:
        """Action spaces
        """
        return {uid: space for uid, space in zip(self.possible_agents, self._env.action_space)}

    def step(self, actions: Mapping[str, Union[np.ndarray, jax.Array]]) -> \
        Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Any]]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dict of nd.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dict of nd.ndarray or jax.Array and any other info
        """
        actions = [_jax2torch(actions[uid], self._env.rl_device, self._jax) for uid in self.possible_agents]

        with torch.no_grad():
            observations, states, rewards, terminated, _, _ = self._env.step(actions)

        observations = _torch2jax(observations, self._jax)
        states = _torch2jax(states, self._jax)
        rewards = _torch2jax(rewards, self._jax)
        terminated = _torch2jax(terminated.to(dtype=torch.int8), self._jax)

        self._states = states[:, 0]
        self._observations = {uid: observations[:,i] for i, uid in enumerate(self.possible_agents)}
        rewards = {uid: rewards[:,i].reshape(-1, 1) for i, uid in enumerate(self.possible_agents)}
        terminated = {uid: terminated[:,i].reshape(-1, 1) for i, uid in enumerate(self.possible_agents)}
        truncated = terminated

        return self._observations, rewards, terminated, truncated, self._info

    def state(self) -> Union[np.ndarray, jax.Array]:
        """Get the environment state

        :return: State
        :rtype: np.ndarray of jax.Array
        """
        return self._states

    def reset(self) -> Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: tuple of dict of np.ndarray of jax.Array and any other info
        """
        if self._reset_once:
            observations, states, _ = self._env.reset()

            observations = _torch2jax(observations, self._jax)
            states = _torch2jax(states, self._jax)

            self._states = states[:, 0]
            self._observations = {uid: observations[:,i] for i, uid in enumerate(self.possible_agents)}
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        return None

    def close(self) -> None:
        """Close the environment
        """
        pass
