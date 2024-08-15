from typing import Any, Dict, Union

import pytest

from collections.abc import Mapping
import gymnasium as gym

import jax
import jax.numpy as jnp
import numpy as np
import torch

from skrl import config
from skrl.envs.wrappers.jax import IsaacLabWrapper, wrap_env


VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]


class IsaacLabEnv(gym.Env):
    def __init__(self, num_states) -> None:
        self.num_actions = 1
        self.num_observations = 4
        self.num_states = num_states
        self.num_envs = 10
        self.extras = {}
        self.device = "cpu"

        self._configure_gym_env_spaces()

    # https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/direct_rl_env.py
    def _configure_gym_env_spaces(self):
        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        observations = {"policy": torch.ones((self.num_envs, self.num_observations), device=self.device)}
        return observations, self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        assert action.clone().shape == torch.Size([self.num_envs, 1])
        observations = {"policy": torch.ones((self.num_envs, self.num_observations), device=self.device, dtype=torch.float32)}
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        return observations, rewards, terminated, truncated, self.extras

    def render(self, recompute: bool = False) -> Union[np.ndarray, None]:
        return None

    def close(self) -> None:
        pass


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("num_states", [0, 5])
def test_env(capsys: pytest.CaptureFixture, backend: str, num_states):
    config.jax.backend = backend
    Array = jax.Array if backend == "jax" else np.ndarray

    num_envs = 10
    action = jnp.ones((num_envs, 1)) if backend == "jax" else np.ones((num_envs, 1))

    # load wrap the environment
    original_env = IsaacLabEnv(num_states)
    env = wrap_env(original_env, "auto")
    # TODO: assert isinstance(env, IsaacLabWrapper)
    env = wrap_env(original_env, "isaaclab")
    assert isinstance(env, IsaacLabWrapper)

    # check properties
    if num_states:
        assert isinstance(env.state_space, gym.Space) and env.state_space.shape == (num_states,)
    else:
        assert env.state_space is None or env.state_space.shape == (num_states,)
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (4,)
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, jax.Device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        observation, info = env.reset()  # edge case: parallel environments are autoreset
        assert isinstance(observation, Array) and observation.shape == (num_envs, 4)
        assert isinstance(info, Mapping)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            assert isinstance(observation, Array) and observation.shape == (num_envs, 4)
            assert isinstance(reward, Array) and reward.shape == (num_envs, 1)
            assert isinstance(terminated, Array) and terminated.shape == (num_envs, 1)
            assert isinstance(truncated, Array) and truncated.shape == (num_envs, 1)
            assert isinstance(info, Mapping)

    env.close()
