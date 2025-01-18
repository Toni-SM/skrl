from typing import Any, Dict, Union

import pytest

from collections.abc import Mapping
import gymnasium as gym

import jax
import jax.numpy as jnp
import numpy as np
import torch

from skrl import config
from skrl.envs.wrappers.jax import OmniverseIsaacGymWrapper, wrap_env


# hack to fix: `np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead
np.Inf = np.inf


class OmniverseIsaacGymEnv(gym.Env):
    def __init__(self, num_states) -> None:
        self.num_actions = 1
        self.num_observations = 4
        self.num_states = num_states
        self.num_envs = 10
        self.extras = {}
        self.device = "cpu"

        # initialize data spaces (defaults to gym.Box)
        self.action_space = gym.spaces.Box(
            np.ones(self.num_actions, dtype=np.float32) * -1.0, np.ones(self.num_actions, dtype=np.float32) * 1.0
        )
        self.observation_space = gym.spaces.Box(
            np.ones(self.num_observations, dtype=np.float32) * -np.Inf,
            np.ones(self.num_observations, dtype=np.float32) * np.Inf,
        )
        self.state_space = gym.spaces.Box(
            np.ones(self.num_states, dtype=np.float32) * -np.Inf, np.ones(self.num_states, dtype=np.float32) * np.Inf
        )

    def reset(self):
        observations = {"obs": torch.ones((self.num_envs, self.num_observations), device=self.device)}
        return observations

    def step(self, actions):
        assert actions.clone().shape == torch.Size([self.num_envs, 1])
        observations = {
            "obs": torch.ones((self.num_envs, self.num_observations), device=self.device, dtype=torch.float32)
        }
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return observations, rewards, dones, self.extras

    def render(self, recompute: bool = False) -> Union[np.ndarray, None]:
        return None

    def close(self) -> None:
        pass


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("num_states", [0, 5])
def test_env(capsys: pytest.CaptureFixture, backend: str, num_states: int):
    config.jax.backend = backend
    Array = jax.Array if backend == "jax" else np.ndarray

    num_envs = 10
    action = jnp.ones((num_envs, 1)) if backend == "jax" else np.ones((num_envs, 1))

    # load wrap the environment
    original_env = OmniverseIsaacGymEnv(num_states)
    env = wrap_env(original_env, "auto")
    # TODO: assert isinstance(env, OmniverseIsaacGymWrapper)
    env = wrap_env(original_env, "omniverse-isaacgym")
    assert isinstance(env, OmniverseIsaacGymWrapper)

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
