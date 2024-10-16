from typing import Any, Dict, Tuple, Union

import pytest

from collections.abc import Mapping
import gym
import gymnasium

import numpy as np
import torch

from skrl.envs.wrappers.torch import IsaacGymPreview3Wrapper, wrap_env


# hack to fix: `np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead
np.Inf = np.inf


class IsaacGymEnv:
    def __init__(self, num_states) -> None:
        self.num_actions = 1
        self.num_obs = 4
        self.num_states = num_states
        self.num_envs = 10
        self.extras = {}
        self.device = "cpu"

        self.state_space = gym.spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.observation_space = gym.spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.action_space = gym.spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

    def reset(self) -> Dict[str, torch.Tensor]:
        obs_dict = {}
        obs_dict["obs"] = torch.ones((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        if self.num_states > 0:
            obs_dict["states"] = torch.ones((self.num_envs, self.num_states), device=self.device, dtype=torch.float32)
        return obs_dict

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert actions.clone().shape == torch.Size([self.num_envs, 1])
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        obs_dict = {}
        obs_dict["obs"] = torch.ones((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        if self.num_states > 0:
            obs_dict["states"] = torch.ones((self.num_envs, self.num_states), device=self.device, dtype=torch.float32)
        self.extras["time_outs"] = torch.zeros_like(terminated)
        return obs_dict, rewards, terminated, self.extras

    def render(self, mode: str = "rgb_array") -> Union[np.ndarray, None]:
        return None

    def close(self) -> None:
        pass


@pytest.mark.parametrize("num_states", [0, 5])
def test_env(capsys: pytest.CaptureFixture, num_states: int):
    num_envs = 10
    action = torch.ones((num_envs, 1))

    # load wrap the environment
    original_env = IsaacGymEnv(num_states)
    # TODO: env = wrap_env(original_env, "auto")
    # TODO: assert isinstance(env, IsaacGymPreview3Wrapper)
    env = wrap_env(original_env, "isaacgym-preview4")
    assert isinstance(env, IsaacGymPreview3Wrapper)  # preview 4 is the same as 3

    # check properties
    if num_states:
        assert isinstance(env.state_space, gymnasium.Space) and env.state_space.shape == (num_states,)
    else:
        assert env.state_space is None
    assert isinstance(env.observation_space, gymnasium.Space) and env.observation_space.shape == (4,)
    assert isinstance(env.action_space, gymnasium.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, torch.device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env  # IsaacGymEnvs don't inherit from gym.Env
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        observation, info = env.reset()  # edge case: parallel environments are autoreset
        assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 4])
        assert isinstance(info, Mapping)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 4])
            assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([num_envs, 1])
            assert isinstance(terminated, torch.Tensor) and terminated.shape == torch.Size([num_envs, 1])
            assert isinstance(truncated, torch.Tensor) and truncated.shape == torch.Size([num_envs, 1])
            assert isinstance(info, Mapping)

    env.close()
