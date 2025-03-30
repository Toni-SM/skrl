import pytest

import sys
from collections.abc import Mapping
import gym
import gymnasium

import torch

from skrl.envs.wrappers.torch import GymWrapper, wrap_env


def test_env(capsys: pytest.CaptureFixture):
    num_envs = 1
    action = torch.ones((num_envs, 1))

    # load wrap the environment
    original_env = gym.make("Pendulum-v1")
    env = wrap_env(original_env, "auto")
    assert isinstance(env, GymWrapper)
    env = wrap_env(original_env, "gym")
    assert isinstance(env, GymWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gymnasium.Space) and env.observation_space.shape == (3,)
    assert isinstance(env.action_space, gymnasium.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, torch.device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
        assert isinstance(info, Mapping)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
            assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([num_envs, 1])
            assert isinstance(terminated, torch.Tensor) and terminated.shape == torch.Size([num_envs, 1])
            assert isinstance(truncated, torch.Tensor) and truncated.shape == torch.Size([num_envs, 1])
            assert isinstance(info, Mapping)

    env.close()


@pytest.mark.parametrize("vectorization_mode", ["async", "sync"])
def test_vectorized_env(capsys: pytest.CaptureFixture, vectorization_mode: str):
    num_envs = 10
    action = torch.ones((num_envs, 1))

    # load wrap the environment
    original_env = gym.vector.make("Pendulum-v1", num_envs=num_envs, asynchronous=(vectorization_mode == "async"))
    env = wrap_env(original_env, "auto")
    assert isinstance(env, GymWrapper)
    env = wrap_env(original_env, "gym")
    assert isinstance(env, GymWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gymnasium.Space) and env.observation_space.shape == (3,)
    assert isinstance(env.action_space, gymnasium.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, torch.device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    assert env._vectorized is True
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        observation, info = env.reset()  # edge case: vectorized environments are autoreset
        assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
        assert isinstance(info, Mapping)
        for _ in range(3):
            try:
                observation, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                if sys.platform.startswith("win"):
                    continue
                raise e
            env.render()
            assert isinstance(observation, torch.Tensor) and observation.shape == torch.Size([num_envs, 3])
            assert isinstance(reward, torch.Tensor) and reward.shape == torch.Size([num_envs, 1])
            assert isinstance(terminated, torch.Tensor) and terminated.shape == torch.Size([num_envs, 1])
            assert isinstance(truncated, torch.Tensor) and truncated.shape == torch.Size([num_envs, 1])
            assert isinstance(info, Mapping)

    try:
        env.close()
    except Exception as e:
        if not sys.platform.startswith("win"):
            raise e
