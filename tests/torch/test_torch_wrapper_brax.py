import pytest
import warnings

from collections.abc import Mapping
import gymnasium as gym

import torch

from skrl.envs.wrappers.torch import BraxWrapper, wrap_env


def test_env(capsys: pytest.CaptureFixture):
    num_envs = 10
    action = torch.ones((num_envs, 1))

    # load wrap the environment
    try:
        import brax.envs
    except ImportError as e:
        warnings.warn(f"\n\nUnable to import Brax environment ({e}).\nThis test will be skipped.\n")
        return

    original_env = brax.envs.create("inverted_pendulum", batch_size=num_envs, backend="spring")
    env = wrap_env(original_env, "auto")
    assert isinstance(env, BraxWrapper)
    env = wrap_env(original_env, "brax")
    assert isinstance(env, BraxWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (4,)
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, torch.device)
    # check internal properties
    assert env._env is not original_env  # # brax's VectorGymWrapper interferes with the checking (it has _env)
    assert env._unwrapped is not original_env.unwrapped  # brax's VectorGymWrapper interferes with the checking
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
