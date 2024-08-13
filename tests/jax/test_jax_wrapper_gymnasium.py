import pytest

from collections.abc import Mapping
import gymnasium as gym

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.envs.wrappers.jax import GymnasiumWrapper, wrap_env


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_env(capsys: pytest.CaptureFixture, backend: str):
    config.jax.backend = backend
    Array = jax.Array if backend == "jax" else np.ndarray

    num_envs = 1
    action = jnp.ones((num_envs, 1))

    # load wrap the environment
    original_env = gym.make("Pendulum-v1")
    env = wrap_env(original_env, "auto")
    assert isinstance(env, GymnasiumWrapper)
    env = wrap_env(original_env, "gymnasium")
    assert isinstance(env, GymnasiumWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (3,)
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
        assert isinstance(observation, Array) and observation.shape == (num_envs, 3)
        assert isinstance(info, Mapping)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            assert isinstance(observation, Array) and observation.shape == (num_envs, 3)
            assert isinstance(reward, Array) and reward.shape == (num_envs, 1)
            assert isinstance(terminated, Array) and terminated.shape == (num_envs, 1)
            assert isinstance(truncated, Array) and truncated.shape == (num_envs, 1)
            assert isinstance(info, Mapping)

    env.close()

@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("vectorization_mode", ["async", "sync"])
def test_vectorized_env(capsys: pytest.CaptureFixture, backend: str, vectorization_mode: str):
    config.jax.backend = backend
    Array = jax.Array if backend == "jax" else np.ndarray

    num_envs = 10
    action = jnp.ones((num_envs, 1))

    # load wrap the environment
    original_env = gym.make_vec("Pendulum-v1", num_envs=num_envs, vectorization_mode=vectorization_mode)
    env = wrap_env(original_env, "auto")
    assert isinstance(env, GymnasiumWrapper)
    env = wrap_env(original_env, "gymnasium")
    assert isinstance(env, GymnasiumWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (3,)
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, jax.Device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    assert env._vectorized is True
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        observation, info = env.reset()  # edge case: vectorized environments are autoreset
        assert isinstance(observation, Array) and observation.shape == (num_envs, 3)
        assert isinstance(info, Mapping)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            assert isinstance(observation, Array) and observation.shape == (num_envs, 3)
            assert isinstance(reward, Array) and reward.shape == (num_envs, 1)
            assert isinstance(terminated, Array) and terminated.shape == (num_envs, 1)
            assert isinstance(truncated, Array) and truncated.shape == (num_envs, 1)
            assert isinstance(info, Mapping)

    env.close()
