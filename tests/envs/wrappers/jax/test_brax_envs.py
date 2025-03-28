import pytest
import warnings

from collections.abc import Mapping
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.envs.wrappers.jax import BraxWrapper, wrap_env

from ....utilities import is_running_on_github_actions


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_env(capsys: pytest.CaptureFixture, backend: str):
    config.jax.backend = backend
    Array = jax.Array if backend == "jax" else np.ndarray

    num_envs = 10
    action = jnp.ones((num_envs, 1)) if backend == "jax" else np.ones((num_envs, 1))

    # load wrap the environment
    try:
        import brax.envs
    except ImportError as e:
        if is_running_on_github_actions():
            raise e
        else:
            pytest.skip(f"Unable to import Brax environment: {e}")

    original_env = brax.envs.create("inverted_pendulum", batch_size=num_envs, backend="spring")
    env = wrap_env(original_env, "auto")
    assert isinstance(env, BraxWrapper)
    env = wrap_env(original_env, "brax")
    assert isinstance(env, BraxWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gymnasium.Space) and env.observation_space.shape == (4,)
    assert isinstance(env.action_space, gymnasium.Space) and env.action_space.shape == (1,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, jax.Device)
    # check internal properties
    assert env._env is not original_env  # # brax's VectorGymWrapper interferes with the checking (it has _env)
    assert env._unwrapped is not original_env.unwrapped  # brax's VectorGymWrapper interferes with the checking
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        observation, info = env.reset()  # edge case: parallel environments are autoreset
        assert isinstance(observation, Array) and observation.shape == (num_envs, 4)
        assert isinstance(info, Mapping)
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            try:
                env.render()
            except AttributeError as e:
                warnings.warn(f"Brax exception when rendering: {e}")
            assert isinstance(observation, Array) and observation.shape == (num_envs, 4)
            assert isinstance(reward, Array) and reward.shape == (num_envs, 1)
            assert isinstance(terminated, Array) and terminated.shape == (num_envs, 1)
            assert isinstance(truncated, Array) and truncated.shape == (num_envs, 1)
            assert isinstance(info, Mapping)

    env.close()
