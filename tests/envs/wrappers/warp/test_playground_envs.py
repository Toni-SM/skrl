import pytest

from collections.abc import Mapping
import gymnasium as gym

import warp as wp

from skrl.envs.loaders.warp import load_playground_env
from skrl.envs.wrappers.warp import wrap_env
from skrl.envs.wrappers.warp.playground_envs import PlaygroundWrapper

from ....utilities import is_running_on_github_actions


@pytest.mark.parametrize("task_name", ["CartpoleBalance", "SO101PickCube"])
def test_env(capsys: pytest.CaptureFixture, task_name: str):
    num_envs = 10
    num_observations = 5 if task_name == "CartpoleBalance" else 24
    num_states = 5 if task_name == "CartpoleBalance" else 24
    num_actions = 1 if task_name == "CartpoleBalance" else 6
    episode_length = 3  # control steps

    # check wrapper definition
    with pytest.raises(AttributeError):
        assert isinstance(wrap_env(None, "playground"), PlaygroundWrapper)

    # load wrap the environment
    try:
        from mujoco_playground.rl import env_registry
    except ImportError as e:
        if is_running_on_github_actions():
            raise e
        else:
            pytest.skip(f"Unable to import MuJoCo Playground environment: {e}")

    # load and wrap the environment
    control_timestep = env_registry.get(task_name).env_cfg_fn().sim.control_timestep
    original_env = load_playground_env(
        task_name=task_name,
        num_envs=num_envs,
        cfg_overrides={"episode_length_secs": (episode_length - 0.5) * control_timestep},
    )
    env = wrap_env(original_env, "auto")
    assert isinstance(env, PlaygroundWrapper)
    env = wrap_env(original_env, "playground")
    assert isinstance(env, PlaygroundWrapper)
    action = wp.zeros((num_envs, num_actions), dtype=wp.float32, device=env.device)

    # check properties
    assert isinstance(env.state_space, gym.Space) and env.state_space.shape == (num_states,)
    assert isinstance(env.observation_space, gym.Space) and env.observation_space.shape == (num_observations,)
    assert isinstance(env.action_space, gym.Space) and env.action_space.shape == (num_actions,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, wp.Device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        state = env.state()
        observation, info = env.reset()  # edge case: parallel environments are autoreset
        state = env.state()
        assert isinstance(observation, wp.array) and observation.shape == (num_envs, num_observations)
        assert isinstance(info, Mapping)
        assert isinstance(state, wp.array) and state.shape == (num_envs, num_states)
        for _ in range(episode_length):
            observation, reward, terminated, truncated, info = env.step(action)
            state = env.state()
            env.render()
            assert isinstance(observation, wp.array) and observation.shape == (num_envs, num_observations)
            assert isinstance(reward, wp.array) and reward.shape == (num_envs, 1)
            assert isinstance(terminated, wp.array) and terminated.shape == (num_envs, 1)
            assert isinstance(truncated, wp.array) and truncated.shape == (num_envs, 1)
            assert isinstance(info, Mapping)
            assert isinstance(state, wp.array) and state.shape == (num_envs, num_states)
            assert observation.dtype == wp.float32 and reward.dtype == wp.float32
            assert terminated.dtype == wp.int8 and truncated.dtype == wp.int8
            assert observation.device == env.device
        # episodes are truncated (time limit) at the last step
        assert bool(wp.to_torch(truncated).all())

    env.close()
