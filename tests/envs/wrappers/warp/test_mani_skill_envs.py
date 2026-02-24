import pytest

from collections.abc import Mapping
import gymnasium

import warp as wp

from skrl.envs.wrappers.warp import wrap_env
from skrl.envs.wrappers.warp.mani_skill_envs import ManiSkillWrapper

from ....utilities import is_running_on_github_actions


def test_env(capsys: pytest.CaptureFixture):
    num_envs = 10
    action = wp.ones((num_envs, 8))

    # check wrapper definition
    assert isinstance(wrap_env(None, "mani-skill"), ManiSkillWrapper)

    # load wrap the environment
    try:
        import mani_skill.envs
    except ImportError as e:
        if is_running_on_github_actions():
            raise e
        else:
            pytest.skip(f"Unable to import ManiSkill environment: {e}")

    env_kwargs = {
        "obs_mode": "state",
        "render_mode": "human",
        "sim_backend": "physx_cuda",
        "control_mode": "pd_joint_delta_pos",
    }
    original_env = gymnasium.make("PushCube-v1", num_envs=num_envs, **env_kwargs)
    env = wrap_env(original_env, "auto")
    assert isinstance(env, ManiSkillWrapper)
    env = wrap_env(original_env, "mani-skill")
    assert isinstance(env, ManiSkillWrapper)

    # check properties
    assert env.state_space is None
    assert isinstance(env.observation_space, gymnasium.Space) and env.observation_space.shape == (35,)
    assert isinstance(env.action_space, gymnasium.Space) and env.action_space.shape == (8,)
    assert isinstance(env.num_envs, int) and env.num_envs == num_envs
    assert isinstance(env.num_agents, int) and env.num_agents == 1
    assert isinstance(env.device, wp.context.Device)
    # check internal properties
    assert env._env is original_env
    assert env._unwrapped is original_env.unwrapped
    # check methods
    for _ in range(2):
        observation, info = env.reset()
        state = env.state()
        observation, info = env.reset()  # edge case: parallel environments are autoreset
        state = env.state()
        assert isinstance(observation, wp.array) and observation.shape == (num_envs, 35)
        assert isinstance(info, Mapping)
        assert state is None
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action)
            state = env.state()
            if not is_running_on_github_actions():
                env.render()
            assert isinstance(observation, wp.array) and observation.shape == (num_envs, 35)
            assert isinstance(reward, wp.array) and reward.shape == (num_envs, 1)
            assert isinstance(terminated, wp.array) and terminated.shape == (num_envs, 1)
            assert isinstance(truncated, wp.array) and truncated.shape == (num_envs, 1)
            assert isinstance(info, Mapping)
            assert state is None

    env.close()
