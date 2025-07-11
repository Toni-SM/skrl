import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

from skrl.trainers.jax import SequentialTrainer, generate_equally_spaced_scopes
from skrl.trainers.jax.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG as DEFAULT_CONFIG

from ...utilities import AgentMock, SingleAgentEnv, check_config_keys


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=10),
    # trainer config
    timesteps=st.integers(min_value=1, max_value=50),
    headless=st.booleans(),
    disable_progressbar=st.booleans(),
    close_environment_at_exit=st.booleans(),
    stochastic_evaluation=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("asymmetric", [True, False])
def test_non_simultaneous_trainer(
    capsys,
    device,
    num_envs,
    asymmetric,
    # trainer config
    timesteps,
    headless,
    disable_progressbar,
    close_environment_at_exit,
    stochastic_evaluation,
):
    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,)) if asymmetric else None
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

    # env
    env = SingleAgentEnv(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        num_envs=num_envs,
        device=device,
        ml_framework="jax",
        probability=0.25,
    )

    # agent
    agent = AgentMock(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        num_envs=num_envs,
        device=device,
        ml_framework="jax",
    )

    # trainer
    cfg = {
        "timesteps": timesteps,
        "headless": headless,
        "disable_progressbar": disable_progressbar,
        "close_environment_at_exit": close_environment_at_exit,
        "environment_info": "episode",
        "stochastic_evaluation": stochastic_evaluation,
    }
    check_config_keys(cfg, DEFAULT_CONFIG)
    trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent)
    # - training
    trainer.train()
    # - evaluation
    trainer.eval()


@hypothesis.given(
    num_envs=st.integers(min_value=2, max_value=10),
    num_simultaneous_agents=st.integers(min_value=2, max_value=10),
    # trainer config
    timesteps=st.integers(min_value=1, max_value=50),
    headless=st.booleans(),
    disable_progressbar=st.booleans(),
    close_environment_at_exit=st.booleans(),
    stochastic_evaluation=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("asymmetric", [True, False])
def test_simultaneous_trainer(
    capsys,
    device,
    num_envs,
    num_simultaneous_agents,
    asymmetric,
    # trainer config
    timesteps,
    headless,
    disable_progressbar,
    close_environment_at_exit,
    stochastic_evaluation,
):
    num_simultaneous_agents = min(num_envs, num_simultaneous_agents)
    scopes = generate_equally_spaced_scopes(
        num_envs=num_envs,
        num_simultaneous_agents=num_simultaneous_agents,
    )

    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,)) if asymmetric else None
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

    # env
    env = SingleAgentEnv(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        num_envs=num_envs,
        device=device,
        ml_framework="jax",
        probability=0.25,
    )

    # agents
    agents = [
        AgentMock(
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            num_envs=scope,
            device=device,
            ml_framework="jax",
        )
        for _, scope in zip(range(num_simultaneous_agents), scopes)
    ]
    assert len(agents) > 1

    # trainer
    cfg = {
        "timesteps": timesteps,
        "headless": headless,
        "disable_progressbar": disable_progressbar,
        "close_environment_at_exit": close_environment_at_exit,
        "environment_info": "episode",
        "stochastic_evaluation": stochastic_evaluation,
    }
    check_config_keys(cfg, DEFAULT_CONFIG)
    trainer = SequentialTrainer(cfg=cfg, env=env, agents=agents, scopes=scopes)
    # - training
    trainer.train()
    # - evaluation
    trainer.eval()
