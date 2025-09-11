import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

from skrl.agents.torch.sarsa import SARSA as Agent
from skrl.agents.torch.sarsa import SARSA_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import tabular_model

from ...utilities import SingleAgentEnv, check_config_keys, is_device_available


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    # model config
    epsilon=st.floats(min_value=0, max_value=1),
    # agent config
    discount_factor=st.floats(min_value=0, max_value=1),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    random_timesteps=st.just(0),
    learning_starts=st.just(0),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("variant", ["epsilon-greedy"])
def test_agent(
    capsys,
    device,
    num_envs,
    # model config
    variant,
    epsilon,
    # agent config
    discount_factor,
    learning_rate,
    random_timesteps,
    learning_starts,
    rewards_shaper,
):
    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    # spaces
    observation_space = gymnasium.spaces.Discrete(3)
    state_space = gymnasium.spaces.Discrete(4)
    action_space = gymnasium.spaces.Discrete(5)

    # env
    env = SingleAgentEnv(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        num_envs=num_envs,
        device=device,
        ml_framework="torch",
    )

    # models
    models = {}
    models["policy"] = tabular_model(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        variant="epsilon-greedy",
        variant_kwargs={"epsilon": epsilon},
    )

    # memory
    memory = RandomMemory(memory_size=50, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "discount_factor": discount_factor,
        "learning_rate": learning_rate,
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "rewards_shaper": rewards_shaper,
        "experiment": {
            "directory": "",
            "experiment_name": "",
            "write_interval": 0,
            "checkpoint_interval": 0,
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        },
    }
    check_config_keys(cfg, DEFAULT_CONFIG)
    check_config_keys(cfg["experiment"], DEFAULT_CONFIG["experiment"])
    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )

    # trainer
    cfg_trainer = {
        "timesteps": 50,
        "headless": True,
        "disable_progressbar": True,
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.train()
