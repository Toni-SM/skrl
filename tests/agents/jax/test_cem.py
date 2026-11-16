import hypothesis
import hypothesis.strategies as st
import pytest

import dataclasses
import math
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np
import optax

from skrl.agents.jax.cem import CEM as Agent
from skrl.agents.jax.cem import CEM_CFG as AgentCfg
from skrl.memories.jax import RandomMemory
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveLR
from skrl.trainers.jax import SequentialTrainer
from skrl.utils.model_instantiators.jax import categorical_model, multicategorical_model

from ...utilities import SingleAgentEnv, check_config_keys, is_device_available


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    # agent config
    rollouts=st.integers(min_value=1, max_value=5),
    percentile=st.floats(min_value=0, max_value=1),
    discount_factor=st.floats(min_value=0, max_value=1),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(optax.schedules.constant_schedule)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    observation_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.just(0),
    learning_starts=st.just(0),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=15,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("asymmetric", [True, False])
def test_agent(
    capsys,
    device,
    num_envs,
    asymmetric,
    # agent config
    rollouts,
    percentile,
    discount_factor,
    learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    observation_preprocessor,
    state_preprocessor,
    random_timesteps,
    learning_starts,
    rewards_shaper,
):
    # check device availability
    if not is_device_available(device, backend="jax"):
        pytest.skip(f"Device {device} not available")

    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
    state_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,)) if asymmetric else None
    action_space = gymnasium.spaces.Discrete(3)

    # env
    env = SingleAgentEnv(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        num_envs=num_envs,
        device=device,
        ml_framework="jax",
    )

    # models
    network = {
        "policy": [
            {
                "name": "net",
                "input": "STATES" if asymmetric else "OBSERVATIONS",
                "layers": [5],
                "activations": "relu",
            }
        ],
    }
    models = {}
    models["policy"] = categorical_model(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        network=network["policy"],
        output="ACTIONS",
    )
    # instantiate models' state dict
    for role, model in models.items():
        model.init_state_dict(role=role)

    # memory
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "rollouts": rollouts,
        "percentile": percentile,
        "discount_factor": discount_factor,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "observation_preprocessor": observation_preprocessor,
        "observation_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.state_space, "device": env.device},
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
    cfg["learning_rate_scheduler_kwargs"][
        "kl_threshold" if learning_rate_scheduler is KLAdaptiveLR else "value"
    ] = learning_rate_scheduler_kwargs_value
    check_config_keys(cfg, dataclasses.asdict(AgentCfg()))
    check_config_keys(cfg["experiment"], dataclasses.asdict(AgentCfg().experiment))
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
        "timesteps": int(5 * rollouts),
        "headless": True,
        "disable_progressbar": True,
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.train()


@pytest.mark.parametrize("multicategorical", [False, True])
@pytest.mark.parametrize("unnormalized_log_prob", [False, True])
def test_elite_action_likelihood(multicategorical: bool, unnormalized_log_prob: bool) -> None:
    """Train on elite actions using their categorical or joint categorical likelihood."""
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
    action_space = gymnasium.spaces.MultiDiscrete([2, 3]) if multicategorical else gymnasium.spaces.Discrete(2)
    factory = multicategorical_model if multicategorical else categorical_model
    policy = factory(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        unnormalized_log_prob=unnormalized_log_prob,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": []}],
        output="ACTIONS" if unnormalized_log_prob else "jax.nn.softmax(ACTIONS, axis=-1)",
    )
    observations = jnp.zeros((1, 1))
    policy.init_state_dict({"observations": observations})
    weights = [2.0, 1.0, 3.0, 2.0, 1.0] if multicategorical else [2.0, 1.0]
    initial_bias = jnp.log(jnp.array(weights))
    params = jax.tree_util.tree_map_with_path(
        lambda path, value: initial_bias if path[-1].key == "bias" else jnp.zeros_like(value),
        policy.state_dict.params,
    )
    policy.state_dict = policy.state_dict.replace(params=params)
    before = jax.tree_util.tree_leaves(params)

    agent = Agent(
        models={"policy": policy},
        memory=RandomMemory(memory_size=2, num_envs=1, device="cpu"),
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        cfg={
            "rollouts": 2,
            "discount_factor": 1.0,
            "percentile": 0.0,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
    )
    agent.init()
    agent.enable_training_mode()
    for timestep in range(2):
        agent.pre_interaction(timestep=timestep, timesteps=2)
        agent.record_transition(
            observations=observations,
            states=None,
            actions=jnp.zeros((1, 2 if multicategorical else 1), dtype=jnp.int32),
            rewards=jnp.ones((1, 1)),
            next_observations=observations,
            next_states=None,
            terminated=jnp.array([[timestep == 1]]),
            truncated=jnp.zeros((1, 1), dtype=jnp.bool_),
            infos={},
            timestep=timestep,
            timesteps=2,
        )
        agent.post_interaction(timestep=timestep, timesteps=2)

    expected_loss = math.log(3.0 if multicategorical else 1.5)
    assert agent.tracking_data["Loss / Policy loss"] == pytest.approx([expected_loss], rel=1e-6)
    after = jax.tree_util.tree_leaves(policy.state_dict.params)
    assert all(np.isfinite(value).all() for value in after)
    assert any(not np.array_equal(old, new) for old, new in zip(before, after))
