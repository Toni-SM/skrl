import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

import optax

from skrl.memories.jax import RandomMemory
from skrl.multi_agents.jax.mappo import MAPPO as MultiAgent
from skrl.multi_agents.jax.mappo import MAPPO_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveLR
from skrl.trainers.jax import SequentialTrainer
from skrl.utils.model_instantiators.jax import categorical_model, deterministic_model, gaussian_model

from ...utilities import MultiAgentEnv, check_config_keys


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    max_num_agents=st.integers(min_value=2, max_value=5),
    # agent config
    rollouts=st.integers(min_value=1, max_value=5),
    learning_epochs=st.integers(min_value=1, max_value=5),
    mini_batches=st.integers(min_value=1, max_value=5),
    discount_factor=st.floats(min_value=0, max_value=1),
    lambda_=st.floats(min_value=0, max_value=1),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(optax.schedules.constant_schedule)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    observation_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    value_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.just(0),
    learning_starts=st.just(0),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    ratio_clip=st.floats(min_value=0, max_value=1),
    value_clip=st.floats(min_value=0, max_value=1),
    clip_predicted_values=st.booleans(),
    entropy_loss_scale=st.floats(min_value=0, max_value=1),
    value_loss_scale=st.floats(min_value=0, max_value=1),
    kl_threshold=st.floats(min_value=0, max_value=1),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
    time_limit_bootstrap=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=25,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("separate", [True])
@pytest.mark.parametrize("asymmetric", [True])
@pytest.mark.parametrize("policy_structure", ["GaussianMixin", "CategoricalMixin"])
def test_agent(
    capsys,
    device,
    num_envs,
    max_num_agents,
    asymmetric,
    # model config
    separate,
    policy_structure,
    # agent config
    rollouts,
    learning_epochs,
    mini_batches,
    discount_factor,
    lambda_,
    learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    observation_preprocessor,
    state_preprocessor,
    value_preprocessor,
    random_timesteps,
    learning_starts,
    grad_norm_clip,
    ratio_clip,
    value_clip,
    clip_predicted_values,
    entropy_loss_scale,
    value_loss_scale,
    kl_threshold,
    rewards_shaper,
    time_limit_bootstrap,
):
    # spaces
    observation_spaces = {}
    state_spaces = {}
    action_spaces = {}
    for i in range(max_num_agents):
        uid = f"agent_{i}"
        observation_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(max_num_agents,))
        state_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(6,)) if asymmetric else None  # common
        if policy_structure in ["GaussianMixin"]:
            action_spaces[uid] = gymnasium.spaces.Box(low=-1, high=1, shape=(max_num_agents - 1,))
        elif policy_structure == "CategoricalMixin":
            action_spaces[uid] = gymnasium.spaces.Discrete(max_num_agents - 1)

    # env
    env = MultiAgentEnv(
        observation_spaces=observation_spaces,
        state_spaces=state_spaces,
        action_spaces=action_spaces,
        num_envs=num_envs,
        device=device,
        ml_framework="jax",
    )

    # models
    network = {
        "policy": [
            {
                "name": "net",
                "input": "OBSERVATIONS",
                "layers": [5],
                "activations": "relu",
            }
        ],
        "value": [
            {
                "name": "net",
                "input": "STATES" if asymmetric else "OBSERVATIONS",
                "layers": [5],
                "activations": "relu",
            }
        ],
    }
    models = {}
    for uid in env.possible_agents:
        models[uid] = {}
        if separate:
            if policy_structure == "GaussianMixin":
                models[uid]["policy"] = gaussian_model(
                    observation_space=env.observation_space(uid),
                    state_space=env.state_space(uid),
                    action_space=env.action_space(uid),
                    device=env.device,
                    network=network["policy"],
                    output="ACTIONS",
                )
            elif policy_structure == "CategoricalMixin":
                models[uid]["policy"] = categorical_model(
                    observation_space=env.observation_space(uid),
                    state_space=env.state_space(uid),
                    action_space=env.action_space(uid),
                    device=env.device,
                    network=network["policy"],
                    output="ACTIONS",
                )
            models[uid]["value"] = deterministic_model(
                observation_space=env.observation_space(uid),
                state_space=env.state_space(uid),
                action_space=env.action_space(uid),
                device=env.device,
                network=network["value"],
                output="ONE",
            )
        else:
            raise ValueError("Separate is not supported for MAPPO, since it uses a centralized value function")
        # instantiate models' state dict
        for role, model in models[uid].items():
            model.init_state_dict(role=role)

    # memory
    memories = {}
    for uid in env.possible_agents:
        memories[uid] = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "discount_factor": discount_factor,
        "lambda": lambda_,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "observation_preprocessor": observation_preprocessor,
        "observation_preprocessor_kwargs": {"size": env.observation_space("agent_0"), "device": env.device},
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.state_space("agent_0"), "device": env.device},
        "value_preprocessor": value_preprocessor,
        "value_preprocessor_kwargs": {"size": 1, "device": env.device},
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "grad_norm_clip": grad_norm_clip,
        "ratio_clip": ratio_clip,
        "value_clip": value_clip,
        "clip_predicted_values": clip_predicted_values,
        "entropy_loss_scale": entropy_loss_scale,
        "value_loss_scale": value_loss_scale,
        "kl_threshold": kl_threshold,
        "rewards_shaper": rewards_shaper,
        "time_limit_bootstrap": time_limit_bootstrap,
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
    check_config_keys(cfg, DEFAULT_CONFIG)
    check_config_keys(cfg["experiment"], DEFAULT_CONFIG["experiment"])
    agent = MultiAgent(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=cfg,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
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
