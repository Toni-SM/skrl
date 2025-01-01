import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

import optax

from skrl.agents.jax.td3 import TD3 as Agent
from skrl.agents.jax.td3 import TD3_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.envs.wrappers.jax import wrap_env
from skrl.memories.jax import RandomMemory
from skrl.resources.noises.jax import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveLR
from skrl.trainers.jax import SequentialTrainer
from skrl.utils.model_instantiators.jax import deterministic_model
from skrl.utils.spaces.jax import sample_space

from ..utils import BaseEnv


class Env(BaseEnv):
    def _sample_observation(self):
        return sample_space(self.observation_space, self.num_envs, backend="numpy")


def _check_agent_config(config, default_config):
    for k in config.keys():
        assert k in default_config
        if k == "experiment":
            _check_agent_config(config["experiment"], default_config["experiment"])
    for k in default_config.keys():
        assert k in config
        if k == "experiment":
            _check_agent_config(config["experiment"], default_config["experiment"])


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    gradient_steps=st.integers(min_value=1, max_value=2),
    batch_size=st.integers(min_value=1, max_value=5),
    discount_factor=st.floats(min_value=0, max_value=1),
    polyak=st.floats(min_value=0, max_value=1),
    actor_learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    critic_learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(optax.schedules.constant_schedule)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.integers(min_value=0, max_value=5),
    learning_starts=st.integers(min_value=0, max_value=5),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    exploration=st.one_of(st.none(), st.just(OrnsteinUhlenbeckNoise), st.just(GaussianNoise)),
    exploration_initial_scale=st.floats(min_value=0, max_value=1),
    exploration_final_scale=st.floats(min_value=0, max_value=1),
    exploration_timesteps=st.one_of(st.none(), st.integers(min_value=1, max_value=50)),
    policy_delay=st.integers(min_value=1, max_value=3),
    smooth_regularization_noise=st.one_of(st.none(), st.just(OrnsteinUhlenbeckNoise), st.just(GaussianNoise)),
    smooth_regularization_clip=st.floats(min_value=0, max_value=1),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
)
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_agent(
    capsys,
    device,
    num_envs,
    # agent config
    gradient_steps,
    batch_size,
    discount_factor,
    polyak,
    actor_learning_rate,
    critic_learning_rate,
    learning_rate_scheduler,
    learning_rate_scheduler_kwargs_value,
    state_preprocessor,
    random_timesteps,
    learning_starts,
    grad_norm_clip,
    exploration,
    exploration_initial_scale,
    exploration_final_scale,
    exploration_timesteps,
    policy_delay,
    smooth_regularization_noise,
    smooth_regularization_clip,
    rewards_shaper,
):
    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))

    # env
    env = wrap_env(Env(observation_space, action_space, num_envs, device), wrapper="gymnasium")

    # models
    network = [
        {
            "name": "net",
            "input": "STATES",
            "layers": [64, 64],
            "activations": "elu",
        }
    ]
    models = {}
    models["policy"] = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        network=network,
        output="ACTIONS",
    )
    models["target_policy"] = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        network=network,
        output="ACTIONS",
    )
    models["critic_1"] = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        network=network,
        output="ONE",
    )
    models["target_critic_1"] = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        network=network,
        output="ONE",
    )
    models["critic_2"] = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        network=network,
        output="ONE",
    )
    models["target_critic_2"] = deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        network=network,
        output="ONE",
    )
    # instantiate models' state dict
    for role, model in models.items():
        model.init_state_dict(role)

    # memory
    memory = RandomMemory(memory_size=50, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = {
        "gradient_steps": gradient_steps,
        "batch_size": batch_size,
        "discount_factor": discount_factor,
        "polyak": polyak,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "learning_rate_scheduler": learning_rate_scheduler,
        "learning_rate_scheduler_kwargs": {},
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "grad_norm_clip": grad_norm_clip,
        "exploration": {
            "initial_scale": exploration_initial_scale,
            "final_scale": exploration_final_scale,
            "timesteps": exploration_timesteps,
        },
        "policy_delay": policy_delay,
        "smooth_regularization_clip": smooth_regularization_clip,
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
    # noise
    # - exploration
    if exploration is None:
        cfg["exploration"]["noise"] = None
    elif exploration is OrnsteinUhlenbeckNoise:
        cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.2, base_scale=1.0, device=env.device)
    elif exploration is GaussianNoise:
        cfg["exploration"]["noise"] = GaussianNoise(mean=0, std=0.1, device=env.device)
    # - regularization
    if smooth_regularization_noise is None:
        cfg["smooth_regularization_noise"] = None
    elif smooth_regularization_noise is OrnsteinUhlenbeckNoise:
        cfg["smooth_regularization_noise"] = OrnsteinUhlenbeckNoise(
            theta=0.1, sigma=0.2, base_scale=1.0, device=env.device
        )
    elif smooth_regularization_noise is GaussianNoise:
        cfg["smooth_regularization_noise"] = GaussianNoise(mean=0, std=0.1, device=env.device)
    _check_agent_config(cfg, DEFAULT_CONFIG)
    _check_agent_config(cfg["experiment"], DEFAULT_CONFIG["experiment"])
    _check_agent_config(cfg["exploration"], DEFAULT_CONFIG["exploration"])
    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
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
