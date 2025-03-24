import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

import torch

from skrl.agents.torch.amp import AMP as Agent
from skrl.agents.torch.amp import AMP_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model
from skrl.utils.spaces.torch import sample_space

from ..utils import BaseEnv, get_test_mixed_precision


class Env(BaseEnv):
    def __init__(self, observation_space, action_space, num_envs, device, amp_observation_space):
        super().__init__(observation_space, action_space, num_envs, device)
        self.amp_observation_space = amp_observation_space

    def _sample_observation(self):
        return sample_space(self.observation_space, batch_size=self.num_envs, backend="numpy")

    def step(self, actions):
        observations, rewards, terminated, truncated, info = super().step(actions)
        info["terminate"] = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        info["amp_obs"] = sample_space(
            self.amp_observation_space, batch_size=self.num_envs, backend="native", device=self.device
        )
        return observations, rewards, terminated, truncated, info

    def fetch_amp_obs_demo(self, num_samples):
        return sample_space(self.amp_observation_space, batch_size=num_samples, backend="native", device=self.device)

    def reset_done(self):
        return (
            {
                "obs": sample_space(
                    self.observation_space, batch_size=self.num_envs, backend="native", device=self.device
                )
            },
        )


def _check_agent_config(config, default_config):
    for k in config.keys():
        assert k in default_config
    for k in default_config.keys():
        assert k in config


@hypothesis.given(
    num_envs=st.integers(min_value=1, max_value=5),
    rollouts=st.integers(min_value=1, max_value=5),
    learning_epochs=st.integers(min_value=1, max_value=5),
    mini_batches=st.integers(min_value=1, max_value=5),
    discount_factor=st.floats(min_value=0, max_value=1),
    lambda_=st.floats(min_value=0, max_value=1),
    learning_rate=st.floats(min_value=1.0e-10, max_value=1),
    learning_rate_scheduler=st.one_of(st.none(), st.just(KLAdaptiveLR), st.just(torch.optim.lr_scheduler.ConstantLR)),
    learning_rate_scheduler_kwargs_value=st.floats(min_value=0.1, max_value=1),
    state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    value_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    amp_state_preprocessor=st.one_of(st.none(), st.just(RunningStandardScaler)),
    random_timesteps=st.just(0),
    learning_starts=st.just(0),
    grad_norm_clip=st.floats(min_value=0, max_value=1),
    ratio_clip=st.floats(min_value=0, max_value=1),
    value_clip=st.floats(min_value=0, max_value=1),
    clip_predicted_values=st.booleans(),
    entropy_loss_scale=st.floats(min_value=0, max_value=1),
    value_loss_scale=st.floats(min_value=0, max_value=1),
    discriminator_loss_scale=st.floats(min_value=0, max_value=1),
    amp_batch_size=st.integers(min_value=1, max_value=5),
    task_reward_weight=st.floats(min_value=0, max_value=1),
    style_reward_weight=st.floats(min_value=0, max_value=1),
    discriminator_batch_size=st.integers(min_value=0, max_value=5),
    discriminator_reward_scale=st.floats(min_value=0, max_value=1),
    discriminator_logit_regularization_scale=st.floats(min_value=0, max_value=1),
    discriminator_gradient_penalty_scale=st.floats(min_value=0, max_value=1),
    discriminator_weight_decay_scale=st.floats(min_value=0, max_value=1),
    rewards_shaper=st.one_of(st.none(), st.just(lambda rewards, *args, **kwargs: 0.5 * rewards)),
    time_limit_bootstrap=st.booleans(),
    mixed_precision=st.booleans(),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("separate", [True])
@pytest.mark.parametrize("policy_structure", ["GaussianMixin"])
def test_agent(
    capsys,
    device,
    num_envs,
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
    state_preprocessor,
    value_preprocessor,
    amp_state_preprocessor,
    random_timesteps,
    learning_starts,
    grad_norm_clip,
    ratio_clip,
    value_clip,
    clip_predicted_values,
    entropy_loss_scale,
    value_loss_scale,
    discriminator_loss_scale,
    amp_batch_size,
    task_reward_weight,
    style_reward_weight,
    discriminator_batch_size,
    discriminator_reward_scale,
    discriminator_logit_regularization_scale,
    discriminator_gradient_penalty_scale,
    discriminator_weight_decay_scale,
    rewards_shaper,
    time_limit_bootstrap,
    mixed_precision,
):
    # spaces
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(5,))
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))
    amp_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(10,))

    # env
    env = wrap_env(Env(observation_space, action_space, num_envs, device, amp_observation_space), wrapper="gymnasium")

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
    if separate:
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            network=network,
            output="ACTIONS",
        )
        models["value"] = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            network=network,
            output="ONE",
        )
        models["discriminator"] = deterministic_model(
            observation_space=env.amp_observation_space,
            action_space=env.action_space,
            device=env.device,
            network=network,
            output="ONE",
        )
    else:
        raise NotADirectoryError

    # memory
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.device)

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
        "state_preprocessor": state_preprocessor,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
        "value_preprocessor": value_preprocessor,
        "value_preprocessor_kwargs": {"size": 1, "device": env.device},
        "amp_state_preprocessor": amp_state_preprocessor,
        "amp_state_preprocessor_kwargs": {"size": env.amp_observation_space, "device": env.device},
        "random_timesteps": random_timesteps,
        "learning_starts": learning_starts,
        "grad_norm_clip": grad_norm_clip,
        "ratio_clip": ratio_clip,
        "value_clip": value_clip,
        "clip_predicted_values": clip_predicted_values,
        "entropy_loss_scale": entropy_loss_scale,
        "value_loss_scale": value_loss_scale,
        "discriminator_loss_scale": discriminator_loss_scale,
        "amp_batch_size": amp_batch_size,
        "task_reward_weight": task_reward_weight,
        "style_reward_weight": style_reward_weight,
        "discriminator_batch_size": discriminator_batch_size,
        "discriminator_reward_scale": discriminator_reward_scale,
        "discriminator_logit_regularization_scale": discriminator_logit_regularization_scale,
        "discriminator_gradient_penalty_scale": discriminator_gradient_penalty_scale,
        "discriminator_weight_decay_scale": discriminator_weight_decay_scale,
        "rewards_shaper": rewards_shaper,
        "time_limit_bootstrap": time_limit_bootstrap,
        "mixed_precision": get_test_mixed_precision(mixed_precision),
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
        "kl_threshold" if learning_rate_scheduler is KLAdaptiveLR else "factor"
    ] = learning_rate_scheduler_kwargs_value
    _check_agent_config(cfg, DEFAULT_CONFIG)
    _check_agent_config(cfg["experiment"], DEFAULT_CONFIG["experiment"])
    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        amp_observation_space=env.amp_observation_space,
        motion_dataset=RandomMemory(memory_size=50, device=device),
        reply_buffer=RandomMemory(memory_size=100, device=device),
        collect_reference_motions=lambda num_samples: env.fetch_amp_obs_demo(num_samples),
        collect_observation=lambda: env.reset_done()[0]["obs"],
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
