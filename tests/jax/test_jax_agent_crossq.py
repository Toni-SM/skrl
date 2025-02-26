import sys
from typing import Callable, Optional, Sequence

import gymnasium

import optax

import flax.linen as nn
import jax.numpy as jnp
import gym_envs

from skrl.agents.jax.crossq import CrossQ as Agent
from skrl.agents.jax.crossq import CROSSQ_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.envs.wrappers.jax import wrap_env
from skrl.memories.jax import RandomMemory
from skrl.models.jax.base import Model, BatchNormModel
from skrl.models.jax.mutabledeterministic import MutableDeterministicMixin
from skrl.models.jax.mutablegaussian import MutableGaussianMixin
from skrl.resources.layers.jax.batch_renorm import BatchRenorm
from skrl.trainers.jax.sequential import SequentialTrainer


class Critic(MutableDeterministicMixin, BatchNormModel):
    net_arch: Sequence[int] = None
    use_batch_norm: bool = True

    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000


    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        device=None,
        clip_actions=False,
        use_batch_norm=False,
        batch_norm_momentum=0.99,
        renorm_warmup_steps: int = 100_000,
        **kwargs,
    ):
        self.net_arch = net_arch
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.renorm_warmup_steps = renorm_warmup_steps
        
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        MutableDeterministicMixin.__init__(self, clip_actions)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role='', train=False):
        x = jnp.concatenate([inputs["states"], inputs["taken_actions"]], axis=-1)
        if self.use_batch_norm:
            x = BatchRenorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                warmup_steps=self.renorm_warmup_steps,
            )(x)
        else:
            x_dummy = BatchRenorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                warmup_steps=self.renorm_warmup_steps,
            )(x)
        for n_neurons in self.net_arch:
            x = nn.Dense(n_neurons)(x)
            x = nn.relu(x)
            if self.use_batch_norm:
                x = BatchRenorm(
                    use_running_average=not train,
                    momentum=self.batch_norm_momentum,
                    warmup_steps=self.renorm_warmup_steps,
                )(x)
            else:
                x_dummy = BatchRenorm(
                    use_running_average=not train,
                    momentum=self.batch_norm_momentum,
                    warmup_steps=self.renorm_warmup_steps,
                )(x)
        x = nn.Dense(1)(x)
        return x, {}


class Actor(MutableGaussianMixin, BatchNormModel):

    net_arch: Sequence[int] = None
    batch_norm_momentum: float = 0.99
    use_batch_norm: bool = False

    renorm_warmup_steps: int = 100_000
    
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        device=None,
        clip_actions=False,
        clip_log_std=False,
        use_batch_norm=False,
        batch_norm_momentum=0.99,
        log_std_min: float = -20,
        log_std_max: float = 2,
        renorm_warmup_steps: int = 100_000,
        **kwargs,
    ):
        self.net_arch = net_arch
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.renorm_warmup_steps = renorm_warmup_steps
                
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        MutableGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std=log_std_min, max_log_std=log_std_max)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, train: bool = False, role=''):
        x = jnp.concatenate([inputs["states"]], axis=-1)
        if self.use_batch_norm:
            x = BatchRenorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                warmup_steps=self.renorm_warmup_steps,
            )(x)
        else:
            x_dummy = BatchRenorm(
                use_running_average=not train
            )(x)
        for n_neurons in self.net_arch:
            x = nn.Dense(n_neurons)(x)
            x = nn.relu(x)
            if self.use_batch_norm:
                x = BatchRenorm(
                    use_running_average=not train,
                    momentum=self.batch_norm_momentum,
                    warmup_steps=self.renorm_warmup_steps,
                )(x)
            else:
                x_dummy = BatchRenorm(
                    use_running_average=not train,
                )(x)
        mean = nn.Dense(self.num_actions)(x)
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))
        return nn.tanh(mean), log_std, {}

def _check_agent_config(config, default_config):
    for k in config.keys():
        assert k in default_config
        if k == "experiment":
            _check_agent_config(config["experiment"], default_config["experiment"])
    for k in default_config.keys():
        assert k in config
        if k == "experiment":
            _check_agent_config(config["experiment"], default_config["experiment"])


def test_agent():
    # env
    env = gymnasium.make("Joint_PandaReach-v0")
    env = wrap_env(env, wrapper="gymnasium")

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
    models["policy"] = Actor(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[256,256],
        device=env.device,
        use_batch_norm=True,
    )
    models["critic_1"] = Critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[1024, 1024],
        device=env.device,
        use_batch_norm=True,
    )
    models["critic_2"] = Critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[1024, 1024],
        device=env.device,
        use_batch_norm=True,
    )
    # instantiate models' state dict
    for role, model in models.items():
        model.init_state_dict(role)

    # memory
    memory = RandomMemory(memory_size=1_000_000, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = DEFAULT_CONFIG
    
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
        "disable_progressbar": False,
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.train()


test_agent()