"""
Notes for Isaac Sim 2022.2.1 or earlier (Python 3.7 environment):
  * Python 3.7 is only supported up to jax<=0.3.25.
    See: https://github.com/google/jax/blob/main/CHANGELOG.md#jaxlib-041-dec-13-2022.
  * Builds for jaxlib<=0.3.25 are only available up to NVIDIA CUDA 11 and cuDNN 8.2 versions.
    See: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    and search for `cuda11/jaxlib-0.3.25+cuda11.cudnn82-cp37-cp37m-manylinux2014_x86_64.whl`.
  * The `jax.Device = jax.xla.Device` statement is required by skrl to support jax<0.4.3.
  * Models require overloading the `__hash__` method to avoid "TypeError: Failed to hash Flax Module".
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


jax.Device = jax.xla.Device  # for Isaac Sim 2022.2.1 or earlier

# import the skrl components to build the RL system
from skrl import config
from skrl.agents.jax.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.loaders.jax import load_omniverse_isaacgym_env
from skrl.envs.wrappers.jax import wrap_env
from skrl.memories.jax import RandomMemory
from skrl.models.jax import DeterministicMixin, Model
from skrl.resources.noises.jax import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.trainers.jax import SequentialTrainer
from skrl.utils import set_seed


config.jax.backend = "jax"  # or "numpy"


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    def __hash__(self):  # for Isaac Sim 2022.2.1 or earlier
        return id(self)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.relu(nn.Dense(512)(inputs["states"]))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.Dense(self.num_actions)(x)
        return nn.tanh(x), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    def __hash__(self):  # for Isaac Sim 2022.2.1 or earlier
        return id(self)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = jnp.concatenate([inputs["states"], inputs["taken_actions"]], axis=-1)
        x = nn.relu(nn.Dense(512)(x))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.Dense(1)(x)
        return x, {}


# load and wrap the Omniverse Isaac Gym environment
env = load_omniverse_isaacgym_env(task_name="Ant", num_envs=64)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 80
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/jax/Ant"

agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
