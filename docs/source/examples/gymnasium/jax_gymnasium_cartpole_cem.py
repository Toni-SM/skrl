import gymnasium as gym

import flax.linen as nn
import jax
import jax.numpy as jnp

# import the skrl components to build the RL system
from skrl import config
from skrl.agents.jax.cem import CEM, CEM_DEFAULT_CONFIG
from skrl.envs.wrappers.jax import wrap_env
from skrl.memories.jax import RandomMemory
from skrl.models.jax import CategoricalMixin, Model
from skrl.trainers.jax import SequentialTrainer
from skrl.utils import set_seed


config.jax.backend = "numpy"  # or "jax"


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define model (categorical model) using mixin
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device=None, unnormalized_log_prob=True, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

    @nn.compact
    def __call__(self, inputs, role):
        x = nn.relu(nn.Dense(64)(inputs["states"]))
        x = nn.relu(nn.Dense(64)(x))
        x = nn.Dense(self.num_actions)(x)
        return x, {}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("CartPole-v1")
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("CartPole-v")][0]
    print("CartPole-v0 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's model (function approximator).
# CEM requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/cem.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device)

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal", stddev=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/cem.html#configuration-and-hyperparameters
cfg = CEM_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1000
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/jax/CartPole"

agent = CEM(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
