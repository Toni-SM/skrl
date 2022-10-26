import gymnasium as gym

import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, CategoricalMixin
from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the model (categorical model) for the CEM agent using mixin
# - Policy: takes as input the environment's observation/state and returns an action
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 64)
        self.linear_layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, self.num_actions)

    def compute(self, states, taken_actions, role):
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x)


# Load and wrap the Gymnasium environment.
# Note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("CartPole-v1")
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("CartPole-v")][0]
    print("CartPole-v0 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# Instantiate the agent's model (function approximators).
# CEM requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.cem.html#spaces-and-models
models_cem = {}
models_cem["policy"] = Policy(env.observation_space, env.action_space, device)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.cem.html#configuration-and-hyperparameters
cfg_cem = CEM_DEFAULT_CONFIG.copy()
cfg_cem["rollouts"] = 1000
cfg_cem["learning_starts"] = 100
# logging to TensorBoard each 1000 timesteps and ignore checkpoints
cfg_cem["experiment"]["write_interval"] = 1000
cfg_cem["experiment"]["checkpoint_interval"] = 0

agent_cem = CEM(models=models_cem,
                memory=None,
                cfg=cfg_cem,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

# load checkpoint
agent_cem.load("./runs/22-09-07_21-41-05-854385_CEM/checkpoints/best_agent.pt")

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(env=env, agents=[agent_cem], cfg=cfg_trainer)

# evaluate the agent
trainer.eval()
