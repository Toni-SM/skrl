from dm_control import suite

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the models (deterministic models) for the DDPG agent using mixins
# and programming with two approaches (torch functional and torch.nn.Sequential class).
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x)), {}

class DeterministicCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# Load and wrap the DeepMind environment
env = suite.load(domain_name="cartpole", task_name="swingup")
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=25000, num_envs=env.num_envs, device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_ddpg["critic"] = DeterministicCritic(env.observation_space, env.action_space, device)
models_ddpg["target_critic"] = DeterministicCritic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ddpg.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg_ddpg["batch_size"] = 100
cfg_ddpg["random_timesteps"] = 100
cfg_ddpg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
cfg_ddpg["experiment"]["write_interval"] = 1000
cfg_ddpg["experiment"]["checkpoint_interval"] = 5000

agent_ddpg = DDPG(models=models_ddpg,
                  memory=memory,
                  cfg=cfg_ddpg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ddpg)

# start training
trainer.train()
