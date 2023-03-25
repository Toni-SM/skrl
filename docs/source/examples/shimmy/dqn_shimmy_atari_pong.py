import gymnasium as gym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the model (deterministic models) for the DQN agent using mixin
class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# Load and wrap the environment
env = gym.make("ALE/Pong-v5")
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#spaces-and-models
models = {}
models["q_network"] = QNetwork(env.observation_space, env.action_space, device)
models["target_q_network"] = QNetwork(env.observation_space, env.action_space, device)

# # Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#configuration-and-hyperparameters
cfg_agent = DQN_DEFAULT_CONFIG.copy()
cfg_agent["learning_starts"] = 100
cfg_agent["exploration"]["initial_epsilon"] = 1.0
cfg_agent["exploration"]["final_epsilon"] = 0.04
cfg_agent["exploration"]["timesteps"] = 1500
# logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
cfg_agent["experiment"]["write_interval"] = 1000
cfg_agent["experiment"]["checkpoint_interval"] = 5000

agent_dqn = DQN(models=models,
                memory=memory,
                cfg=cfg_agent,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_dqn)

# start training
trainer.train()
