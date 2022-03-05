import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the models (deterministic models) for the DDPG agent using a helper class
# and programming with two approaches (layer by layer and torch.nn.Sequential class).
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy 
class DeterministicActor(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, states, taken_actions):
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        return 2 * torch.tanh(self.action_layer(x))  # Pendulum-v1 action_space is -2 to 2

class DeterministicCritic(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, 1))

    def compute(self, states, taken_actions):
        return self.net(torch.cat([states, taken_actions], dim=1))


# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version
try:
    env = gym.make("Pendulum-v1")
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=40000, num_envs=env.num_envs, device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#models-networks
models_ddpg = {"policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
               "target_policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
               "critic": DeterministicCritic(env.observation_space, env.action_space, device),
               "target_critic": DeterministicCritic(env.observation_space, env.action_space, device)}

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
# logging to TensorBoard and write checkpoints each 1000 and 4000 timesteps respectively
cfg_ddpg["experiment"]["write_interval"] = 1000
cfg_ddpg["experiment"]["checkpoint_interval"] = 4000

agent_ddpg = DDPG(models=models_ddpg, 
                  memory=memory, 
                  cfg=cfg_ddpg, 
                  observation_space=env.observation_space, 
                  action_space=env.action_space,
                  device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 40000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ddpg)

# start training
trainer.train()
