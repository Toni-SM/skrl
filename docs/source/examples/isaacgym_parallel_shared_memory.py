import isaacgym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import GaussianModel, DeterministicModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3


# Define the models (stochastic and deterministic models) for the agents using helper classes 
# and programming with two approaches (layer by layer and torch.nn.Sequential class).
# - StochasticActor: takes as input the environment's observation/state and returns an action
# - DeterministicActor: takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class StochasticActor(GaussianModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        super().__init__(observation_space, action_space, device, clip_actions,
                         clip_log_std, min_log_std, max_log_std)

        self.linear_layer_1 = nn.Linear(self.num_observations, 32)
        self.linear_layer_2 = nn.Linear(32, 32)
        self.mean_action_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        x = F.elu(self.linear_layer_1(states))
        x = F.elu(self.linear_layer_2(x))
        return torch.tanh(self.mean_action_layer(x)), self.log_std_parameter

class DeterministicActor(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 32)
        self.linear_layer_2 = nn.Linear(32, 32)
        self.action_layer = nn.Linear(32, self.num_actions)

    def compute(self, states, taken_actions):
        x = F.elu(self.linear_layer_1(states))
        x = F.elu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x))

class Critic(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 1))

    def compute(self, states, taken_actions):
        return self.net(torch.cat([states, taken_actions], dim=1))


# Load and wrap the Isaac Gym environment.
# The following lines are intended to support both versions (preview 2 and 3). 
# It tries to load from preview 3, but if it fails, it will try to load from preview 2
try:
    env = load_isaacgym_env_preview3(task_name="Cartpole")
except Exception as e:
    print("Isaac Gym (preview 3) failed: {}\nTrying preview 2...".format(e))
    env = load_isaacgym_env_preview2("Cartpole")
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory (without replacement) as shared experience replay memory
memory = RandomMemory(memory_size=8000, num_envs=env.num_envs, device=device, replacement=True)


# Instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#models-networks
models_ddpg = {"policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
               "target_policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
               "critic": Critic(env.observation_space, env.action_space, device),
               "target_critic": Critic(env.observation_space, env.action_space, device)}
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#models-networks
models_td3 = {"policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
              "target_policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
              "critic_1": Critic(env.observation_space, env.action_space, device),
              "critic_2": Critic(env.observation_space, env.action_space, device),
              "target_critic_1": Critic(env.observation_space, env.action_space, device),
              "target_critic_2": Critic(env.observation_space, env.action_space, device)}
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#models-networks
models_sac = {"policy": StochasticActor(env.observation_space, env.action_space, device, clip_actions=True),
              "critic_1": Critic(env.observation_space, env.action_space, device),
              "critic_2": Critic(env.observation_space, env.action_space, device),
              "target_critic_1": Critic(env.observation_space, env.action_space, device),
              "target_critic_2": Critic(env.observation_space, env.action_space, device)}

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ddpg.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
for model in models_td3.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
cfg_ddpg["gradient_steps"] = 1
cfg_ddpg["batch_size"] = 512
cfg_ddpg["random_timesteps"] = 0
cfg_ddpg["learning_starts"] = 0
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_ddpg["experiment"]["write_interval"] = 25
cfg_ddpg["experiment"]["checkpoint_interval"] = 1000
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#configuration-and-hyperparameters
cfg_td3 = TD3_DEFAULT_CONFIG.copy()
cfg_td3["exploration"]["noise"] = GaussianNoise(0, 0.2, device=device)
cfg_td3["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
cfg_td3["smooth_regularization_clip"] = 0.1
cfg_td3["gradient_steps"] = 1
cfg_td3["batch_size"] = 512
cfg_td3["random_timesteps"] = 0
cfg_td3["learning_starts"] = 0
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_td3["experiment"]["write_interval"] = 25
cfg_td3["experiment"]["checkpoint_interval"] = 1000
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1
cfg_sac["batch_size"] = 512
cfg_sac["random_timesteps"] = 0
cfg_sac["learning_starts"] = 0
cfg_sac["learn_entropy"] = True
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 25
cfg_sac["experiment"]["checkpoint_interval"] = 1000

agent_ddpg = DDPG(models=models_ddpg, 
                  memory=memory, 
                  cfg=cfg_ddpg, 
                  observation_space=env.observation_space, 
                  action_space=env.action_space,
                  device=device)

agent_td3 = TD3(models=models_td3, 
                memory=memory, 
                cfg=cfg_td3, 
                observation_space=env.observation_space, 
                action_space=env.action_space,
                device=device)

agent_sac = SAC(models=models_sac, 
                memory=memory, 
                cfg=cfg_sac, 
                observation_space=env.observation_space, 
                action_space=env.action_space,
                device=device)


# Configure and instantiate the RL trainer
cfg = {"timesteps": 8000, "headless": True}
trainer = SequentialTrainer(cfg=cfg, 
                            env=env, 
                            agents=[agent_ddpg, agent_td3, agent_sac],
                            agents_scope=[])

# start training
trainer.train()
