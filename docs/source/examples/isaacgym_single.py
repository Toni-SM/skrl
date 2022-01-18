import isaacgym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import GaussianModel, DeterministicModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3


# Define the models (stochastic and deterministic models) for the agent using the helper class 
# and programming with two approaches (layer by layer and the Sequential class).
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy(GaussianModel):
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

class Value(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 1))

    def compute(self, states, taken_actions):
        return self.net(states)


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


# Instanciate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# Instanciate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#models-networks
networks_ppo = {"policy": Policy(env.observation_space, env.action_space, device, clip_actions=True),
                "value": Value(env.observation_space, env.action_space, device)}

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for network in networks_ppo.values():
    network.init_parameters(method_name="normal_", mean=0.0, std=0.1)   


# Configure and instanciate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["learning_starts"] = 0
cfg_ppo["random_timesteps"] = 0
cfg_ppo["rollouts"] = 16
cfg_ppo["learning_epochs"] = 8
cfg_ppo["grad_norm_clip"] = 2.0
cfg_ppo["value_loss_scale"] = 2.0
# logging to TensorBoard and write checkpoints each 16 and 160 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 16
cfg_ppo["experiment"]["checkpoint_interval"] = 160

agent = PPO(networks=networks_ppo,
            memory=memory, 
            cfg=cfg_ppo, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)


# Configure and instanciate the RL trainer
cfg_trainer = {"timesteps": 8000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.start()
