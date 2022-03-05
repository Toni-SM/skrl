import isaacgym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import GaussianModel, DeterministicModel
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3


# Define only the policies for evaluation 
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


# Instantiate the agent's policies.
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#models-networks
models_ddpg = {"policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
               "target_policy": None,
               "critic": None,
               "target_critic": None}
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#models-networks
models_td3 = {"policy": DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True),
              "target_policy": None,
              "critic_1": None,
              "critic_2": None,
              "target_critic_1": None,
              "target_critic_2": None}
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#models-networks
models_sac = {"policy": StochasticActor(env.observation_space, env.action_space, device, clip_actions=True),
              "critic_1": None,
              "critic_2": None,
              "target_critic_1": None,
              "target_critic_2": None}

# load checkpoints
models_ddpg["policy"].load("./runs/22-02-06_19-37-44-874837_DDPG/checkpoints/8000_policy.pt")
models_td3["policy"].load("./runs/22-02-06_19-28-48-436345_TD3/checkpoints/5000_policy.pt")
models_sac["policy"].load("./runs/22-02-06_19-28-48-441161_SAC/checkpoints/3000_policy.pt")


# Configure and instantiate the agents.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["random_timesteps"] = 0
# logging to TensorBoard each 25 timesteps and ignore checkpoints
cfg_ddpg["experiment"]["write_interval"] = 25
cfg_ddpg["experiment"]["checkpoint_interval"] = 0
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#configuration-and-hyperparameters
cfg_td3 = TD3_DEFAULT_CONFIG.copy()
cfg_td3["random_timesteps"] = 0
# logging to TensorBoard each 25 timesteps and ignore checkpoints
cfg_td3["experiment"]["write_interval"] = 25
cfg_td3["experiment"]["checkpoint_interval"] = 0
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["random_timesteps"] = 0
# logging to TensorBoard each 25 timesteps and ignore checkpoints
cfg_sac["experiment"]["write_interval"] = 25
cfg_sac["experiment"]["checkpoint_interval"] = 0

agent_ddpg = DDPG(models=models_ddpg, 
                  memory=None, 
                  cfg=cfg_ddpg, 
                  observation_space=env.observation_space, 
                  action_space=env.action_space,
                  device=device)

agent_td3 = TD3(models=models_td3, 
                memory=None, 
                cfg=cfg_td3, 
                observation_space=env.observation_space, 
                action_space=env.action_space,
                device=device)

agent_sac = SAC(models=models_sac, 
                memory=None, 
                cfg=cfg_sac, 
                observation_space=env.observation_space, 
                action_space=env.action_space,
                device=device)


# Configure and instantiate the RL trainer
cfg = {"timesteps": 8000, "headless": True}
trainer = SequentialTrainer(cfg=cfg, 
                            env=env, 
                            agents=[agent_ddpg, agent_td3, agent_sac],
                            agents_scope=[100, 200, 212])

# evaluate the agents
trainer.eval()
