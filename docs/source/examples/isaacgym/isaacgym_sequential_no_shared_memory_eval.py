import isaacgym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview4


# Define only the policies for evaluation
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions, role):
        return self.net(states), self.log_std_parameter

class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, states, taken_actions, role):
        return self.net(states)


# Load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="Cartpole")   # preview 3 and 4 use the same loader
env = wrap_env(env)

device = env.device


# Instantiate the agent's policies.
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#spaces-and-models
models_td3 = {}
models_td3["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#spaces-and-models
models_sac = {}
models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)


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

# load checkpoint (agent)
agent_ddpg.load("./runs/22-09-12_22-30-58-982355_DDPG/checkpoints/agent_8000.pt")
agent_td3.load("./runs/22-09-12_22-30-58-986295_TD3/checkpoints/agent_8000.pt")
agent_sac.load("./runs/22-09-12_22-30-58-987142_SAC/checkpoints/agent_8000.pt")


# Configure and instantiate the RL trainer
cfg = {"timesteps": 8000, "headless": True}
trainer = SequentialTrainer(cfg=cfg, 
                            env=env, 
                            agents=[agent_ddpg, agent_td3, agent_sac],
                            agents_scope=[100, 200, 212])

# evaluate the agents
trainer.eval()
