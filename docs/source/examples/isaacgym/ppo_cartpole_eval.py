import isaacgym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview4


# Define the shared model (stochastic and deterministic models) for the agent using mixins.
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU())
        
        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        
        self.value_layer = nn.Linear(32, 1)

    def act(self, states, taken_actions, role):
        if role == "policy":
            return GaussianMixin.act(self, states, taken_actions, role)
        elif role == "value":
            return DeterministicMixin.act(self, states, taken_actions, role)

    def compute(self, states, taken_actions, role):
        if role == "policy":
            return self.mean_layer(self.net(states)), self.log_std_parameter
        elif role == "value":
            return self.value_layer(self.net(states))


# Load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="Cartpole")   # preview 3 and 4 use the same loader
env = wrap_env(env)

device = env.device


# Instantiate the agent's policy.
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Shared(env.observation_space, env.action_space, device)


# Configure and instantiate the agent.
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["random_timesteps"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard each 16 timesteps and ignore checkpoints
cfg_ppo["experiment"]["write_interval"] = 16
cfg_ppo["experiment"]["checkpoint_interval"] = 0

agent = PPO(models=models_ppo,
            memory=None, 
            cfg=cfg_ppo, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)

# load checkpoint (agent)
agent.load("./runs/22-09-12_18-56-10-110956_PPO/checkpoints/agent_1600.pt")


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1600, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# evaluate the agent
trainer.eval()
