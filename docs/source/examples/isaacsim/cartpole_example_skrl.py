# Omniverse Isaac Sim tutorial: Creating New RL Environment 
# https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_new_rl_example.html

# Instance of VecEnvBase and create the task
from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=True)

from cartpole_task import CartpoleTask
task = CartpoleTask(name="Cartpole")
env.set_task(task, backend="torch")


import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import GaussianModel, DeterministicModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the models (stochastic and deterministic models) for the agent using helper classes.
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy(GaussianModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        super().__init__(observation_space, action_space, device, clip_actions,
                         clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        return torch.tanh(self.net(states)), self.log_std_parameter

class Value(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 1))

    def compute(self, states, taken_actions):
        return self.net(states)


# Load and wrap the environment
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {"policy": Policy(env.observation_space, env.action_space, device, clip_actions=True),
              "value": Value(env.observation_space, env.action_space, device)}

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ppo.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)   


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 1000
cfg_ppo["learning_epochs"] = 20
cfg_ppo["mini_batches"] = 1
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 0.001
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = False
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 0.5
cfg_ppo["kl_threshold"] = 0
cfg_ppo["experiment"]["write_interval"] = 1000
cfg_ppo["experiment"]["checkpoint_interval"] = 10000

agent = PPO(models=models_ppo,
            memory=memory, 
            cfg=cfg_ppo, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
