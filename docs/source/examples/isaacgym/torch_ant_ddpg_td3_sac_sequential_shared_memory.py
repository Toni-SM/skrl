import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="Ant", num_envs=64)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay (unique to all agents)
memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)


# instantiate the agents' models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models_ddpg["critic"] = Critic(env.observation_space, env.action_space, device)
models_ddpg["target_critic"] = Critic(env.observation_space, env.action_space, device)

# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#models
models_td3 = {}
models_td3["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models_td3["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models_td3["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_td3["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_td3["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_td3["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models_sac = {}
models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agents (visit their documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
cfg_ddpg["gradient_steps"] = 1
cfg_ddpg["batch_size"] = 4096
cfg_ddpg["discount_factor"] = 0.99
cfg_ddpg["polyak"] = 0.005
cfg_ddpg["actor_learning_rate"] = 5e-4
cfg_ddpg["critic_learning_rate"] = 5e-4
cfg_ddpg["random_timesteps"] = 80
cfg_ddpg["learning_starts"] = 80
cfg_ddpg["state_preprocessor"] = RunningStandardScaler
cfg_ddpg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg_ddpg["experiment"]["write_interval"] = 800
cfg_ddpg["experiment"]["checkpoint_interval"] = 8000
cfg_ddpg["experiment"]["directory"] = "runs/torch/Ant"

# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#configuration-and-hyperparameters
cfg_td3 = TD3_DEFAULT_CONFIG.copy()
cfg_td3["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
cfg_td3["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
cfg_td3["smooth_regularization_clip"] = 0.5
cfg_td3["gradient_steps"] = 1
cfg_td3["batch_size"] = 4096
cfg_td3["discount_factor"] = 0.99
cfg_td3["polyak"] = 0.005
cfg_td3["actor_learning_rate"] = 5e-4
cfg_td3["critic_learning_rate"] = 5e-4
cfg_td3["random_timesteps"] = 80
cfg_td3["learning_starts"] = 80
cfg_td3["state_preprocessor"] = RunningStandardScaler
cfg_td3["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg_td3["experiment"]["write_interval"] = 800
cfg_td3["experiment"]["checkpoint_interval"] = 8000
cfg_td3["experiment"]["directory"] = "runs/torch/Ant"

# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1
cfg_sac["batch_size"] = 4096
cfg_sac["discount_factor"] = 0.99
cfg_sac["polyak"] = 0.005
cfg_sac["actor_learning_rate"] = 5e-4
cfg_sac["critic_learning_rate"] = 5e-4
cfg_sac["random_timesteps"] = 80
cfg_sac["learning_starts"] = 80
cfg_sac["grad_norm_clip"] = 0
cfg_sac["learn_entropy"] = True
cfg_sac["entropy_learning_rate"] = 5e-3
cfg_sac["initial_entropy_value"] = 1.0
cfg_sac["state_preprocessor"] = RunningStandardScaler
cfg_sac["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg_sac["experiment"]["write_interval"] = 800
cfg_sac["experiment"]["checkpoint_interval"] = 8000
cfg_sac["experiment"]["directory"] = "runs/torch/Ant"

agent_ddpg = DDPG(models=models_ddpg,
                  memory=memory,  # shared memory
                  cfg=cfg_ddpg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)

agent_td3 = TD3(models=models_td3,
                memory=memory,  # shared memory
                cfg=cfg_td3,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

agent_sac = SAC(models=models_sac,
                memory=memory,  # shared memory
                cfg=cfg_sac,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer,
                            env=env,
                            agents=[agent_ddpg, agent_td3, agent_sac],
                            agents_scope=[])

# start training
trainer.train()
