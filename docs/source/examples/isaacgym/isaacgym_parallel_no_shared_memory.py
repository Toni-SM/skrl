import isaacgym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.trainers.torch import ParallelTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview4


# Define the models (stochastic and deterministic models) for the agents using mixins.
# - StochasticActor: takes as input the environment's observation/state and returns an action
# - DeterministicActor: takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
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

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


if __name__ == '__main__':

    # Load and wrap the Isaac Gym environment
    env = load_isaacgym_env_preview4(task_name="Cartpole")   # preview 3 and 4 use the same loader
    env = wrap_env(env)

    device = env.device


    # Instantiate the RandomMemory (without replacement) as experience replay memories
    memory_ddpg = RandomMemory(memory_size=8000, num_envs=100, device=device, replacement=True)
    memory_td3 = RandomMemory(memory_size=8000, num_envs=200, device=device, replacement=True)
    memory_sac = RandomMemory(memory_size=8000, num_envs=212, device=device, replacement=True)


    # Instantiate the agent's models (function approximators).
    # DDPG requires 4 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
    models_ddpg = {}
    models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
    models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
    models_ddpg["critic"] = Critic(env.observation_space, env.action_space, device)
    models_ddpg["target_critic"] = Critic(env.observation_space, env.action_space, device)
    # TD3 requires 6 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#spaces-and-models
    models_td3 = {}
    models_td3["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
    models_td3["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
    models_td3["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models_td3["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models_td3["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models_td3["target_critic_2"] = Critic(env.observation_space, env.action_space, device)
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#spaces-and-models
    models_sac = {}
    models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
    models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

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
                      memory=memory_ddpg,
                      cfg=cfg_ddpg,
                      observation_space=env.observation_space,
                      action_space=env.action_space,
                      device=device)

    agent_td3 = TD3(models=models_td3,
                    memory=memory_td3,
                    cfg=cfg_td3,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)

    agent_sac = SAC(models=models_sac,
                    memory=memory_sac,
                    cfg=cfg_sac,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)


    # Configure and instantiate the RL trainer and define the agent scopes
    cfg = {"timesteps": 8000, "headless": True}
    trainer = ParallelTrainer(cfg=cfg,
                              env=env,
                              agents=[agent_ddpg, agent_td3, agent_sac],
                              agents_scope=[100, 200, 212])   # agent scopes

    # start training
    trainer.train()
