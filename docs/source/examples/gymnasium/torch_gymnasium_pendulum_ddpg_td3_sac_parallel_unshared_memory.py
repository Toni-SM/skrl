import gymnasium as gym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.trainers.torch import ParallelTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=True, min_log_std=-5, max_log_std=2)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return 2 * self.net(inputs["observations"]), {"log_std": self.log_std_parameter}

class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, self.num_actions),
                                 nn.Tanh())

    def compute(self, inputs, role):
        return 2 * self.net(inputs["observations"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1)), {}


if __name__ == '__main__':

    # load and wrap the gymnasium environment.
    # note: the environment version may change depending on the gymnasium version
    try:
        env = gym.make_vec("Pendulum-v1", num_envs=3, vectorization_mode="sync")
    except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
        env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][0]
        print("Pendulum-v1 not found. Trying {}".format(env_id))
        env = gym.make_vec(env_id, num_envs=3, vectorization_mode="sync")
    env = wrap_env(env)

    device = env.device


    # instantiate memories as experience replay (unique for each agents).
    # scopes (3 envs): DDPG 1, TD3 1, and SAC 1
    memory_ddpg = RandomMemory(memory_size=15000, num_envs=1, device=device)
    memory_td3 = RandomMemory(memory_size=15000, num_envs=1, device=device)
    memory_sac = RandomMemory(memory_size=15000, num_envs=1, device=device)


    # instantiate the agents' models (function approximators).
    # DDPG requires 4 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
    models_ddpg = {}
    models_ddpg["policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device, clip_actions=True)
    models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device, clip_actions=True)
    models_ddpg["critic"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_ddpg["target_critic"] = Critic(env.observation_space, env.state_space, env.action_space, device)

    # TD3 requires 6 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/td3.html#models
    models_td3 = {}
    models_td3["policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device, clip_actions=True)
    models_td3["target_policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device, clip_actions=True)
    models_td3["critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_td3["critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_td3["target_critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_td3["target_critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)

    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models_sac = {}
    models_sac["policy"] = StochasticActor(env.observation_space, env.state_space, env.action_space, device, clip_actions=True)
    models_sac["critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_sac["critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_sac["target_critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_sac["target_critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)


    # configure and instantiate the agents (visit their documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
    cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
    cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
    cfg_ddpg["discount_factor"] = 0.98
    cfg_ddpg["batch_size"] = 100
    cfg_ddpg["random_timesteps"] = 100
    cfg_ddpg["learning_starts"] = 100
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg_ddpg["experiment"]["write_interval"] = "auto"
    cfg_ddpg["experiment"]["checkpoint_interval"] = "auto"
    cfg_ddpg["experiment"]["directory"] = "runs/torch/Pendulum-DDPG"

    # https://skrl.readthedocs.io/en/latest/api/agents/td3.html#configuration-and-hyperparameters
    cfg_td3 = TD3_DEFAULT_CONFIG.copy()
    cfg_td3["exploration"]["noise"] = GaussianNoise(mean=0, std=0.1, device=device)
    cfg_td3["smooth_regularization_noise"] = GaussianNoise(mean=0, std=0.2, device=device)
    cfg_td3["smooth_regularization_clip"] = 0.5
    cfg_td3["discount_factor"] = 0.98
    cfg_td3["batch_size"] = 100
    cfg_td3["random_timesteps"] = 1000
    cfg_td3["learning_starts"] = 1000
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg_td3["experiment"]["write_interval"] = "auto"
    cfg_td3["experiment"]["checkpoint_interval"] = "auto"
    cfg_td3["experiment"]["directory"] = "runs/torch/Pendulum-TD3"

    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg_sac = SAC_DEFAULT_CONFIG.copy()
    cfg_sac["discount_factor"] = 0.98
    cfg_sac["batch_size"] = 100
    cfg_sac["random_timesteps"] = 0
    cfg_sac["learning_starts"] = 1000
    cfg_sac["learn_entropy"] = True
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg_sac["experiment"]["write_interval"] = "auto"
    cfg_sac["experiment"]["checkpoint_interval"] = "auto"
    cfg_sac["experiment"]["directory"] = "runs/torch/Pendulum-SAC"

    agent_ddpg = DDPG(models=models_ddpg,
                      memory=memory_ddpg,
                      cfg=cfg_ddpg,
                      observation_space=env.observation_space,
                      state_space=env.state_space,
                      action_space=env.action_space,
                      device=device)

    agent_td3 = TD3(models=models_td3,
                    memory=memory_td3,
                    cfg=cfg_td3,
                    observation_space=env.observation_space,
                    state_space=env.state_space,
                    action_space=env.action_space,
                    device=device)

    agent_sac = SAC(models=models_sac,
                    memory=memory_sac,
                    cfg=cfg_sac,
                    observation_space=env.observation_space,
                    state_space=env.state_space,
                    action_space=env.action_space,
                    device=device)


    # configure and instantiate the RL trainer and define the agent scopes
    cfg_trainer = {"timesteps": 15000, "headless": True}
    trainer = ParallelTrainer(cfg=cfg_trainer,
                              env=env,
                              agents=[agent_ddpg, agent_td3, agent_sac],
                              scopes=[1, 1, 1])  # scopes (3 envs): DDPG 1, TD3 1 and SAC 1

    # start training
    # trainer.train()
    trainer.eval()
