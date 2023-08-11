import gymnasium as gym

import torch

# import the skrl components to build the RL system
from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, TabularMixin
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define model (tabular model) using mixin
class EpilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, epsilon=0.1):
        Model.__init__(self, observation_space, action_space, device)
        TabularMixin.__init__(self, num_envs)

        self.epsilon = epsilon
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions),
                                  dtype=torch.float32, device=self.device)

    def compute(self, inputs, role):
        actions = torch.argmax(self.q_table[torch.arange(self.num_envs).view(-1, 1), inputs["states"]],
                               dim=-1, keepdim=True).view(-1,1)

        # choose random actions for exploration according to epsilon
        indexes = (torch.rand(inputs["states"].shape[0], device=self.device) < self.epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions, {}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("FrozenLake-v0")
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("FrozenLake-v")][0]
    print("FrozenLake-v0 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# instantiate the agent's model (table)
# Q-learning requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/q_learning.html#models
models = {}
models["policy"] = EpilonGreedyPolicy(env.observation_space, env.action_space, device, num_envs=env.num_envs, epsilon=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/q_learning.html#configuration-and-hyperparameters
cfg = Q_LEARNING_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.999
cfg["alpha"] = 0.4
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1600
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/FrozenLake"

agent = Q_LEARNING(models=models,
                   memory=None,
                   cfg=cfg,
                   observation_space=env.observation_space,
                   action_space=env.action_space,
                   device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 80000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
