import gym

import torch

# Import the skrl components to build the RL system
from skrl.models.torch import Model, TabularMixin
from skrl.agents.torch.sarsa import SARSA, SARSA_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the model (tabular models) for the SARSA agent using a helper class
class EpilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, epsilon=0.1):
        Model.__init__(self, observation_space, action_space, device)
        TabularMixin.__init__(self, num_envs)

        self.epsilon = epsilon
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions), 
                                  dtype=torch.float32, device=self.device)
        
    def compute(self, states, taken_actions, role):
        actions = torch.argmax(self.q_table[torch.arange(self.num_envs).view(-1, 1), states], 
                               dim=-1, keepdim=True).view(-1,1)
        
        # choose random actions for exploration according to epsilon
        indexes = (torch.rand(states.shape[0], device=self.device) < self.epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions


# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version
try:
    env = gym.make("Taxi-v3")
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("Taxi-v")][0]
    print("Taxi-v3 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# Instantiate the agent's models (table)
# SARSA requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sarsa.html#spaces-and-models
models_sarsa = {}
models_sarsa["policy"] = EpilonGreedyPolicy(env.observation_space, env.action_space, device, num_envs=env.num_envs, epsilon=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sarsa.html#configuration-and-hyperparameters
cfg_sarsa = SARSA_DEFAULT_CONFIG.copy()
cfg_sarsa["random_timesteps"] = 0
# logging to TensorBoard and write checkpoints each 1600 and ignore checkpoints
cfg_sarsa["experiment"]["write_interval"] = 1600
cfg_sarsa["experiment"]["checkpoint_interval"] = 0

agent_sarsa = SARSA(models=models_sarsa,
                    memory=None, 
                    cfg=cfg_sarsa, 
                    observation_space=env.observation_space, 
                    action_space=env.action_space,
                    device=device)

# load checkpoint
agent_sarsa.load("./runs/22-09-10_13-13-41-011999_SARSA/checkpoints/agent_80000.pt")


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 80000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_sarsa)

# evaluate the agent
trainer.eval()
