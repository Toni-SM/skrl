import gym

class DummyEnv:
    observation_space = gym.spaces.Discrete(4)
    action_space = gym.spaces.Discrete(3)
    device = "cuda:0"
    num_envs = 2

env = DummyEnv()

# [start-epsilon-greedy]
import torch

from skrl.models.torch import TabularModel


# define the model
class EpilonGreedyPolicy(TabularModel):
    def __init__(self, observation_space, action_space, device, num_envs=1, epsilon=0.1):
        super().__init__(observation_space, action_space, device, num_envs)

        self.epsilon = epsilon
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions), dtype=torch.float32)
        
        self.tables["q_table"] = self.q_table

    def compute(self, states, taken_actions):
        actions = torch.argmax(self.q_table[torch.arange(self.num_envs).view(-1, 1), states], 
                               dim=-1, keepdim=True).view(-1,1)
        
        indexes = (torch.rand(states.shape[0], device=self.device) < self.epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions


# instantiate the model (assumes there is a wrapped environment: env)
policy = EpilonGreedyPolicy(observation_space=env.observation_space, 
                            action_space=env.action_space, 
                            device=env.device, 
                            num_envs=env.num_envs,
                            epsilon=0.15)
# [end-epsilon-greedy]

import torch
policy.to(env.device)
actions = policy.act(torch.tensor([[0, 1, 2, 3]], device=env.device))
assert actions[0].shape == torch.Size([10, env.action_space.shape[0]])
