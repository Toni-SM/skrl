# [start-definition-torch]
class TabularModel(TabularMixin, Model):
    def __init__(self, observation_space, action_space, device=None, num_envs=1):
        Model.__init__(self, observation_space, action_space, device)
        TabularMixin.__init__(self, num_envs)
# [end-definition-torch]

# =============================================================================

# [start-epsilon-greedy-torch]
import torch

from skrl.models.torch import Model, TabularMixin


# define the model
class EpilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, epsilon=0.1):
        Model.__init__(self, observation_space, action_space, device)
        TabularMixin.__init__(self, num_envs)

        self.epsilon = epsilon
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions), dtype=torch.float32)

    def compute(self, inputs, role):
        states = inputs["states"]
        actions = torch.argmax(self.q_table[torch.arange(self.num_envs).view(-1, 1), states],
                               dim=-1, keepdim=True).view(-1,1)

        indexes = (torch.rand(states.shape[0], device=self.device) < self.epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions, {}


# instantiate the model (assumes there is a wrapped environment: env)
policy = EpilonGreedyPolicy(observation_space=env.observation_space,
                            action_space=env.action_space,
                            device=env.device,
                            num_envs=env.num_envs,
                            epsilon=0.15)
# [end-epsilon-greedy-torch]
