# [start-definition-torch]
class TabularModel(TabularMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        TabularMixin.__init__(self)
        # [end-definition-torch]


# =============================================================================

# [start-epsilon-greedy-torch]
import torch
import torch.nn as nn

from skrl.models.torch import Model, TabularMixin


# define the model
class EpsilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, epsilon=0.1):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        TabularMixin.__init__(self)

        self.epsilon = epsilon
        self.q_table = nn.Parameter(
            torch.ones((self.num_observations, self.num_actions), dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

    def compute(self, inputs, role):
        observations = inputs["observations"]
        actions = torch.argmax(self.q_table[observations], dim=-1, keepdim=False)
        # choose random actions for exploration according to epsilon
        indexes = (torch.rand(observations.shape[0], device=self.device) < self.epsilon).nonzero().flatten()
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions, {}


# instantiate the model (given a wrapped environment: `env`)
policy = EpsilonGreedyPolicy(env.observation_space, env.state_space, env.action_space, env.device)
# [end-epsilon-greedy-torch]
