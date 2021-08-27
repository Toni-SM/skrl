import torch
from torch.distributions import MultivariateNormal

from . import Model


class GaussianModel(Model):
    def __init__(self, env, device) -> None:
        """
        Diagonal Gaussian model (Stochastic)

        # TODO: describe internal properties

        https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
        """
        super().__init__(env, device)
        
        self.parameters_log_std = None

        self.clamp_log_std = True
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        
    def act(self, states, taken_actions=None, inference=False):
        # map from states/observations to mean actions and log standard deviations
        actions_mean, log_std = self.compute(states, taken_actions)

        # log standard deviations as standalone parameters
        if self.parameters_log_std is not None:
            log_std = self.parameters_log_std
        
        # clamp log standard deviations
        if self.clamp_log_std:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # distribution
        covariance = torch.diag(log_std.exp() * log_std.exp())
        if torch.numel(log_std) != self.num_action:
            covariance = covariance.unsqueeze(-1)
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        # actions # TODO: sample vs rsample
        # actions = distribution.sample()
        actions = distribution.rsample()

        # clip actions # FIXME: use tensor too
        actions.clamp_(self.env.action_space.low[0], self.env.action_space.high[0])
        
        # log of the probability density function
        log_prob = distribution.log_prob(actions)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        return actions, log_prob, actions_mean
