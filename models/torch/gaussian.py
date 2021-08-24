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
        
    def act(self, states, taken_actions=None, inference=False):
        # map from states/observations to mean actions and log standard deviations
        actions_mean, log_std = self.compute(states, taken_actions)

        # log standard deviations as standalone parameters
        if self.parameters_log_std is not None:
            log_std = self.parameters_log_std
        
        # distribution
        if torch.numel(log_std) != self.num_action:
            # FIXME: try to use all log_std
            covariance = torch.diag(torch.mean(log_std.exp() * log_std.exp(), dim=0))
        else:
            covariance = torch.diag(log_std.exp() * log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        # actions and log of the probability density function
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)

        return actions, log_prob, actions_mean
