from typing import Union, Tuple

import gym
import torch
from torch.distributions import MultivariateNormal

from . import Model


class GaussianModel(Model):
    def __init__(self, observation_space: Union[int, Tuple[int], gym.Space, None] = None, action_space: Union[int, Tuple[int], gym.Space, None] = None, device: str = "cuda:0") -> None:
        """
        Diagonal Gaussian model (stochastic model)

        https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
        # TODO: describe internal properties

        Parameters
        ----------
        observation_space: int, tuple or list of integers, gym.Space or None, optional
            Observation/state space or shape (default: None).
            If it is not None, the num_observations property will contain the size of that space (number of elements)
        action_space: int, tuple or list of integers, gym.Space or None, optional
            Action space or shape (default: None).
            If it is not None, the num_actions property will contain the size of that space (number of elements)
        device: str, optional
            Device on which a torch tensor is or will be allocated (default: "cuda:0")
        """
        super(GaussianModel, self).__init__(observation_space, action_space, device)
        
        self.parameters_log_std = None

        self.clamp_log_std = False
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        
    def act(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None, inference=False) -> Tuple[torch.Tensor]:
        """
        Act stochastically in response to the state of the environment

        Parameters
        ----------
        states: torch.Tensor
            Observation/state of the environment used to make the decision
        taken_actions: torch.Tensor or None, optional
            Actions taken by a policy to the given states (default: None).
            The use of these actions only makes sense in critical networks, e.g.
        inference: bool, optional
            Flag to indicate whether the network is making inference (default: False)
        
        Returns
        -------
        tuple of torch.Tensor
            Action to be taken by the agent given the state of the environment.
            The tuple's components are the actions, the log of the probability density function and mean actions
        """
        # map from states/observations to mean actions and log standard deviations
        actions_mean, log_std = self.compute(states.to(self.device), 
                                             taken_actions.to(self.device) if taken_actions is not None else taken_actions)

        # log standard deviations as standalone parameters
        if self.parameters_log_std is not None:
            log_std = self.parameters_log_std
        
        # clamp log standard deviations
        if self.clamp_log_std:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # distribution
        covariance = torch.diag(log_std.exp() * log_std.exp())
        if self.num_actions is not None and torch.numel(log_std) != self.num_actions:
            covariance = covariance.unsqueeze(-1)
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        # actions # TODO: sample vs rsample
        # actions = distribution.sample()
        actions = distribution.rsample()

        # clip actions 
        # TODO: use tensor too for low and high
        if issubclass(type(self.action_space), gym.Space):
            actions = torch.clamp(actions, min=self.action_space.low[0], max=self.action_space.high[0])
        
        # log of the probability density function
        log_prob = distribution.log_prob(actions)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        return actions, log_prob, actions_mean
