from typing import Union, Tuple

import gym
import torch
from torch.distributions import MultivariateNormal

from . import Model


class GaussianModel(Model):
    def __init__(self, observation_space: Union[int, Tuple[int], gym.Space, None] = None, action_space: Union[int, Tuple[int], gym.Space, None] = None, device: str = "cuda:0", clip_actions: bool = False, clip_log_std: bool = True, min_log_std: float = -20, max_log_std: float = 2) -> None:
        """Diagonal Gaussian model (stochastic model)

        :param observation_space: Observation/state space or shape (default: None).
                                  If it is not None, the num_observations property will contain the size of that space (number of elements)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None).
                             If it is not None, the num_actions property will contain the size of that space (number of elements)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str, optional
        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: False)
        :type clip_actions: bool, optional
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
        :type clip_log_std: bool, optional
        :param min_log_std: Minimum value of the log standard deviation if clip_log_std is True (default: -20)
        :type min_log_std: float, optional
        :param max_log_std: Maximum value of the log standard deviation if clip_log_std is True (default: 2)
        :type max_log_std: float, optional
        """
        super(GaussianModel, self).__init__(observation_space, action_space, device)
        
        self.clip_actions = clip_actions

        self.clip_log_std = clip_log_std
        self.log_std_min = min_log_std
        self.log_std_max = max_log_std

        self._distribution = None
        
    def act(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None, inference=False) -> Tuple[torch.Tensor]:
        """Act stochastically in response to the state of the environment

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: None).
                              The use of these actions only makes sense in critical networks, e.g.
        :type taken_actions: torch.Tensor or None, optional
        :param inference: Flag to indicate whether the network is making inference (default: False).
                          If True, the returned tensors will be detached from the current graph
        :type inference: bool, optional
        
        :return: Action to be taken by the agent given the state of the environment.
                 The tuple's components are the actions, the log of the probability density function and mean actions
        :rtype: tuple of torch.Tensor
        """
        # map from states/observations to mean actions and log standard deviations
        actions_mean, log_std = self.compute(states.to(self.device), 
                                             taken_actions.to(self.device) if taken_actions is not None else taken_actions)
        
        # clamp log standard deviations
        if self.clip_log_std:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # distribution
        covariance = torch.diag(log_std.exp() * log_std.exp())
        if covariance.dim() < actions_mean.dim():
            covariance = covariance.unsqueeze(-1)
        self._distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        # sample using the reparameterization trick
        actions = self._distribution.rsample()

        # clip actions 
        # TODO: use tensor too for low and high
        if self.clip_actions and issubclass(type(self.action_space), gym.Space):
            actions = torch.clamp(actions, min=self.action_space.low[0], max=self.action_space.high[0])
        
        # log of the probability density function
        log_prob = self._distribution.log_prob(actions if taken_actions is None else taken_actions)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        if inference:
            return actions.detach(), log_prob.detach(), actions_mean.detach()
        return actions, log_prob, actions_mean

    def get_entropy(self) -> torch.Tensor:
        """Compute and return the entropy of the model

        :return: Entropy of the model
        :rtype: torch.Tensor
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)
    