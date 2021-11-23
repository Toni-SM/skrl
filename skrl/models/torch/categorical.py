from typing import Union, Tuple

import gym
import torch
from torch.distributions import Categorical

from . import Model


class CategoricalModel(Model):
    def __init__(self, observation_space: Union[int, Tuple[int], gym.Space, None] = None, action_space: Union[int, Tuple[int], gym.Space, None] = None, device: str = "cuda:0") -> None:
        """Categorical model (stochastic model)

        # TODO: describe internal properties

        :param observation_space: Observation/state space or shape (default: None).
                                  If it is not None, the num_observations property will contain the size of that space (number of elements)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None).
                             If it is not None, the num_actions property will contain the size of that space (number of elements)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str, optional
        """
        super(CategoricalModel, self).__init__(observation_space, action_space, device)

        self.use_unnormalized_log_probabilities = True

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
                 The tuple's components are the actions, the log of the probability density function and None for the last component
        :rtype: tuple of torch.Tensor
        """
        # map from states/observations to normalized probabilities or unnormalized log probabilities
        output = self.compute(states.to(self.device), 
                              taken_actions.to(self.device) if taken_actions is not None else taken_actions)

        # unnormalized log probabilities
        if self.use_unnormalized_log_probabilities:
            distribution = Categorical(logits=output)
        # normalized probabilities
        else:
            distribution = Categorical(probs=output)
        
        # actions and log of the probability density function
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)

        if inference:
            return actions.detach(), log_prob.detach(), torch.Tensor()
        return actions, log_prob, torch.Tensor()
