from typing import Union, Tuple

import gym
import torch
from torch.distributions import Categorical

from . import Model


class CategoricalModel(Model):
    def __init__(self, observation_space: Union[int, Tuple[int], gym.Space, None] = None, action_space: Union[int, Tuple[int], gym.Space, None] = None, device: str = "cuda:0") -> None:
        """
        Categorical model (stochastic model)

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
        super(CategoricalModel, self).__init__(observation_space, action_space, device)

        self.use_unnormalized_log_probabilities = True

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
            The tuple's components are the actions, the log of the probability density function and None for the last component
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

        return actions, log_prob, torch.Tensor()
