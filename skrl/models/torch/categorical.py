from typing import Union, Tuple

import gym

import torch
from torch.distributions import Categorical

from . import Model


class CategoricalModel(Model):
    def __init__(self, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0",
                 unnormalized_log_prob: bool = True) -> None:
        """Categorical model (stochastic model)

        :param observation_space: Observation/state space or shape (default: None).
                                  If it is not None, the num_observations property will contain the size of that space
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None).
                             If it is not None, the num_actions property will contain the size of that space
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        :param unnormalized_log_prob: Flag to indicate how to be interpreted the network's output (default: True).
                                      If True, the network's output is interpreted as unnormalized log probabilities 
                                      (it can be any real number), otherwise as normalized probabilities 
                                      (the output must be non-negative, finite and have a non-zero sum)
        :type unnormalized_log_prob: bool, optional
        """
        super(CategoricalModel, self).__init__(observation_space, action_space, device)

        self._unnormalized_log_prob = unnormalized_log_prob

        self._distribution = None

    def act(self, 
            states: torch.Tensor, 
            taken_actions: Union[torch.Tensor, None] = None, 
            inference=False) -> Tuple[torch.Tensor]:
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
                 The tuple's components are the actions, the log of the probability density function and the network's output
        :rtype: tuple of torch.Tensor
        """
        # map from states/observations to normalized probabilities or unnormalized log probabilities
        if self._instantiator_net is None:
            output = self.compute(states.to(self.device), 
                                  taken_actions.to(self.device) if taken_actions is not None else taken_actions)
        else:
            output = self._get_instantiator_output(states.to(self.device), \
                taken_actions.to(self.device) if taken_actions is not None else taken_actions)

        # unnormalized log probabilities
        if self._unnormalized_log_prob:
            self._distribution = Categorical(logits=output)
        # normalized probabilities
        else:
            self._distribution = Categorical(probs=output)
        
        # actions and log of the probability density function
        actions = self._distribution.sample()
        log_prob = self._distribution.log_prob(actions if taken_actions is None else taken_actions.view(-1))

        if inference:
            return actions.unsqueeze(-1).detach(), log_prob.unsqueeze(-1).detach(), output.detach()
        return actions.unsqueeze(-1), log_prob.unsqueeze(-1), output

    def distribution(self) -> torch.distributions.Categorical:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Categorical
        """
        return self._distribution