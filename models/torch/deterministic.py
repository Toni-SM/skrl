from typing import Union, Tuple

import gym
import torch

from . import Model


class DeterministicModel(Model):
    def __init__(self, observation_space: Union[int, Tuple[int], gym.Space, None] = None, action_space: Union[int, Tuple[int], gym.Space, None] = None, device: str = "cuda:0") -> None:
        """
        Deterministic model (deterministic model)

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
        super(DeterministicModel, self).__init__(observation_space, action_space, device)
        
    def act(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None, inference=False) -> Tuple[torch.Tensor]:
        """
        Act deterministically in response to the state of the environment

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
            The tuple's components are the computed actions and None for the last two components
        """
        # map from observations/states to actions
        actions = self.compute(states.to(self.device), 
                               taken_actions.to(self.device) if taken_actions is not None else taken_actions)

        # clip actions 
        # TODO: use tensor too for low and high
        if issubclass(type(self.action_space), gym.Space):
            actions.clamp_(self.action_space.low[0], self.action_space.high[0])

        return actions, None, None
        