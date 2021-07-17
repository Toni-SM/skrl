from typing import Union, Tuple

import torch
import numpy as np
from gym import spaces

from .base import Memory


class RandomMemory(Memory):
    def __init__(self, buffer_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True, state_space: Union[spaces.Space, None] = None, action_space: Union[spaces.Space, None] = None) -> None:
        super().__init__(buffer_size=buffer_size, 
                         num_envs=num_envs, 
                         device=device, 
                         preallocate=preallocate,
                         state_space=state_space, 
                         action_space=action_space)
        """
        Random sampling memory

        Sample a batch from the memory randomly

        Parameters
        ----------
        buffer_size: int
            Maximum number of items in the buffer 
        num_envs: int
            Number of parallel environments
        device: str
            Device on which a PyTorch tensor is or will be allocated
        preallocate: bool
            If true, preallocate memory for efficient use
        state_space: Union[gym.spaces.Space, None]
            State/observation space
        action_space: Union[gym.spaces.Space, None]
            Action space
        """

    def sample(self, batch_size) -> Tuple[torch.Tensor]:
        """
        Sample a batch from the memory randomly

        Parameters
        ----------
        batch_size: int
            Number of element to sample

        Returns
        -------
            tuple
                Sampled tensors (states, actions, rewards, next_states, dones)
        """
        # get indexes
        max_value = len(self) * self.num_envs
        indexes =  np.random.choice(max_value, size=batch_size, replace=True)

        # sample
        states = self.states.view(-1, self.states.size(-1))[indexes]
        actions = self.actions.view(-1, self.actions.size(-1))[indexes]
        rewards = self.rewards.view(-1, self.rewards.size(-1))[indexes]
        next_states = self.next_states.view(-1, self.next_states.size(-1))[indexes]
        dones = self.dones.view(-1, self.dones.size(-1))[indexes]

        return states, actions, rewards, next_states, dones
