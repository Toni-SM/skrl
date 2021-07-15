import torch
from torch.utils.data import RandomSampler

from .base import Memory


class RandomMemory(Memory):
    def __init__(self, buffer_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True, state_space=None, action_space=None) -> None:
        super().__init__(buffer_size=buffer_size, 
                         num_envs=num_envs, 
                         device=device, 
                         preallocate=preallocate,
                         state_space=state_space, 
                         action_space=action_space)

    def sample(self, batch_size):
        max_value = (self.buffer_size if self.filled else self.position) * self.num_envs
        indexes = RandomSampler(range(max_value), replacement=True, num_samples=batch_size)

        states = self.states.view(-1, self.states.size(-1))[indexes]
        actions = self.actions.view(-1, self.actions.size(-1))[indexes]
        rewards = self.rewards.view(-1, self.rewards.size(-1))[indexes]
        next_states = self.next_states.view(-1, self.next_states.size(-1))[indexes]
        dones = self.dones.view(-1, self.dones.size(-1))[indexes]

        return states, actions, rewards, next_states, dones
