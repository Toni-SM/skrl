from typing import Union, Tuple

import torch
from gym import spaces


class Memory:
    def __init__(self, buffer_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True, state_space: Union[spaces.Space, None] = None, action_space: Union[spaces.Space, None] = None) -> None:
        """
        Base class that represent a memory with circular buffers

        The implementation creates the buffers with the following shapes if the preallocate flag is set to true
        - states (buffer_size, num_envs, *state_space.shape)
        - actions (buffer_size, num_envs, *action_space.shape
        - rewards (buffer_size, num_envs, 1)
        - next_states (buffer_size, num_envs, *state_space.shape)
        - dones (buffer_size, num_envs, 1)

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
        state_space: gym.spaces.Space or None
            State/observation space
        action_space: gym.spaces.Space or None
            Action space
        """
        # TODO: handle dynamic memory
        # TODO: show memory consumption

        if preallocate:
            if not isinstance(state_space, spaces.Space):
                raise TypeError("env.state_space must be a gym Space")
            if not isinstance(action_space, spaces.Space):
                raise TypeError("env.action_space must be a gym Space")
            
        self.preallocate = preallocate
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device

        self.filled = False
        self.position_env = 0
        self.position_buffer = 0

        # buffers
        self.states = torch.zeros(buffer_size, num_envs, *state_space.shape, device=self.device)
        self.actions = torch.zeros(buffer_size, num_envs, *action_space.shape, device=self.device)
        self.rewards = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.next_states = torch.zeros(buffer_size, num_envs, *state_space.shape, device=self.device)
        self.dones = torch.zeros(buffer_size, num_envs, 1, device=self.device).byte()

        # alternative names
        self.push = self.add_transitions
        self.add_transition = self.add_transitions

    def __len__(self):
        """
        Current (valid) size of the buffer

        Returns
        -------
        int
            valid size
        """
        return self.buffer_size * self.num_envs if self.filled else self.position_buffer * self.num_envs + self.position_env
        
    def reset(self) -> None:
        """
        Reset the memory
        """
        self.filled = False
        self.position_env = 0
        self.position_buffer = 0

    def add_transitions(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> None:
        """
        Record a single transition or a transition batch in the memory

        Single transition:
            Storage a single transition (tensors with one dimension) and increment the position of the environment pointer (second index)
        Transition batch:
            Store the transitions (where tensors' first dimension must be equal to the number of parallel environments) and increment the position of the buffer pointer (first index)

        Parameters
        ----------
        states: torch.Tensor
            States/observations of the environment used to make the decision
        actions: torch.Tensor
            Actions taken by the agent
        rewards: torch.Tensor
            Instant rewards achieved by the current actions
        next_states: torch.Tensor
            Next states/observations of the environment
        dones: torch.Tensor
            Signals to indicate that episodes have ended
        """
        # single env transition
        if states.dim() == 1:
            self.states[self.position_buffer, self.position_env].copy_(states)
            self.actions[self.position_buffer, self.position_env].copy_(actions)
            self.rewards[self.position_buffer, self.position_env].copy_(rewards.view(-1))
            self.next_states[self.position_buffer, self.position_env].copy_(next_states)
            self.dones[self.position_buffer, self.position_env].copy_(dones.view(-1))
            self.position_env += 1
        # multi envs transitions
        elif states.dim() > 1 and states.shape[0] == self.num_envs:
            self.states[self.position_buffer].copy_(states)
            self.actions[self.position_buffer].copy_(actions)
            self.rewards[self.position_buffer].copy_(rewards.view(-1, 1))
            self.next_states[self.position_buffer].copy_(next_states)
            self.dones[self.position_buffer].copy_(dones.view(-1, 1))
            self.position_buffer += 1
        else:
            raise BufferError("The first dimension of the transition tensors {} does not match the number of parallel environments {}".format(states.shape[0], self.num_envs))
        
        # update pointers
        if self.position_env >= self.num_envs:
            self.position_env = 0
            self.position_buffer += 1
        if self.position_buffer >= self.buffer_size:
            self.position_buffer = 0
            self.filled = True

    def sample(self, batch_size: int) -> Tuple[torch.Tensor]:
        """
        Sample a batch from the memory

        Parameters
        ----------
        batch_size: int
            Number of element to sample

        Returns
        -------
        tuple
            Sampled tensors
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")
