import torch
from gym import spaces


class Memory:
    def __init__(self, buffer_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True, state_space=None, action_space=None) -> None:
        # TODO: handle dynamic memory

        if preallocate:
            if not isinstance(state_space, spaces.Space):
                raise TypeError("env.state_space must be a gym Space")
            if not isinstance(action_space, spaces.Space):
                raise TypeError("env.action_space must be a gym Space")
            
        self.preallocate = preallocate
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device

        self.position = 0
        self.filled = False

        self.states = torch.zeros(buffer_size, num_envs, *state_space.shape, device=self.device)
        self.actions = torch.zeros(buffer_size, num_envs, *action_space.shape, device=self.device)
        self.rewards = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.next_states = torch.zeros(buffer_size, num_envs, *state_space.shape, device=self.device)
        self.dones = torch.zeros(buffer_size, num_envs, 1, device=self.device).byte()

    def __len__(self):
        return self.buffer_size

    def add_transition(self, state, action, reward, next_state, done):
        if self.position >= self.buffer_size:
            self.position = 0
            self.filled = True

        self.states[self.position, 0].copy_(state)
        self.actions[self.position, 0].copy_(action)
        self.rewards[self.position, 0].copy_(reward.view(-1, 1))
        self.next_states[self.position, 0].copy_(next_state)
        self.dones[self.position, 0].copy_(done.view(-1, 1))

        self.position += 1

    def add_transitions(self, states, actions, rewards, next_states, dones):
        if self.position >= self.buffer_size:
            self.position = 0

        self.states[self.position].copy_(states)
        self.actions[self.position].copy_(actions)
        self.rewards[self.position].copy_(rewards.view(-1, 1))
        self.next_states[self.position].copy_(next_states)
        self.dones[self.position].copy_(dones.view(-1, 1))

        self.position += 1

    def push(self, state, action, reward, next_state, done):
        self.add_transition(state, action, reward, next_state, done)

    def sample(self, batch_size):
        raise NotImplementedError
