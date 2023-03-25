import random
import gymnasium as gym

import torch


class DummyEnv(gym.Env):
    def __init__(self, num_envs, device = "cpu"):
        self.num_agents = 1
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def __getattr__(self, key):
        if key in ["_spec_to_space", "observation_spec"]:
            return lambda *args, **kwargs: None
        return None

    def step(self, action):
        observation = self.observation_space.sample()
        reward = random.random()
        terminated = random.random() > 0.95
        truncated = random.random() > 0.95

        observation = torch.tensor(observation, dtype=torch.float32).view(self.num_envs, -1)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        return observation, reward, terminated, truncated, {}

    def reset(self):
        observation = self.observation_space.sample()
        observation = torch.tensor(observation, dtype=torch.float32).view(self.num_envs, -1)
        return observation, {}

    def render(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class _DummyBaseAgent:
    def __init__(self):
        pass

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        pass

    def pre_interaction(self, timestep, timesteps):
        pass

    def post_interaction(self, timestep, timesteps):
        pass

    def set_running_mode(self, mode):
        pass


class DummyAgent(_DummyBaseAgent):
    def __init__(self):
        super().__init__()

    def init(self, trainer_cfg=None):
        pass

    def act(self, states, timestep, timesteps):
        return torch.tensor([]), None, {}

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        pass

    def pre_interaction(self, timestep, timesteps):
        pass

    def post_interaction(self, timestep, timesteps):
        pass


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cpu")
        self.layer = torch.nn.Linear(1, 1)

    def set_mode(self, *args, **kwargs):
        pass

    def get_specification(self, *args, **kwargs):
        return {}

    def act(self, *args, **kwargs):
        return torch.tensor([]), None, {}
