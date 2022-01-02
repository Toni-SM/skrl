import gym
import numpy as np

import torch

__all__ = ["wrap_env"]


class Wrapper(object):
    def __init__(self, env):
        self._env = env

        # device (faster than @property)
        if hasattr(self._env, "device"):
            self.device = torch.device(self._env.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def num_envs(self):
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, this property will be set to 1
        """
        return self._env.num_envs if hasattr(self._env, "num_envs") else 1

    @property
    def state_space(self):
        """State space

        If the wrapped environment does not have the ``state_space`` property, the value of the ``observation_space`` property will be used.
        """
        return self._env.state_space if hasattr(self._env, "state_space") else self._env.observation_space

    @property
    def observation_space(self):
        """Observation space
        """
        return self._env.observation_space

    @property
    def action_space(self):
        """Action space
        """
        return self._env.action_space


class IsaacGymPreview2Wrapper(Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self._reset_once = True
        self._obs_buf = None

    def step(self, actions):
        self._obs_buf, rew_buf, reset_buf, extras = self._env.step(actions)
        return self._obs_buf, rew_buf, reset_buf, None

    def reset(self):
        if self._reset_once:
            self._obs_buf = self._env.reset()
            self._reset_once = False
        return self._obs_buf

    def render(self, mode='human'):
        pass


class IsaacGymPreview3Wrapper(Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self._reset_once = True
        self._obs_dict = None

    def step(self, actions):
        self._obs_dict, rew_buf, reset_buf, extras = self._env.step(actions)
        return self._obs_dict["obs"], rew_buf, reset_buf, None

    def reset(self):
        if self._reset_once:
            self._obs_dict = self._env.reset()
            self._reset_once = False
        return self._obs_dict["obs"]

    def render(self, mode='human'):
        self._env.render()
    

class GymWrapper(Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def step(self, actions):
        # convert the actions to numpy array if it is a torch tensor
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        observation, reward, done, info = self._env.step(actions)
        # convert response to torch
        return torch.tensor(observation, device=self.device).view(1, -1), \
               torch.tensor(reward, device=self.device).view(1, -1), \
               torch.tensor(done, device=self.device).view(1, -1), \
               info
        
    def reset(self):
        state = self._env.reset()
        if isinstance(state, np.ndarray):
            return torch.tensor(state, device=self.device, dtype=torch.float32).view(1, -1)
        return state.to(self.device).view(1, -1)

    def render(self, mode='human'):
        self._env.render(mode=mode)


def wrap_env(env, wrapper="auto") -> Wrapper:
    """Wrap an environment to use a common interface

    :param env: The type of wrapper to use (default: "auto").
                If "auto", the wrapper will be automatically selected based on the environment class.
                The specific wrappers supported are "gym", "isaacgym-preview2" and "isaacgym-preview3"
    :type env: gym.Env, rlgpu.tasks.base.vec_task.VecTask or isaacgymenvs.tasks.base.vec_task.VecTask
    :param wrapper: The environment to be wrapped
    :type wrapper: str, optional
    
    :raises ValueError: Unknow wrapper type
    
    :return: Wrapped environment
    :rtype: Wrapper
    """
    print("[INFO] Environment:", [str(base).replace("<class '", "").replace("'>", "") for base in env.__class__.__bases__])
    
    if wrapper == "auto":
        # TODO: automatically select other wrappers
        if isinstance(env, gym.core.Wrapper):
            print("[INFO] Wrapper: Gym")
            return GymWrapper(env)
        elif "<class 'rlgpu.tasks.base.vec_task.VecTask'>" in [str(base) for base in env.__class__.__bases__]:
            print("[INFO] Wrapper: Isaac Gym (preview 2)")
            return IsaacGymPreview2Wrapper(env)
        print("[INFO] Wrapper: Isaac Gym (preview 3)")
        return IsaacGymPreview3Wrapper(env)
    elif wrapper == "gym":
        print("[INFO] Wrapper: Gym")
        return GymWrapper(env)
    elif wrapper == "isaacgym-preview2":
        print("[INFO] Wrapper: Isaac Gym (preview 2)")
        return IsaacGymPreview2Wrapper(env)
    elif wrapper == "isaacgym-preview3":
        print("[INFO] Wrapper: Isaac Gym (preview 3)")
        return IsaacGymPreview3Wrapper(env)
    else:
        raise ValueError("Unknown {} wrapper type".format(wrapper))
