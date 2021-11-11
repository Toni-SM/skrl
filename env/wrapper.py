import gym
import torch
import numpy as np

__all__ = ["wrap_env"]


class _Wrapper(object):
    def __init__(self, env):
        self._env = env

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        if hasattr(self._env, "device"):
            return self._env.device
        return "cuda:0" if torch.cuda.is_available() else "cpu" 

    @property
    def num_envs(self):
        return self._env.num_envs if hasattr(self._env, "num_envs") else 1

    @property
    def state_space(self):
        return self._env.state_space if hasattr(self._env, "state_space") else self._env.observation_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space


class _IsaacWrapper(_Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def step(self, actions):
        self._env.step(actions)
        return self._env.obs_buf, self._env.rew_buf, self._env.reset_buf, None

    def reset(self):
        # TODO: reset each env
        return self._env.obs_buf

    def render(self, mode='human'):
        self._env.render()
    

class _GymWrapper(_Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def step(self, actions):
        # convert the actions to numpy array if it is a torch tensor
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        observation, reward, done, info = self._env.step(actions)
        # convert response to torch
        return torch.from_numpy(observation).to(self.device).view(-1), \
               torch.tensor(reward).to(self.device).view(-1), \
               torch.tensor(done).to(self.device).view(-1), \
               info
        
    def reset(self):
        state = self._env.reset()
        # convert the state to torch if it is a numpy array
        if isinstance(state, np.ndarray):
            return torch.from_numpy(state).to(self.device)
        return state

    def render(self, mode='human'):
        self._env.render(mode=mode)


def wrap_env(env, wrapper="auto"):
    """
    Wrap an environment to use a common interface

    Parameters
    ----------
    env: gym.Env
        The environment to wrap
    wrapper: str, optional
        The type of wrapper to use (default: "auto").
        If "auto", the wrapper will be automatically selected based on the environment class.
        The specific wrappers supported are "gym" and "isaac"
    """
    print("[INFO] Environment:", [str(base).replace("<class '", "").replace("'>", "") for base in env.__class__.__bases__])
    
    if wrapper == "auto":
        # TODO: automatically select other wrappers
        if isinstance(env, gym.core.Wrapper):
            return _GymWrapper(env)
        return _IsaacWrapper(env)
    elif wrapper == "gym":
        return _GymWrapper(env)
    elif wrapper == "isaac":
        return _IsaacWrapper(env)
    else:
        raise ValueError("Unknown {} wrapper type".format(wrapper))
