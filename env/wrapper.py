
__all__ = ["wrap_env"]


class _EnvWrapper:
    def __init__(self, env) -> None:
        self._env = env

    def step(self, actions):
        self._env.step(actions)
        return self._env.states_buffer, self._env.rewards_buffer, self._env.dones_buffer, None

    def reset(self):
        # TODO: reset each env
        return self._env.states_buffer

    def render(self, mode='human'):
        self._env.render(mode=mode)

    @property
    def state_space(self):
        return self._env.state_space

    @property
    def observation_space(self):
        return self._env.state_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def num_envs(self):
        return self._env.num_envs
    

def wrap_env(env):
    return _EnvWrapper(env)
