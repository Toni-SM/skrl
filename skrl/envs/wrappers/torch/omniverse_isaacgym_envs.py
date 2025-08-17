from typing import Any, Optional, Tuple, Union

import torch

from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class OmniverseIsaacGymWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Omniverse Isaac Gym environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    def run(self, trainer: Optional["omni.isaac.gym.vec_env.vec_env_mt.TrainerMT"] = None) -> None:
        """Run the simulation in the main thread.

        This method is valid only for the Omniverse Isaac Gym multi-threaded environments.

        :param trainer: Trainer which should implement a ``run`` method that initiates the RL loop on a new thread.
        """
        self._env.run(trainer)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, self._info = self._env.step(
            unflatten_tensorized_space(self.action_space, actions)
        )
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["obs"]))
        truncated = self._info["time_outs"] if "time_outs" in self._info else torch.zeros_like(terminated)
        states = observations.get("states", None)
        if states is not None and states.shape[-1]:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> Union[torch.Tensor, None]:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations = self._env.reset()
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["obs"]))
            states = observations.get("states", None)
            if states is not None and states.shape[-1]:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
