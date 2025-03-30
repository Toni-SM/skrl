from typing import Any, Tuple

import collections
import gymnasium

import numpy as np
import torch

from skrl import logger
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)


class DeepMindWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """DeepMind environment wrapper

        :param env: The environment to wrap
        :type env: Any supported DeepMind environment
        """
        super().__init__(env)

        from dm_env import specs

        self._specs = specs

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        return self._spec_to_space(self._env.observation_spec())

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        return self._spec_to_space(self._env.action_spec())

    def _spec_to_space(self, spec: Any) -> gymnasium.Space:
        """Convert the DeepMind spec to a gymnasium space

        :param spec: The DeepMind spec to convert
        :type spec: Any supported DeepMind spec

        :raises: ValueError if the spec type is not supported

        :return: The gymnasium space
        :rtype: gymnasium.Space
        """
        if isinstance(spec, self._specs.DiscreteArray):
            return gymnasium.spaces.Discrete(spec.num_values)
        elif isinstance(spec, self._specs.BoundedArray):
            return gymnasium.spaces.Box(
                shape=spec.shape,
                dtype=spec.dtype,
                low=spec.minimum if spec.minimum.ndim else np.full(spec.shape, spec.minimum),
                high=spec.maximum if spec.maximum.ndim else np.full(spec.shape, spec.maximum),
            )
        elif isinstance(spec, self._specs.Array):
            return gymnasium.spaces.Box(
                shape=spec.shape,
                dtype=spec.dtype,
                low=np.full(spec.shape, float("-inf")),
                high=np.full(spec.shape, float("inf")),
            )
        elif isinstance(spec, collections.OrderedDict):
            return gymnasium.spaces.Dict({k: self._spec_to_space(v) for k, v in spec.items()})
        else:
            raise ValueError(f"Spec type {type(spec)} not supported. Please report this issue")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        actions = untensorize_space(self.action_space, unflatten_tensorized_space(self.action_space, actions))
        timestep = self._env.step(actions)

        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, timestep.observation, device=self.device)
        )
        reward = timestep.reward if timestep.reward is not None else 0
        terminated = timestep.last()
        truncated = False
        info = {}

        # convert response to torch
        return (
            observation,
            torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1),
            torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1),
            torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1),
            info,
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        timestep = self._env.reset()
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, timestep.observation, device=self.device)
        )
        return observation, {}

    def render(self, *args, **kwargs) -> np.ndarray:
        """Render the environment

        OpenCV is used to render the environment.
        Install OpenCV with ``pip install opencv-python``
        """
        frame = self._env.physics.render(480, 640, camera_id=0)

        # render the frame using OpenCV
        try:
            import cv2

            cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        except ImportError as e:
            logger.warning(f"Unable to import opencv-python: {e}. Frame will not be rendered.")
        return frame

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
