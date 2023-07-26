from typing import Any, Optional, Tuple

import collections
import gym

import numpy as np
import torch

from skrl.envs.wrappers.torch.base import Wrapper


class DeepMindWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """DeepMind environment wrapper

        :param env: The environment to wrap
        :type env: Any supported DeepMind environment
        """
        super().__init__(env)

        from dm_env import specs
        self._specs = specs

        # observation and action spaces
        self._observation_space = self._spec_to_space(self._env.observation_spec())
        self._action_space = self._spec_to_space(self._env.action_spec())

    @property
    def state_space(self) -> gym.Space:
        """State space

        An alias for the ``observation_space`` property
        """
        return self._observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        return self._action_space

    def _spec_to_space(self, spec: Any) -> gym.Space:
        """Convert the DeepMind spec to a Gym space

        :param spec: The DeepMind spec to convert
        :type spec: Any supported DeepMind spec

        :raises: ValueError if the spec type is not supported

        :return: The Gym space
        :rtype: gym.Space
        """
        if isinstance(spec, self._specs.DiscreteArray):
            return gym.spaces.Discrete(spec.num_values)
        elif isinstance(spec, self._specs.BoundedArray):
            return gym.spaces.Box(shape=spec.shape,
                                  dtype=spec.dtype,
                                  low=spec.minimum if spec.minimum.ndim else np.full(spec.shape, spec.minimum),
                                  high=spec.maximum if spec.maximum.ndim else np.full(spec.shape, spec.maximum))
        elif isinstance(spec, self._specs.Array):
            return gym.spaces.Box(shape=spec.shape,
                                  dtype=spec.dtype,
                                  low=np.full(spec.shape, float("-inf")),
                                  high=np.full(spec.shape, float("inf")))
        elif isinstance(spec, collections.OrderedDict):
            return gym.spaces.Dict({k: self._spec_to_space(v) for k, v in spec.items()})
        else:
            raise ValueError(f"Spec type {type(spec)} not supported. Please report this issue")

    def _observation_to_tensor(self, observation: Any, spec: Optional[Any] = None) -> torch.Tensor:
        """Convert the DeepMind observation to a flat tensor

        :param observation: The DeepMind observation to convert to a tensor
        :type observation: Any supported DeepMind observation

        :raises: ValueError if the observation spec type is not supported

        :return: The observation as a flat tensor
        :rtype: torch.Tensor
        """
        spec = spec if spec is not None else self._env.observation_spec()

        if isinstance(spec, self._specs.DiscreteArray):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        elif isinstance(spec, self._specs.Array):  # includes BoundedArray
            return torch.tensor(observation, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        elif isinstance(spec, collections.OrderedDict):
            return torch.cat([self._observation_to_tensor(observation[k], spec[k]) \
                for k in sorted(spec.keys())], dim=-1).reshape(self.num_envs, -1)
        else:
            raise ValueError(f"Observation spec type {type(spec)} not supported. Please report this issue")

    def _tensor_to_action(self, actions: torch.Tensor) -> Any:
        """Convert the action to the DeepMind expected format

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raise ValueError: If the action space type is not supported

        :return: The action in the DeepMind expected format
        :rtype: Any supported DeepMind action
        """
        spec = self._env.action_spec()

        if isinstance(spec, self._specs.DiscreteArray):
            return np.array(actions.item(), dtype=spec.dtype)
        elif isinstance(spec, self._specs.Array):  # includes BoundedArray
            return np.array(actions.cpu().numpy(), dtype=spec.dtype).reshape(spec.shape)
        else:
            raise ValueError(f"Action spec type {type(spec)} not supported. Please report this issue")

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        timestep = self._env.step(self._tensor_to_action(actions))

        observation = timestep.observation
        reward = timestep.reward if timestep.reward is not None else 0
        terminated = timestep.last()
        truncated = False
        info = {}

        # convert response to torch
        return self._observation_to_tensor(observation), \
               torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1), \
               torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1), \
               torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1), \
               info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        timestep = self._env.reset()
        return self._observation_to_tensor(timestep.observation), {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment

        OpenCV is used to render the environment.
        Install OpenCV with ``pip install opencv-python``
        """
        frame = self._env.physics.render(480, 640, camera_id=0)

        # render the frame using OpenCV
        import cv2
        cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()
