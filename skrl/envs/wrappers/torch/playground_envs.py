from __future__ import annotations

from typing import Any

import gymnasium
import mujoco
from packaging import version

import jax
import jax.dlpack as jax_dlpack
import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack

from skrl import config, logger
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"
if _CPU:
    logger.warning("Isaac Lab runs on GPU, but there is no GPU backend for JAX. JAX operations will run on CPU.")


def _jax2torch(array, device):
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return torch_dlpack.from_dlpack(array).to(device=device)
    return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)


def _torch2jax(tensor):
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return jax_dlpack.from_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous())
    return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """MuJoCo Playground environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

        # build spaces
        observation_size = self._unwrapped.observation_size
        action_size = self._unwrapped.action_size
        if isinstance(observation_size, dict):
            self._observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=observation_size["state"])
            self._state_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=observation_size["privileged_state"]
            )
        else:
            self._observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,))
            self._state_space = None
        self._action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(action_size,))

        # set wrapped environment for just-in-time compilation with XLA
        self._env_state = None
        self._env_step = jax.jit(env.step)
        self._env_reset = jax.jit(env.reset)
        self._env_reset_key = jax.random.split(config.jax.key, self.num_envs)

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return self._action_space

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        return self._state_space

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = _torch2jax(unflatten_tensorized_space(self.action_space, actions))
        self._env_state = self._env_step(self._env_state, actions)

        if self._state_space is None:
            self._observations = flatten_tensorized_space(
                tensorize_space(self.observation_space, _jax2torch(self._env_state.obs, self.device))
            )
        else:
            self._observations = flatten_tensorized_space(
                tensorize_space(self.observation_space, _jax2torch(self._env_state.obs["state"], self.device))
            )
            self._states = flatten_tensorized_space(
                tensorize_space(self.state_space, _jax2torch(self._env_state.obs["privileged_state"], self.device))
            )
        reward = _jax2torch(self._env_state.reward, self.device)
        terminated = _jax2torch(self._env_state.done, self.device)
        truncated = self._env_state.info.get("truncation")
        if truncated is None:
            truncated = jnp.zeros_like(terminated)
        truncated = _jax2torch(truncated, self.device)
        info = self._env_state.info

        return self._observations, reward.reshape(-1, 1), terminated.reshape(-1, 1), truncated.reshape(-1, 1), info

    def state(self) -> torch.Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            self._env_state = self._env_reset(self._env_reset_key)
            if self._state_space is None:
                self._observations = flatten_tensorized_space(
                    tensorize_space(self.observation_space, _jax2torch(self._env_state.obs, self.device))
                )
            else:
                self._observations = flatten_tensorized_space(
                    tensorize_space(self.observation_space, _jax2torch(self._env_state.obs["state"], self.device))
                )
                self._states = flatten_tensorized_space(
                    tensorize_space(self.state_space, _jax2torch(self._env_state.obs["privileged_state"], self.device))
                )
            self._info = self._env_state.info
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        if self.num_envs > 1:
            logger.warning("Rendering is not supported for parallel environments. Rendering will be skipped")
            return

        # render frame
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        frame = self._unwrapped.render(self._env_state, width=640, height=480, scene_option=scene_option)
        # show rendered frame using OpenCV
        try:
            import cv2

            cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        except ImportError as e:
            logger.warning(f"Unable to import opencv-python: {e}. Frame will not be rendered.")
        return frame

    def close(self) -> None:
        """Close the environment."""
        pass
