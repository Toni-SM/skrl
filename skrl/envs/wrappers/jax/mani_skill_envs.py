from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import jax
import jax.dlpack as jax_dlpack
import numpy as np


try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except:
    pass  # TODO: show warning message
else:
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space

from skrl import logger
from skrl.envs.wrappers.jax.base import Wrapper


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"
if _CPU:
    logger.warning("ManiSkill runs on GPU, but there is no GPU backend for JAX. JAX operations will run on CPU.")


def _jax2torch(array, device):
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return torch_dlpack.from_dlpack(array).to(device=device)
    return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)


def _torch2jax(tensor):
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return jax_dlpack.from_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous())
    return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))


class ManiSkillWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """ManiSkill environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._env_device = torch.device(self._unwrapped.device)
        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        try:
            return self._unwrapped.single_state_space
        except AttributeError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        try:
            return self._unwrapped.single_observation_space
        except:
            return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = _jax2torch(actions, self._env_device)
        actions = unflatten_tensorized_space(self.action_space, actions)

        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)

            # auto-reset environments
            dones = (terminated | truncated).flatten()
            if dones.any():
                env_idx = torch.arange(self.num_envs, device=dones.device)[dones]
                observations, self._info = self._env.reset(options={"env_idx": env_idx})

        self._observations = _torch2jax(flatten_tensorized_space(tensorize_space(self.observation_space, observations)))
        terminated = terminated.to(dtype=torch.int8)
        truncated = truncated.to(dtype=torch.int8)

        return (
            self._observations,
            _torch2jax(reward.view(-1, 1)),
            _torch2jax(terminated.view(-1, 1)),
            _torch2jax(truncated.view(-1, 1)),
            self._info,
        )

    def state(self) -> jax.Array | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[jax.Array, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = _torch2jax(
                flatten_tensorized_space(tensorize_space(self.observation_space, observations))
            )
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
