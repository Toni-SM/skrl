from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import jax
import jax.dlpack as jax_dlpack
import torch
import torch.utils.dlpack as torch_dlpack

from skrl import config, logger
from skrl.envs.wrappers.jax.base import Wrapper

# MuJoCo Playground runs on PyTorch: reuse its PyTorch wrapper and only convert inputs/outputs (avoid duplicate logic)
from skrl.envs.wrappers.torch.playground_envs import PlaygroundWrapper as _TorchPlaygroundWrapper
from skrl.utils.framework.jax.types import Tensor


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"
if _CPU:
    logger.warning("There is no GPU backend for JAX. JAX operations will run on CPU.")


def _jax2torch(array: jax.Array, device: torch.device) -> torch.Tensor:
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return torch_dlpack.from_dlpack(array).to(device=device)
    return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)


def _torch2jax(tensor: torch.Tensor | None) -> jax.Array | None:
    if tensor is None:
        return None
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return jax_dlpack.from_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous())
    return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """MuJoCo Playground environment wrapper.

        See :py:class:`~skrl.envs.wrappers.torch.playground_envs.PlaygroundWrapper` for more details.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._wrapper = _TorchPlaygroundWrapper(env)
        # the outputs (converted with DLPack) are placed on the environment device (or on CPU if JAX has no GPU backend)
        torch_device = self._wrapper.device
        self._device = config.jax.parse_device("cpu" if _CPU or torch_device.type == "cpu" else str(torch_device))
        self._observations = None
        self._states = None

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        return self._wrapper.state_space

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        return self._wrapper.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return self._wrapper.action_space

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._wrapper.num_envs

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, truncated, info = self._wrapper.step(
            _jax2torch(actions, self._wrapper.device)
        )
        self._observations = _torch2jax(observations)
        self._states = _torch2jax(self._wrapper.state())
        return (
            self._observations,
            _torch2jax(reward),
            _torch2jax(terminated.to(dtype=torch.int8)),
            _torch2jax(truncated.to(dtype=torch.int8)),
            info,
        )

    def state(self) -> Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[Tensor, Any]:
        """Reset the environment.

        :return: Observation, info.
        """
        observations, info = self._wrapper.reset()
        self._observations = _torch2jax(observations)
        self._states = _torch2jax(self._wrapper.state())
        return self._observations, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._wrapper.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._wrapper.close()
