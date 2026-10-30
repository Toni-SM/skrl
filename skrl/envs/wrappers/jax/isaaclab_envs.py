from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import jax
import jax.dlpack as jax_dlpack
import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack

from skrl import config, logger
from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper, Wrapper

# Isaac Lab runs on PyTorch: reuse its PyTorch wrappers and only convert inputs/outputs (avoid duplicate logic)
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabMultiAgentWrapper as _TorchIsaacLabMultiAgentWrapper
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper as _TorchIsaacLabWrapper


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
    if tensor is None:
        return None
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return jax_dlpack.from_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous())
    return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._wrapper = _TorchIsaacLabWrapper(env)
        self._wrapper._seed = np.asarray(jax.device_get(config.jax.key)).sum().item()
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None

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

    def _convert(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        final_observations: torch.Tensor,
        final_states: torch.Tensor | None,
    ) -> tuple[jax.Array, jax.Array | None, jax.Array, jax.Array | None]:
        """Convert the PyTorch wrapper's observations/states and final observations/states.

        :param observations: Observations returned by the PyTorch wrapper's :py:meth:`step`/:py:meth:`reset`.
        :param states: States returned by the PyTorch wrapper's :py:meth:`state`.
        :param final_observations: Final observations returned by the PyTorch wrapper's :py:meth:`final`.
        :param final_states: Final states returned by the PyTorch wrapper's :py:meth:`final`.

        :return: Observation, state, final observation, final state.
        """
        converted_observations = _torch2jax(observations)
        converted_states = _torch2jax(states)
        # avoid converting the same tensor twice when no episode ended
        return (
            converted_observations,
            converted_states,
            converted_observations if final_observations is observations else _torch2jax(final_observations),
            converted_states if final_states is states else _torch2jax(final_states),
        )

    def step(self, actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, truncated, info = self._wrapper.step(
            _jax2torch(actions, self._wrapper.device)
        )
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return (
            self._observations,
            _torch2jax(reward),
            _torch2jax(terminated.to(dtype=torch.int8)),
            _torch2jax(truncated.to(dtype=torch.int8)),
            info,
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
        observations, info = self._wrapper.reset()
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return self._observations, info

    def final(self) -> tuple[jax.Array, jax.Array | None]:
        """Get the observations and states with which the last transitions ended.

        The final observations/states are read from the info (``"final_obs"`` key) when the environment
        computes them (i.e.: ``compute_final_obs=True`` in the environment configuration).

        See :py:meth:`~skrl.envs.wrappers.jax.base.Wrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return self._wrapper.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._wrapper.close()


class IsaacLabMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper for multi-agent implementation.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._wrapper = _TorchIsaacLabMultiAgentWrapper(env)
        self._wrapper._seed = np.asarray(jax.device_get(config.jax.key)).sum().item()
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None

    def _convert(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        final_observations: dict[str, torch.Tensor],
        final_states: dict[str, torch.Tensor | None],
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array | None], dict[str, jax.Array], dict[str, jax.Array | None]]:
        """Convert the PyTorch wrapper's observations/states and final observations/states.

        :param observations: Observations returned by the PyTorch wrapper's :py:meth:`step`/:py:meth:`reset`.
        :param states: States returned by the PyTorch wrapper's :py:meth:`state`.
        :param final_observations: Final observations returned by the PyTorch wrapper's :py:meth:`final`.
        :param final_states: Final states returned by the PyTorch wrapper's :py:meth:`final`.

        :return: Observation, state, final observation, final state.
        """
        converted_observations = {uid: _torch2jax(value) for uid, value in observations.items()}
        # the same state is shared by all agents: convert it once
        converted_state = _torch2jax(next(iter(states.values()), None))
        converted_states = {uid: converted_state for uid in states}
        # avoid converting the same tensor twice when no episode ended
        return (
            converted_observations,
            converted_states,
            {
                uid: converted_observations[uid] if value is observations[uid] else _torch2jax(value)
                for uid, value in final_observations.items()
            },
            (
                converted_states
                if final_states is states
                else {uid: _torch2jax(value) for uid, value in final_states.items()}
            ),
        )

    def step(
        self, actions: dict[str, jax.Array]
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array], dict[str, Any]]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, rewards, terminated, truncated, info = self._wrapper.step(
            {uid: _jax2torch(value, self._wrapper.device) for uid, value in actions.items()}
        )
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return (
            self._observations,
            {uid: _torch2jax(value) for uid, value in rewards.items()},
            {uid: _torch2jax(value.to(dtype=torch.int8)) for uid, value in terminated.items()},
            {uid: _torch2jax(value.to(dtype=torch.int8)) for uid, value in truncated.items()},
            info,
        )

    def reset(self) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        observations, info = self._wrapper.reset()
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return self._observations, info

    def state(self) -> dict[str, jax.Array | None]:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def final(self) -> tuple[dict[str, jax.Array], dict[str, jax.Array | None]]:
        """Get the observations and states with which the last transitions ended.

        The final observations are read from each agent's info (``"final_obs"`` key) when the environment
        computes them (i.e.: ``compute_final_obs=True`` in the environment configuration). The final state is
        not provided by the environment, so it is the same as the one returned by :py:meth:`state`.

        See :py:meth:`~skrl.envs.wrappers.jax.base.MultiAgentEnvWrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return self._wrapper.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._wrapper.close()
