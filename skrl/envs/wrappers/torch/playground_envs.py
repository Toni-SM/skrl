from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import jax
import jax.dlpack as jax_dlpack
import torch.utils.dlpack as torch_dlpack

from skrl import logger

# MuJoCo Playground runs on JAX: reuse its JAX wrapper and only convert inputs/outputs (avoid duplicate logic)
from skrl.envs.wrappers.jax.playground_envs import PlaygroundWrapper as _JaxPlaygroundWrapper
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.framework.torch.types import Device, Tensor


# ML frameworks conversion utilities
# jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: DLPack tensor is on GPU, but no GPU backend was provided.
_CPU = jax.devices()[0].device_kind.lower() == "cpu"
if _CPU:
    logger.warning("There is no GPU backend for JAX. MuJoCo Playground (JAX) operations will run on CPU.")


def _jax2torch(array: jax.Array | None, device: Device) -> Tensor | None:
    if array is None:
        return None
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return torch_dlpack.from_dlpack(array).to(device=device)
    return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)


def _torch2jax(tensor: Tensor) -> jax.Array:
    if version.parse(jax.__version__) >= version.parse("0.7.0"):
        return jax_dlpack.from_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous())
    return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous()))


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """MuJoCo Playground environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._wrapper = _JaxPlaygroundWrapper(env)
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
        observations: jax.Array,
        states: jax.Array | None,
        final_observations: jax.Array,
        final_states: jax.Array | None,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
        """Convert the JAX wrapper's observations/states and final observations/states.

        :param observations: Observations returned by the JAX wrapper's :py:meth:`step`/:py:meth:`reset`.
        :param states: States returned by the JAX wrapper's :py:meth:`state`.
        :param final_observations: Final observations returned by the JAX wrapper's :py:meth:`final`.
        :param final_states: Final states returned by the JAX wrapper's :py:meth:`final`.

        :return: Observation, state, final observation, final state.
        """
        converted_observations = _jax2torch(observations, self.device)
        converted_states = _jax2torch(states, self.device)
        # avoid converting the same array twice when no episode ended
        return (
            converted_observations,
            converted_states,
            (
                converted_observations
                if final_observations is observations
                else _jax2torch(final_observations, self.device)
            ),
            converted_states if final_states is states else _jax2torch(final_states, self.device),
        )

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        observations, reward, terminated, truncated, info = self._wrapper.step(_torch2jax(actions))
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return (
            self._observations,
            _jax2torch(reward, self.device),
            _jax2torch(terminated, self.device).bool(),
            _jax2torch(truncated, self.device).bool(),
            info,
        )

    def state(self) -> Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        observations, info = self._wrapper.reset()
        self._observations, self._states, self._final_observations, self._final_states = self._convert(
            observations, self._wrapper.state(), *self._wrapper.final()
        )
        return self._observations, info

    def final(self) -> tuple[Tensor, Tensor | None]:
        """Get the observations and states with which the last transitions ended.

        The final observations/states of the auto-reset environments are read from the final observation group,
        provided in the info under the first available key: ``"final_obs"``, ``"final_observation"`` or
        ``"final_observations"``. The final observation group must have the same structure as the environment's
        observation group (i.e.: including the ``"privileged_state"`` key, if the environment provides states).

        .. note::

            MuJoCo Playground's auto-reset wrapper (``wrap_for_brax_training``) does not provide the final
            observation group. In that case, the final observations/states are the ones returned by
            :py:meth:`step`/:py:meth:`state` (i.e.: the initial ones of the new episodes for the reset environments).

        See :py:meth:`~skrl.envs.wrappers.torch.base.Wrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._wrapper.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._wrapper.close()
