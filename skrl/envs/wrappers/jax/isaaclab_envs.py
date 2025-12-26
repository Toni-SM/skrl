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

from skrl import config, logger
from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper, Wrapper


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


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = np.asarray(jax.device_get(config.jax.key)).sum().item()
        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

        if self._unwrapped:
            self._env_device = torch.device(self._unwrapped.device)

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        try:
            return self._unwrapped.single_observation_space["policy"]
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
        actions = unflatten_tensorized_space(self.action_space, _jax2torch(actions, self._env_device))

        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)

        self._observations = _torch2jax(
            flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))
        )
        terminated = terminated.to(dtype=torch.int8)
        truncated = truncated.to(dtype=torch.int8)
        states = observations.get("critic", None)
        if states is not None:
            self._states = _torch2jax(flatten_tensorized_space(tensorize_space(self.state_space, states)))

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
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations = _torch2jax(
                flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))
            )
            states = observations.get("critic", None)
            if states is not None:
                self._states = _torch2jax(flatten_tensorized_space(tensorize_space(self.state_space, states)))
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()


class IsaacLabMultiAgentWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper for multi-agent implementation.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = np.asarray(jax.device_get(config.jax.key)).sum().item()
        self._reset_once = True
        self._observations = None
        self._info = {}

        if self._unwrapped:
            self._env_device = torch.device(self._unwrapped.device)

    def step(
        self, actions: dict[str, jax.Array]
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array], dict[str, Any]]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = {uid: _jax2torch(value, self._env_device) for uid, value in actions.items()}
        actions = {k: unflatten_tensorized_space(self.action_spaces[k], v) for k, v in actions.items()}

        with torch.no_grad():
            observations, rewards, terminated, truncated, self._info = self._env.step(actions)
        observations = {
            k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v)) for k, v in observations.items()
        }

        self._observations = {uid: _torch2jax(value) for uid, value in observations.items()}
        return (
            self._observations,
            {uid: _torch2jax(value.view(-1, 1)) for uid, value in rewards.items()},
            {uid: _torch2jax(value.to(dtype=torch.int8).view(-1, 1)) for uid, value in terminated.items()},
            {uid: _torch2jax(value.to(dtype=torch.int8).view(-1, 1)) for uid, value in truncated.items()},
            self._info,
        )

    def reset(self) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            observations = {
                k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v))
                for k, v in observations.items()
            }
            self._observations = {uid: _torch2jax(value) for uid, value in observations.items()}
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def state(self) -> dict[jax.Array | None]:
        """Get the environment state.

        :return: State.
        """
        try:
            state = self._env.state()
        except AttributeError:  # 'OrderEnforcing' object has no attribute 'state'
            state = self._unwrapped.state()
        if state is not None:
            state = _torch2jax(flatten_tensorized_space(tensorize_space(next(iter(self.state_spaces.values())), state)))
        return {uid: state for uid in self.possible_agents}

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
