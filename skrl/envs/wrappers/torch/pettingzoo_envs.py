from __future__ import annotations

from typing import Any

from collections.abc import Mapping

import torch

from skrl import config
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper
from skrl.utils.framework.torch.types import Tensor
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)


class PettingZooWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """PettingZoo (Parallel API) environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = config.torch.key
        self._observations = {}
        self._states = {}
        self._final_observations = {}
        self._final_states = {}

    def _get_state(self) -> Any | None:
        """Get the (raw) state from the environment.

        In PettingZoo, the state is a global view of the environment, so it is the same for all agents.

        :return: State, or ``None`` if the environment does not provide it.
        """
        if next(iter(self.state_spaces.values()), None) is None:
            return None
        try:
            return self._env.state()
        except NotImplementedError:  # environment without state
            return None

    def _parse_group(
        self, *, observations: dict[str, Any], states: Any | None
    ) -> tuple[dict[str, Tensor], dict[str, Tensor | None]]:
        """Parse the observations and states from the environment's output.

        :param observations: Observations returned by the environment.
        :param states: State returned by the environment (see :py:meth:`_get_state`). It is shared by all agents.

        :return: Observation, state.
        """
        observations = {
            uid: flatten_tensorized_space(tensorize_space(self.observation_spaces[uid], value, device=self.device))
            for uid, value in observations.items()
        }
        if states is not None:
            states = flatten_tensorized_space(
                tensorize_space(next(iter(self.state_spaces.values())), states, device=self.device)
            )
        return observations, {uid: states for uid in self.possible_agents}

    def _parse_final_group(
        self,
        *,
        observations: dict[str, Tensor],
        states: dict[str, Tensor | None],
        terminated: dict[str, Tensor],
        truncated: dict[str, Tensor],
        info: Any,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor | None]]:
        """Parse the final observations and states from the environment's output.

        The final observation/state of each agent is read from the agent's info (see :py:meth:`_get_final_info`).
        The current observations/states are used if the final values are not available.

        :param observations: Current observations (already parsed, see :py:meth:`_parse_group`).
        :param states: Current states (already parsed, see :py:meth:`_parse_group`).
        :param terminated: Terminated flags (already converted).
        :param truncated: Truncated flags (already converted).
        :param info: Infos returned by the environment (per-agent info).

        :return: Final observation, state.
        """
        final_observations, final_states = dict(observations), dict(states)
        final_observation_values, final_state_values = self._get_final_info(info)
        for uid in self.possible_agents:
            final_observation, final_state = final_observation_values[uid], final_state_values[uid]
            if final_observation is not None:
                final_observations[uid] = flatten_tensorized_space(
                    tensorize_space(self.observation_spaces[uid], final_observation, device=self.device)
                )
            if final_state is not None and self.state_spaces[uid] is not None:
                final_states[uid] = flatten_tensorized_space(
                    tensorize_space(self.state_spaces[uid], final_state, device=self.device)
                )
        return final_observations, final_states

    def step(
        self, actions: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = {
            uid: untensorize_space(self.action_spaces[uid], unflatten_tensorized_space(self.action_spaces[uid], action))
            for uid, action in actions.items()
        }
        observations, rewards, terminated, truncated, infos = self._env.step(actions)

        # convert response to torch
        self._observations, self._states = self._parse_group(observations=observations, states=self._get_state())
        rewards = {
            uid: torch.tensor(value, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
            for uid, value in rewards.items()
        }
        terminated = {
            uid: torch.tensor(value, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
            for uid, value in terminated.items()
        }
        truncated = {
            uid: torch.tensor(value, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
            for uid, value in truncated.items()
        }
        self._final_observations, self._final_states = self._parse_final_group(
            observations=self._observations, states=self._states, terminated=terminated, truncated=truncated, info=infos
        )
        return self._observations, rewards, terminated, truncated, infos

    def state(self) -> dict[str, Tensor | None]:
        """Get the environment state.

        In PettingZoo, the state is a global view of the environment, so it is the same for all agents.
        The state is computed at the last :py:meth:`reset`/:py:meth:`step` call.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        outputs = self._env.reset(seed=self._seed)
        if isinstance(outputs, Mapping):
            observations = outputs
            infos = {uid: {} for uid in self.possible_agents}
        else:
            observations, infos = outputs
        self._seed = None

        self._observations, self._states = self._parse_group(observations=observations, states=self._get_state())
        self._final_observations, self._final_states = self._observations, self._states
        return self._observations, infos

    def final(self) -> tuple[dict[str, Tensor], dict[str, Tensor | None]]:
        """Get the observations and states with which the last transitions ended.

        The final observation/state of each agent is read from the agent's info under the first available info key:
        ``"final_obs"``, ``"final_observation"`` or ``"final_observations"`` for observations, and ``"final_state"`` or
        ``"final_states"`` for states. If not available, the values returned by :py:meth:`step`/:py:meth:`state` are
        used.

        See :py:meth:`~skrl.envs.wrappers.torch.base.MultiAgentEnvWrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
