from __future__ import annotations

from typing import Any

import gymnasium

import torch

from skrl import config
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.framework.torch.types import Tensor
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class ManiSkillWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """ManiSkill environment wrapper.

        :param env: The environment instance to wrap.
        """
        super().__init__(env)

        self._seed = config.torch.key
        self._reset_once = True
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None
        self._info = {}
        # whether the environment resets itself when the episodes end (e.g.: ``ManiSkillVectorEnv(auto_reset=True)``).
        # Look through the wrappers' ``env`` attribute, since vector wrappers do not forward attributes
        self._auto_reset = False
        wrapped_env = self._env
        while wrapped_env is not None:
            if hasattr(wrapped_env, "auto_reset"):
                self._auto_reset = bool(wrapped_env.auto_reset)
                break
            wrapped_env = getattr(wrapped_env, "env", None)

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

    def _get_state(self) -> Any | None:
        """Get the (raw) state from the environment.

        :return: State, or ``None`` since the wrapper does not provide states.
        """
        return None

    def _parse_group(self, *, observations: Any, states: Any | None) -> tuple[Tensor, Tensor | None]:
        """Parse the observations and states from the environment's output.

        :param observations: Observations returned by the environment.
        :param states: States returned by the environment (see :py:meth:`_get_state`).

        :return: Observation, state (``None`` if not provided).
        """
        observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations))
        if states is not None and self.state_space is not None:
            states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        else:
            states = None
        return observations, states

    def _parse_final_group(
        self, *, observations: Tensor, states: Tensor | None, terminated: Tensor, truncated: Tensor, info: Any
    ) -> tuple[Tensor, Tensor | None]:
        """Parse the final observations and states from the environment's output.

        The final observation is read from the info (see :py:meth:`_get_final_info`) when the environment is
        auto-reset (e.g.: ``ManiSkillVectorEnv(auto_reset=True)``), and it is used for the sub-environments whose
        episodes ended. Otherwise, the current observations (i.e.: reached before the wrapper resets the environments
        whose episodes ended) are used. The wrapper does not provide states, so the final states are the current states.

        :param observations: Current observations (already parsed, see :py:meth:`_parse_group`).
        :param states: Current states (already parsed, see :py:meth:`_parse_group`).
        :param terminated: Terminated flags.
        :param truncated: Truncated flags.
        :param info: Info returned by the environment.

        :return: Final observation, state.
        """
        dones = (terminated | truncated).flatten()
        if not dones.any():
            return observations, states
        final_observation = self._get_final_info(info)[0]
        if final_observation is not None:
            final_observation = flatten_tensorized_space(tensorize_space(self.observation_space, final_observation))
            return torch.where(dones.to(observations.device).view(-1, 1), final_observation, observations), states
        # clone the observations, since the environment may reuse its buffers when resetting
        return observations.clone(), states

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)

        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)
            self._observations, self._states = self._parse_group(observations=observations, states=self._get_state())
            self._final_observations, self._final_states = self._parse_final_group(
                observations=self._observations,
                states=self._states,
                terminated=terminated,
                truncated=truncated,
                info=self._info,
            )
            # reset the environments whose episodes ended (if the environment does not reset them itself)
            dones = (terminated | truncated).flatten()
            if not self._auto_reset and dones.any():
                env_idx = torch.arange(self.num_envs, device=dones.device)[dones]
                observations, self._info = self._env.reset(options={"env_idx": env_idx})
                self._observations, self._states = self._parse_group(
                    observations=observations, states=self._get_state()
                )
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations, self._states = self._parse_group(observations=observations, states=self._get_state())
            self._reset_once = False
            self._seed = None
        self._final_observations, self._final_states = self._observations, self._states
        return self._observations, self._info

    def final(self) -> tuple[Tensor, Tensor | None]:
        """Get the observations and states with which the last transitions ended.

        The final observation is read from the info under the first available key (``"final_obs"``,
        ``"final_observation"`` or ``"final_observations"``) when the environment is auto-reset (e.g.:
        ``ManiSkillVectorEnv(auto_reset=True)``). Otherwise, it is the observation reached before the wrapper resets
        the environments whose episodes ended. The wrapper does not provide states, so the final state is the same as
        the one returned by :py:meth:`state`.

        See :py:meth:`~skrl.envs.wrappers.torch.base.Wrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
