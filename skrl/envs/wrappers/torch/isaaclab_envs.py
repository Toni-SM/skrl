from __future__ import annotations

from typing import Any

import gymnasium

import torch

from skrl import config
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper, Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper.

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

    def _parse_group(self, group: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Parse the observations and states from the environment's observation groups.

        :param group: Observation group (keys: ``"policy"`` and, optionally, ``"critic"``).

        :return: Observation, state (``None`` if the ``"critic"`` key is not provided).
        """
        observations = flatten_tensorized_space(tensorize_space(self.observation_space, group["policy"]))
        states = group.get("critic")
        if states is not None:
            states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        return observations, states

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        with torch.no_grad():
            group, reward, terminated, truncated, self._info = self._env.step(actions)
        self._observations, self._states = self._parse_group(group)
        # final observations/states (info is not cleared between steps)
        self._final_observations = self._observations
        self._final_states = self._states
        final_group = self._info.get("final_obs")
        if final_group is not None:
            dones = (terminated | truncated).view(-1, 1)
            if dones.any():
                final_observations, final_states = self._parse_group(final_group)
                self._final_observations = torch.where(dones, final_observations, self._observations)
                if final_states is not None:
                    self._final_states = torch.where(dones, final_states, self._states)
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

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
            group, self._info = self._env.reset(seed=self._seed)
            self._observations, self._states = self._parse_group(group)
            self._reset_once = False
            self._seed = None
        self._final_observations = self._observations
        self._final_states = self._states
        return self._observations, self._info

    def final(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get the observations and states with which the last transitions ended.

        The final observations/states are read from the info (``"final_obs"`` key) when the environment
        computes them (i.e.: ``compute_final_obs=True`` in the environment configuration).

        See :py:meth:`~skrl.envs.wrappers.torch.base.Wrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

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

        self._seed = config.torch.key
        self._reset_once = True
        self._observations = None
        self._states = None
        self._final_observations = None
        self._final_states = None
        self._info = {}

    def _get_states(self) -> dict[str, torch.Tensor | None]:
        """Get the environment state (shared by all agents) from the wrapped environment.

        :return: State.
        """
        try:
            state = self._env.state()
        except AttributeError:  # 'OrderEnforcing' object has no attribute 'state'
            state = self._unwrapped.state()
        if state is not None:
            state = flatten_tensorized_space(tensorize_space(next(iter(self.state_spaces.values())), state))
        return {uid: state for uid in self.possible_agents}

    def step(self, actions: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = {k: unflatten_tensorized_space(self.action_spaces[k], v) for k, v in actions.items()}
        with torch.no_grad():
            observations, rewards, terminated, truncated, self._info = self._env.step(actions)
        self._observations = {
            k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v)) for k, v in observations.items()
        }
        self._states = self._get_states()
        # final observations/states (info is not cleared between steps)
        # - an environment is reset when all its agents are terminated or truncated
        # - the final states are not provided by the environment
        self._final_observations = self._observations
        self._final_states = self._states
        if any("final_obs" in self._info.get(k, {}) for k in self._observations):
            dones = (
                torch.stack([v.view(-1) for v in terminated.values()]).all(dim=0)
                | torch.stack([v.view(-1) for v in truncated.values()]).all(dim=0)
            ).view(-1, 1)
            if dones.any():
                self._final_observations = {
                    k: (
                        torch.where(
                            dones,
                            flatten_tensorized_space(
                                tensorize_space(self.observation_spaces[k], self._info[k]["final_obs"])
                            ),
                            v,
                        )
                        if "final_obs" in self._info.get(k, {})
                        else v
                    )
                    for k, v in self._observations.items()
                }
        return (
            self._observations,
            {k: v.view(-1, 1) for k, v in rewards.items()},
            {k: v.view(-1, 1) for k, v in terminated.items()},
            {k: v.view(-1, 1) for k, v in truncated.items()},
            self._info,
        )

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations = {
                k: flatten_tensorized_space(tensorize_space(self.observation_spaces[k], v))
                for k, v in observations.items()
            }
            self._states = self._get_states()
            self._reset_once = False
            self._seed = None
        self._final_observations = self._observations
        self._final_states = self._states
        return self._observations, self._info

    def state(self) -> dict[str, torch.Tensor | None]:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def final(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        """Get the observations and states with which the last transitions ended.

        The final observations are read from each agent's info (``"final_obs"`` key) when the environment
        computes them (i.e.: ``compute_final_obs=True`` in the environment configuration). The final state is
        not provided by the environment, so it is the same as the one returned by :py:meth:`state`.

        See :py:meth:`~skrl.envs.wrappers.torch.base.MultiAgentEnvWrapper.final` for more details.

        :return: Final observation, state.
        """
        return self._final_observations, self._final_states

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
