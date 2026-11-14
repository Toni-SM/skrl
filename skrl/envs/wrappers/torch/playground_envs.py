from __future__ import annotations

from typing import Any

import math
from collections.abc import Mapping
import gymnasium

import numpy as np
import torch

from skrl import config, logger
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.framework.torch.types import Tensor
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space


def _spec_to_space(spec: Any) -> gymnasium.spaces.Box:
    """Convert a batched (first dimension is the number of environments) ``dm_env`` spec to a Gymnasium space.

    :param spec: Batched ``dm_env`` array spec (e.g.: ``Array`` or ``BoundedArray``).

    :return: Gymnasium Box space (without the batch dimension).
    """
    dtype = np.dtype(spec.dtype)
    limits = np.finfo(dtype) if np.issubdtype(dtype, np.floating) else np.iinfo(dtype)
    default_low, default_high = (-np.inf, np.inf) if np.issubdtype(dtype, np.floating) else (limits.min, limits.max)
    low = np.broadcast_to(getattr(spec, "minimum", default_low), spec.shape)[0].astype(dtype)
    high = np.broadcast_to(getattr(spec, "maximum", default_high), spec.shape)[0].astype(dtype)
    return gymnasium.spaces.Box(low=low, high=high, shape=spec.shape[1:], dtype=spec.dtype)


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """MuJoCo Playground environment wrapper.

        The observation and state spaces are defined by the environment's ``"policy"`` and ``"critic"``
        observation groups, respectively (other observation groups, e.g. ``"camera"``, are ignored).
        If the environment's action spec has more than one entry, the action space is the concatenation
        of the (flattened) entries, in the spec's order.

        .. note::

            The environment's auto-reset is disabled. Instead, the wrapper resets the environments
            whose episode ended right after stepping, so that the returned observations/states are
            the initial ones of the new episodes (Gym/VecEnv style auto-reset).

        :param env: The environment instance to wrap.
        """
        super().__init__(env)
        self._device = config.torch.parse_device(self._unwrapped.xpu_device)

        # MuJoCo Playground dependencies (imported here to not require them when importing this module)
        import dm_env
        from reaf.simulation.torch import environment as reaf_environment

        self._last_step_type = dm_env.StepType.LAST
        self._unwrapped.auto_reset_mode = reaf_environment.AutoResetMode.DISABLED

        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}

        # observation/state spaces
        observation_specs = self._unwrapped.timestep_spec().observation
        if not isinstance(observation_specs, Mapping):
            observation_specs = {"policy": observation_specs}
        self._observation_space = _spec_to_space(observation_specs["policy"])
        self._state_space = _spec_to_space(observation_specs["critic"]) if "critic" in observation_specs else None
        ignored_groups = [k for k in observation_specs if k not in ("policy", "critic")]
        if ignored_groups:
            logger.warning(
                f"MuJoCo Playground observation groups other than 'policy' (observations) and 'critic' (states) "
                f"are not supported. The following groups will be ignored: {', '.join(ignored_groups)}"
            )
        # action space
        action_specs = self._unwrapped.action_spec()
        self._action_spaces = {k: _spec_to_space(v) for k, v in action_specs.items()}
        if len(self._action_spaces) == 1:
            self._action_space = next(iter(self._action_spaces.values()))
        else:
            spaces = list(self._action_spaces.values())
            self._action_space = gymnasium.spaces.Box(
                low=np.concatenate([space.low.reshape(-1) for space in spaces]),
                high=np.concatenate([space.high.reshape(-1) for space in spaces]),
                dtype=np.result_type(*[space.dtype for space in spaces]),
            )
        self._action_sizes = [math.prod(space.shape) for space in self._action_spaces.values()]

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        return self._state_space

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        return self._action_space

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._unwrapped.nworld

    def _parse_observations(self, observations: Any) -> None:
        """Parse the environment's observation groups into the wrapper's observations and states.

        :param observations: Environment's observations (mapping of observation groups or a single tensor).
        """
        if not isinstance(observations, Mapping):
            observations = {"policy": observations}
        self._observations = flatten_tensorized_space(tensorize_space(self._observation_space, observations["policy"]))
        if self._state_space is not None:
            self._states = flatten_tensorized_space(tensorize_space(self._state_space, observations["critic"]))

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        split_actions = torch.split(actions, self._action_sizes, dim=-1)
        timestep = self._unwrapped.step(
            {k: v.reshape(-1, *space.shape) for (k, space), v in zip(self._action_spaces.items(), split_actions)}
        )

        reward = timestep.reward
        reward_terms = {}
        if isinstance(reward, Mapping):
            reward_terms = reward.get("terms", {})
            reward = reward["reward"]
        last = timestep.step_type == self._last_step_type
        terminated = last & (timestep.discount == 0)
        truncated = last & (timestep.discount > 0)

        self._info = {"reward_terms": reward_terms}
        # reset the environments whose episode ended
        if last.any():
            # get the (mean-reduced) episode metrics before resetting
            # (logged by the trainers under the default 'environment_info' key: "episode")
            self._info["episode"] = {
                f"Episode / {k}": v for k, v in self._unwrapped.task_layer.get_episode_metrics(last).items()
            }
            timestep = self._unwrapped.reset_with_options(
                options=self._unwrapped.default_reset_options(), env_mask=last
            )
        self._parse_observations(timestep.observation)

        return self._observations, reward.reshape(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[Tensor, dict[str, Any]]:
        """Reset the environment.

        .. note::

            Since the wrapper resets the environments whose episode ended right after stepping,
            the environment is reset only on the first call.

        :return: Observation, info.
        """
        if self._reset_once:
            timestep = self._unwrapped.reset()
            self._parse_observations(timestep.observation)
            self._info = {}
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Close the environment."""
        self._unwrapped.close()
