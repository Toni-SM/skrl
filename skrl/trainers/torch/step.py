from typing import Any, List, Optional, Tuple, Union

import copy
import sys
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer


# fmt: off
# [start-config-dict-torch]
STEP_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-torch]
# fmt: on


class StepTrainer(Trainer):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Trainer for training simultaneous and non-simultaneous agents step by step.

        Args:
            env: Environment to train/evaluate on.
            agents: Agent or simultaneous agents to train/evaluate.
            agents_scope: Optional list specifying number of environments for simultaneous agents.
                If not provided, environments will be divided equally among simultaneous agents.
            cfg: Trainer configuration dictionary.
                See :data:`~skrl.trainers.torch.step.STEP_TRAINER_DEFAULT_CONFIG` for default values.
        """
        _cfg = copy.deepcopy(STEP_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

        self._timestep = 0
        self._progress = None

        self.observations = None
        self.states = None

    def train(
        self, timestep: Optional[int] = None, timesteps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Execute one training loop iteration.

        This method executes the following steps on each call.
        If :guilabel:`disable_progressbar` is false, a progress bar will be shown.

        - Agent's pre-interaction
        - Compute actions
        - Interact with the environment(s)
        - Render scene (if :guilabel:`headless` is false)
        - Record environment transition(s) and agent data
        - Log environment info (if :guilabel:`environment_info` is in ``info``)
        - Agent's post-interaction
        - Reset environment(s)

        Args:
            timestep: Current timestep. If None, the current timestep will be carried by an internal variable.
            timesteps: Total number of timesteps. If None, the number of timesteps is obtained from the trainer's config.

        Returns:
            Observation, reward, terminated, truncated, info
        """
        if timestep is None:
            self._timestep += 1
            timestep = self._timestep
        timesteps = self.timesteps if timesteps is None else timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps, disable=self.disable_progressbar, file=sys.stdout)
        self._progress.update(n=1)

        # hack to simplify non-simultaneous agents handling
        if self.num_simultaneous_agents == 1:
            self.agents = [self.agents]

        # set running mode
        for agent in self.agents:
            agent.set_running_mode("train")

        # reset environments
        if self.observations is None:
            self.observations, infos = self.env.reset()
            self.states = self.env.state()

        # pre-interaction
        for agent in self.agents:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            # compute actions
            actions = torch.vstack(
                [
                    agent.act(
                        self.observations[scope[0] : scope[1]],
                        self.states[scope[0] : scope[1]],
                        timestep=timestep,
                        timesteps=timesteps,
                    )[0]
                    for agent, scope in zip(self.agents, self.agents_scope)
                ]
            )

            # step the environments
            next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
            next_states = self.env.state()

            # render scene
            if not self.headless:
                self.env.render()

            # record the environments' transitions
            for agent, scope in zip(self.agents, self.agents_scope):
                agent.record_transition(
                    observations=self.observations[scope[0] : scope[1]],
                    states=self.states[scope[0] : scope[1]],
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_observations=next_observations[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]],
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )

            # log environment info
            if self.environment_info in infos:
                for k, v in infos[self.environment_info].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        for agent in self.agents:
                            agent.track_data(f"Info / {k}", v.item())

        # post-interaction
        for agent in self.agents:
            agent.post_interaction(timestep=timestep, timesteps=timesteps)

        # reset environments
        if self.env.num_envs > 1:
            self.observations = next_observations
            self.states = next_states
        else:
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    self.observations, infos = self.env.reset()
                    self.states = self.env.state()
            else:
                self.observations = next_observations
                self.states = next_states

        return next_observations, rewards, terminated, truncated, infos

    def eval(
        self, timestep: Optional[int] = None, timesteps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Execute one evaluation loop iteration.

        This method executes the following steps on each call.
        If :guilabel:`disable_progressbar` is false, a progress bar will be shown.

        - Agent's pre-interaction
        - Compute actions (stochastic actions if :guilabel:`stochastic_evaluation` is true)
        - Interact with the environment(s)
        - Render scene (if :guilabel:`headless` is false)
        - Record environment transition(s)
        - Log environment info (if :guilabel:`environment_info` is in ``info``)
        - Agent's post-interaction (TensorBoard data writing and checkpoint saving)
        - Reset environment(s)

        Args:
            timestep: Current timestep. If None, the current timestep will be carried by an internal variable.
            timesteps: Total number of timesteps. If None, the number of timesteps is obtained from the trainer's config.

        Returns:
            Observation, reward, terminated, truncated, info
        """
        if timestep is None:
            self._timestep += 1
            timestep = self._timestep
        timesteps = self.timesteps if timesteps is None else timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps, disable=self.disable_progressbar, file=sys.stdout)
        self._progress.update(n=1)

        # hack to simplify code
        if self.num_simultaneous_agents == 1:
            self.agents = [self.agents]

        # set running mode
        for agent in self.agents:
            agent.set_running_mode("eval")

        # reset env
        if self.observations is None:
            self.observations, infos = self.env.reset()
            self.states = self.env.state()

        # pre-interaction
        for agent in self.agents:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            # compute actions
            outputs = [
                agent.act(
                    self.observations[scope[0] : scope[1]],
                    self.states[scope[0] : scope[1]],
                    timestep=timestep,
                    timesteps=timesteps,
                )
                for agent, scope in zip(self.agents, self.agents_scope)
            ]
            actions = torch.vstack(
                [
                    output[0] if self.stochastic_evaluation else output[-1].get("mean_actions", output[0])
                    for output in outputs
                ]
            )

            # step the environments
            next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
            next_states = self.env.state()

            # render scene
            if not self.headless:
                self.env.render()

            # write data to TensorBoard
            for agent, scope in zip(self.agents, self.agents_scope):
                agent.record_transition(
                    observations=self.observations[scope[0] : scope[1]],
                    states=self.states[scope[0] : scope[1]],
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_observations=next_observations[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]],
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )

            # log environment info
            if self.environment_info in infos:
                for k, v in infos[self.environment_info].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        for agent in self.agents:
                            agent.track_data(f"Info / {k}", v.item())

        # post-interaction (base class, TensorBoard data writing and checkpoint saving)
        for agent in self.agents:
            super(type(agent), agent).post_interaction(timestep=timestep, timesteps=timesteps)

        # reset environments
        if self.env.num_envs > 1:
            self.observations = next_observations
            self.states = next_states
        else:
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    self.observations, infos = self.env.reset()
                    self.states = self.env.state()
            else:
                self.observations = next_observations
                self.states = next_states

        return next_observations, rewards, terminated, truncated, infos
