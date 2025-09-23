from typing import Any, List, Optional, Tuple, Union

import copy
import dataclasses
import sys
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.torch import MultiAgent
from skrl.trainers.torch import Trainer, TrainerCfg
from skrl.utils import ScopedTimer


@dataclasses.dataclass(kw_only=True)
class StepTrainerCfg(TrainerCfg):
    """Configuration for the step trainer."""


class StepTrainer(Trainer):
    def __init__(
        self,
        *,
        env: Union[Wrapper, MultiAgentEnvWrapper],
        agents: Union[Agent, MultiAgent, List[Agent], List[MultiAgent]],
        scopes: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Step-by-step trainer.

        Train agents by controlling the training/evaluation loop step by step.

        :param env: Environment to train/evaluate on.
        :param agents: Agent(s) to train/evaluate.
        :param scopes: Number of environments for each simultaneous agent to train/evaluate on.
        :param cfg: Configuration dictionary.
        """
        self.cfg: StepTrainerCfg
        super().__init__(
            env=env,
            agents=agents,
            scopes=scopes if scopes is not None else [],
            cfg=StepTrainerCfg(**cfg) if isinstance(cfg, dict) else cfg,
        )

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
        """Execute a training iteration.

        This method executes the following steps once:

        - Pre-interaction (sequentially if ``num_simultaneous_agents > 1``)
        - Compute actions (sequentially if ``num_simultaneous_agents > 1``)
        - Interact with the environments
        - Render environments
        - Record transitions (sequentially if ``num_simultaneous_agents > 1``)
        - Post-interaction (sequentially if n``um_simultaneous_agents > 1``)
        - Reset environments

        :param timestep: Current timestep. If None, the current timestep will be carried by an internal variable.
        :param timesteps: Total number of timesteps. If None, it is obtained from the trainer's config.

        :return: Environment's observations, rewards, terminated, truncated and info.
        """
        if timestep is None:
            self._timestep += 1
            timestep = self._timestep
        timesteps = timesteps if timesteps is not None else self.cfg.timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps, disable=self.cfg.disable_progressbar, file=sys.stdout)
        self._progress.update(n=1)

        # hack to simplify calls
        if not isinstance(self.agents, (tuple, list)):
            self.agents = [self.agents]

        # set mode
        for agent in self.agents:
            agent.enable_training_mode(True)

        # reset the environments
        if self.observations is None:
            self.observations, infos = self.env.reset()
            self.states = self.env.state()

        # pre-interaction
        for agent in self.agents:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            # compute actions
            _actions, _outputs = [], []
            for agent, scope in zip(self.agents, self.scopes):
                with ScopedTimer() as timer:
                    actions, outputs = agent.act(
                        self.observations[scope[0] : scope[1]],
                        self.states[scope[0] : scope[1]] if self.states is not None else None,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )
                    agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                _actions.append(actions)
                _outputs.append(outputs)
            actions = torch.vstack(_actions)

            # step the environments
            with ScopedTimer() as timer:
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()
                elapsed_time_ms = timer.elapsed_time_ms
                for agent in self.agents:
                    agent.track_data("Stats / Env stepping time (ms)", elapsed_time_ms)

            # render the environments
            if not self.cfg.headless:
                self.env.render()

            # record the environments' transitions
            for agent, scope in zip(self.agents, self.scopes):
                agent.record_transition(
                    observations=self.observations[scope[0] : scope[1]],
                    states=self.states[scope[0] : scope[1]] if self.states is not None else None,
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_observations=next_observations[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )

            # log environment info
            if self.cfg.environment_info in infos:
                for k, v in infos[self.cfg.environment_info].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        for agent in self.agents:
                            agent.track_data(k if "/" in k else f"Info / {k}", v.item())

        # post-interaction
        for agent in self.agents:
            agent.post_interaction(timestep=timestep, timesteps=timesteps)

        # reset environments
        # - parallel/vectorized environments (single or multi-agent)
        if self.env.num_envs > 1:
            self.observations = next_observations
            self.states = next_states
        # - single environment
        else:
            # check condition to reset
            # - multi-agent
            if self.env.num_agents > 1:
                should_reset = not self.env.agents
            # - single-agent
            else:
                should_reset = terminated.any() or truncated.any()
            # explicit reset
            if should_reset:
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
        """Execute an evaluation iteration.

        This method executes the following steps in loop:

        - Pre-interaction (sequentially if ``num_simultaneous_agents > 1``)
        - Compute actions (sequentially if ``num_simultaneous_agents > 1``)
        - Interact with the environments
        - Render environments
        - Record transitions (sequentially if ``num_simultaneous_agents > 1``)
        - Reset environments

        :param timestep: Current timestep. If None, the current timestep will be carried by an internal variable.
        :param timesteps: Total number of timesteps. If None, it is obtained from the trainer's config.

        :return: Environment's observations, rewards, terminated, truncated and info.
        """
        if timestep is None:
            self._timestep += 1
            timestep = self._timestep
        timesteps = timesteps if timesteps is not None else self.cfg.timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps, disable=self.cfg.disable_progressbar, file=sys.stdout)
        self._progress.update(n=1)

        # hack to simplify calls
        if not isinstance(self.agents, (tuple, list)):
            self.agents = [self.agents]

        # set mode
        for agent in self.agents:
            agent.enable_training_mode(False)

        # reset the environments
        if self.observations is None:
            self.observations, infos = self.env.reset()
            self.states = self.env.state()

        # pre-interaction
        for agent in self.agents:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            # compute actions
            _actions, _outputs = [], []
            for agent, scope in zip(self.agents, self.scopes):
                with ScopedTimer() as timer:
                    actions, outputs = agent.act(
                        self.observations[scope[0] : scope[1]],
                        self.states[scope[0] : scope[1]] if self.states is not None else None,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )
                    agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                _actions.append(actions if self.cfg.stochastic_evaluation else outputs.get("mean_actions", actions))
                _outputs.append(outputs)
            actions = torch.vstack(_actions)

            # step the environments
            with ScopedTimer() as timer:
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()
                elapsed_time_ms = timer.elapsed_time_ms
                for agent in self.agents:
                    agent.track_data("Stats / Env stepping time (ms)", elapsed_time_ms)

            # render the environments
            if not self.cfg.headless:
                self.env.render()

            # write data to TensorBoard
            for agent, scope in zip(self.agents, self.scopes):
                agent.record_transition(
                    observations=self.observations[scope[0] : scope[1]],
                    states=self.states[scope[0] : scope[1]] if self.states is not None else None,
                    actions=actions[scope[0] : scope[1]],
                    rewards=rewards[scope[0] : scope[1]],
                    next_observations=next_observations[scope[0] : scope[1]],
                    next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                    terminated=terminated[scope[0] : scope[1]],
                    truncated=truncated[scope[0] : scope[1]],
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.cfg.timesteps,
                )

            # log environment info
            if self.cfg.environment_info in infos:
                for k, v in infos[self.cfg.environment_info].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        for agent in self.agents:
                            agent.track_data(k if "/" in k else f"Info / {k}", v.item())

        # post-interaction
        for agent in self.agents:
            super(agent.__class__, agent).post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

        # reset environments
        # - parallel/vectorized environments (single or multi-agent)
        if self.env.num_envs > 1:
            self.observations = next_observations
            self.states = next_states
        # - single environment
        else:
            # check condition to reset
            # - multi-agent
            if self.env.num_agents > 1:
                should_reset = not self.env.agents
            # - single-agent
            else:
                should_reset = terminated.any() or truncated.any()
            # explicit reset
            if should_reset:
                with torch.no_grad():
                    self.observations, infos = self.env.reset()
                    self.states = self.env.state()
            else:
                self.observations = next_observations
                self.states = next_states

        return next_observations, rewards, terminated, truncated, infos

    def reset(self):
        """Reset the trainer."""
        self._timestep = 0
        self._progress = None

        self.observations = None
        self.states = None
