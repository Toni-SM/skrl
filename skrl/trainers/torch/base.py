from typing import List, Optional, Union

import atexit
import sys
from abc import ABC, abstractmethod
import tqdm

import torch

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper


def generate_equally_spaced_scopes(num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for multiple simultaneous agents.

    This function divides the number of environments equally among the agents.
    If the division is not exact, the remaining environments are assigned to the last agent.

    :param num_envs: Number of environments.
    :param num_simultaneous_agents: Number of simultaneous agents to divide environments between.

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments.

    :return: List of environment counts per agent.
    """
    scopes = [int(num_envs / num_simultaneous_agents)] * num_simultaneous_agents
    if sum(scopes):
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(
            f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})"
        )
    return scopes


class Trainer(ABC):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class for trainers.

        :param env: Environment to train/evaluate on.
        :param agents: Agent(s) to train/evaluate.
        :param agents_scope: Number of environments for each agent to train/evaluate on.
        :param cfg: Configuration dictionary.
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")
        self.stochastic_evaluation = self.cfg.get("stochastic_evaluation", False)

        self.initial_timestep = 0

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.torch.is_distributed:
            if config.torch.rank:
                self.disable_progressbar = True

    def __str__(self) -> str:
        """Generate a string representation of the trainer.

        :return: Representation of the trainer as string.
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training/evaluation.

        :raises ValueError: Invalid setup.
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # simultaneous agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(
                            f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})"
                        )
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(
                        f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})"
                    )
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(
                        f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})"
                    )
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    @abstractmethod
    def train(self) -> None:
        """Train the agents.

        :raises NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def eval(self) -> None:
        """Evaluate the agents.

        :raises NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def non_simultaneous_train(self) -> None:
        """Train non-simultaneous agent (single-agent or multi-agent).

        This method executes the following steps in loop (:guilabel:`timesteps` times).
        If :guilabel:`disable_progressbar` is false, a progress bar will be shown.

        .. list-table:: Steps
            :header-rows: 1

            * - Step
              - Call
            * - Pre-interaction
              - Agent: :func:`~skrl.agents.torch.base.Agent.pre_interaction`
            * - Compute actions
              - Agent: :func:`~skrl.agents.torch.base.Agent.act`
            * - Interact with the environment(s)
              - Environment: :func:`~skrl.envs.wrappers.torch.Wrapper.step`
            * - Render scene (if :guilabel:`headless` is false)
              - Environment: :func:`~skrl.envs.wrappers.torch.Wrapper.render`
            * - Record transitions
              - Agent: :func:`~skrl.agents.torch.base.Agent.record_transition`
            * - Log environment info (if :guilabel:`environment_info` is in ``info``)
              - Agent: :func:`~skrl.agents.torch.base.Agent.track_data`
            * - Post-interaction
              - Agent: :func:`~skrl.agents.torch.base.Agent.post_interaction`
            * - Reset environment(s)
              - Environment: :func:`~skrl.envs.wrappers.torch.Wrapper.reset`

        :raises AssertionError: If there are simultaneous agents.
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"

        # reset env
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = self.agents.act(observations, states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states

    def non_simultaneous_eval(self) -> None:
        """Evaluate non-simultaneous agent (single-agent or multi-agent).

        This method executes the following steps in loop (:guilabel:`timesteps` times).
        If :guilabel:`disable_progressbar` is false, a progress bar will be shown.

        .. list-table:: Steps
            :header-rows: 1

            * - Step
              - Call
            * - Pre-interaction
              - Agent: :func:`~skrl.agents.torch.base.Agent.pre_interaction`
            * - Compute (stochastic, if :guilabel:`stochastic_evaluation` is true) actions
              - Agent: :func:`~skrl.agents.torch.base.Agent.act`
            * - Interact with the environments
              - Environment: :func:`~skrl.envs.wrappers.torch.Wrapper.step`
            * - Render scene (if :guilabel:`headless` is false)
              - Environment: :func:`~skrl.envs.wrappers.torch.Wrapper.render`
            * - Record transitions
              - Agent: :func:`~skrl.agents.torch.base.Agent.record_transition`
            * - Log environment info (if :guilabel:`environment_info` is in ``info``)
              - Agent: :func:`~skrl.agents.torch.base.Agent.track_data`
            * - Post-interaction (TensorBoard data writing and checkpoint saving)
              - Agent (base): :func:`~skrl.agents.torch.base.Agent.post_interaction`
            * - Reset environment(s)
              - Environment: :func:`~skrl.envs.wrappers.torch.Wrapper.reset`

        :raises AssertionError: If there are simultaneous agents.
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"

        # reset env
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = self.agents.act(observations, states, timestep=timestep, timesteps=self.timesteps)
                actions = outputs[0] if self.stochastic_evaluation else outputs[-1].get("mean_actions", outputs[0])

                # step the environments
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states
