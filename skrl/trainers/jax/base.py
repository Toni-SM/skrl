from typing import List, Optional, Union

import atexit
import contextlib
import sys
from abc import ABC
import tqdm

from skrl import config, logger
from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.jax import MultiAgent
from skrl.utils import ScopedTimer


def generate_equally_spaced_scopes(*, num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for simultaneous agents.

    :param num_envs: Number of environments.
    :param num_simultaneous_agents: Number of simultaneous agents.

    :return: List of equally spaced scopes.

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments.
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
        *,
        env: Union[Wrapper, MultiAgentEnvWrapper],
        agents: Union[Agent, MultiAgent, List[Agent], List[MultiAgent]],
        scopes: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base trainer class for implementing custom trainers.

        :param env: Environment to train/evaluate on.
        :param agents: Agent(s) to train/evaluate.
        :param scopes: Number of environments for each simultaneous agent to train/evaluate on.
        :param cfg: Configuration dictionary.
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.scopes = scopes if scopes is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")
        self.stochastic_evaluation = self.cfg.get("stochastic_evaluation", False)

        self.initial_timestep = 0

        # setup agents
        self.num_simultaneous_agents = 1
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:

            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.jax.is_distributed:
            if config.jax.rank:
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
            for agent, scope in zip(self.agents, self.scopes):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup simultaneous agents.

        :raises ValueError: Invalid setup.
        """
        # validate agents and their scopes
        if isinstance(self.agents, (tuple, list)):
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.scopes = [(0, self.env.num_envs)]
            # simultaneous agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.scopes):
                    logger.warning("The agents' scopes are empty. They will be generated to be as equal as possible")
                    self.scopes = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.scopes):
                        self.scopes[-1] += self.env.num_envs - sum(self.scopes)
                    else:
                        raise ValueError(
                            f"The number of simultaneous agents ({len(self.agents)}) is greater than "
                            f"the number of parallelizable environments ({self.env.num_envs})"
                        )
                elif len(self.scopes) != len(self.agents):
                    raise ValueError(
                        f"The number of simultaneous agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.scopes)})"
                    )
                elif sum(self.scopes) != self.env.num_envs:
                    raise ValueError(
                        f"The scopes ({sum(self.scopes)}) don't cover the number of parallelizable environments ({self.env.num_envs})"
                    )
                # generate agents' scopes
                index = 0
                for i in range(len(self.scopes)):
                    index += self.scopes[i]
                    self.scopes[i] = (index - self.scopes[i], index)
            else:
                raise ValueError("A list of agents is expected")
        # non-simultaneous agent
        else:
            self.num_simultaneous_agents = 1
            self.scopes = [(0, self.env.num_envs)]

    def train(self) -> None:
        """Train a single/multi-agent.

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render environments
        - Record transitions
        - Post-interaction
        - Reset environments

        :raises AssertionError: If the method is called in a simultaneous agents setup.
        """
        assert self.num_simultaneous_agents == 1, (
            "This method is not allowed for simultaneous agents. "
            "Inherit from `Trainer` and reimplement the `.train()` method instead."
        )

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with contextlib.nullcontext():
                # compute actions
                with ScopedTimer() as timer:
                    actions, outputs = self.agents.act(
                        observations, states, timestep=timestep, timesteps=self.timesteps
                    )
                    self.agents.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    self.agents.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                # render the environments
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

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
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
                    with contextlib.nullcontext():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states

    def eval(self) -> None:
        """Evaluate a single/multi-agent.

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render environments
        - Record transitions
        - Reset environments

        :raises AssertionError: If the method is called in a simultaneous agents setup.
        """
        assert self.num_simultaneous_agents == 1, (
            "This method is not allowed for simultaneous agents. "
            "Inherit from `Trainer` and reimplement the `.eval()` method instead."
        )

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with contextlib.nullcontext():
                # compute actions
                with ScopedTimer() as timer:
                    actions, outputs = self.agents.act(
                        observations, states, timestep=timestep, timesteps=self.timesteps
                    )
                    self.agents.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                actions = actions if self.stochastic_evaluation else outputs.get("mean_actions", actions)

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    self.agents.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                # render the environments
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

            # post-interaction
            super(self.agents.__class__, self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
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
                    with contextlib.nullcontext():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states
