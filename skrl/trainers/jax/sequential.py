from typing import List, Optional, Union

import contextlib
import copy
import sys
import tqdm

import jax.numpy as jnp

from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.jax import MultiAgent
from skrl.trainers.jax import Trainer
from skrl.utils import ScopedTimer


# fmt: off
# [start-config-dict-jax]
SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-jax]
# fmt: on


class SequentialTrainer(Trainer):
    def __init__(
        self,
        *,
        env: Union[Wrapper, MultiAgentEnvWrapper],
        agents: Union[Agent, MultiAgent, List[Agent], List[MultiAgent]],
        scopes: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Sequential trainer.

        Train agents sequentially, i.e., one after the other, in each interaction with the environment.

        :param env: Environment to train/evaluate on.
        :param agents: Agent(s) to train/evaluate.
        :param scopes: Number of environments for each simultaneous agent to train/evaluate on.
        :param cfg: Configuration dictionary.
        """
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        scopes = scopes if scopes is not None else []
        super().__init__(env=env, agents=agents, scopes=scopes, cfg=_cfg)

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """Train agents sequentially.

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render environments
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(True)
        else:
            self.agents.enable_training_mode(True)

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            super().train()
            return

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with contextlib.nullcontext():
                # compute actions
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )
                        agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                    _actions.append(actions)
                    _outputs.append(outputs)
                actions = jnp.vstack(_actions)

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    elapsed_time_ms = timer.elapsed_time_ms
                    for agent in self.agents:
                        agent.track_data("Stats / Env stepping time (ms)", elapsed_time_ms)

                # render the environments
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]] if states is not None else None,
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            # - single environment
            else:
                raise RuntimeError("Sequential trainer is not supported for single environment")

    def eval(self) -> None:
        """Evaluate agents sequentially.

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions (sequentially)
        - Interact with the environments
        - Render environments
        - Record transitions
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(False)
        else:
            self.agents.enable_training_mode(False)

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            super().eval()
            return

        # reset the environments
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with contextlib.nullcontext():
                # compute actions
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )
                        agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                    _actions.append(actions if self.stochastic_evaluation else outputs.get("mean_actions", actions))
                    _outputs.append(outputs)
                actions = jnp.vstack(_actions)

                # step the environments
                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    elapsed_time_ms = timer.elapsed_time_ms
                    for agent in self.agents:
                        agent.track_data("Stats / Env stepping time (ms)", elapsed_time_ms)

                # render the environments
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]] if states is not None else None,
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

            # post-interaction
            for agent in self.agents:
                super(agent.__class__, agent).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            # - parallel/vectorized environments (single or multi-agent)
            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            # - single environment
            else:
                raise RuntimeError("Sequential trainer is not supported for single environment")
