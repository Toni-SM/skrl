from typing import List, Optional, Union

import contextlib
import copy
import tqdm

import jax.numpy as jnp

from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import Wrapper
from skrl.trainers.jax import Trainer


SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
}


class SequentialTrainer(Trainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Sequential trainer

        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.jax.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See SEQUENTIAL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """Train the agents sequentially

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_train()
            # multi-agent
            else:
                self.multi_agent_train()
            return

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with contextlib.nullcontext():
                actions = jnp.vstack([agent.act(states[scope[0]:scope[1]], timestep=timestep, timesteps=self.timesteps)[0] \
                                      for agent, scope in zip(self.agents, self.agents_scope)])

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            # render scene
            if not self.headless:
                self.env.render()

            # record the environments' transitions
            with contextlib.nullcontext():
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(states=states[scope[0]:scope[1]],
                                            actions=actions[scope[0]:scope[1]],
                                            rewards=rewards[scope[0]:scope[1]],
                                            next_states=next_states[scope[0]:scope[1]],
                                            terminated=terminated[scope[0]:scope[1]],
                                            truncated=truncated[scope[0]:scope[1]],
                                            infos=infos,
                                            timestep=timestep,
                                            timesteps=self.timesteps)

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            with contextlib.nullcontext():
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                else:
                    states = next_states

    def eval(self) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_eval()
            # multi-agent
            else:
                self.multi_agent_eval()
            return

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # compute actions
            with contextlib.nullcontext():
                actions = jnp.vstack([agent.act(states[scope[0]:scope[1]], timestep=timestep, timesteps=self.timesteps)[0] \
                                      for agent, scope in zip(self.agents, self.agents_scope)])

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            # render scene
            if not self.headless:
                self.env.render()

            with contextlib.nullcontext():
                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(states=states[scope[0]:scope[1]],
                                            actions=actions[scope[0]:scope[1]],
                                            rewards=rewards[scope[0]:scope[1]],
                                            next_states=next_states[scope[0]:scope[1]],
                                            terminated=terminated[scope[0]:scope[1]],
                                            truncated=truncated[scope[0]:scope[1]],
                                            infos=infos,
                                            timestep=timestep,
                                            timesteps=self.timesteps)
                    super(type(agent), agent).post_interaction(timestep=timestep, timesteps=self.timesteps)

                # reset environments
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                else:
                    states = next_states
