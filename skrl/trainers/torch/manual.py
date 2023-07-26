from typing import List, Optional, Union

import copy
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer


MANUAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
}


class ManualTrainer(Trainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Manual trainer

        Train agents by manually controlling the training/evaluation loop

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See MANUAL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(MANUAL_TRAINER_DEFAULT_CONFIG)
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

        self.states = None

    def train(self, timestep: Optional[int] = None, timesteps: Optional[int] = None) -> None:
        """Execute a training iteration

        This method executes the following steps once:

        - Pre-interaction (sequentially if num_simultaneous_agents > 1)
        - Compute actions (sequentially if num_simultaneous_agents > 1)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially if num_simultaneous_agents > 1)
        - Post-interaction (sequentially if num_simultaneous_agents > 1)
        - Reset environments

        :param timestep: Current timestep (default: ``None``).
                         If None, the current timestep will be carried by an internal variable
        :type timestep: int, optional
        :param timesteps: Total number of timesteps (default: ``None``).
                          If None, the total number of timesteps is obtained from the trainer's config
        :type timesteps: int, optional
        """
        if timestep is None:
            self._timestep += 1
            timestep = self._timestep
        timesteps = self.timesteps if timesteps is None else timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps, disable=self.disable_progressbar)
        self._progress.update(n=1)

        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # reset env
        if self.states is None:
            self.states, infos = self.env.reset()

        if self.num_simultaneous_agents == 1:
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=timesteps)

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(self.states, timestep=timestep, timesteps=timesteps)[0]

        else:
            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=timesteps)

            # compute actions
            with torch.no_grad():
                actions = torch.vstack([agent.act(self.states[scope[0]:scope[1]], timestep=timestep, timesteps=timesteps)[0] \
                                        for agent, scope in zip(self.agents, self.agents_scope)])

        with torch.no_grad():
            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            # render scene
            if not self.headless:
                self.env.render()

        if self.num_simultaneous_agents == 1:
            # record the environments' transitions
            with torch.no_grad():
                self.agents.record_transition(states=self.states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=timesteps)

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=timesteps)

        else:
            # record the environments' transitions
            with torch.no_grad():
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(states=self.states[scope[0]:scope[1]],
                                            actions=actions[scope[0]:scope[1]],
                                            rewards=rewards[scope[0]:scope[1]],
                                            next_states=next_states[scope[0]:scope[1]],
                                            terminated=terminated[scope[0]:scope[1]],
                                            truncated=truncated[scope[0]:scope[1]],
                                            infos=infos,
                                            timestep=timestep,
                                            timesteps=timesteps)

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=timesteps)

        # reset environments
        with torch.no_grad():
            if terminated.any() or truncated.any():
                self.states, infos = self.env.reset()
            else:
                self.states = next_states

    def eval(self, timestep: Optional[int] = None, timesteps: Optional[int] = None) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially if num_simultaneous_agents > 1)
        - Interact with the environments
        - Render scene
        - Reset environments

        :param timestep: Current timestep (default: ``None``).
                         If None, the current timestep will be carried by an internal variable
        :type timestep: int, optional
        :param timesteps: Total number of timesteps (default: ``None``).
                          If None, the total number of timesteps is obtained from the trainer's config
        :type timesteps: int, optional
        """
        if timestep is None:
            self._timestep += 1
            timestep = self._timestep
        timesteps = self.timesteps if timesteps is None else timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps, disable=self.disable_progressbar)
        self._progress.update(n=1)

        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        # reset env
        if self.states is None:
            self.states, infos = self.env.reset()

        with torch.no_grad():
            if self.num_simultaneous_agents == 1:
                # compute actions
                actions = self.agents.act(self.states, timestep=timestep, timesteps=timesteps)[0]

            else:
                # compute actions
                actions = torch.vstack([agent.act(self.states[scope[0]:scope[1]], timestep=timestep, timesteps=timesteps)[0] \
                                        for agent, scope in zip(self.agents, self.agents_scope)])

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            # render scene
            if not self.headless:
                self.env.render()

            if self.num_simultaneous_agents == 1:
                # write data to TensorBoard
                self.agents.record_transition(states=self.states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=timesteps)
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=timesteps)

            else:
                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(states=self.states[scope[0]:scope[1]],
                                            actions=actions[scope[0]:scope[1]],
                                            rewards=rewards[scope[0]:scope[1]],
                                            next_states=next_states[scope[0]:scope[1]],
                                            terminated=terminated[scope[0]:scope[1]],
                                            truncated=truncated[scope[0]:scope[1]],
                                            infos=infos,
                                            timestep=timestep,
                                            timesteps=timesteps)
                    super(type(agent), agent).post_interaction(timestep=timestep, timesteps=timesteps)

            # reset environments
            if terminated.any() or truncated.any():
                self.states, infos = self.env.reset()
            else:
                self.states = next_states
