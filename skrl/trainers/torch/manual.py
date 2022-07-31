from typing import Union, List, Optional

import tqdm

import torch

from ...envs.torch import Wrapper
from ...agents.torch import Agent

from . import Trainer


class ManualTrainer(Trainer):
    def __init__(self,
                 cfg: dict,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent], List[List[Agent]]],
                 agents_scope : List[int] = []) -> None:
        """Manual trainer

        Train agents by manually controlling the training/evaluation loop

        :param cfg: Configuration dictionary
        :type cfg: dict
        :param env: Environment to train on
        :type env: skrl.env.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: [])
        :type agents_scope: tuple or list of integers
        """
        # TODO: close the environment
        super().__init__(cfg, env, agents, agents_scope)

        # init agents
        if self.num_agents > 1:
            for agent in self.agents:
                agent.init()
        else:
            self.agents.init()

        self._progress = None

        self.states = None

    def train(self, timestep: int, timesteps: Optional[int] = None) -> None:
        """Execute a training iteration

        This method executes the following steps once:

        - Pre-interaction (sequentially if num_agents > 1)
        - Compute actions (sequentially if num_agents > 1)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially if num_agents > 1)
        - Post-interaction (sequentially if num_agents > 1)
        - Reset environments

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Total number of timesteps (default: None).
                          If None, the total number of timesteps is obtained from the trainer's config
        :type timesteps: int, optional
        """
        timesteps = self.timesteps if timesteps is None else timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps)
        self._progress.update(n=1)

        # reset env
        if self.states is None:
            self.states = self.env.reset()

        if self.num_agents == 1:
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=timesteps)

            # compute actions
            with torch.no_grad():
                actions, _, _ = self.agents.act(self.states, inference=True, timestep=timestep, timesteps=timesteps)

        else:
            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=timesteps)

            # compute actions
            with torch.no_grad():
                actions = torch.vstack([agent.act(self.states[scope[0]:scope[1]],
                                                  inference=True,
                                                  timestep=timestep,
                                                  timesteps=timesteps)[0] \
                                        for agent, scope in zip(self.agents, self.agents_scope)])

        # step the environments
        next_states, rewards, dones, infos = self.env.step(actions)

        # render scene
        if not self.headless:
            self.env.render()

        if self.num_agents == 1:
            # record the environments' transitions
            with torch.no_grad():
                self.agents.record_transition(states=self.states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              dones=dones,
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
                                            dones=dones[scope[0]:scope[1]],
                                            infos=infos,
                                            timestep=timestep,
                                            timesteps=timesteps)

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=timesteps)

        # reset environments
        with torch.no_grad():
            if dones.any():
                self.states = self.env.reset()
            else:
                self.states.copy_(next_states)


    def eval(self, timestep: int, timesteps: Optional[int] = None) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially if num_agents > 1)
        - Interact with the environments
        - Render scene
        - Reset environments

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Total number of timesteps (default: None).
                          If None, the total number of timesteps is obtained from the trainer's config
        :type timesteps: int, optional
        """
        timesteps = self.timesteps if timesteps is None else timesteps

        if self._progress is None:
            self._progress = tqdm.tqdm(total=timesteps)
        self._progress.update(n=1)

        # reset env
        if self.states is None:
            self.states = self.env.reset()
        
        with torch.no_grad():
            if self.num_agents == 1:
                # compute actions
                actions, _, _ = self.agents.act(self.states, inference=True, timestep=timestep, timesteps=timesteps)

            else:
                # compute actions
                actions = torch.vstack([agent.act(self.states[scope[0]:scope[1]],
                                                  inference=True,
                                                  timestep=timestep,
                                                  timesteps=timesteps)[0] \
                                        for agent, scope in zip(self.agents, self.agents_scope)])

        # step the environments
        next_states, rewards, dones, infos = self.env.step(actions)

        # render scene
        if not self.headless:
            self.env.render()

        with torch.no_grad():
            if self.num_agents == 1:
                # write data to TensorBoard
                super(type(self.agents), self.agents).record_transition(states=self.states,
                                                                        actions=actions,
                                                                        rewards=rewards,
                                                                        next_states=next_states,
                                                                        dones=dones,
                                                                        infos=infos,
                                                                        timestep=timestep,
                                                                        timesteps=timesteps)
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=timesteps)

            else:
                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    super(type(agent), agent).record_transition(states=self.states[scope[0]:scope[1]],
                                                                actions=actions[scope[0]:scope[1]],
                                                                rewards=rewards[scope[0]:scope[1]],
                                                                next_states=next_states[scope[0]:scope[1]],
                                                                dones=dones[scope[0]:scope[1]],
                                                                infos=infos,
                                                                timestep=timestep,
                                                                timesteps=timesteps)
                    super(type(agent), agent).post_interaction(timestep=timestep, timesteps=timesteps)

            # reset environments
            if dones.any():
                self.states = self.env.reset()
            else:
                self.states.copy_(next_states)
