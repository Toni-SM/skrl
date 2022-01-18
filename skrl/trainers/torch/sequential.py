from typing import Union, List

import torch

from ...envs.torch import Wrapper
from ...agents.torch import Agent

from . import Trainer


class SequentialTrainer(Trainer):
    def __init__(self, 
                 cfg: dict, 
                 env: Wrapper, 
                 agents: Union[Agent, List[Agent], List[List[Agent]]], 
                 agents_scope : List[int] = []) -> None:
        """Sequential trainer
        
        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :param cfg: Configuration dictionary
        :type cfg: dict
        :param env: Environment to train on
        :type env: skrl.env.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: [])
        :type agents_scope: tuple or list of integers
        """
        super().__init__(cfg, env, agents, agents_scope)

    def train(self) -> None:
        """Train the agents sequentially

        Execute the following steps in loop:
        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Reset environments
        - Post-interaction (sequentially)
        """
        # reset env
        states = self.env.reset()

        for timestep in range(self.initial_timestep, self.timesteps):
            # show progress
            self.show_progress(timestep=timestep, timesteps=self.timesteps)

            # pre-interaction
            if self.num_agents > 1:
                for agent in self.agents:
                    agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            else:
                self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            
            # compute actions
            with torch.no_grad():
                if self.num_agents > 1:
                    actions = torch.vstack([agent.act(states[scope[0]:scope[1]], 
                                                      inference=True,
                                                      timestep=timestep, 
                                                      timesteps=self.timesteps)[0] \
                                            for agent, scope in zip(self.agents, self.agents_scope)])
                else:
                    actions, _, _ = self.agents.act(states, inference=True, timestep=timestep, timesteps=self.timesteps)
            
            # step the environments
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # render scene
            if not self.headless:
                self.env.render()

            # record the environments' transitions
            with torch.no_grad():
                if self.num_agents > 1:
                    for agent, scope in zip(self.agents, self.agents_scope):
                        agent.record_transition(states=states[scope[0]:scope[1]], 
                                                actions=actions[scope[0]:scope[1]], 
                                                rewards=rewards[scope[0]:scope[1]], 
                                                next_states=next_states[scope[0]:scope[1]], 
                                                dones=dones[scope[0]:scope[1]],
                                                timestep=timestep,
                                                timesteps=self.timesteps)
                else:
                    self.agents.record_transition(states=states, 
                                                  actions=actions,
                                                  rewards=rewards,
                                                  next_states=next_states,
                                                  dones=dones,
                                                  timestep=timestep,
                                                  timesteps=self.timesteps)
            
            # reset environments
            with torch.no_grad():
                if dones.any():
                    states = self.env.reset()
                else:
                    states.copy_(next_states)
                
            # post-interaction
            if self.num_agents > 1:
                for agent in self.agents:
                    agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
            else:
                self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
        
    def eval(self) -> None:
        """Evaluate the agents sequentially

        Execute the following steps in loop:
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        # reset env
        states = self.env.reset()

        for timestep in range(self.initial_timestep, self.timesteps):
            # show progress
            self.show_progress(timestep=timestep, timesteps=self.timesteps)
            
            # compute actions
            with torch.no_grad():
                if self.num_agents > 1:
                    actions = torch.vstack([agent.act(states[scope[0]:scope[1]], 
                                                      inference=True,
                                                      timestep=timestep, 
                                                      timesteps=self.timesteps)[0] \
                                            for agent, scope in zip(self.agents, self.agents_scope)])
                else:
                    actions, _, _ = self.agents.act(states, inference=True, timestep=timestep, timesteps=self.timesteps)
            
            # step the environments
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # render scene
            if not self.headless:
                self.env.render()
            
            # reset environments
            with torch.no_grad():
                if dones.any():
                    states = self.env.reset()
                else:
                    states.copy_(next_states)
        
    def start(self) -> None:
        """Start training

        This method is deprecated in favour of the '.train()' method 
        """
        super().start()
        self.train()
