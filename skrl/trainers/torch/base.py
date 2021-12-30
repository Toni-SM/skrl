from typing import Union, List

import torch

from ...env.torch import Wrapper
from ...agents.torch import Agent


class Trainer():
    def __init__(self, cfg: dict, env: Wrapper, agents: Union[Agent, List[Agent], List[List[Agent]]], agents_scope : List[int] = []) -> None:
        """Base class for trainers

        :param cfg: Configuration dictionary
        :type cfg: dict
        :param env: Environment to train on
        :type env: skrl.env.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: [])
        :type agents_scope: tuple or list of integers
        """
        self.starting_timestep = 0
        
        self.env = env

        self.num_agents = 0
        self.agents = agents
        self.agents_scope = agents_scope
        self.parallel_agents = False

        self.timesteps = cfg.get('timesteps', 0)
        self.headless = cfg.get("headless", False)

        self._setup_agents()

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = "Trainer: {}".format(repr(self))
        string += "\n  |-- Number of parallelizable environments: {}".format(self.env.num_envs)
        string += "\n  |-- Parallel agents: {}".format(self.parallel_agents)
        string += "\n  |-- Number of agents: {}".format(len(self.agents) if self.parallel_agents else 1)
        if self.parallel_agents:
            string += "\n  |-- Agents and scopes:"
            for agent, scope in zip(self.agents, self.agents_scope):
                string += "\n  |     |-- agent: {}".format(type(agent))
                string += "\n  |     |     |-- scope: {} environments ({}:{})".format(scope[1] - scope[0], scope[0], scope[1])
        return string

    def _pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Call pre_interaction method for each agent before all interactions

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def _post_interaction(self, timestep: int, timesteps: int) -> None:
        """Call post_interaction method for each agent after all interactions
        
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scope
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.agents = self.agents[0]
                self.parallel_agents = False
            # parallel agents
            elif len(self.agents) > 1:
                self.parallel_agents = True
                # check scope
                if not len(self.agents_scope):
                    print("[WARNING] The agents scope is empty")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError("The number of agents ({}) is greater than the number of parallelizable environments ({})".format(len(self.agents), self.env.num_envs))
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError("The number of agents ({}) doesn't match the number of scopes ({})".format(len(self.agents), len(self.agents_scope)))
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError("The scopes ({}) don't cover the number of parallelizable environments ({})".format(sum(self.agents_scope), self.env.num_envs))

                index = 0 
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("There are no agents to train")
        
        self.num_agents = len(self.agents) if self.parallel_agents else 1
        
        # enable train mode
        if self.parallel_agents:
            for agent in self.agents:
                agent.set_mode("train")
        else:
            self.agents.set_mode("train")
        
    def start(self) -> None:
        """Start training
        """
        # reset env
        states = self.env.reset()

        for timestep in range(self.starting_timestep, self.timesteps):
            if timestep % 1000 == 0:
                print("timestep:", timestep)
            
            # pre-interaction
            self._pre_interaction(timestep=timestep, timesteps=self.timesteps)
            
            # compute actions
            if self.parallel_agents:
                actions = torch.vstack([agent.act(states[scope[0]:scope[1]], 
                                                  inference=True,
                                                  timestep=timestep, 
                                                  timesteps=self.timesteps)[0] for agent, scope in zip(self.agents, self.agents_scope)])
            else:
                actions, _, _ = self.agents.act(states, inference=True, timestep=timestep, timesteps=self.timesteps)
            
            # step the environment
            next_states, rewards, dones, infos = self.env.step(actions)
            
            # render scene
            if not self.headless:
                self.env.render()

            # record the transition 
            if self.parallel_agents:
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
            if dones.any():
                states = self.env.reset()
            else:
                states.copy_(next_states)
                
            # post-interaction
            self._post_interaction(timestep=timestep, timesteps=self.timesteps)
