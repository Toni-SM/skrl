from typing import Union, List

import time

from ...envs.torch import Wrapper
from ...agents.torch import Agent


class Trainer():
    def __init__(self, 
                 cfg: dict, 
                 env: Wrapper, 
                 agents: Union[Agent, List[Agent], List[List[Agent]]], 
                 agents_scope : List[int] = []) -> None:
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
        self.cfg = cfg
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope
        
        # get configuration
        self.timesteps = self.cfg.get('timesteps', 0)
        self.headless = self.cfg.get("headless", False)
        self.progress_interval = self.cfg.get("progress_interval", 1000)

        self.initial_timestep = 0

        self._timestamp = None
        self._timestamp_elapsed = None

        # setup agents
        self.num_agents = 0
        self._setup_agents()

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = "Trainer: {}".format(repr(self))
        string += "\n  |-- Number of parallelizable environments: {}".format(self.env.num_envs)
        string += "\n  |-- Number of agents: {}".format(self.num_agents)
        string += "\n  |-- Agents and scopes:"
        if self.num_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += "\n  |     |-- agent: {}".format(type(agent))
                string += "\n  |     |     |-- scope: {} environments ({}:{})".format(scope[1] - scope[0], scope[0], scope[1])
        else:
            string += "\n  |     |-- agent: {}".format(type(self.agents))
            string += "\n  |     |     |-- scope: {} environment(s)".format(self.env.num_envs)
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    print("[WARNING] The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError("The number of agents ({}) is greater than the number of parallelizable environments ({})". \
                            format(len(self.agents), self.env.num_envs))
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError("The number of agents ({}) doesn't match the number of scopes ({})". \
                        format(len(self.agents), len(self.agents_scope)))
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError("The scopes ({}) don't cover the number of parallelizable environments ({})". \
                        format(sum(self.agents_scope), self.env.num_envs))
                # generate agents' scopes
                index = 0 
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        
        # enable train mode
        if self.num_agents > 1:
            for agent in self.agents:
                agent.set_mode("train")
        else:
            self.agents.set_mode("train")
        
    def show_progress(self, timestep: int, timesteps: int) -> None:
        """Show training progress

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep > 0:
            timestep += 1
        
        if not timestep % self.progress_interval:
            current_timestamp = time.time()
            if self._timestamp is None:
                self._timestamp = current_timestamp
                self._timestamp_elapsed = self._timestamp

            delta = current_timestamp - self._timestamp
            elapsed = current_timestamp - self._timestamp_elapsed if timestep else 0.0
            remaining = elapsed * (self.timesteps / timestep - 1) if timestep else 0.0
            
            self._timestamp = current_timestamp

            print("|--------------------------|--------------------------|")
            print("|     timestep / timesteps | {} / {}".format(timestep, self.timesteps))
            print("|     timesteps per second |", round(self.progress_interval / delta, 2) if timestep else 0.0)
            print("| elapsed / remaining time | {} sec / {} sec".format(round(elapsed, 2), round(remaining, 2)))

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start training

        This method is deprecated in favour of the '.train()' method 
        """
        print("[WARNING] Trainer.start() method is deprecated in favour of the '.train()' method")
