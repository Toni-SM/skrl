from typing import Union, List

import threading

from ...envs.torch import Wrapper
from ...agents.torch import Agent

from . import Trainer


class _Thread(threading.Thread):
    def __init__(self, barrier, target, args=(), kwargs={}):
        super().__init__(target=target, args=args, kwargs=kwargs)

        self._barrier = barrier

    def run(self):
        self._target(*self._args, **self._kwargs)
        self._barrier.wait()


class ConcurrentTrainer(Trainer):
    def __init__(self, cfg: dict, env: Wrapper, agents: Union[Agent, List[Agent], List[List[Agent]]], agents_scope : List[int] = []) -> None:
        """Concurrent trainer
        
        Train multiple agents in parallel
        
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

        self._barrier = threading.Barrier(self.num_agents + 1)

    def _pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Call pre_interaction method for each agent before all interactions

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.parallel_agents:
            for agent in self.agents:
                _Thread(self._barrier, agent.pre_interaction, args=(timestep, timesteps)).start()
            self._barrier.wait()
        else:
            self.agents.pre_interaction(timestep=timestep, timesteps=timesteps)

    def _post_interaction(self, timestep: int, timesteps: int) -> None:
        """Call post_interaction method for each agent after all interactions
        
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.parallel_agents:
            for agent in self.agents:
                _Thread(self._barrier, agent.post_interaction, args=(timestep, timesteps)).start()
            self._barrier.wait()
        else:
            self.agents.post_interaction(timestep=timestep, timesteps=timesteps)
