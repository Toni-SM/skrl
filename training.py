from typing import Union, List

import gym
import torch

from .env import wrap_env
from .env import Environment
from .agents import Agent


class Trainer:
    def __init__(self) -> None:
        """
        RL trainer
        """
        self._current_learning_iteration = 0
        self._multiagent = False

        self._agents = None
        self._env = None
        self._cfg = {}

    def _pre_rollouts(self, timestep: int, timesteps: int) -> None:
        """
        Call pre_rollouts method for each agent before all rollouts

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        if self._multiagent:
            for agent in self._agents:
                agent.pre_rollouts(timestep=timestep, timesteps=timesteps)
        else:
            self._agents.pre_rollouts(timestep=timestep, timesteps=timesteps)

    def _inter_rollouts(self, timestep: int, timesteps: int, rollout: int, rollouts: int) -> None:
        """
        Call inter_rollouts method for each agent after each rollout

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        rollout: int
            Current rollout
        rollouts: int
            Number of rollouts
        """
        if self._multiagent:
            for agent in self._agents:
                agent.inter_rollouts(timestep=timestep, timesteps=timesteps, rollout=rollout, rollouts=rollouts)
        else:
            self._agents.inter_rollouts(timestep=timestep, timesteps=timesteps, rollout=rollout, rollouts=rollouts)

    def _post_rollouts(self, timestep: int, timesteps: int) -> None:
        """
        Call post_rollouts method for each agent after all rollouts

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        if self._multiagent:
            for agent in self._agents:
                agent.post_rollouts(timestep=timestep, timesteps=timesteps)
        else:
            self._agents.post_rollouts(timestep=timestep, timesteps=timesteps)

    def train(self, agents: Union[Agent, List(Agent)], env: Union[Environment, gym.Env], cfg: dict = {}) -> None:
        """
        Train the agents

        Parameters
        ----------
        agents: skrl.agents.Agent or list of skrl.agents.Agent
            Agent or agents to be trained.
            In the case of a multi-agent system, the number of agents must match the number of parallelizable environments
        env: skrl.env.Environment or gym.Env
            RL environment
        cfg: dict
            Configuration dictionary
        """
        # wrap env
        self._env = wrap_env(env)

        # verify multi-agent training
        self._agents = agents
        if type(self._agents) is list:
            if not hasattr(self._env, 'num_envs'):
                raise AttributeError("The environment does not support parallelization")
            if len(self._agents) != self._env.num_envs:
                raise AssertionError("The number of agents ({}) does not match the number of parallelizable environments ({})".format(len(self._agents), self._env.num_envs))
            self._multiagent = True

        # enable train mode
        if self._multiagent:
            for agent in self._agents:
                agent.set_mode("train")
        else:
            self._agents.set_mode("train")
        
        # read configuration
        self._cfg = cfg
        self._max_learning_iterations = int(self._cfg.get("timesteps", 1e6))
        
        # get rollouts
        # for off-policy algorithms the notion of rollout corresponds to the steps taken in the environment between two updates (https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html?highlight=Rollout#custom-callback)
        rollouts = self._cfg.get("rollouts", 1)
        if rollouts <= 0:
            rollouts = 1
        
        # reset env
        states = self._env.reset()

        for t in range(self._current_learning_iteration, self._max_learning_iterations):
            print("iteration:", t)
            
            # pre-rollout
            self._pre_rollouts(timestep=t, timesteps=self._max_learning_iterations)
            
            # rollouts
            for r in range(rollouts):
                # compute the action
                # TODO: sample controlled random actions
                if self._multiagent:
                    with torch.no_grad():
                        actions = torch.stack([self._agents[i].act(states[i,:], inference=False) for i in range(len(self._agents))])
                else:
                    actions = self._agents.act(states, inference=False)

                # step the environment
                next_states, rewards, dones, infos = self._env.step(actions)

                # record the transition
                if self._multiagent:
                    for i in range(len(self._agents)):
                        if self._agents[i].memory is not None:
                            self._agents[i].memory.add_transitions(states[i,:], actions[i,:], rewards[i,:], next_states[i,:], dones[i,:])
                else:
                    if self._agents.memory is not None:
                        self._agents.memory.add_transitions(states, actions, rewards, next_states, dones)
                
                # reset env
                # check boolean (Gym compatibility)
                if isinstance(dones, bool):
                    states = self._env.reset() if dones else next_states
                # check tensor
                else:
                    if dones.any():
                        states = self._env.reset()
                    else:
                        states.copy_(next_states)

                # inter-rollout
                self._inter_rollouts(timestep=t, timesteps=self._max_learning_iterations, rollout=r, rollouts=rollouts)
                
            # post-rollout
            self._post_rollouts(timestep=t, timesteps=self._max_learning_iterations)
