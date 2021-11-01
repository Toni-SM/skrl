from typing import Union, List

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from .env import wrap_env
from .env import Environment
from .agents import Agent


class Trainer:
    def __init__(self) -> None:
        """
        RL trainer
        """
        self._current_learning_iteration = 0
        self._is_multi_agent = False

        self._cfg = {}
        self._env = None

        self._agents = None
        self._agents_scope = None

        # tensorboard
        # TODO: modify writer's parameters
        self._writer = SummaryWriter()

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
        if self._is_multi_agent:
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
        if self._is_multi_agent:
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
        if self._is_multi_agent:
            for agent in self._agents:
                agent.post_rollouts(timestep=timestep, timesteps=timesteps)
        else:
            self._agents.post_rollouts(timestep=timestep, timesteps=timesteps)

    def train(self, cfg: dict, env: Union[Environment, gym.Env], agents: Union[Agent, List[Agent]], agents_scope : List[int] = []) -> None:
        """
        # TODO: rewrite
        Train the agent(s)
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
        self._env = env
        self._agents = agents
        self._agents_scope = agents_scope

        self._is_multi_agent = False

        # wrap the environment to gym style
        if issubclass(type(env), Environment):
            self._env = wrap_env(env)
        
        if not hasattr(self._env, 'num_envs'):
            raise AttributeError("The environment does not support parallelization (num_envs property is not defined)")

        # validate agents and their scope
        if type(agents) in [tuple, list]:
            # single agent
            if len(agents) == 1:
                self._agents = agents[0]
                self._is_multi_agent = False
            # multi-agent
            elif len(agents) > 1:
                self._is_multi_agent = True
                # check scope
                if not len(agents_scope):
                    print("[WARNING] The agents scope is empty")
                    self._agents_scope = [int(self._env.num_envs / len(agents))] * len(agents)
                    if sum(self._agents_scope):
                        self._agents_scope[-1] += self._env.num_envs - sum(self._agents_scope)
                    else:
                        raise ValueError("The number of agents ({}) is greater than the number of parallelizable environments ({})".format(len(agents), self._env.num_envs))
                elif len(agents_scope) != len(agents):
                    raise ValueError("The number of agents ({}) doesn't match the number of scopes ({})".format(len(agents), len(agents_scope)))
                elif sum(agents_scope) != self._env.num_envs:
                    raise ValueError("The scopes ({}) don't cover the number of parallelizable environments ({})".format(sum(agents_scope), self._env.num_envs))

                index = 0 
                for i in range(len(self._agents_scope)):
                    index += self._agents_scope[i]
                    self._agents_scope[i] = (index - self._agents_scope[i], index)
            else:
                raise ValueError("The collection of agents is empty")
        
        # enable train mode
        if self._is_multi_agent:
            for agent in self._agents:
                agent.set_mode("train")
        else:
            self._agents.set_mode("train")
        
        # set writer
        if self._is_multi_agent:
            for agent in self._agents:
                agent.set_writer(self._writer)
        else:
            self._agents.set_writer(self._writer)

        # read configuration
        self._cfg = cfg
        self._max_learning_iterations = int(self._cfg.get("timesteps", 1e6))
        
        # get rollouts
        # for off-policy algorithms the notion of rollout corresponds to the steps taken in the environment between two updates (https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html?highlight=Rollout#custom-callback)
        rollouts = self._cfg.get("rollouts", 1)

        # show information about RL system
        print("")
        print("RL system")
        print("  |-- Number of parallelizable environments:", self._env.num_envs)
        print("  |-- Multi-agent:", self._is_multi_agent)
        print("  |-- Number of agents:", len(self._agents) if self._is_multi_agent else 1)
        if self._is_multi_agent:
            print("  |-- Agents and scopes:")
            for agent, scope in zip(self._agents, self._agents_scope):
                print("  |     |-- agent:", type(agent))
                print("  |     |     |-- scope: {} environments ({}:{})".format(scope[1] - scope[0], scope[0], scope[1]))
        print("")

        # reset env
        states = self._env.reset()

        for t in range(self._current_learning_iteration, self._max_learning_iterations):
            print("iteration:", t)
            
            # pre-rollout
            self._pre_rollouts(timestep=t, timesteps=self._max_learning_iterations)
            
            # rollouts
            for r in range(rollouts):
                # compute actions
                # TODO: sample controlled random actions
                if self._is_multi_agent:
                    actions = torch.stack([self._agents[i].act(states[i,:], inference=True) for i in range(len(self._agents))])
                else:
                    actions, _, _ = self._agents.act(states, inference=True)
                
                # step the environment
                next_states, rewards, dones, infos = self._env.step(actions)

                # record the transition 
                if self._is_multi_agent:
                    for agent, scope in zip(self._agents, self._agents_scope):
                        agent.record_transition(states=states[scope[0]:scope[1]], 
                                                actions=actions[scope[0]:scope[1]], 
                                                rewards=rewards[scope[0]:scope[1]], 
                                                next_states=next_states[scope[0]:scope[1]], 
                                                dones=dones[scope[0]:scope[1]])
                else:
                    self._agents.record_transition(states, actions, rewards, next_states, dones)
                
                # reset environments
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
