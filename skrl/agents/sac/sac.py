from typing import Union, Dict

import gym
import torch
import torch.nn.functional as F
import itertools
import numpy as np

from ...env import Environment
from ...memories import Memory
from ...models.torch import Model

from .. import Agent


SAC_DEFAULT_CONFIG = {
    "discount_factor": 0.99,        # discount factor (gamma)
    "gradient_steps": 1,            # gradient steps
    
    "polyak": 0.995,                # soft update hyperparameter (tau)
    
    "batch_size": 64,               # size of minibatch
    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.1,   # initial entropy value
    "target_entropy": None,         # target entropy

    "device": None,                 # device to use
}


class SAC(Agent):
    def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """
        Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290
        """
        SAC_DEFAULT_CONFIG.update(cfg)
        super().__init__(env=env, networks=networks, memory=memory, cfg=SAC_DEFAULT_CONFIG)

        # networks
        if not "policy" in self.networks.keys():
            raise KeyError("Policy network not found in networks. Use 'policy' key to define the policy network")
        if not "q_1" in self.networks.keys() and not "critic_1" in self.networks.keys():
            raise KeyError("Q1-network (critic 1) not found in networks. Use 'critic_1' or 'q_1' keys to define the Q1-network (critic 1)")
        if not "q_2" in self.networks.keys() and not "critic_2" in self.networks.keys():
            raise KeyError("Q2-network (critic 2) not found in networks. Use 'critic_2' or 'q_2' keys to define the Q2-network (critic 2)")
        if not "target_1" in self.networks.keys():
            raise KeyError("Q1-target network (target 1) not found in networks. Use 'target_1' key to define the Q1-target network (target 1)")
        if not "target_2" in self.networks.keys():
            raise KeyError("Q2-target network (target 2) not found in networks. Use 'target_2' key to define the Q2-target network (target 2)")
        
        self.policy = self.networks["policy"]
        self.critic_1 = self.networks.get("critic_1", self.networks.get("q_1", None))
        self.critic_2 = self.networks.get("critic_2", self.networks.get("q_2", None))
        self.target_1 = self.networks["target_1"]
        self.target_2 = self.networks["target_2"]

        # freeze target networks with respect to optimizers (update via .update_parameters())
        self.target_1.freeze_parameters(True)
        self.target_2.freeze_parameters(True)

        # update target networks (hard update)
        self.target_1.update_parameters(self.critic_1, polyak=0)
        self.target_2.update_parameters(self.critic_2, polyak=0)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]
        self._polyak = self.cfg["polyak"]
        self._discount_factor = self.cfg["discount_factor"]
        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
            if self._target_entropy is None:
                self._target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
            
            self.log_entropy_coefficient = torch.log(torch.ones(1, device=self.device) * self._entropy_coefficient).requires_grad_(True)
            self.optimizer_entropy = torch.optim.Adam([self.log_entropy_coefficient], lr=self._entropy_learning_rate)

        # set up optimizers
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
        self.optimizer_critic = torch.optim.Adam(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=self._critic_learning_rate)

        # create tensors in memory
        self.memory.create_tensor(name="states", size=self.env.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_states", size=self.env.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.env.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)

        self.tensors_names = ["states", "actions", "rewards", "next_states", "dones"]

    def act(self, states: torch.Tensor, inference: bool = False, timestep: Union[int, None] = None, timesteps: Union[int, None] = None) -> torch.Tensor:
        """
        Process the environments' states to make a decision (actions) using the main policy

        Parameters
        ----------
        states: torch.Tensor
            Environments' states
        inference: bool
            Flag to indicate whether the network is making inference
        timestep: int or None
            Current timestep
        timesteps: int or None
            Number of timesteps
            
        Returns
        -------
        torch.Tensor
            Actions
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(states)

        # sample stochastic actions
        return self.policy.act(states, inference=inference)

    def record_transition(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, timestep: int, timesteps: int) -> None:
        """
        Record an environment transition in memory
        
        Parameters
        ----------
        states: torch.Tensor
            Observations/states of the environment used to make the decision
        actions: torch.Tensor
            Actions taken by the agent
        rewards: torch.Tensor
            Instant rewards achieved by the current actions
        next_states: torch.Tensor
            Next observations/states of the environment
        dones: torch.Tensor
            Signals to indicate that episodes have ended
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        super().record_transition(states, actions, rewards, next_states, dones, timestep, timesteps)
        if self.memory is not None:
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)

    def pre_rollouts(self, timestep: int, timesteps: int) -> None:
        """
        Callback called before all rollouts

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        pass

    def inter_rollouts(self, timestep: int, timesteps: int, rollout: int, rollouts: int) -> None:
        """
        Callback called after each rollout

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
        pass

    def post_rollouts(self, timestep: int, timesteps: int) -> None:
        """
        Callback called after all rollouts

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        if timestep >= self._learning_starts:
            self._update(timestep, timesteps)
    
    def _update(self, timestep: int, timesteps: int):
        # update steps
        for gradient_step in range(self._gradient_steps):
            
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.memory.sample(self._batch_size, self.tensors_names)

            # compute targets for Q-functions
            with torch.no_grad():
                next_actions, next_log_prob, _ = self.policy.act(states=sampled_next_states)

                target_1_values, _, _ = self.target_1.act(states=sampled_next_states, taken_actions=next_actions)
                target_2_values, _, _ = self.target_2.act(states=sampled_next_states, taken_actions=next_actions)
                target_values = torch.min(target_1_values, target_2_values) - self._entropy_coefficient * next_log_prob
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_values
                # Subtraction, the `-` operator, with a bool tensor is not supported. 
                # If you are trying to invert a mask, use the `~` or `logical_not()` operator instead

            # update critic (Q-functions)
            critic_1_values, _, _ = self.critic_1.act(states=sampled_states, taken_actions=sampled_actions)
            critic_2_values, _, _ = self.critic_2.act(states=sampled_states, taken_actions=sampled_actions)
            
            loss_critic = F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
            
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

            # update policy
            actions, log_prob, _ = self.policy.act(states=sampled_states)

            critic_1_values, _, _ = self.critic_1.act(states=sampled_states, taken_actions=actions)
            critic_2_values, _, _ = self.critic_2.act(states=sampled_states, taken_actions=actions)

            loss_policy = (self._entropy_coefficient * log_prob - torch.min(critic_1_values, critic_2_values)).mean()

            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()

            # update entropy
            if self._learn_entropy:
                loss_entropy = -(self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()).mean()

                self.optimizer_entropy.zero_grad()
                loss_entropy.backward()
                self.optimizer_entropy.step()

                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            # update target networks (soft update)
            self.target_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_2.update_parameters(self.critic_2, polyak=self._polyak)

            # record data
            self.writer.add_scalar('Loss/policy', loss_policy.item(), timestep)
            self.writer.add_scalar('Loss/critic', loss_critic.item(), timestep)

            self.writer.add_scalar('Q-networks/q1_max', torch.max(critic_1_values).item(), timestep)
            self.writer.add_scalar('Q-networks/q1_min', torch.min(critic_1_values).item(), timestep)
            self.writer.add_scalar('Q-networks/q1_mean', torch.mean(critic_1_values).item(), timestep)

            self.writer.add_scalar('Q-networks/q2_max', torch.max(critic_2_values).item(), timestep)
            self.writer.add_scalar('Q-networks/q2_min', torch.min(critic_2_values).item(), timestep)
            self.writer.add_scalar('Q-networks/q2_mean', torch.mean(critic_2_values).item(), timestep)
            
            self.writer.add_scalar('Target/max', torch.max(target_values).item(), timestep)
            self.writer.add_scalar('Target/min', torch.min(target_values).item(), timestep)
            self.writer.add_scalar('Target/mean', torch.mean(target_values).item(), timestep)

            if self._learn_entropy:
                self.writer.add_scalar('Entropy/loss', loss_entropy.item(), timestep)
                self.writer.add_scalar('Entropy/coefficient', self._entropy_coefficient.item(), timestep)
