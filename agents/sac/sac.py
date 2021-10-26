from typing import Union, Dict

import gym
import torch
import torch.nn.functional as F
import itertools
import numpy as np

from ...env import Environment
from ...memory import Memory
from ...models.torch import Model

from .. import Agent


class SAC(Agent):
    def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """
        Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290
        """
        super().__init__(env=env, networks=networks, memory=memory, cfg=cfg)

        # networks
        if not "policy" in self.networks.keys():
            raise KeyError("The network dictionary (networks) does not contain the policy network in 'policy' key. Use 'policy' key to define the policy network")
        if not "q_1" in self.networks.keys() and not "critic_1" in self.networks.keys():
            raise KeyError("The network dictionary (networks) does not contain the Q1-network (critic 1) in 'critic_1' or 'q_1' keys. Use 'critic_1' or 'q_1' keys to define the Q1-network (critic 1)")
        if not "q_2" in self.networks.keys() and not "critic_2" in self.networks.keys():
            raise KeyError("The network dictionary (networks) does not contain the Q2-network (critic 2) in 'critic_2' or 'q_2' keys. Use 'critic_2' or 'q_2' keys to define the Q2-network (critic 2)")
        if not "target_1" in self.networks.keys():
            raise KeyError("The network dictionary (networks) does not contain the Q1-target network (target 1) in 'target_1' key. Use 'target_1' key to define the Q1-target network (target 1)")
        if not "target_2" in self.networks.keys():
            raise KeyError("The network dictionary (networks) does not contain the Q2-target network (target 2) in 'target_2' key. Use 'target_2' key to define the Q2-target network (target 2)")
        
        self.policy = self.networks["policy"]
        self.critic_1 = self.networks.get("critic_1", self.networks.get("q_1", None))
        self.critic_2 = self.networks.get("critic_2", self.networks.get("q_2", None))
        self.target_1 = self.networks["target_1"]
        self.target_2 = self.networks["target_2"]

        # freeze target networks with respect to optimizers (update via .update_parameters())
        for param in self.target_1.parameters():
            param.requires_grad = False
        for param in self.target_2.parameters():
            param.requires_grad = False

        # update target networks (hard update)
        self.target_1.update_parameters(self.critic_1, polyak=0)
        self.target_2.update_parameters(self.critic_2, polyak=0)

        # configuration
        self._gradient_steps = self.cfg.get("gradient_steps", 1)
        self._batch_size = self.cfg.get("batch_size", 64)
        self._polyak = self.cfg.get("polyak", 0.995)
        self._discount_factor = self.cfg.get("discount_factor", 0.99)
        self._entropy_coefficient = self.cfg.get("entropy_coefficient", "auto")
        self._learning_rate = self.cfg.get("learning_rate", 3e-4)

        # entropy
        self._optimize_entropy = False
        if self._entropy_coefficient == "auto":
            self._optimize_entropy = True
            self._entropy_coefficient = 0.2 # TODO: get initial value from cfg
            self._target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
            
            self.log_entropy_coefficient = torch.log(torch.ones(1, device=self.device) * self._entropy_coefficient).requires_grad_(True)
            self.optimizer_entropy_coefficient = torch.optim.Adam([self.log_entropy_coefficient], lr=self._learning_rate)

        # set up optimizers
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        self.optimizer_critic = torch.optim.Adam(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=self._learning_rate)

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
        return self.policy.act(self.policy.to_tensor(states), inference=inference)

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
        self._update(timestep, timesteps)
    
    def _update(self, timestep: int, timesteps: int):
        # check memory size
        if len(self.memory) < self._batch_size:
            return
        
        # update steps
        for gradient_steps in range(self._gradient_steps):
            
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.memory.sample(self._batch_size)

            # compute targets for Q-functions
            with torch.no_grad():
                next_actions, next_log_prob, _ = self.policy.act(states=sampled_next_states)

                target_1_values, _, _ = self.target_1.act(states=sampled_next_states, taken_actions=next_actions)
                target_2_values, _, _ = self.target_2.act(states=sampled_next_states, taken_actions=next_actions)
                target_values = torch.min(target_1_values, target_2_values) - self._entropy_coefficient * next_log_prob
                target_values = sampled_rewards + self._discount_factor * (1 - sampled_dones) * target_values

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
            if self._optimize_entropy:
                loss_entropy = -(self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()).mean()

                self.optimizer_entropy_coefficient.zero_grad()
                loss_entropy.backward()
                self.optimizer_entropy_coefficient.step()

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

            if self._optimize_entropy:
                self.writer.add_scalar('Entropy/loss', loss_entropy.item(), timestep)
                self.writer.add_scalar('Entropy/coefficient', self._entropy_coefficient.item(), timestep)
