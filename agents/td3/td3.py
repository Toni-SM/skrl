from typing import Union, Dict

import gym
import torch
import torch.nn.functional as F
import itertools

from ...env import Environment
from ...memories import Memory
from ...models.torch import Model

from .. import Agent


class TD3(Agent):
    def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """
        Twin Delayed DDPG (TD3)

        https://arxiv.org/abs/1802.09477
        """
        super().__init__(env=env, networks=networks, memory=memory, cfg=cfg)

        # networks
        if not "policy" in self.networks.keys():
            raise KeyError("Policy network not found in networks. Use 'policy' key to define the policy network")
        if not "target_policy" in self.networks.keys():
            raise KeyError("Policy-target network not found in networks. Use 'target_policy' key to define the policy-target network")
        if not "q_1" in self.networks.keys() and not "critic_1" in self.networks.keys():
            raise KeyError("Q1-network (critic 1) not found in networks. Use 'critic_1' or 'q_1' keys to define the Q1-network (critic 1)")
        if not "q_2" in self.networks.keys() and not "critic_2" in self.networks.keys():
            raise KeyError("Q2-network (critic 2) not found in networks. Use 'critic_2' or 'q_2' keys to define the Q2-network (critic 2)")
        if not "target_1" in self.networks.keys():
            raise KeyError("Q1-target network (target 1) not found in networks. Use 'target_1' key to define the Q1-target network (target 1)")
        if not "target_2" in self.networks.keys():
            raise KeyError("Q2-target network (target 2) not found in networks. Use 'target_2' key to define the Q2-target network (target 2)")
        
        self.policy = self.networks["policy"]
        self.target_policy = self.networks["target_policy"]
        self.critic_1 = self.networks.get("critic_1", self.networks.get("q_1", None))
        self.critic_2 = self.networks.get("critic_2", self.networks.get("q_2", None))
        self.target_1 = self.networks["target_1"]
        self.target_2 = self.networks["target_2"]
        
        # freeze target networks with respect to optimizers (update via .update_parameters())
        self.target_policy.freeze_parameters(True)
        self.target_1.freeze_parameters(True)
        self.target_2.freeze_parameters(True)

        # update target networks (hard update)
        self.target_1.update_parameters(self.critic_1, polyak=0)
        self.target_2.update_parameters(self.critic_2, polyak=0)
        self.target_policy.update_parameters(self.policy, polyak=0)

        # configuration
        self._gradient_steps = self.cfg.get("gradient_steps", 1)
        self._batch_size = self.cfg.get("batch_size", 64)
        self._polyak = self.cfg.get("polyak", 0.995)
        self._discount_factor = self.cfg.get("discount_factor", 0.99)
        self._learning_rate = self.cfg.get("learning_rate", 3e-4)
        self._policy_delay = self.cfg.get("policy_delay", 2)

        self._noise = self.cfg.get("noise", None)
        self._noise_initial_scale = self.cfg.get("noise_initial_scale", 1.0)
        self._noise_final_scale = self.cfg.get("noise_final_scale", 0.01)
        self._noise_scale_timesteps = self.cfg.get("noise_scale_timesteps", 6000)

        self._smooth_noise = self.cfg.get("smooth_noise", None)
        self._smooth_noise_clip = self.cfg.get("smooth_noise_clip", 0.5)

        self.counter_q_function_updates = 0

        # set up optimizers
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        self.optimizer_critic = torch.optim.Adam(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=self._learning_rate)

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
        if inference:
            with torch.no_grad():
                actions = self.policy.act(states, inference=inference)
        else:
            actions = self.policy.act(states, inference=inference)
        
        # add noise
        if self._noise is not None:
            # sample noises
            noises = self._noise.sample(actions[0].shape)
            
            # scale noises
            scale = self._noise_final_scale
            if self._noise_scale_timesteps is None:
                self._noise_scale_timesteps = timesteps
            if timestep <= self._noise_scale_timesteps:
                scale = (1 - timestep / self._noise_scale_timesteps) * (self._noise_initial_scale - self._noise_final_scale) + self._noise_final_scale
            noises.mul_(scale)

            # modify actions
            actions[0].add_(noises)
            actions[0].clamp_(self.env.action_space.low[0], self.env.action_space.high[0]) # FIXME: use tensor too

            # record noises
            if timestep is not None:
                self.writer.add_scalar('Noise/max', torch.max(noises).item(), timestep)
                self.writer.add_scalar('Noise/min', torch.min(noises).item(), timestep)
                self.writer.add_scalar('Noise/mean', torch.mean(noises).item(), timestep)
        
        return actions

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
        for gradient_step in range(self._gradient_steps):
            
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.memory.sample(self._batch_size, self.tensors_names)

            # compute targets for Q-function
            with torch.no_grad():
                next_actions, _, _ = self.target_policy.act(states=sampled_next_states)

                # target policy smoothing
                noises = torch.clamp(self._smooth_noise.sample(next_actions.shape), -self._smooth_noise_clip, self._smooth_noise_clip)
                next_actions.add_(noises)
                next_actions.clamp_(self.env.action_space.low[0], self.env.action_space.high[0])  # FIXME: use tensor too

                target_1_values, _, _ = self.target_1.act(states=sampled_next_states, taken_actions=next_actions)
                target_2_values, _, _ = self.target_2.act(states=sampled_next_states, taken_actions=next_actions)
                target_values = torch.min(target_1_values, target_2_values)
                target_values = sampled_rewards + self._discount_factor * (1 - sampled_dones) * target_values

            # update critic (Q-functions)
            critic_1_values, _, _ = self.critic_1.act(states=sampled_states, taken_actions=sampled_actions)
            critic_2_values, _, _ = self.critic_2.act(states=sampled_states, taken_actions=sampled_actions)
            
            loss_critic = F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
            
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

            # delayed update
            self.counter_q_function_updates += 1
            if not self.counter_q_function_updates % self._policy_delay:

                # update policy
                actions, _, _ = self.policy.act(states=sampled_states)

                critic_values, _, _ = self.critic_1.act(states=sampled_states, taken_actions=actions)

                loss_policy = -critic_values.mean()

                self.optimizer_policy.zero_grad()
                loss_policy.backward()
                self.optimizer_policy.step()

                # update target networks
                self.target_1.update_parameters(self.critic_1, polyak=self._polyak)
                self.target_2.update_parameters(self.critic_2, polyak=self._polyak)
                self.target_policy.update_parameters(self.policy, polyak=self._polyak)

            # record data
            if not self.counter_q_function_updates % self._policy_delay:
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
