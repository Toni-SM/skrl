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


PPO_DEFAULT_CONFIG = {
    "discount_factor": 0.99,        # discount factor (gamma)
    "learning_epochs": 10,           # number of learning epochs
    
    "polyak": 0.995,                # soft update hyperparameter (tau)
    
    "batch_size": 64,               # size of minibatch
    "policy_learning_rate": 1e-3,   # policy learning rate
    "value_learning_rate": 1e-3,    # value learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.1,   # initial entropy value
    "target_entropy": None,         # target entropy

    "device": None,                 # device to use

    "ratio_clip": 0.2,              # ratio clip parameter
    "value_clip": 0.2,              # value clip parameter
    "clip_value_loss": True,        # clip value loss

    "value_loss_scale": 1.0,        # value loss scale coefficient
    "entropy_scale": 0.0,           # entropy scale coefficient

    "kl_threshold": None, #0.01,           # KL divergence threshold
}


class PPO(Agent):
    def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """
        Proximal Policy Optimization (PPO)

        https://arxiv.org/abs/1707.06347
        """
        PPO_DEFAULT_CONFIG.update(cfg)
        super().__init__(env=env, networks=networks, memory=memory, cfg=PPO_DEFAULT_CONFIG)

        # networks
        if not "policy" in self.networks.keys():
            raise KeyError("Policy network not found in networks. Use 'policy' key to define the policy network")
        if not "value" in self.networks.keys():
            raise KeyError("Value-network not found in networks. Use 'value' key to define the Value-network")
        
        self.policy = self.networks["policy"]
        self.value = self.networks["value"]

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]

        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_value_loss = self.cfg["clip_value_loss"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_scale = self.cfg["entropy_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._policy_learning_rate = self.cfg["policy_learning_rate"]
        self._value_learning_rate = self.cfg["value_learning_rate"]


        self._batch_size = self.cfg["batch_size"]
        self._polyak = self.cfg["polyak"]
        self._discount_factor = self.cfg["discount_factor"]
        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        # set up optimizers
        # self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self._policy_learning_rate)
        # self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=self._value_learning_rate)
        self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._policy_learning_rate)

        # create tensors in memory
        self.memory.create_tensor(name="states", size=self.env.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_states", size=self.env.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.env.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)

        self.memory.create_tensor(name="states", size=self.env.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.env.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

        # self.tensors_names = ["states", "actions", "rewards", "next_states", "dones"]
        self.tensors_names = ["states", "actions", "rewards", "dones", "log_prob", "values", "returns", "advantages"]

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
        if inference:
            with torch.no_grad():
                actions, log_prob, actions_mean = self.policy.act(states, inference=inference)
        else:
            actions, log_prob, actions_mean = self.policy.act(states, inference=inference)
        self._log_prob = log_prob
        return actions, log_prob, actions_mean

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
            values, _, _ = self.value.act(states=states)
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones, 
                                    log_prob=self._log_prob, values=values)

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
        # print("Updating...")
        # sample a batch from memory
        self.memory.compute_functions(returns_dst="returns", advantages_dst="advantages") #, gamma=self._gamma, lam=self._lam)
        sampled_states, sampled_actions, sampled_rewards, sampled_dones, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages = self.memory.sample(self._batch_size, self.tensors_names)

        # update steps
        for epoch in range(self._learning_epochs):

            next_actions, next_log_prob, _ = self.policy.act(states=sampled_states, taken_actions=sampled_actions)
            entropies = self.policy.get_entropy()

            # surrogate loss
            ratio = torch.exp(next_log_prob - sampled_log_prob)
            surrogate = sampled_advantages * ratio
            surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
            
            policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
            
            # value loss
            predicted_values, _, _ = self.value.act(states=sampled_states)

            if self._clip_value_loss:
                predicted_values = sampled_values + torch.clip(predicted_values - sampled_values, -self._value_clip, self._value_clip)
            value_loss = F.mse_loss(sampled_returns, predicted_values)

            # # TODO: check isaac-gym method
            # if self._clip_value_loss:
            #     clipped_predicted_values = sampled_values + torch.clip(predicted_values - sampled_values, -self._value_clip, self._value_clip)
            #     square_error = torch.pow(predicted_values - sampled_returns, 2)
            #     clipped_square_error = torch.pow(clipped_predicted_values - sampled_returns, 2)
            #     value_loss = torch.max(square_error, clipped_square_error).mean()
            # else:
            #     value_loss = F.mse_loss(sampled_returns, predicted_values)

            # entropy loss
            entropy_loss = (-next_log_prob).mean() if entropies is None else entropies.mean()

            # total loss
            loss = policy_loss + self._value_loss_scale * value_loss - self._entropy_scale * entropy_loss
            
            # KL divergence
            if self._kl_threshold is not None:
                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl = ((torch.exp(ratio) - 1) - ratio).mean()
                
                # TODO: 1.5 or 2.0?
                if kl > 1.5 * self._kl_threshold:
                    print("[INFO] Early stopping at step {} due to reaching maximum KL divergence {}".format(epoch, kl))
                    break
            
            # update 
            self.optimizer.zero_grad()
            loss.backward()
            # TODO: clip grad norm
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # record data
            self.writer.add_scalar('Loss/policy', policy_loss.item(), timestep)
            self.writer.add_scalar('Loss/value', self._value_loss_scale * value_loss.item(), timestep)
            self.writer.add_scalar('Loss/Total', loss, timestep)

            self.writer.add_scalar('Entropy/loss', -self._entropy_scale * entropy_loss.item(), timestep)

