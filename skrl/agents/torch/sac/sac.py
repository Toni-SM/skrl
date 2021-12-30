from typing import Union, Dict

import itertools
import numpy as np

import torch
import torch.nn.functional as F

from ....env.torch import Wrapper
from ....memories.torch import Memory
from ....models.torch import Model

from .. import Agent


SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size
    
    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.995,                # soft update hyperparameter (tau)
    
    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.2,   # initial entropy value
    "target_entropy": None,         # target entropy

    "device": None,                 # computing device

    "experiment": {
        "base_directory": "",       # base directory for the experiment
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # write interval for the experiment
    }
}


class SAC(Agent):
    def __init__(self, env: Wrapper, networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290
        
        :param env: RL environment
        :type env: skrl.env.torch.Wrapper
        :param networks: Networks used by the agent
        :type networks: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions
        :type memory: skrl.memory.torch.Memory or None
        :param cfg: Configuration dictionary
        :type cfg: dict
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
        if not "target_q_1" in self.networks.keys() and not "target_critic_1" in self.networks.keys():
            raise KeyError("Target Q1-network (target critic 1) not found in networks. Use 'target_critic_1' or 'target_q_1' keys to define the target Q1-network (target critic 1)")
        if not "target_q_2" in self.networks.keys() and not "target_critic_2" in self.networks.keys():
            raise KeyError("Target Q2-network (target critic 2) not found in networks. Use 'target_critic_2' or 'target_q_2' keys to define the target Q2-network (target critic 2)")
        
        self.policy = self.networks["policy"]
        self.critic_1 = self.networks.get("critic_1", self.networks.get("q_1", None))
        self.critic_2 = self.networks.get("critic_2", self.networks.get("q_2", None))
        self.target_critic_1 = self.networks.get("target_critic_1", self.networks.get("target_q_1", None))
        self.target_critic_2 = self.networks.get("target_critic_2", self.networks.get("target_q_2", None))

        # freeze target networks with respect to optimizers (update via .update_parameters())
        self.target_critic_1.freeze_parameters(True)
        self.target_critic_2.freeze_parameters(True)

        # update target networks (hard update)
        self.target_critic_1.update_parameters(self.critic_1, polyak=0)
        self.target_critic_2.update_parameters(self.critic_2, polyak=0)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]
        
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
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self._entropy_learning_rate)

        # set up optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=self._critic_learning_rate)

        # create tensors in memory
        self.memory.create_tensor(name="states", size=self.env.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_states", size=self.env.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.env.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)

        self.tensors_names = ["states", "actions", "rewards", "next_states", "dones"]

    def act(self, states: torch.Tensor, inference: bool = False, timestep: Union[int, None] = None, timesteps: Union[int, None] = None) -> torch.Tensor:
        """Process the environments' states to make a decision (actions) using the main policy

        :param states: Environments' states
        :type states: torch.Tensor
        :param inference: Flag to indicate whether the network is making inference
        :type inference: bool
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # # TODO, check for stochasticity
        # if timestep < self._random_timesteps:
        #     return self.policy.random_act(states)

        # sample stochastic actions
        return self.policy.act(states, inference=inference)

    def record_transition(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, timestep: int, timesteps: int) -> None:
        """Record an environment transition in memory
        
        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param dones: Signals to indicate that episodes have ended
        :type dones: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, dones, timestep, timesteps)
        if self.memory is not None:
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self._update(timestep, timesteps)
        
        # write to tensorboard
        if self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep, timesteps)
             
    def _update(self, timestep: int, timesteps: int):
        # gradient steps
        for gradient_step in range(self._gradient_steps):
            
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.memory.sample(self._batch_size, self.tensors_names)

            # compute target values
            with torch.no_grad():
                next_actions, next_log_prob, _ = self.policy.act(states=sampled_next_states)

                target_q1_values, _, _ = self.target_critic_1.act(states=sampled_next_states, taken_actions=next_actions)
                target_q2_values, _, _ = self.target_critic_2.act(states=sampled_next_states, taken_actions=next_actions)
                target_q_values = torch.min(target_q1_values, target_q2_values) - self._entropy_coefficient * next_log_prob
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute critic loss
            critic_1_values, _, _ = self.critic_1.act(states=sampled_states, taken_actions=sampled_actions)
            critic_2_values, _, _ = self.critic_2.act(states=sampled_states, taken_actions=sampled_actions)
            
            critic_loss = (F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)) / 2
            
            # optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute policy (actor) loss
            actions, log_prob, _ = self.policy.act(states=sampled_states)
            critic_1_values, _, _ = self.critic_1.act(states=sampled_states, taken_actions=actions)
            critic_2_values, _, _ = self.critic_2.act(states=sampled_states, taken_actions=actions)

            policy_loss = (self._entropy_coefficient * log_prob - torch.min(critic_1_values, critic_2_values)).mean()

            # optimize policy (actor)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # entropy learning
            if self._learn_entropy:
                # compute entropy loss
                entropy_loss = -(self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()).mean()

                # optimize entropy
                self.entropy_optimizer.zero_grad()
                entropy_loss.backward()
                self.entropy_optimizer.step()

                # compute entropy coefficient
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            # update target networks (soft update)
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # record data
            if self.write_interval > 0:
                self.tracking_data["Loss / Policy loss"].append(policy_loss.item())
                self.tracking_data["Loss / Critic loss"].append(critic_loss.item())

                self.tracking_data["Q-network / Q1 (max)"].append(torch.max(critic_1_values).item())
                self.tracking_data["Q-network / Q1 (min)"].append(torch.min(critic_1_values).item())
                self.tracking_data["Q-network / Q1 (mean)"].append(torch.mean(critic_1_values).item())

                self.tracking_data["Q-network / Q2 (max)"].append(torch.max(critic_2_values).item())
                self.tracking_data["Q-network / Q2 (min)"].append(torch.min(critic_2_values).item())
                self.tracking_data["Q-network / Q2 (mean)"].append(torch.mean(critic_2_values).item())
                
                self.tracking_data["Target / Target (max)"].append(torch.max(target_values).item())
                self.tracking_data["Target / Target (min)"].append(torch.min(target_values).item())
                self.tracking_data["Target / Target (mean)"].append(torch.mean(target_values).item())

                if self._learn_entropy:
                    self.tracking_data["Loss / Entropy loss"].append(entropy_loss.item())
                    self.tracking_data["Coefficient / Entropy coefficient"].append(self._entropy_coefficient.item())
