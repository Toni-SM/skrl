from typing import Union, Tuple, Dict

import gym
import itertools
import numpy as np

import torch
import torch.nn.functional as F

from ....memories.torch import Memory
from ....models.torch import Model

from .. import Agent


SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size
    
    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)
    
    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.2,   # initial entropy value
    "target_entropy": None,         # target entropy

    "experiment": {
        "base_directory": "",       # base directory for the experiment
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "checkpoint_policy_only": True,     # checkpoint for policy only
    }
}


class SAC(Agent):
    def __init__(self, 
                 networks: Dict[str, Model], 
                 memory: Union[Memory, None] = None, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0", 
                 cfg: dict = {}) -> None:
        """Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290
        
        :param networks: Networks used by the agent
        :type networks: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions
        :type memory: skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Computing device (default: "cuda:0")
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the networks dictionary is missing a required key
        """
        SAC_DEFAULT_CONFIG.update(cfg)
        super().__init__(networks=networks, 
                         memory=memory, 
                         observation_space=observation_space, 
                         action_space=action_space, 
                         device=device, 
                         cfg=SAC_DEFAULT_CONFIG)

        # networks
        if not "policy" in self.networks.keys():
            raise KeyError("The policy network is not defined under 'policy' key (networks['policy'])")
        if not "critic_1" in self.networks.keys():
            raise KeyError("The Q1-network (critic 1) is not defined under 'critic_1' key (networks['critic_1'])")
        if not "critic_2" in self.networks.keys():
            raise KeyError("The Q2-network (critic 2) is not defined under 'critic_2' key (networks['critic_2'])")
        if not "target_critic_1" in self.networks.keys():
            raise KeyError("The target Q1-network (target critic 1) is not defined under 'target_critic_1' key (networks['target_critic_1'])")
        if not "target_critic_2" in self.networks.keys():
            raise KeyError("The target Q2-network (target critic 2) is not defined under 'target_critic_2' key (networks['target_critic_2'])")
        
        self.policy = self.networks["policy"]
        self.critic_1 = self.networks["critic_1"]
        self.critic_2 = self.networks["critic_2"]
        self.target_critic_1 = self.networks["target_critic_1"]
        self.target_critic_2 = self.networks["target_critic_2"]

        # checkpoint networks
        self.checkpoint_networks = {"policy": self.policy} if self.checkpoint_policy_only else self.networks

        # freeze target networks with respect to optimizers (update via .update_parameters())
        self.target_critic_1.freeze_parameters(True)
        self.target_critic_2.freeze_parameters(True)

        # update target networks (hard update)
        self.target_critic_1.update_parameters(self.critic_1, polyak=1)
        self.target_critic_2.update_parameters(self.critic_2, polyak=1)

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
                self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
            
            self.log_entropy_coefficient = torch.log(torch.ones(1, device=self.device) * self._entropy_coefficient).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self._entropy_learning_rate)

        # set up optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), 
                                                 lr=self._critic_learning_rate)

        # create tensors in memory
        self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)

        self.tensors_names = ["states", "actions", "rewards", "next_states", "dones"]

    def act(self, 
            states: torch.Tensor, 
            timestep: int, 
            timesteps: int, 
            inference: bool = False) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :param inference: Flag to indicate whether the network is making inference
        :type inference: bool

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # # TODO, check for stochasticity
        # if timestep < self._random_timesteps:
        #     return self.policy.random_act(states)

        # sample stochastic actions
        return self.policy.act(states, inference=inference)

    def record_transition(self, 
                          states: torch.Tensor, 
                          actions: torch.Tensor, 
                          rewards: torch.Tensor, 
                          next_states: torch.Tensor, 
                          dones: torch.Tensor, 
                          timestep: int, 
                          timesteps: int) -> None:
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
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)

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
        
        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)
             
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):
            
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

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
                self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
                self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

                self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
                self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
                self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())
                
                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())
