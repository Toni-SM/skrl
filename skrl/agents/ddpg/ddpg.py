from typing import Union, Dict

import gym
import torch
import torch.nn.functional as F

from ...env import Environment
from ...memories import Memory
from ...models.torch import Model

from .. import Agent


DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size
    
    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.995,                # soft update hyperparameter (tau)
    
    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

    "device": None,                 # computing device
}


class DDPG(Agent):
    def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """
        Deep Deterministic Policy Gradient (DDPG)

        https://arxiv.org/abs/1509.02971
        """
        DDPG_DEFAULT_CONFIG.update(cfg)
        super().__init__(env=env, networks=networks, memory=memory, cfg=DDPG_DEFAULT_CONFIG)

        # networks
        if not "policy" in self.networks.keys():
            raise KeyError("Policy network not found in networks. Use 'policy' key to define the policy network")
        if not "target_policy" in self.networks.keys():
            raise KeyError("Target policy network not found in networks. Use 'target_policy' key to define the target policy network")
        if not "q" in self.networks.keys() and not "critic" in self.networks.keys():
            raise KeyError("Q-network (critic) not found in networks. Use 'critic' or 'q' keys to define the Q-network (critic)")
        if not "target_q" in self.networks.keys() and not "target_critic" in self.networks.keys():
            raise KeyError("Target Q-network (target critic) not found in networks. Use 'target_critic' or 'target_q' keys to define the target Q-network (target critic)")
        
        self.policy = self.networks["policy"]
        self.target_policy = self.networks["target_policy"]
        self.critic = self.networks.get("critic", self.networks.get("q", None))
        self.target_critic = self.networks.get("target_critic", self.networks.get("target_q", None))
        
        # freeze target networks with respect to optimizers (update via .update_parameters())
        self.target_policy.freeze_parameters(True)
        self.target_critic.freeze_parameters(True)

        # update target networks (hard update)
        self.target_policy.update_parameters(self.policy, polyak=0)
        self.target_critic.update_parameters(self.critic, polyak=0)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]
        
        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        
        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]
        
        # set up optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._critic_learning_rate)

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
        if timestep < self._random_timesteps:
            return self.policy.random_act(states)

        # sample deterministic actions
        actions = self.policy.act(states, inference=inference)

        # add exloration noise
        if self._exploration_noise is not None:
            # sample noises
            noises = self._exploration_noise.sample(actions[0].shape)
            
            # scale noises
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps
            if timestep <= self._exploration_timesteps:
                scale = (1 - timestep / self._exploration_timesteps) * (self._exploration_initial_scale - self._exploration_final_scale) + self._exploration_final_scale
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

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """
        Callback called before the interaction with the environment

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """
        Callback called after the interaction with the environment

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
        # gradient steps
        for gradient_step in range(self._gradient_steps):
            
            # sample a batch from memory
            sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.memory.sample(self._batch_size, self.tensors_names)

            # compute target values
            with torch.no_grad():
                next_actions, _, _ = self.target_policy.act(states=sampled_next_states)
                
                target_q_values, _, _ = self.target_critic.act(states=sampled_next_states, taken_actions=next_actions)
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute critic loss
            critic_values, _, _ = self.critic.act(states=sampled_states, taken_actions=sampled_actions)
            
            critic_loss = F.mse_loss(critic_values, target_values)
            
            # optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute policy (actor) loss
            actions, _, _ = self.policy.act(states=sampled_states)
            critic_values, _, _ = self.critic.act(states=sampled_states, taken_actions=actions)

            policy_loss = -critic_values.mean()

            # optimize policy (actor)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # update target networks
            self.target_policy.update_parameters(self.policy, polyak=self._polyak)
            self.target_critic.update_parameters(self.critic, polyak=self._polyak)

            # record data
            self.writer.add_scalar('Loss/policy', policy_loss.item(), timestep)
            self.writer.add_scalar('Loss/critic', critic_loss.item(), timestep)

            self.writer.add_scalar('Q-networks/q1_max', torch.max(critic_values).item(), timestep)
            self.writer.add_scalar('Q-networks/q1_min', torch.min(critic_values).item(), timestep)
            self.writer.add_scalar('Q-networks/q1_mean', torch.mean(critic_values).item(), timestep)
            
            self.writer.add_scalar('Target/max', torch.max(target_values).item(), timestep)
            self.writer.add_scalar('Target/min', torch.min(target_values).item(), timestep)
            self.writer.add_scalar('Target/mean', torch.mean(target_values).item(), timestep)
