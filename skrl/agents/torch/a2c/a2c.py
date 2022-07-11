from typing import Union, Tuple, Dict, Any

import gym
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....memories.torch import Memory
from ....models.torch import Model

from .. import Agent


A2C_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "mini_batches": 1,              # number of mini batches to use for updating
    
    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages
    
    "policy_learning_rate": 1e-3,   # policy learning rate
    "value_learning_rate": 1e-3,    # value learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,          # clipping coefficient for the norm of the gradients

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "checkpoint_policy_only": True,     # checkpoint for policy only
    }
}


class A2C(Agent):
    def __init__(self, 
                 models: Dict[str, Model], 
                 memory: Union[Memory, Tuple[Memory], None] = None, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0", 
                 cfg: dict = {}) -> None:
        """Advantage Actor Critic (A2C)

        https://arxiv.org/abs/1602.01783
        
        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and 
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Computing device (default: "cuda:0")
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(A2C_DEFAULT_CONFIG)
        _cfg.update(cfg)
        super().__init__(models=models, 
                         memory=memory, 
                         observation_space=observation_space, 
                         action_space=action_space, 
                         device=device, 
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_models = {"policy": self.policy} if self.checkpoint_policy_only else self.models

        # configuration
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._policy_learning_rate = self.cfg["policy_learning_rate"]
        self._value_learning_rate = self.cfg["value_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.value is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._policy_learning_rate)
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self._value_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.value_scheduler = self._learning_rate_scheduler(self.value_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

        # set up preprocessors
        self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"]) if self._state_preprocessor \
            else self._empty_preprocessor
        self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"]) if self._value_preprocessor \
            else self._empty_preprocessor

    def init(self) -> None:
        """Initialize the agent
        """
        super().init()
        self.set_mode("eval")
        
        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

        self.tensors_names = ["states", "actions", "rewards", "dones", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_states = None

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
        :param inference: Flag to indicate whether the model is making inference
        :type inference: bool

        :return: Actions
        :rtype: torch.Tensor
        """
        states = self._state_preprocessor(states)

        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(states)

        # sample stochastic actions
        return self.policy.act(states, inference=inference)

    def record_transition(self, 
                          states: torch.Tensor, 
                          actions: torch.Tensor, 
                          rewards: torch.Tensor, 
                          next_states: torch.Tensor, 
                          dones: torch.Tensor, 
                          infos: Any, 
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
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, dones, infos, timestep, timesteps)

        # reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)

        self._current_next_states = next_states

        if self.memory is not None:
            values, _, _ = self.value.act(states=self._state_preprocessor(states), inference=True)
            values = self._value_preprocessor(values, inverse=True)

            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones, 
                                    values=values)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones, 
                                   values=values)

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
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        def compute_gae(rewards: torch.Tensor, 
                        dones: torch.Tensor, 
                        values: torch.Tensor, 
                        next_values: torch.Tensor, 
                        discount_factor: float = 0.99, 
                        lambda_coefficient: float = 0.95) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        last_values, _, _ = self.value.act(states=self._state_preprocessor(self._current_next_states.float() \
            if not torch.is_floating_point(self._current_next_states) else self._current_next_states), inference=True)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                          dones=self.memory.get_tensor_by_name("dones"),
                                          values=values,
                                          next_values=last_values,
                                          discount_factor=self._discount_factor,
                                          lambda_coefficient=self._lambda)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # mini-batches loop
        for sampled_states, sampled_actions, _, _, _, sampled_returns, sampled_advantages in sampled_batches:

            sampled_states = self._state_preprocessor(sampled_states, train=True)

            _, next_log_prob, _ = self.policy.act(states=sampled_states, taken_actions=sampled_actions)

            # compute entropy loss
            if self._entropy_loss_scale:
                entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy().mean()
            else:
                entropy_loss = 0
            
            # compute policy loss
            policy_loss = -(sampled_advantages * next_log_prob).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
            self.policy_optimizer.step()

            # compute value loss
            predicted_values, _, _ = self.value.act(states=sampled_states)

            value_loss = F.mse_loss(sampled_returns, predicted_values)

            # optimization step (value)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip)
            self.value_optimizer.step()

            # update cumulative losses
            cumulative_policy_loss += policy_loss.item()
            cumulative_value_loss += value_loss.item()
            if self._entropy_loss_scale:
                cumulative_entropy_loss += entropy_loss.item()

        # update learning rate
        if self._learning_rate_scheduler:
            self.policy_scheduler.step()
            self.value_scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / len(sampled_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / len(sampled_batches))
        
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / len(sampled_batches))

        self.track_data("Policy / Standard deviation", self.policy.distribution().stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
            self.track_data("Learning / Value learning rate", self.value_scheduler.get_last_lr()[0])
