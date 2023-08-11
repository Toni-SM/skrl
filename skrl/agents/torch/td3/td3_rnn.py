from typing import Any, Dict, Optional, Tuple, Union

import copy
import itertools
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


# [start-config-dict-torch]
TD3_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

    "policy_delay": 2,                      # policy delay update with respect to critic update
    "smooth_regularization_noise": None,    # smooth noise for regularization
    "smooth_regularization_clip": 0.5,      # clip for smooth regularization

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]


class TD3_RNN(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Twin Delayed DDPG (TD3) with support for Recurrent Neural Networks (RNN, GRU, LSTM, etc.)

        https://arxiv.org/abs/1802.09477

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(TD3_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)
        self.target_policy = self.models.get("target_policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        if self.target_policy is not None and self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._policy_delay = self.cfg["policy_delay"]
        self._critic_update_counter = 0

        self._smooth_regularization_noise = self.cfg["smooth_regularization_noise"]
        self._smooth_regularization_clip = self.cfg["smooth_regularization_clip"]
        if self._smooth_regularization_noise is None:
            logger.warning("agents:TD3: No smooth regularization noise specified to reduce variance during training")

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            self.critic_optimizer = torch.optim.Adam(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                                                     lr=self._critic_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.critic_scheduler = self._learning_rate_scheduler(self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated"]

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"policy": []}
        self._rnn_initial_states = {"policy": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)

        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True)
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # clip noise bounds
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}

        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states), **rnn}, role="policy")

        # sample deterministic actions
        actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states), **rnn}, role="policy")

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

        # add exloration noise
        if self._exploration_noise is not None:
            # sample noises
            noises = self._exploration_noise.sample(actions.shape)

            # define exploration timesteps
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps

            # apply exploration noise
            if timestep <= self._exploration_timesteps:
                scale = (1 - timestep / self._exploration_timesteps) \
                      * (self._exploration_initial_scale - self._exploration_final_scale) \
                      + self._exploration_final_scale
                noises.mul_(scale)

                # modify actions
                actions.add_(noises)
                actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                # record noises
                self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
                self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
                self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item())

            else:
                # record noises
                self.track_data("Exploration / Exploration noise (max)", 0)
                self.track_data("Exploration / Exploration noise (min)", 0)
                self.track_data("Exploration / Exploration noise (mean)", 0)

        return actions, None, outputs

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
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
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update({f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])})

            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, **rnn_states)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, **rnn_states)

        # update RNN states
        if self._rnn:
            # reset states if the episodes have ended
            finished_episodes = terminated.nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states

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
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self._tensors_names, batch_size=self._batch_size, sequence_length=self._rnn_sequence_length)[0]

        rnn_policy = {}
        if self._rnn:
            sampled_rnn = self.memory.sample_by_index(names=self._rnn_tensors_names, indexes=self.memory.get_sampling_indexes())[0]
            rnn_policy = {"rnn": [s.transpose(0, 1) for s in sampled_rnn], "terminated": sampled_dones}

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states, train=True)
            sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

            with torch.no_grad():
                # target policy smoothing
                next_actions, _, _ = self.target_policy.act({"states": sampled_next_states, **rnn_policy}, role="target_policy")
                if self._smooth_regularization_noise is not None:
                    noises = torch.clamp(self._smooth_regularization_noise.sample(next_actions.shape),
                                        min=-self._smooth_regularization_clip,
                                        max=self._smooth_regularization_clip)
                    next_actions.add_(noises)
                    next_actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                # compute target values
                target_q1_values, _, _ = self.target_critic_1.act({"states": sampled_next_states, "taken_actions": next_actions, **rnn_policy}, role="target_critic_1")
                target_q2_values, _, _ = self.target_critic_2.act({"states": sampled_next_states, "taken_actions": next_actions, **rnn_policy}, role="target_critic_2")
                target_q_values = torch.min(target_q1_values, target_q2_values)
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute critic loss
            critic_1_values, _, _ = self.critic_1.act({"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="critic_1")
            critic_2_values, _, _ = self.critic_2.act({"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy}, role="critic_2")

            critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self._grad_norm_clip)
            self.critic_optimizer.step()

            # delayed update
            self._critic_update_counter += 1
            if not self._critic_update_counter % self._policy_delay:

                # compute policy (actor) loss
                actions, _, _ = self.policy.act({"states": sampled_states, **rnn_policy}, role="policy")
                critic_values, _, _ = self.critic_1.act({"states": sampled_states, "taken_actions": actions, **rnn_policy}, role="critic_1")

                policy_loss = -critic_values.mean()

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                self.policy_optimizer.step()

                # update target networks
                self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)
                self.target_policy.update_parameters(self.policy, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if not self._critic_update_counter % self._policy_delay:
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

            if self._learning_rate_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
