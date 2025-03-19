from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


# fmt: off
# [start-config-dict-torch]
CROSSQ_DEFAULT_CONFIG = {
    "policy_delay" : 3,
    "gradient_steps": 1,            # gradient steps
    "batch_size": 256,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "optimizer_kwargs" : {
        "betas": [0.5, 0.999]
    },

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 1.0,   # initial entropy value
    "target_entropy": None,         # target entropy

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class CrossQ(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """CrossQ

        https://arxiv.org/abs/1902.05605

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(CROSSQ_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)

        assert (
            getattr(self.policy, "set_bn_training_mode", None) is not None
        ), "Policy has no required method 'set_bn_training_mode'"
        assert (
            getattr(self.critic_1, "set_bn_training_mode", None) is not None
        ), "Critic 1 has no required method 'set_bn_training_mode'"
        assert (
            getattr(self.critic_2, "set_bn_training_mode", None) is not None
        ), "Critic 2 has no required method 'set_bn_training_mode'"

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        # configuration
        self.policy_delay = self.cfg["policy_delay"]
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]
        self.optimizer_kwargs = self.cfg["optimizer_kwargs"]

        self.n_updates = 0

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self._entropy_learning_rate)

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self._actor_learning_rate, **self.optimizer_kwargs
            )
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                lr=self._critic_learning_rate,
                **self.optimizer_kwargs,
            )
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

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
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")

        return actions, None, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
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
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

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

        # update learning rate
        if self._learning_rate_scheduler:
            self.policy_scheduler.step()
            self.critic_scheduler.step()
        # print("Time step: ", timestep)
        # gradient steps
        for gradient_step in range(self._gradient_steps):
            self.n_updates += 1
            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            if self._learn_entropy:
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                with torch.no_grad():
                    self.policy.set_bn_training_mode(False)
                    next_actions, next_log_prob, _ = self.policy.act(
                        {"states": sampled_next_states}, role="policy", should_log_prob=True
                    )
                    # print(f"next_actions : {next_actions[0]}")
                    # print(f"next_log_prob : {next_log_prob[0]}")

                all_states = torch.cat((sampled_states, sampled_next_states))
                all_actions = torch.cat((sampled_actions, next_actions))

                # print(f"all_states : {all_states[0]}, {all_states[256]}")
                # print(f"all_actions : {all_actions[0]}, {all_actions[256]}")

                self.critic_1.set_bn_training_mode(True)
                self.critic_2.set_bn_training_mode(True)
                all_q1, _, _ = self.critic_1.act({"states": all_states, "taken_actions": all_actions}, role="critic_1")
                all_q2, _, _ = self.critic_2.act({"states": all_states, "taken_actions": all_actions}, role="critic_2")
                self.critic_1.set_bn_training_mode(False)
                self.critic_2.set_bn_training_mode(False)

                q1, next_q1 = torch.split(all_q1, split_size_or_sections=self._batch_size)
                q2, next_q2 = torch.split(all_q2, split_size_or_sections=self._batch_size)

                # print(f"q1 : {q1[0]}")
                # print(f"q2 : {q2[0]}")

                # compute target values
                with torch.no_grad():
                    next_q = torch.minimum(next_q1.detach(), next_q2.detach())
                    target_q_values = next_q - self._entropy_coefficient * next_log_prob.reshape(-1, 1)
                    target_values: torch.Tensor = (
                        sampled_rewards + self._discount_factor * (sampled_terminated).logical_not() * target_q_values
                    )
                # compute critic loss
                critic_loss = 0.5 * (F.mse_loss(q1, target_values.detach()) + F.mse_loss(q2, target_values.detach()))
            # print(f"critic_loss : {critic_loss}")
            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self._grad_norm_clip
                )

            # TODO : CHECK UPDATED WEIGHTS
            self.scaler.step(self.critic_optimizer)
            # HERE

            should_update_policy = self.n_updates % self.policy_delay == 0
            if should_update_policy:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # compute policy (actor) loss
                    self.policy.set_bn_training_mode(True)
                    actions, log_prob, _ = self.policy.act(
                        {"states": sampled_states}, role="policy", should_log_prob=True
                    )
                    log_prob = log_prob.reshape(-1, 1)
                    self.policy.set_bn_training_mode(False)

                    # entropy learning
                    if self._learn_entropy:
                        # compute entropy loss
                        entropy_loss = -(
                            self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()
                        ).mean()

                if self._learn_entropy:
                    # optimization step (entropy)
                    self.entropy_optimizer.zero_grad()
                    self.scaler.scale(entropy_loss).backward()
                    self.scaler.step(self.entropy_optimizer)

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    self.critic_1.set_bn_training_mode(False)
                    self.critic_2.set_bn_training_mode(False)
                    critic_1_values, _, _ = self.critic_1.act(
                        {"states": sampled_states, "taken_actions": actions}, role="critic_1"
                    )
                    critic_2_values, _, _ = self.critic_2.act(
                        {"states": sampled_states, "taken_actions": actions}, role="critic_2"
                    )
                    q_pi = torch.minimum(critic_1_values, critic_2_values)
                    policy_loss = (self._entropy_coefficient * log_prob - q_pi).mean()

                # print(f"policy_loss : {policy_loss}")
                # print(f"entropy_loss : {entropy_loss}")
                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

                self.scaler.step(self.policy_optimizer)

            self.scaler.update()  # called once, after optimizers have been stepped

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())

                if should_update_policy:
                    self.track_data("Loss / Policy loss", policy_loss.item())

                    self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
                    self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
                    self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

                    self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
                    self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
                    self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

                    if self._learn_entropy:
                        self.track_data("Loss / Entropy loss", entropy_loss.item())

                if self._learn_entropy:
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self._learning_rate_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                    self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
