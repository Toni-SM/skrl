from __future__ import annotations

from typing import Any, Callable

import itertools
import math
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .amp_cfg import AMP_CFG


def compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> torch.Tensor:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param dones: Signals to indicate that episodes have ended.
    :param values: Values obtained by the agent.
    :param next_values: Next values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.

    :return: Generalized Advantage Estimator.
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_dones = dones.logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        advantage = (
            rewards[i] - values[i] + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


class AMP(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: AMP_CFG | dict = {},
        amp_observation_space: gymnasium.Space | None = None,
        motion_dataset: Memory | None = None,
        reply_buffer: Memory | None = None,
        collect_reference_motions: Callable[[int], torch.Tensor] | None = None,
        collect_observation: Callable[[], torch.Tensor] | None = None,
    ) -> None:
        """Adversarial Motion Priors (AMP).

        https://arxiv.org/abs/2104.02180

        .. note::

            The implementation is adapted from the NVIDIA IsaacGymEnvs repository.

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.
        :param amp_observation_space: AMP observation space.
        :param motion_dataset: Reference motion dataset (M).
        :param reply_buffer: Reply buffer for preventing discriminator overfitting (B).
        :param collect_reference_motions: Callable to collect reference motions.
        :param collect_observation: Callable to collect AMP observations.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: AMP_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=AMP_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        self.amp_observation_space = amp_observation_space
        self.motion_dataset = motion_dataset
        self.reply_buffer = reply_buffer
        self.collect_reference_motions = collect_reference_motions
        self.collect_observation = collect_observation

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        self.discriminator = self.models.get("discriminator", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        self.checkpoint_modules["discriminator"] = self.discriminator

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.value is not None:
                self.value.broadcast_parameters()
            if self.discriminator is not None:
                self.discriminator.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None and self.discriminator is not None:
            # - optimizers
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.policy.parameters(), self.value.parameters(), self.discriminator.parameters()),
                lr=self.cfg.learning_rate[0],
            )
            self.checkpoint_modules["optimizer"] = self.optimizer
            # - learning rate schedulers
            self.scheduler = self.cfg.learning_rate_scheduler[0]
            if self.scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler[0](
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )

        # set up preprocessors
        # - observations
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        # - values
        if self.cfg.value_preprocessor:
            self._value_preprocessor = self.cfg.value_preprocessor(**self.cfg.value_preprocessor_kwargs)
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor
        # - AMP observations
        if self.cfg.amp_observation_preprocessor:
            self._amp_observation_preprocessor = self.cfg.amp_observation_preprocessor(
                **self.cfg.amp_observation_preprocessor_kwargs
            )
            self.checkpoint_modules["amp_observation_preprocessor"] = self._amp_observation_preprocessor
        else:
            self._amp_observation_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="amp_observations", size=self.amp_observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)

        self.tensors_names = [
            "observations",
            "states",
            "actions",
            "rewards",
            "next_observations",
            "next_states",
            "terminated",
            "log_prob",
            "values",
            "returns",
            "advantages",
            "amp_observations",
            "next_values",
        ]

        # create tensors for motion dataset and reply buffer
        if self.motion_dataset is not None:
            self.motion_dataset.create_tensor(name="observations", size=self.amp_observation_space, dtype=torch.float32)
            self.reply_buffer.create_tensor(name="observations", size=self.amp_observation_space, dtype=torch.float32)

            # initialize motion dataset
            for _ in range(math.ceil(self.motion_dataset.memory_size / self.cfg.amp_batch_size)):
                self.motion_dataset.add_samples(observations=self.collect_reference_motions(self.cfg.amp_batch_size))

        # create temporary variables needed for storage and computation
        self._current_observations = None
        self._current_states = None
        self._current_log_prob = None
        self._rollout = 0

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        # use collected observations/states
        if self._current_observations is not None:
            observations = self._current_observations
        if self._current_states is not None:
            states = self._current_states

        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # use collected observations/states
        if self._current_observations is not None:
            observations = self._current_observations
        if self._current_states is not None:
            states = self._current_states

        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.memory is not None:
            amp_observations = infos["amp_obs"]

            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(observations),
                    "states": self._state_preprocessor(states),
                }
                values, _ = self.value.act(inputs, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self.cfg.time_limit_bootstrap:
                rewards += self.cfg.discount_factor * values * truncated

            # compute next values
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(next_observations),
                    "states": self._state_preprocessor(next_states),
                }
                next_values, _ = self.value.act(inputs, role="value")
                next_values = self._value_preprocessor(next_values, inverse=True)
                if "terminate" in infos:
                    next_values *= infos["terminate"].view(-1, 1).logical_not()  # compatibility with IsaacGymEnvs
                else:
                    next_values *= terminated.view(-1, 1).logical_not()

            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                amp_observations=amp_observations,
                next_values=next_values,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # compatibility with IsaacGymEnvs
        if self.collect_observation is not None:
            self._current_observations = self.collect_observation()

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        self._rollout += 1
        if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
            with ScopedTimer() as timer:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # update dataset of reference motions
        self.motion_dataset.add_samples(observations=self.collect_reference_motions(self.cfg.amp_batch_size))

        # compute combined rewards
        rewards = self.memory.get_tensor_by_name("rewards")
        amp_observations = self.memory.get_tensor_by_name("amp_observations")

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            amp_logits, _ = self.discriminator.act(
                {"observations": self._amp_observation_preprocessor(amp_observations)}, role="discriminator"
            )
            style_reward = -torch.log(
                torch.maximum(1 - 1 / (1 + torch.exp(-amp_logits)), torch.tensor(0.0001, device=self.device))
            ).view(rewards.shape)

        combined_rewards = self.cfg.task_reward_scale * rewards + self.cfg.style_reward_scale * style_reward

        # compute returns and advantages
        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")
        returns, advantages = compute_gae(
            rewards=combined_rewards,
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=next_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.lambda_,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self.cfg.mini_batches)
        sampled_motion_batches = self.motion_dataset.sample(
            names=["observations"],
            batch_size=self.memory.memory_size * self.memory.num_envs,
            mini_batches=self.cfg.mini_batches,
        )
        if len(self.reply_buffer):
            sampled_replay_batches = self.reply_buffer.sample(
                names=["observations"],
                batch_size=self.memory.memory_size * self.memory.num_envs,
                mini_batches=self.cfg.mini_batches,
            )
        else:
            sampled_replay_batches = [
                [batches[self.tensors_names.index("amp_observations")]] for batches in sampled_batches
            ]

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_discriminator_loss = 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for batch_index, (
                sampled_observations,
                sampled_states,
                sampled_actions,
                _,
                _,
                _,
                _,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_amp_observations,
                _,
            ) in enumerate(sampled_batches):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": self._state_preprocessor(sampled_states, train=not epoch),
                    }

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # compute entropy loss
                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _ = self.value.act(inputs, role="value")

                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # compute discriminator loss
                    if self.cfg.discriminator_batch_size > 0:
                        sampled_amp_observations = self._amp_observation_preprocessor(
                            sampled_amp_observations[0 : self.cfg.discriminator_batch_size], train=True
                        )
                        sampled_amp_replay_observations = self._amp_observation_preprocessor(
                            sampled_replay_batches[batch_index][0][0 : self.cfg.discriminator_batch_size], train=True
                        )
                        sampled_amp_motion_observations = self._amp_observation_preprocessor(
                            sampled_motion_batches[batch_index][0][0 : self.cfg.discriminator_batch_size], train=True
                        )
                    else:
                        sampled_amp_observations = self._amp_observation_preprocessor(
                            sampled_amp_observations, train=True
                        )
                        sampled_amp_replay_observations = self._amp_observation_preprocessor(
                            sampled_replay_batches[batch_index][0], train=True
                        )
                        sampled_amp_motion_observations = self._amp_observation_preprocessor(
                            sampled_motion_batches[batch_index][0], train=True
                        )

                    sampled_amp_motion_observations.requires_grad_(True)
                    amp_logits, _ = self.discriminator.act(
                        {"observations": sampled_amp_observations}, role="discriminator"
                    )
                    amp_replay_logits, _ = self.discriminator.act(
                        {"observations": sampled_amp_replay_observations}, role="discriminator"
                    )
                    amp_motion_logits, _ = self.discriminator.act(
                        {"observations": sampled_amp_motion_observations}, role="discriminator"
                    )

                    amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)

                    # discriminator prediction loss
                    discriminator_loss = 0.5 * (
                        nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits))
                        + torch.nn.BCEWithLogitsLoss()(amp_motion_logits, torch.ones_like(amp_motion_logits))
                    )

                    # discriminator logit regularization
                    if self.cfg.discriminator_logit_regularization_scale:
                        logit_weights = torch.flatten(list(self.discriminator.modules())[-1].weight)
                        discriminator_loss += self.cfg.discriminator_logit_regularization_scale * torch.sum(
                            torch.square(logit_weights)
                        )

                    # discriminator gradient penalty
                    if self.cfg.discriminator_gradient_penalty_scale:
                        amp_motion_gradient = torch.autograd.grad(
                            amp_motion_logits,
                            sampled_amp_motion_observations,
                            grad_outputs=torch.ones_like(amp_motion_logits),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )
                        gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                        discriminator_loss += self.cfg.discriminator_gradient_penalty_scale * gradient_penalty

                    # discriminator weight decay
                    if self.cfg.discriminator_weight_decay_scale:
                        weights = [
                            torch.flatten(module.weight)
                            for module in self.discriminator.modules()
                            if isinstance(module, torch.nn.Linear)
                        ]
                        weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                        discriminator_loss += self.cfg.discriminator_weight_decay_scale * weight_decay

                    discriminator_loss *= self.cfg.discriminator_loss_scale

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + discriminator_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    self.value.reduce_parameters()
                    self.discriminator.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.policy.parameters(), self.value.parameters(), self.discriminator.parameters()
                        ),
                        self.cfg.grad_norm_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_discriminator_loss += discriminator_loss.item()

            # update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # update AMP replay buffer
        self.reply_buffer.add_samples(observations=amp_observations.view(-1, amp_observations.shape[-1]))

        # record data
        n = self.cfg.learning_epochs * self.cfg.mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / n)
        self.track_data("Loss / Value loss", cumulative_value_loss / n)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)
        self.track_data("Loss / Discriminator loss", cumulative_discriminator_loss / n)

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
