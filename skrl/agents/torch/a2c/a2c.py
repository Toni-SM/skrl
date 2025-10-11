from __future__ import annotations

from typing import Any

import itertools
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

from .a2c_cfg import A2C_CFG


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
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


class A2C(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: A2C_CFG | dict = {},
    ) -> None:
        """Advantage Actor Critic (A2C).

        https://arxiv.org/abs/1602.01783

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: A2C_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=A2C_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            # - optimizers
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self.cfg.learning_rate[0]
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

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
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
            self._current_next_observations = next_observations
            self._current_next_states = next_states

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

            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

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
        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.lambda_,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self.cfg.mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        kl_divergences = []

        # mini-batches loop
        for (
            sampled_observations,
            sampled_states,
            sampled_actions,
            sampled_log_prob,
            sampled_returns,
            sampled_advantages,
        ) in sampled_batches:

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                    "states": self._state_preprocessor(sampled_states, train=True),
                }

                _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                next_log_prob = outputs["log_prob"]

                # compute approximate KL divergence for KLAdaptive learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, KLAdaptiveLR):
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
                policy_loss = -(sampled_advantages * next_log_prob).mean()

                # compute value loss
                predicted_values, _ = self.value.act(inputs, role="value")

                value_loss = F.mse_loss(sampled_returns, predicted_values)

            # optimization step
            self.optimizer.zero_grad()
            self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()
                if self.policy is not self.value:
                    self.value.reduce_parameters()

            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.optimizer)
                if self.policy is self.value:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                else:
                    nn.utils.clip_grad_norm_(
                        itertools.chain(self.policy.parameters(), self.value.parameters()), self.cfg.grad_norm_clip
                    )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update cumulative losses
            cumulative_policy_loss += policy_loss.item()
            cumulative_value_loss += value_loss.item()
            if self.cfg.entropy_loss_scale:
                cumulative_entropy_loss += entropy_loss.item()

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

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / len(sampled_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / len(sampled_batches))

        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / len(sampled_batches))

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
