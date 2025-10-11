from __future__ import annotations

from typing import Any

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .mappo_cfg import MAPPO_CFG


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


class MAPPO(MultiAgent):
    def __init__(
        self,
        *,
        possible_agents: list[str],
        models: dict[str, dict[str, Model]],
        memories: dict[str, Memory] | None = None,
        observation_spaces: dict[str, gymnasium.Space] | None = None,
        state_spaces: dict[str, gymnasium.Space] | None = None,
        action_spaces: dict[str, gymnasium.Space] | None = None,
        device: str | torch.device | None = None,
        cfg: MAPPO_CFG | dict = {},
    ) -> None:
        """Multi-Agent Proximal Policy Optimization (MAPPO).

        https://arxiv.org/abs/2103.01955

        :param possible_agents: Name of all possible agents the environment could generate.
        :param models: Agents' models.
        :param memories: Memories to storage agents' data and environment transitions.
        :param observation_spaces: Observation spaces.
        :param state_spaces: State spaces.
        :param action_spaces: Action spaces.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Multi-agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: MAPPO_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=MAPPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}

        # checkpoint models
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]

        # broadcast models' parameters in distributed runs
        for uid in self.possible_agents:
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
                    if self.values[uid] is not None and self.policies[uid] is not self.values[uid]:
                        self.values[uid].broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizer and learning rate scheduler
        self.optimizers = {}
        self.schedulers = {}
        for uid in self.possible_agents:
            if self.policies[uid] is not None and self.values[uid] is not None:
                # - optimizers
                if self.policies[uid] is self.values[uid]:
                    self.optimizers[uid] = torch.optim.Adam(
                        self.policies[uid].parameters(), lr=self.cfg.learning_rate[uid][0]
                    )
                else:
                    self.optimizers[uid] = torch.optim.Adam(
                        itertools.chain(self.policies[uid].parameters(), self.values[uid].parameters()),
                        lr=self.cfg.learning_rate[uid][0],
                    )
                self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]
                # - learning rate schedulers
                self.schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0]
                if self.schedulers[uid] is not None:
                    self.schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0](
                        self.optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][0]
                    )

        # set up preprocessors
        self._observation_preprocessor = {}
        self._state_preprocessor = {}
        self._value_preprocessor = {}
        for uid in self.possible_agents:
            # - observations
            if self.cfg.observation_preprocessor[uid]:
                self._observation_preprocessor[uid] = self.cfg.observation_preprocessor[uid](
                    **self.cfg.observation_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["observation_preprocessor"] = self._observation_preprocessor[uid]
            else:
                self._observation_preprocessor[uid] = self._empty_preprocessor
            # - states
            if self.cfg.state_preprocessor[uid]:
                self._state_preprocessor[uid] = self.cfg.state_preprocessor[uid](
                    **self.cfg.state_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor
            # - values
            if self.cfg.value_preprocessor[uid]:
                self._value_preprocessor[uid] = self.cfg.value_preprocessor[uid](
                    **self.cfg.value_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memories
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(
                    name="observations", size=self.observation_spaces[uid], dtype=torch.float32
                )
                self.memories[uid].create_tensor(name="states", size=self.state_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = {}
        self._current_next_states = {}
        self._current_log_prob = {}
        self._rollout = 0

    def act(
        self,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        actions = {}
        log_prob = {}
        outputs = {}

        for uid in self.possible_agents:
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            # sample random actions
            # TODO, check for stochasticity
            if timestep < self.cfg.random_timesteps:
                actions[uid], outputs[uid] = self.policies[uid].random_act(inputs, role="policy")

            # sample stochastic actions
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")
                log_prob[uid] = outputs[uid]["log_prob"]

        self._current_log_prob = log_prob
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: dict[str, torch.Tensor],
        states: dict[str, torch.Tensor | None],
        actions: dict[str, torch.Tensor],
        rewards: dict[str, torch.Tensor],
        next_observations: dict[str, torch.Tensor],
        next_states: dict[str, torch.Tensor],
        terminated: dict[str, torch.Tensor],
        truncated: dict[str, torch.Tensor],
        infos: dict[str, Any],
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

        if self.memories:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            for uid in self.possible_agents:
                # reward shaping
                if self.cfg.rewards_shaper is not None:
                    rewards[uid] = self.cfg.rewards_shaper(rewards[uid], timestep, timesteps)

                # compute values
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](observations[uid]),
                        "states": self._state_preprocessor[uid](states[uid]),
                    }
                    values, _ = self.values[uid].act(inputs, role="value")
                    values = self._value_preprocessor[uid](values, inverse=True)

                # time-limit (truncation) bootstrapping
                if self.cfg.time_limit_bootstrap[uid]:
                    rewards[uid] += self.cfg.discount_factor[uid] * values * truncated[uid]

                # storage transition in memory
                self.memories[uid].add_samples(
                    observations=observations[uid],
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
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
                for uid in self.possible_agents:
                    self.update(timestep=timestep, timesteps=timesteps, uid=uid)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int, uid: str) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        :param uid: Agent ID.
        """
        policy = self.policies[uid]
        value = self.values[uid]
        memory = self.memories[uid]

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                "states": self._state_preprocessor[uid](self._current_next_states[uid]),
            }
            value.enable_training_mode(False)
            last_values, _ = value.act(inputs, role="value")
            value.enable_training_mode(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

        values = memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=memory.get_tensor_by_name("rewards"),
            dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor[uid],
            lambda_coefficient=self.cfg.lambda_[uid],
        )

        memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
        memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
        memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self.cfg.mini_batches[uid])

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs[uid]):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=not epoch),
                        "states": self._state_preprocessor[uid](sampled_states, train=not epoch),
                    }

                    _, outputs = policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self.cfg.kl_threshold[uid] and kl_divergence > self.cfg.kl_threshold[uid]:
                        break

                    # compute entropy loss
                    if self.cfg.entropy_loss_scale[uid]:
                        entropy_loss = -self.cfg.entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip[uid], 1.0 + self.cfg.ratio_clip[uid]
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _ = value.act(inputs, role="value")

                    if self.cfg.value_clip[uid] > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self.cfg.value_clip[uid],
                            max=self.cfg.value_clip[uid],
                        )
                    value_loss = self.cfg.value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizers[uid].zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    policy.reduce_parameters()
                    if policy is not value:
                        value.reduce_parameters()

                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.optimizers[uid])
                    if policy is value:
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(policy.parameters(), value.parameters()), self.cfg.grad_norm_clip[uid]
                        )

                self.scaler.step(self.optimizers[uid])
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale[uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self.schedulers[uid]:
                if isinstance(self.schedulers[uid], KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.schedulers[uid].step(kl.item())
                else:
                    self.schedulers[uid].step()

        # record data
        self.track_data(
            f"Loss / Policy loss ({uid})",
            cumulative_policy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        self.track_data(
            f"Loss / Value loss ({uid})",
            cumulative_value_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        if self.cfg.entropy_loss_scale[uid]:
            self.track_data(
                f"Loss / Entropy loss ({uid})",
                cumulative_entropy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
            )

        self.track_data(f"Policy / Standard deviation ({uid})", policy.distribution(role="policy").stddev.mean().item())

        if self.schedulers[uid]:
            self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])
