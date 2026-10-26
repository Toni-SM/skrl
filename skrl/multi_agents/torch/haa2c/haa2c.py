from __future__ import annotations

from typing import Any

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

from .haa2c_cfg import HAA2C_CFG


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param terminated: Signals to indicate that episodes have ended.
    :param truncated: Signals to indicate that episodes have been truncated.
    :param values: Values obtained by the agent.
    :param last_values: Last values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.
    :param time_limit_bootstrap: Whether to use time-limit (truncation) bootstrapping.

    :return: Returns and (normalized) Generalized Advantage Estimator.
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


class HAA2C(MultiAgent):
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
        cfg: HAA2C_CFG | dict = {},
    ) -> None:
        """Heterogeneous-Agent Advantage Actor-Critic (HAA2C).

        https://jmlr.org/papers/v25/23-0488.html

        The agents' policies are updated sequentially (one agent after the other), and each policy update takes into
        account the updates of the policies that were updated before it. The value model can be either shared by all
        the agents (the same model instance is given to every agent) or specific to each agent (a distinct instance
        is given to every agent). A shared value model is trained once per update on the team reward
        (mean of the agents' rewards).

        :param possible_agents: Name of all possible agents the environment could generate.
        :param models: Agents' models.
        :param memories: Memories to storage agents' data and environment transitions.
        :param observation_spaces: Observation spaces.
        :param state_spaces: State spaces.
        :param action_spaces: Action spaces.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Multi-agent's configuration.

        :raises KeyError: If a configuration key is missing.
        :raises ValueError: If the same policy instance is given to several agents, if the value models are shared
            by only some of the agents, if an agent's policy and value are the same model, or if the state spaces or
            value-related configuration of a shared value model differ between agents.
        """
        self.cfg: HAA2C_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=HAA2C_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}

        # policies: specific to each agent (the same instance must not be given to several agents)
        self._check_distinct_instances("Model 'policy'", self.policies)

        # value model: shared by all the agents or specific to each agent
        self._shared_value = self._resolve_shared_models(["value"])
        # - team of agents whose transitions train each value model (keyed by the agent that owns the value model).
        #   A shared value model is owned by the first agent and trained on the transitions of all the agents
        self._value_agents = (
            {self.possible_agents[0]: self.possible_agents}
            if self._shared_value
            else {uid: [uid] for uid in self.possible_agents}
        )
        for uid in self.possible_agents:
            if self.policies[uid] is not None and self.policies[uid] is self.values[uid]:
                raise ValueError(
                    f"The policy and value of agent '{uid}' must be different models, "
                    "since HAA2C updates the value after all the policies"
                )
        if self._shared_value:
            first_uid = self.possible_agents[0]
            for uid in self.possible_agents[1:]:
                if self.state_spaces and self.state_spaces[uid] != self.state_spaces[first_uid]:
                    raise ValueError(
                        f"The agents' state spaces must be equal when the value model is shared, "
                        f"got {self.state_spaces[first_uid]} ('{first_uid}') and {self.state_spaces[uid]} ('{uid}')"
                    )
            self._check_shared_settings(
                [
                    "discount_factor",
                    "gae_lambda",
                    "value_clip",
                    "value_loss_scale",
                    "value_loss_type",
                    "huber_delta",
                    "time_limit_bootstrap",
                    "learning_epochs",
                    "mini_batches",
                    "grad_norm_clip",
                    "state_preprocessor",
                    "state_preprocessor_kwargs",
                    "value_preprocessor",
                    "value_preprocessor_kwargs",
                ]
            )
            self._check_shared_settings(
                ["learning_rate", "learning_rate_scheduler", "learning_rate_scheduler_kwargs"], index=1
            )

        # checkpoint models
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]

        # broadcast models' parameters in distributed runs (a shared value model only once)
        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            for uid in self.possible_agents:
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
            for uid in self._value_agents:
                if self.values[uid] is not None:
                    self.values[uid].broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizers and learning rate schedulers (separate for the policy and the value)
        self.policy_optimizers = {}
        self.value_optimizers = {}
        self.policy_schedulers = {}
        self.value_schedulers = {}
        for uid in self.possible_agents:
            # - policy
            if self.policies[uid] is not None:
                self.policy_optimizers[uid] = torch.optim.Adam(
                    self.policies[uid].parameters(), lr=self.cfg.learning_rate[uid][0]
                )
                self.checkpoint_modules[uid]["policy_optimizer"] = self.policy_optimizers[uid]
                self.policy_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0]
                if self.policy_schedulers[uid] is not None:
                    self.policy_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0](
                        self.policy_optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][0]
                    )
            # - value (a shared value model has a single optimizer and learning rate scheduler)
            if self.values[uid] is not None:
                if uid in self._value_agents:
                    self.value_optimizers[uid] = torch.optim.Adam(
                        self.values[uid].parameters(), lr=self.cfg.learning_rate[uid][1]
                    )
                    self.value_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][1]
                    if self.value_schedulers[uid] is not None:
                        self.value_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][1](
                            self.value_optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][1]
                        )
                else:
                    self.value_optimizers[uid] = self.value_optimizers[self.possible_agents[0]]
                    self.value_schedulers[uid] = self.value_schedulers[self.possible_agents[0]]
                self.checkpoint_modules[uid]["value_optimizer"] = self.value_optimizers[uid]

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
            # - states and values (a shared value model has a single state and value preprocessor)
            if uid in self._value_agents:
                if self.cfg.state_preprocessor[uid]:
                    self._state_preprocessor[uid] = self.cfg.state_preprocessor[uid](
                        **self.cfg.state_preprocessor_kwargs[uid]
                    )
                else:
                    self._state_preprocessor[uid] = self._empty_preprocessor
                if self.cfg.value_preprocessor[uid]:
                    self._value_preprocessor[uid] = self.cfg.value_preprocessor[uid](
                        **self.cfg.value_preprocessor_kwargs[uid]
                    )
                else:
                    self._value_preprocessor[uid] = self._empty_preprocessor
            else:
                self._state_preprocessor[uid] = self._state_preprocessor[self.possible_agents[0]]
                self._value_preprocessor[uid] = self._value_preprocessor[self.possible_agents[0]]
            if self.cfg.state_preprocessor[uid]:
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            if self.cfg.value_preprocessor[uid]:
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]

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
                self.memories[uid].create_tensor(name="factor", size=1, dtype=torch.float32)

            self._policy_tensors_names = ["observations", "states", "actions", "log_prob", "advantages", "factor"]
            self._value_tensors_names = ["observations", "states", "values", "returns"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = {}
        self._current_next_states = {}
        self._current_log_prob = {}
        self._current_values = {}
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
        current_values = {}

        for uid in self.possible_agents:
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            # sample stochastic actions
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")
                log_prob[uid] = outputs[uid]["log_prob"]

                # compute values (a shared value model is evaluated once and its values are used for all the agents)
                if self.training and uid in self._value_agents:
                    values, _ = self.values[uid].act(inputs, role="value")
                    values = self._value_preprocessor[uid](values, inverse=True)
                    current_values.update({team_uid: values for team_uid in self._value_agents[uid]})

        self._current_log_prob = log_prob
        self._current_values = current_values
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

        if self.training:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # compute next values for time-limit (truncation) bootstrapping
            # (a shared value model is evaluated once and its values are used for all the agents)
            next_values = {}
            for uid, team in self._value_agents.items():
                if self.cfg.time_limit_bootstrap[uid] and any(truncated[team_uid].any() for team_uid in team):
                    # bootstrap from the observations/states with which the transitions ended
                    # (they differ from the next ones if the environment is auto-reset at the same step)
                    final_observations, final_states = self._get_final_observations_and_states(
                        next_observations, next_states, infos
                    )
                    with torch.no_grad():
                        inputs = {
                            "observations": self._observation_preprocessor[uid](final_observations[uid]),
                            "states": self._state_preprocessor[uid](final_states[uid]),
                        }
                        values, _ = self.values[uid].act(inputs, role="value")
                        values = self._value_preprocessor[uid](values, inverse=True)
                    next_values.update({team_uid: values for team_uid in team})

            for uid in self.possible_agents:
                # reward shaping
                if self.cfg.rewards_shaper is not None:
                    rewards[uid] = self.cfg.rewards_shaper(rewards[uid], timestep, timesteps)

                # time-limit (truncation) bootstrapping
                if uid in next_values:
                    rewards[uid] += self.cfg.discount_factor[uid] * next_values[uid] * truncated[uid]

                # storage transition in memory
                self.memories[uid].add_samples(
                    observations=observations[uid],
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=self._current_values[uid],
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
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        The policies are updated sequentially, in a fixed or random order. Each policy's (unclipped) surrogate
        objective is weighted by the product of the probability ratios (between the updated and the previous policy)
        of the agents updated before it (``factor``). Then, the value models are updated.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # compute returns and advantages
        # (a shared value model computes them once, from the team reward, for all the agents)
        for uid, team in self._value_agents.items():
            value = self.values[uid]
            memory = self.memories[uid]

            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                    "states": self._state_preprocessor[uid](self._current_next_states[uid]),
                }
                value.enable_training_mode(False)
                last_values, _ = value.act(inputs, role="value")
                value.enable_training_mode(True)
                last_values = self._value_preprocessor[uid](last_values, inverse=True)

            # team reward (mean of the agents' rewards) and episode end (all the agents have terminated/truncated)
            team_memories = [self.memories[team_uid] for team_uid in team]
            rewards = torch.stack([m.get_tensor_by_name("rewards") for m in team_memories]).mean(dim=0)
            terminated = torch.stack([m.get_tensor_by_name("terminated") for m in team_memories]).all(dim=0)
            truncated = torch.stack([m.get_tensor_by_name("truncated") for m in team_memories]).all(dim=0)

            values = memory.get_tensor_by_name("values")
            returns, advantages = compute_gae(
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                values=values,
                last_values=last_values,
                discount_factor=self.cfg.discount_factor[uid],
                lambda_coefficient=self.cfg.gae_lambda[uid],
                time_limit_bootstrap=self.cfg.time_limit_bootstrap[uid],
            )

            values = self._value_preprocessor[uid](values, train=True)
            returns = self._value_preprocessor[uid](returns, train=True)
            for team_memory in team_memories:
                team_memory.set_tensor_by_name("values", values)
                team_memory.set_tensor_by_name("returns", returns)
                team_memory.set_tensor_by_name("advantages", advantages)

        # agents' update order: possible agents' order or a random permutation of it
        if self.cfg.fixed_order:
            order = self.possible_agents
        else:
            permutation = torch.randperm(self.num_agents, device=self.device)
            # use the same order in all workers/processes in distributed runs
            if config.torch.is_distributed:
                torch.distributed.broadcast(permutation, src=0)
            order = [self.possible_agents[i] for i in permutation.tolist()]

        # sequential policy updates
        # (factor: product of the probability ratios of the previously updated agents' stored actions)
        kl_divergences = {}
        factor = torch.ones_like(self.memories[order[0]].get_tensor_by_name("factor"))
        for uid in order:
            self.memories[uid].set_tensor_by_name("factor", factor)
            log_prob = self._log_prob(uid)
            kl_divergences[uid] = self._update_policy(uid)
            factor = factor * torch.exp(self._log_prob(uid) - log_prob)

        # value updates
        for uid, team in self._value_agents.items():
            # KL divergence of each learning epoch, averaged over the policies of the team
            team_kl_divergences = [kl_divergences[team_uid] for team_uid in team]
            self._update_value(uid, kl_divergences=[sum(kl) / len(kl) for kl in zip(*team_kl_divergences)])

    def _log_prob(self, uid: str) -> torch.Tensor:
        """Compute the log probability of the actions stored in the agent's memory using its current policy.

        :param uid: Agent ID.

        :return: Log probability of the stored actions, with shape ``(memory_size, num_envs, 1)``.
        """
        memory = self.memories[uid]
        observations, states, actions = memory.sample_all(names=["observations", "states", "actions"])[0]
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor[uid](observations),
                "states": self._state_preprocessor[uid](states),
            }
            _, outputs = self.policies[uid].act({**inputs, "taken_actions": actions}, role="policy")
        return outputs["log_prob"].reshape(memory.memory_size, memory.num_envs, 1)

    def _update_policy(self, uid: str) -> list[float]:
        """Update the agent's policy.

        :param uid: Agent ID.

        :return: Mean approximate KL divergence (between the policy that collected the data and the policy being
            optimized) of each learning epoch.
        """
        policy = self.policies[uid]
        memory = self.memories[uid]

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        epoch_kl_divergences = []

        # learning epochs
        for epoch in range(self.cfg.learning_epochs[uid]):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_advantages,
                sampled_factor,
            ) in memory.sample(
                names=self._policy_tensors_names, batch_size=len(memory), mini_batches=self.cfg.mini_batches[uid]
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=not epoch),
                        "states": self._state_preprocessor[uid](sampled_states),
                    }

                    _, outputs = policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence for KLAdaptive learning rate schedulers
                    log_ratio = next_log_prob - sampled_log_prob
                    ratio = torch.exp(log_ratio)
                    with torch.no_grad():
                        kl_divergences.append(((ratio - 1) - log_ratio).mean())

                    # compute entropy loss
                    if self.cfg.entropy_loss_scale[uid]:
                        entropy_loss = -self.cfg.entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss (weighted by the factor of the previously updated agents)
                    policy_loss = -(sampled_factor * ratio * sampled_advantages).mean()

                # optimization step
                self.policy_optimizers[uid].zero_grad()
                self.scaler.scale(policy_loss + entropy_loss).backward()

                if config.torch.is_distributed:
                    policy.reduce_parameters()

                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.policy_optimizers[uid])
                    nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])

                self.scaler.step(self.policy_optimizers[uid])
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                if self.cfg.entropy_loss_scale[uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            # compute the epoch's KL divergence (also used by the value's learning rate scheduler)
            kl = torch.tensor(kl_divergences, device=self.device).mean()
            # reduce (collect from all workers/processes) KL in distributed runs
            if config.torch.is_distributed:
                torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                kl /= config.torch.world_size
            epoch_kl_divergences.append(kl.item())

            # update learning rate
            if self.policy_schedulers[uid]:
                if isinstance(self.policy_schedulers[uid], KLAdaptiveLR):
                    self.policy_schedulers[uid].step(epoch_kl_divergences[-1])
                else:
                    self.policy_schedulers[uid].step()

        # record data
        self.track_data(
            f"Loss / Policy loss ({uid})",
            cumulative_policy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )
        if self.cfg.entropy_loss_scale[uid]:
            self.track_data(
                f"Loss / Entropy loss ({uid})",
                cumulative_entropy_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
            )

        self.track_data(f"Policy / Standard deviation ({uid})", policy.distribution(role="policy").stddev.mean().item())

        if self.policy_schedulers[uid]:
            self.track_data(f"Learning / Policy learning rate ({uid})", self.policy_schedulers[uid].get_last_lr()[0])

        return epoch_kl_divergences

    def _update_value(self, uid: str, *, kl_divergences: list[float]) -> None:
        """Update the value model owned by the agent (the first agent for a shared value model).

        :param uid: Agent ID.
        :param kl_divergences: Policy KL divergence of each learning epoch (used by a KL adaptive scheduler).
        """
        value = self.values[uid]
        memory = self.memories[uid]

        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs[uid]):

            # mini-batches loop
            for sampled_observations, sampled_states, sampled_values, sampled_returns in memory.sample(
                names=self._value_tensors_names, batch_size=len(memory), mini_batches=self.cfg.mini_batches[uid]
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations),
                        "states": self._state_preprocessor[uid](sampled_states, train=not epoch),
                    }

                    # compute value loss
                    predicted_values, _ = value.act(inputs, role="value")

                    if self.cfg.value_clip[uid] > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self.cfg.value_clip[uid],
                            max=self.cfg.value_clip[uid],
                        )
                    value_loss = self.cfg.value_loss_scale[uid] * (
                        F.huber_loss(predicted_values, sampled_returns, delta=self.cfg.huber_delta[uid])
                        if self.cfg.value_loss_type[uid] == "huber"
                        else F.mse_loss(sampled_returns, predicted_values)
                    )

                # optimization step
                self.value_optimizers[uid].zero_grad()
                self.scaler.scale(value_loss).backward()

                if config.torch.is_distributed:
                    value.reduce_parameters()

                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.value_optimizers[uid])
                    nn.utils.clip_grad_norm_(value.parameters(), self.cfg.grad_norm_clip[uid])

                self.scaler.step(self.value_optimizers[uid])
                self.scaler.update()

                # update cumulative losses
                cumulative_value_loss += value_loss.item()

            # update learning rate
            if self.value_schedulers[uid]:
                if isinstance(self.value_schedulers[uid], KLAdaptiveLR):
                    self.value_schedulers[uid].step(kl_divergences[epoch])
                else:
                    self.value_schedulers[uid].step()

        # record data (the agent ID is omitted for a shared value model)
        suffix = "" if self._shared_value else f" ({uid})"
        self.track_data(
            f"Loss / Value loss{suffix}",
            cumulative_value_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )

        if self.value_schedulers[uid]:
            self.track_data(f"Learning / Value learning rate{suffix}", self.value_schedulers[uid].get_last_lr()[0])
