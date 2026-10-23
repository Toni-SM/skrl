from __future__ import annotations

from typing import Any, Sequence

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
from skrl.utils import ScopedTimer
from skrl.utils.spaces.torch import compute_joint_space, compute_space_limits, compute_space_size

from .hatd3_cfg import HATD3_CFG


def sample_joint_batch(
    memories: dict[str, Memory], possible_agents: Sequence[str], names: list[str], batch_size: int
) -> dict[str, list[torch.Tensor]]:
    """Sample the same transitions (same memory indexes) from the memories of all the agents.

    A batch is sampled from the first agent's memory, and the transitions at the same indexes
    are gathered from the memories of the other agents.

    :param memories: Agents' memories. All the memories must have the same memory size and number of environments.
    :param possible_agents: Name of all possible agents.
    :param names: Tensors names from which to obtain the samples.
    :param batch_size: Number of elements to sample.

    :return: Sampled data from the tensors of each agent's memory, sorted according to their position in the list
        of names.

    :raises ValueError: If the first agent's memory does not provide the indexes of the sampled transitions.
    """
    batch = {possible_agents[0]: memories[possible_agents[0]].sample(names, batch_size=batch_size)[0]}
    indexes = memories[possible_agents[0]].get_sampling_indexes()
    if indexes is None:
        raise ValueError(
            f"The memory of agent '{possible_agents[0]}' ({type(memories[possible_agents[0]]).__name__}) does not "
            "provide the indexes of the sampled transitions (get_sampling_indexes() returned None), which are "
            "required to sample the same transitions from the memories of all the agents"
        )
    for uid in possible_agents[1:]:
        batch[uid] = memories[uid].sample_by_index(names, indexes=indexes)[0]
    return batch


class HATD3(MultiAgent):
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
        cfg: HATD3_CFG | dict = {},
    ) -> None:
        """Heterogeneous-Agent Twin Delayed DDPG (HATD3).

        https://jmlr.org/papers/v25/23-0488.html

        The critics (twin Q-networks) are centralized: they take the states and the joint action of all the agents (the
        agents' actions concatenated in the order of the possible agents), so they must be built with the joint action
        space (see :py:func:`~skrl.utils.spaces.torch.compute_joint_space`, ``"concatenate"`` mode). The critics can be
        either shared by all the agents (the same model instances are given to every agent) or specific to each agent
        (distinct instances are given to every agent). A shared critic is trained on the team reward (mean of the
        agents' rewards). The target actions are smoothed with a per-agent noise (target policy smoothing). The agents'
        policies are updated sequentially (one agent after the other) every ``policy_delay`` critic updates, and each
        policy update takes into account the updated actions of the agents updated before it.

        :param possible_agents: Name of all possible agents the environment could generate.
        :param models: Agents' models.
        :param memories: Memories to storage agents' data and environment transitions.
            All the memories must have the same memory size and number of environments.
        :param observation_spaces: Observation spaces.
        :param state_spaces: State spaces.
        :param action_spaces: Action spaces.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Multi-agent's configuration.

        :raises KeyError: If a configuration key is missing.
        :raises ValueError: If the configuration is not valid (e.g.: ``policy_delay`` is less than 1), if the same
            policy, target policy or memory instance is given to several agents, if the critics are shared by only
            some of the agents (or only some of the critic roles are shared), if the critics are not built with the
            joint action space, if the memories have different sizes or number of environments, or if the state
            spaces or critic-related configuration of a shared critic differ between agents.
        """
        self.cfg: HATD3_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=HATD3_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )
        self.cfg.validate()

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.target_policies = {uid: self.models[uid].get("target_policy", None) for uid in self.possible_agents}
        self.critics_1 = {uid: self.models[uid].get("critic_1", None) for uid in self.possible_agents}
        self.critics_2 = {uid: self.models[uid].get("critic_2", None) for uid in self.possible_agents}
        self.target_critics_1 = {uid: self.models[uid].get("target_critic_1", None) for uid in self.possible_agents}
        self.target_critics_2 = {uid: self.models[uid].get("target_critic_2", None) for uid in self.possible_agents}
        critic_models = {
            "critic_1": self.critics_1,
            "critic_2": self.critics_2,
            "target_critic_1": self.target_critics_1,
            "target_critic_2": self.target_critics_2,
        }

        # policies: specific to each agent (the same instance must not be given to several agents)
        self._check_distinct_instances("Model 'policy'", self.policies)
        self._check_distinct_instances("Model 'target_policy'", self.target_policies)

        # critics: shared by all the agents or specific to each agent
        self._shared_critic = self._resolve_shared_models(list(critic_models))
        # - team of agents whose transitions train each critic (keyed by the agent that owns the critic).
        #   A shared critic is owned by the first agent and trained on the transitions of all the agents
        self._critic_agents = (
            {self.possible_agents[0]: self.possible_agents}
            if self._shared_critic
            else {uid: [uid] for uid in self.possible_agents}
        )
        if self._shared_critic:
            first_uid = self.possible_agents[0]
            for uid in self.possible_agents[1:]:
                if self.state_spaces and self.state_spaces[uid] != self.state_spaces[first_uid]:
                    raise ValueError(
                        f"The agents' state spaces must be equal when the critic is shared, "
                        f"got {self.state_spaces[first_uid]} ('{first_uid}') and {self.state_spaces[uid]} ('{uid}')"
                    )
            self._check_shared_settings(["discount_factor", "state_preprocessor", "state_preprocessor_kwargs"])
            self._check_shared_settings(
                ["polyak", "grad_norm_clip"],
                note="A single value is required since it is used to update the shared critic "
                "(note that it also applies to the agents' policies)",
            )
            self._check_shared_settings(
                ["learning_rate", "learning_rate_scheduler", "learning_rate_scheduler_kwargs"], index=1
            )

        # the same transitions are sampled from all the agents' memories
        if self.memories:
            self._check_distinct_instances("Memory", self.memories)
            first_uid = self.possible_agents[0]
            first_memory = self.memories[first_uid]
            for uid in self.possible_agents[1:]:
                memory = self.memories[uid]
                if (memory.memory_size, memory.num_envs) != (first_memory.memory_size, first_memory.num_envs):
                    raise ValueError(
                        "The agents' memories must have the same 'memory_size' and 'num_envs', since the same "
                        f"transitions are sampled from all of them, got ({first_memory.memory_size}, "
                        f"{first_memory.num_envs}) ('{first_uid}') and ({memory.memory_size}, {memory.num_envs}) "
                        f"('{uid}')"
                    )

        # the critics take the joint action of all the agents
        if self.action_spaces and any(
            model is not None for models in critic_models.values() for model in models.values()
        ):
            joint_action_space = compute_joint_space(
                [self.action_spaces[uid] for uid in self.possible_agents], mode="concatenate"
            )
            num_joint_actions = compute_space_size(joint_action_space)
            for role, models in critic_models.items():
                for uid in self.possible_agents:
                    if models[uid] is not None and models[uid].num_actions != num_joint_actions:
                        raise ValueError(
                            f"The '{role}' model of agent '{uid}' must be built with the joint action space of all "
                            f"the agents (see skrl.utils.spaces.torch.compute_joint_space): expected "
                            f"{num_joint_actions} actions, got {models[uid].num_actions}"
                        )

        # checkpoint models
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["target_policy"] = self.target_policies[uid]
            for role, models in critic_models.items():
                self.checkpoint_modules[uid][role] = models[uid]

        # broadcast models' parameters in distributed runs (a shared critic only once)
        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            for uid in self.possible_agents:
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
            for uid in self._critic_agents:
                for critics in [self.critics_1, self.critics_2]:
                    if critics[uid] is not None:
                        critics[uid].broadcast_parameters()

        # set up noise
        self._exploration_noise = {}
        for uid in self.possible_agents:
            if self.cfg.exploration_noise[uid] is not None:
                self._exploration_noise[uid] = self.cfg.exploration_noise[uid](**self.cfg.exploration_noise_kwargs[uid])
            else:
                self._exploration_noise[uid] = None
        uids = [uid for uid, noise in self._exploration_noise.items() if noise is None]
        if uids:
            logger.warning(
                f"multi_agents:HATD3: No exploration noise specified for {uids}, training performance may be degraded"
            )
        # - smooth regularization noise (target policy smoothing)
        self._smooth_regularization_noise = {}
        for uid in self.possible_agents:
            if self.cfg.smooth_regularization_noise[uid] is not None:
                self._smooth_regularization_noise[uid] = self.cfg.smooth_regularization_noise[uid](
                    **self.cfg.smooth_regularization_noise_kwargs[uid]
                )
            else:
                self._smooth_regularization_noise[uid] = None
        uids = [uid for uid, noise in self._smooth_regularization_noise.items() if noise is None]
        if uids:
            logger.warning(
                f"multi_agents:HATD3: No smooth regularization noise specified for {uids}, "
                "training variance may be high"
            )

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizers and learning rate schedulers
        self.policy_optimizers = {}
        self.critic_optimizers = {}
        self.policy_schedulers = {}
        self.critic_schedulers = {}
        for uid in self.possible_agents:
            has_critics = self.critics_1[uid] is not None and self.critics_2[uid] is not None
            if self.policies[uid] is not None and has_critics:
                # - policy
                self.policy_optimizers[uid] = torch.optim.Adam(
                    self.policies[uid].parameters(), lr=self.cfg.learning_rate[uid][0]
                )
                self.policy_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0]
                if self.policy_schedulers[uid] is not None:
                    self.policy_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0](
                        self.policy_optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][0]
                    )
                self.checkpoint_modules[uid]["policy_optimizer"] = self.policy_optimizers[uid]
            if has_critics:
                # - critics (a shared critic has a single optimizer and learning rate scheduler, and a single optimizer
                #   is used for the twin critics)
                if uid in self._critic_agents:
                    self.critic_optimizers[uid] = torch.optim.Adam(
                        itertools.chain(self.critics_1[uid].parameters(), self.critics_2[uid].parameters()),
                        lr=self.cfg.learning_rate[uid][1],
                    )
                    self.critic_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][1]
                    if self.critic_schedulers[uid] is not None:
                        self.critic_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][1](
                            self.critic_optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][1]
                        )
                else:
                    self.critic_optimizers[uid] = self.critic_optimizers[self.possible_agents[0]]
                    self.critic_schedulers[uid] = self.critic_schedulers[self.possible_agents[0]]
                self.checkpoint_modules[uid]["critic_optimizer"] = self.critic_optimizers[uid]

        # set up target networks
        # - freeze target networks with respect to optimizers (update via .update_parameters())
        # - update target networks (hard update)
        for uid in self.possible_agents:
            if self.target_policies[uid] is not None:
                self.target_policies[uid].freeze_parameters(True)
                self.target_policies[uid].update_parameters(self.policies[uid], polyak=1)
        for uid in self._critic_agents:
            for target_critics, critics in [
                (self.target_critics_1, self.critics_1),
                (self.target_critics_2, self.critics_2),
            ]:
                if target_critics[uid] is not None:
                    target_critics[uid].freeze_parameters(True)
                    target_critics[uid].update_parameters(critics[uid], polyak=1)

        # set up preprocessors
        self._observation_preprocessor = {}
        self._state_preprocessor = {}
        for uid in self.possible_agents:
            # - observations
            if self.cfg.observation_preprocessor[uid]:
                self._observation_preprocessor[uid] = self.cfg.observation_preprocessor[uid](
                    **self.cfg.observation_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["observation_preprocessor"] = self._observation_preprocessor[uid]
            else:
                self._observation_preprocessor[uid] = self._empty_preprocessor
            # - states (a shared critic has a single state preprocessor)
            if uid in self._critic_agents:
                if self.cfg.state_preprocessor[uid]:
                    self._state_preprocessor[uid] = self.cfg.state_preprocessor[uid](
                        **self.cfg.state_preprocessor_kwargs[uid]
                    )
                else:
                    self._state_preprocessor[uid] = self._empty_preprocessor
            else:
                self._state_preprocessor[uid] = self._state_preprocessor[self.possible_agents[0]]
            if self.cfg.state_preprocessor[uid]:
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]

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
                self.memories[uid].create_tensor(
                    name="next_observations", size=self.observation_spaces[uid], dtype=torch.float32
                )
                self.memories[uid].create_tensor(name="states", size=self.state_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="next_states", size=self.state_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)

            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "rewards",
                "next_observations",
                "next_states",
                "terminated",
            ]

        # clip noise bounds (the limits of unbounded spaces are -inf/inf, so clipping has no effect on them.
        # If the action spaces are not specified, there are no limits (None) and the actions are not clipped)
        self._min_actions = {}
        self._max_actions = {}
        for uid in self.possible_agents:
            self._min_actions[uid], self._max_actions[uid] = compute_space_limits(
                self.action_spaces[uid] if self.action_spaces else None, device=self.device
            )

        # create temporary variables needed for storage and computation
        self._update_counter = 0

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
        outputs = {}

        for uid in self.possible_agents:
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            # sample random actions
            if timestep < self.cfg.random_timesteps:
                actions[uid], outputs[uid] = self.policies[uid].random_act(inputs, role="policy")
                continue

            # sample deterministic actions
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")

            # add exploration noise
            if self._exploration_noise[uid]:
                noises = self._exploration_noise[uid].sample(actions[uid].shape)
                if self.cfg.exploration_scheduler[uid]:
                    noises.mul_(self.cfg.exploration_scheduler[uid](timestep, timesteps))
                actions[uid].add_(noises)
                if self._min_actions[uid] is not None:
                    actions[uid].clamp_(min=self._min_actions[uid], max=self._max_actions[uid])

                self.track_data(f"Exploration / Exploration noise (max) ({uid})", torch.max(noises).item())
                self.track_data(f"Exploration / Exploration noise (min) ({uid})", torch.min(noises).item())
                self.track_data(f"Exploration / Exploration noise (mean) ({uid})", torch.mean(noises).item())

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
            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = {
                    uid: self.cfg.rewards_shaper(rewards[uid], timestep, timesteps) for uid in self.possible_agents
                }

            # storage transition in memories
            for uid in self.possible_agents:
                self.memories[uid].add_samples(
                    observations=observations[uid],
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    next_observations=next_observations[uid],
                    next_states=next_states[uid],
                    terminated=terminated[uid],
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
            if timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        For each gradient step, the same transitions are sampled from all the agents' memories. First, the critics
        are updated using the joint action of all the agents' target policies (with target policy smoothing) as the
        next action and the minimum of the twin target critics. Then, every ``policy_delay`` gradient steps, the
        policies are updated sequentially, in a fixed or random order, and the target networks are updated.
        Each policy maximizes the first critic's value of the joint action, which is made of the agent's own action,
        the updated actions of the agents updated before it, and the current actions of the remaining agents.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """

        # gradient steps
        for gradient_step in range(self.cfg.gradient_steps):

            # sample the same transitions from all the agents' memories
            batch = sample_joint_batch(self.memories, self.possible_agents, self._tensors_names, self.cfg.batch_size)

            inputs, next_inputs = {}, {}
            sampled_actions, sampled_rewards, sampled_terminated = {}, {}, {}
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                for uid in self.possible_agents:
                    (
                        sampled_observations,
                        sampled_states,
                        sampled_actions[uid],
                        sampled_rewards[uid],
                        sampled_next_observations,
                        sampled_next_states,
                        sampled_terminated[uid],
                    ) = batch[uid]
                    # a shared state preprocessor is trained once (with the states of the first agent)
                    train_states = uid in self._critic_agents
                    inputs[uid] = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=True),
                        "states": self._state_preprocessor[uid](sampled_states, train=train_states),
                    }
                    next_inputs[uid] = {
                        "observations": self._observation_preprocessor[uid](sampled_next_observations, train=True),
                        "states": self._state_preprocessor[uid](sampled_next_states, train=train_states),
                    }

                # joint actions: the agents' actions concatenated in the order of the possible agents
                # (the next one is made of the smoothed actions of the agents' target policies)
                sampled_joint_actions = torch.cat([sampled_actions[uid] for uid in self.possible_agents], dim=-1)
                with torch.no_grad():
                    next_joint_actions = torch.cat(
                        [self._compute_target_actions(uid, next_inputs[uid]) for uid in self.possible_agents], dim=-1
                    )

            # critic updates (a shared critic is updated once, with the team reward)
            for uid, team in self._critic_agents.items():
                critic_1, critic_2 = self.critics_1[uid], self.critics_2[uid]

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    # team reward (mean of the agents' rewards) and episode end (all the agents have terminated)
                    rewards = torch.stack([sampled_rewards[team_uid] for team_uid in team]).mean(dim=0)
                    terminated = torch.stack([sampled_terminated[team_uid] for team_uid in team]).all(dim=0)

                    # compute target values
                    with torch.no_grad():
                        target_inputs = self._critic_inputs(uid, next_inputs, next_joint_actions)
                        target_q1_values, _ = self.target_critics_1[uid].act(target_inputs, role="target_critic_1")
                        target_q2_values, _ = self.target_critics_2[uid].act(target_inputs, role="target_critic_2")
                        target_q_values = torch.min(target_q1_values, target_q2_values)
                        target_values = (
                            rewards + self.cfg.discount_factor[uid] * terminated.logical_not() * target_q_values
                        )

                    # compute critic loss
                    critic_inputs = self._critic_inputs(uid, inputs, sampled_joint_actions)
                    critic_1_values, _ = critic_1.act(critic_inputs, role="critic_1")
                    critic_2_values, _ = critic_2.act(critic_inputs, role="critic_2")

                    critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(
                        critic_2_values, target_values
                    )

                # optimization step (critics)
                self.critic_optimizers[uid].zero_grad()
                self.scaler.scale(critic_loss).backward()

                if config.torch.is_distributed:
                    critic_1.reduce_parameters()
                    critic_2.reduce_parameters()

                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.critic_optimizers[uid])
                    nn.utils.clip_grad_norm_(
                        itertools.chain(critic_1.parameters(), critic_2.parameters()), self.cfg.grad_norm_clip[uid]
                    )

                self.scaler.step(self.critic_optimizers[uid])

                # record data (the agent ID is omitted for a shared critic)
                suffix = "" if self._shared_critic else f" ({uid})"
                self.track_data(f"Loss / Critic loss{suffix}", critic_loss.item())

                self.track_data(f"Q-network / Q1 (max){suffix}", torch.max(critic_1_values).item())
                self.track_data(f"Q-network / Q1 (min){suffix}", torch.min(critic_1_values).item())
                self.track_data(f"Q-network / Q1 (mean){suffix}", torch.mean(critic_1_values).item())

                self.track_data(f"Q-network / Q2 (max){suffix}", torch.max(critic_2_values).item())
                self.track_data(f"Q-network / Q2 (min){suffix}", torch.min(critic_2_values).item())
                self.track_data(f"Q-network / Q2 (mean){suffix}", torch.mean(critic_2_values).item())

                self.track_data(f"Target / Target (max){suffix}", torch.max(target_values).item())
                self.track_data(f"Target / Target (min){suffix}", torch.min(target_values).item())
                self.track_data(f"Target / Target (mean){suffix}", torch.mean(target_values).item())

            # delayed update (policies and target networks)
            self._update_counter += 1
            if not self._update_counter % self.cfg.policy_delay:

                # agents' update order: possible agents' order or a random permutation of it
                if self.cfg.fixed_order:
                    order = self.possible_agents
                else:
                    permutation = torch.randperm(self.num_agents, device=self.device)
                    # use the same order in all workers/processes in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.broadcast(permutation, src=0)
                    order = [self.possible_agents[i] for i in permutation.tolist()]

                # sequential policy updates, starting from the current policies' actions
                with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    actions = {
                        uid: self.policies[uid].act(inputs[uid], role="policy")[0] for uid in self.possible_agents
                    }
                for uid in order:
                    policy = self.policies[uid]

                    with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                        # compute policy (actor) loss (only the agent's own action is differentiable)
                        actions[uid], _ = policy.act(inputs[uid], role="policy")
                        joint_actions = torch.cat([actions[agent_uid] for agent_uid in self.possible_agents], dim=-1)
                        critic_values, _ = self.critics_1[uid].act(
                            self._critic_inputs(uid, inputs, joint_actions), role="critic_1"
                        )

                        policy_loss = -critic_values.mean()

                    # optimization step (policy)
                    self.policy_optimizers[uid].zero_grad()
                    self.scaler.scale(policy_loss).backward()

                    if config.torch.is_distributed:
                        policy.reduce_parameters()

                    if self.cfg.grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.policy_optimizers[uid])
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])

                    self.scaler.step(self.policy_optimizers[uid])

                    # replace the agent's action with the action of its updated policy (for the next agents)
                    with (
                        torch.no_grad(),
                        torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision),
                    ):
                        actions[uid], _ = policy.act(inputs[uid], role="policy")

                    # record data
                    self.track_data(f"Loss / Policy loss ({uid})", policy_loss.item())

                # update target networks
                for uid in self.possible_agents:
                    self.target_policies[uid].update_parameters(self.policies[uid], polyak=self.cfg.polyak[uid])
                for uid in self._critic_agents:
                    self.target_critics_1[uid].update_parameters(self.critics_1[uid], polyak=self.cfg.polyak[uid])
                    self.target_critics_2[uid].update_parameters(self.critics_2[uid], polyak=self.cfg.polyak[uid])

                # update the policies' learning rate (with the delayed policy updates)
                for uid in self.possible_agents:
                    if self.policy_schedulers[uid]:
                        self.policy_schedulers[uid].step()
                        self.track_data(
                            f"Learning / Policy learning rate ({uid})", self.policy_schedulers[uid].get_last_lr()[0]
                        )

            self.scaler.update()  # called once, after optimizers have been stepped

            # update the critics' learning rate
            for uid in self._critic_agents:
                if self.critic_schedulers[uid]:
                    self.critic_schedulers[uid].step()
                    suffix = "" if self._shared_critic else f" ({uid})"
                    self.track_data(
                        f"Learning / Critic learning rate{suffix}", self.critic_schedulers[uid].get_last_lr()[0]
                    )

    def _compute_target_actions(self, uid: str, next_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the next actions of the agent's target policy (with target policy smoothing).

        If the agent has a smooth regularization noise, the clipped noise is added to the target policy's actions,
        and the result is clamped to the agent's action limits (unbounded dimensions are not clamped).

        :param uid: Agent ID.
        :param next_inputs: Agent's (preprocessed) next observations and states.

        :return: Target policy's actions.
        """
        next_actions, _ = self.target_policies[uid].act(next_inputs, role="target_policy")
        if self._smooth_regularization_noise[uid] is not None:
            noises = torch.clamp(
                self._smooth_regularization_noise[uid].sample(next_actions.shape),
                min=-self.cfg.smooth_regularization_clip[uid],
                max=self.cfg.smooth_regularization_clip[uid],
            )
            next_actions = next_actions + noises
            if self._min_actions[uid] is not None:
                next_actions = torch.clamp(next_actions, min=self._min_actions[uid], max=self._max_actions[uid])
        return next_actions

    def _critic_inputs(
        self, uid: str, inputs: dict[str, dict[str, torch.Tensor]], joint_actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Get the inputs of the agent's critics (the twin critics and their targets take the same inputs).

        The critics take the states and the joint action of all the agents. A per-agent critic also takes
        the agent's own observations. A shared critic takes the states of the first agent (the critic's owner).

        :param uid: Agent ID.
        :param inputs: Agents' (preprocessed) observations and states.
        :param joint_actions: Joint action of all the agents.

        :return: Critic inputs.
        """
        if self._shared_critic:
            return {"states": inputs[self.possible_agents[0]]["states"], "taken_actions": joint_actions}
        return {**inputs[uid], "taken_actions": joint_actions}
