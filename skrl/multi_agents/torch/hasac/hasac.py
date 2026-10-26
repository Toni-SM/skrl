from __future__ import annotations

from typing import Any, Sequence

import itertools
import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.utils import ScopedTimer
from skrl.utils.spaces.torch import compute_joint_space, compute_space_size

from .hasac_cfg import HASAC_CFG


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


class HASAC(MultiAgent):
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
        cfg: HASAC_CFG | dict = {},
    ) -> None:
        """Heterogeneous-Agent Soft Actor-Critic (HASAC).

        https://openreview.net/forum?id=tmqOhBC4a5

        The critics (twin Q-networks) are centralized: they take the states and the joint action of all the agents (the
        agents' actions concatenated in the order of the possible agents), so they must be built with the joint action
        space (see :py:func:`~skrl.utils.spaces.torch.compute_joint_space`, ``"concatenate"`` mode). The critics can be
        either shared by all the agents (the same model instances are given to every agent) or specific to each agent
        (distinct instances are given to every agent). A shared critic is trained on the team reward (mean of the
        agents' rewards). Each agent has a stochastic policy and its own entropy coefficient (fixed or learned). The
        agents' policies are updated sequentially (one agent after the other), and each policy update takes into account
        the actions sampled from the updated policies of the agents updated before it.

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
        :raises ValueError: If the same policy or memory instance is given to several agents,
            if the critics are shared by only some of the agents (or only some of the critic roles are shared),
            if the critics are not built with the joint action space, if the memories have different sizes or number
            of environments, or if the state spaces or critic-related configuration of a shared critic differ
            between agents.
        """
        self.cfg: HASAC_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=HASAC_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
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
                ["polyak"], note="A single value is required since it is used to update the shared critic's targets"
            )
            self._check_shared_settings(
                ["grad_norm_clip"],
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

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # entropy (each agent has its own entropy coefficient)
        self._entropy_coefficients = {}
        self._target_entropies = {}
        self.log_entropy_coefficients = {}
        self.entropy_optimizers = {}
        self.entropy_schedulers = {}
        for uid in self.possible_agents:
            self._entropy_coefficients[uid] = self.cfg.initial_entropy_value[uid]
            # learned entropy coefficient (optimized through its logarithm)
            if self.cfg.learn_entropy[uid]:
                # - target entropy (default: -prod(action shape) for Box action spaces, -n for Discrete action spaces,
                #   and 0 otherwise, e.g.: if the action space is not specified)
                self._target_entropies[uid] = self.cfg.target_entropy[uid]
                if self._target_entropies[uid] is None:
                    action_space = self.action_spaces[uid] if self.action_spaces else None
                    if issubclass(type(action_space), gymnasium.spaces.Box):
                        self._target_entropies[uid] = -float(np.prod(action_space.shape))
                    elif issubclass(type(action_space), gymnasium.spaces.Discrete):
                        self._target_entropies[uid] = -float(action_space.n)
                    else:
                        self._target_entropies[uid] = 0.0
                # - log entropy coefficient (held by a module, to be checkpointed and restored in place),
                #   and its optimizer and learning rate scheduler
                log_entropy_coefficient = nn.ParameterDict(
                    {
                        "log_entropy_coefficient": nn.Parameter(
                            torch.log(torch.ones(1, device=self.device) * self._entropy_coefficients[uid])
                        )
                    }
                )
                self.log_entropy_coefficients[uid] = log_entropy_coefficient["log_entropy_coefficient"]
                self.checkpoint_modules[uid]["log_entropy_coefficient"] = log_entropy_coefficient
                self.entropy_optimizers[uid] = torch.optim.Adam(
                    [self.log_entropy_coefficients[uid]], lr=self.cfg.learning_rate[uid][2]
                )
                self.checkpoint_modules[uid]["entropy_optimizer"] = self.entropy_optimizers[uid]
                self.entropy_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][2]
                if self.entropy_schedulers[uid] is not None:
                    self.entropy_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][2](
                        self.entropy_optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][2]
                    )

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

    def load(self, path: str) -> None:
        """Load the agent from the specified path.

        The learned entropy coefficients are computed from the loaded (logarithm of the) entropy coefficients.

        :param path: Path to load the agent from.
        """
        super().load(path)
        for uid, log_entropy_coefficient in self.log_entropy_coefficients.items():
            self._entropy_coefficients[uid] = torch.exp(log_entropy_coefficient.detach())

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

            # sample stochastic actions
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions[uid], outputs[uid] = self.policies[uid].act(inputs, role="policy")

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

            # storage transition in memories (with the observations/states with which the transitions ended,
            # that differ from the next ones if the environment is auto-reset at the same step)
            next_observations, next_states = self._get_final_observations_and_states(
                next_observations, next_states, infos
            )
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
        are updated towards the soft target values: the minimum of the twin target critics evaluated at the next joint
        action, sampled from the agents' policies, minus the sum of the agents' entropy terms (each agent's entropy
        coefficient times the log probability of its next action). Then, the policies are updated sequentially,
        in a fixed or random order. Each policy minimizes its entropy coefficient times the log probability of its
        action minus the minimum of the twin critics' values of the joint action, which is made of the agent's own
        action, the actions resampled from the updated policies of the agents updated before it, and the actions
        sampled from the current policies of the remaining agents. Each learned entropy coefficient is updated right
        after its agent's policy. Finally, the target critics are updated.

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
                sampled_joint_actions = torch.cat([sampled_actions[uid] for uid in self.possible_agents], dim=-1)
                # - the next joint action is sampled from the agents' policies, and the entropy term is the sum
                #   of the agents' entropy coefficients times the log probabilities of their next actions
                with torch.no_grad():
                    next_actions, next_entropy_terms = [], []
                    for uid in self.possible_agents:
                        next_action, next_outputs = self.policies[uid].act(next_inputs[uid], role="policy")
                        next_actions.append(next_action)
                        next_entropy_terms.append(self._entropy_coefficients[uid] * next_outputs["log_prob"])
                    next_joint_actions = torch.cat(next_actions, dim=-1)
                    next_entropy_term = torch.stack(next_entropy_terms).sum(dim=0)

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
                        target_q_values = torch.min(target_q1_values, target_q2_values) - next_entropy_term
                        target_values = (
                            rewards + self.cfg.discount_factor[uid] * terminated.logical_not() * target_q_values
                        )

                    # compute critic loss
                    critic_inputs = self._critic_inputs(uid, inputs, sampled_joint_actions)
                    critic_1_values, _ = critic_1.act(critic_inputs, role="critic_1")
                    critic_2_values, _ = critic_2.act(critic_inputs, role="critic_2")

                    critic_loss = (
                        F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
                    ) / 2

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

            # agents' update order: possible agents' order or a random permutation of it
            if self.cfg.fixed_order:
                order = self.possible_agents
            else:
                permutation = torch.randperm(self.num_agents, device=self.device)
                # use the same order in all workers/processes in distributed runs
                if config.torch.is_distributed:
                    torch.distributed.broadcast(permutation, src=0)
                order = [self.possible_agents[i] for i in permutation.tolist()]

            # sequential policy updates, starting from actions sampled from the current policies
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions = {uid: self.policies[uid].act(inputs[uid], role="policy")[0] for uid in self.possible_agents}
            for uid in order:
                policy = self.policies[uid]

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    # compute policy (actor) loss (only the agent's own action is differentiable)
                    actions[uid], outputs = policy.act(inputs[uid], role="policy")
                    log_prob = outputs["log_prob"]
                    joint_actions = torch.cat([actions[agent_uid] for agent_uid in self.possible_agents], dim=-1)
                    critic_inputs = self._critic_inputs(uid, inputs, joint_actions)
                    critic_1_values, _ = self.critics_1[uid].act(critic_inputs, role="critic_1")
                    critic_2_values, _ = self.critics_2[uid].act(critic_inputs, role="critic_2")

                    policy_loss = (
                        self._entropy_coefficients[uid] * log_prob - torch.min(critic_1_values, critic_2_values)
                    ).mean()

                # optimization step (policy)
                self.policy_optimizers[uid].zero_grad()
                self.scaler.scale(policy_loss).backward()

                if config.torch.is_distributed:
                    policy.reduce_parameters()

                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.policy_optimizers[uid])
                    nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])

                self.scaler.step(self.policy_optimizers[uid])

                # entropy learning (with the log probability of the agent's action used in the policy loss)
                if self.cfg.learn_entropy[uid]:
                    with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                        # compute entropy loss
                        entropy_loss = -(
                            self.log_entropy_coefficients[uid] * (log_prob + self._target_entropies[uid]).detach()
                        ).mean()

                    # optimization step (entropy)
                    self.entropy_optimizers[uid].zero_grad()
                    self.scaler.scale(entropy_loss).backward()
                    self.scaler.step(self.entropy_optimizers[uid])

                    # compute entropy coefficient
                    self._entropy_coefficients[uid] = torch.exp(self.log_entropy_coefficients[uid].detach())

                # replace the agent's action with an action sampled from its updated policy (for the next agents)
                with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    actions[uid], _ = policy.act(inputs[uid], role="policy")

                # record data
                self.track_data(f"Loss / Policy loss ({uid})", policy_loss.item())
                if self.cfg.learn_entropy[uid]:
                    self.track_data(f"Loss / Entropy loss ({uid})", entropy_loss.item())
                    self.track_data(
                        f"Coefficient / Entropy coefficient ({uid})", self._entropy_coefficients[uid].item()
                    )

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            for uid in self._critic_agents:
                self.target_critics_1[uid].update_parameters(self.critics_1[uid], polyak=self.cfg.polyak[uid])
                self.target_critics_2[uid].update_parameters(self.critics_2[uid], polyak=self.cfg.polyak[uid])

            # update learning rate
            for uid in self.possible_agents:
                if self.policy_schedulers[uid]:
                    self.policy_schedulers[uid].step()
                    self.track_data(
                        f"Learning / Policy learning rate ({uid})", self.policy_schedulers[uid].get_last_lr()[0]
                    )
            for uid in self._critic_agents:
                if self.critic_schedulers[uid]:
                    self.critic_schedulers[uid].step()
                    suffix = "" if self._shared_critic else f" ({uid})"
                    self.track_data(
                        f"Learning / Critic learning rate{suffix}", self.critic_schedulers[uid].get_last_lr()[0]
                    )
            for uid, entropy_scheduler in self.entropy_schedulers.items():
                if entropy_scheduler:
                    entropy_scheduler.step()
                    self.track_data(f"Learning / Entropy learning rate ({uid})", entropy_scheduler.get_last_lr()[0])

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
