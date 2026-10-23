from __future__ import annotations

from typing import Any, Sequence

import gymnasium
from packaging import version

import torch
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.utils import ScopedTimer
from skrl.utils.spaces.torch import compute_joint_space, compute_space_size

from .had3qn_cfg import HAD3QN_CFG


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


def indiv_to_joint(actions: Sequence[torch.Tensor], action_dims: Sequence[int]) -> torch.Tensor:
    """Map the agents' individual discrete actions to the joint action index.

    The joint action index is ``sum_i a_i * prod_{j < i} n_j`` (little-endian mixed radix), where ``a_i`` is
    the action of the ``i``-th agent and ``n_j`` is the number of actions of the ``j``-th agent.
    For example, for 2 agents with ``[4, 3]`` actions, the individual actions ``[1, 2]`` map to ``1 + 4 * 2 = 9``.

    :param actions: Agents' individual actions (shape ``(batch_size, 1)``), in the order of the possible agents.
    :param action_dims: Agents' number of actions, in the order of the possible agents.

    :return: Joint action indexes (``torch.long``) with shape ``(batch_size, 1)``.
    """
    joint_actions = torch.zeros_like(actions[0], dtype=torch.long)
    stride = 1
    for action, dim in zip(actions, action_dims):
        joint_actions = joint_actions + stride * action.long()
        stride *= dim
    return joint_actions


def get_joint_indexes(actions: Sequence[torch.Tensor], action_dims: Sequence[int], index: int) -> torch.Tensor:
    """Get the joint action indexes in which an agent takes each of its actions while the other agents stay fixed.

    :param actions: Agents' individual actions (shape ``(batch_size, 1)``), in the order of the possible agents.
    :param action_dims: Agents' number of actions, in the order of the possible agents.
    :param index: Index (position in the order of the possible agents) of the agent whose actions are enumerated.

    :return: Joint action indexes (``torch.long``) with shape ``(batch_size, action_dims[index])``.
        The column ``j`` holds the joint action index when the agent takes the action ``j``.
    """
    others = list(actions)
    others[index] = torch.zeros_like(actions[index])
    stride = 1
    for dim in action_dims[:index]:
        stride *= dim
    own_actions = torch.arange(action_dims[index], dtype=torch.long, device=actions[index].device)
    return indiv_to_joint(others, action_dims) + stride * own_actions


class HAD3QN(MultiAgent):
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
        cfg: HAD3QN_CFG | dict = {},
    ) -> None:
        """Heterogeneous-Agent Dueling Double Deep Q-Network (HAD3QN).

        https://jmlr.org/papers/v25/23-0488.html

        Each agent has a Q-network that takes its own observations and outputs a value for each of its (discrete)
        actions. The critic is centralized and shared by all the agents (the same model instance is given to every
        agent): it takes the states and outputs a value for each joint action of all the agents, so it must be built
        with the joint action space (see :py:func:`~skrl.utils.spaces.torch.compute_joint_space`, ``"product"`` mode),
        whose size is the product of the agents' numbers of actions. The critic is trained on the team reward (mean of
        the agents' rewards). The agents' Q-networks are updated sequentially (one agent after the other), and each one
        is regressed towards the critic's value of the joint action, in which the agents updated before it take the best
        actions according to the critic.

        :param possible_agents: Name of all possible agents the environment could generate.
        :param models: Agents' models.
        :param memories: Memories to storage agents' data and environment transitions.
            All the memories must have the same memory size and number of environments.
        :param observation_spaces: Observation spaces.
        :param state_spaces: State spaces.
        :param action_spaces: Action spaces. They must be ``Discrete`` spaces (with ``start`` equal to ``0``).
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Multi-agent's configuration.

        :raises KeyError: If a configuration key is missing.
        :raises ValueError: If the same Q-network, target Q-network or memory instance is given to several agents,
            if the action spaces are not discrete, if the critic is not shared by all the agents,
            if the critic is not built with the joint action space, if the memories have different sizes or number
            of environments, or if the state spaces or critic-related configuration differ between agents.
        """
        self.cfg: HAD3QN_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=HAD3QN_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.q_networks = {uid: self.models[uid].get("q_network", None) for uid in self.possible_agents}
        self.target_q_networks = {uid: self.models[uid].get("target_q_network", None) for uid in self.possible_agents}
        self.critics = {uid: self.models[uid].get("critic", None) for uid in self.possible_agents}
        self.target_critics = {uid: self.models[uid].get("target_critic", None) for uid in self.possible_agents}

        # Q-networks: specific to each agent (the same instance must not be given to several agents)
        self._check_distinct_instances("Model 'q_network'", self.q_networks)
        self._check_distinct_instances("Model 'target_q_network'", self.target_q_networks)

        # the agents' actions are discrete: the number of actions of each agent (in the order of the possible agents)
        self._action_dims = None
        if self.action_spaces:
            for uid in self.possible_agents:
                space = self.action_spaces[uid]
                if not isinstance(space, gymnasium.spaces.Discrete) or space.start != 0:
                    raise ValueError(
                        f"HAD3QN only supports Discrete action spaces (with start equal to 0), got {space} ('{uid}')"
                    )
            self._action_dims = [int(self.action_spaces[uid].n) for uid in self.possible_agents]

        # critic: shared by all the agents (it is owned by the first agent)
        first_uid = self.possible_agents[0]
        shared_critic = self._resolve_shared_models(["critic", "target_critic"])
        if self.critics[first_uid] is not None or self.target_critics[first_uid] is not None:
            if not shared_critic:
                raise ValueError(
                    "HAD3QN requires a critic shared by all the agents (the same 'critic' and 'target_critic' "
                    "model instances must be given to every agent), since the critic evaluates the joint action"
                )
            for uid in self.possible_agents[1:]:
                if self.state_spaces and self.state_spaces[uid] != self.state_spaces[first_uid]:
                    raise ValueError(
                        f"The agents' state spaces must be equal when the critic is shared, "
                        f"got {self.state_spaces[first_uid]} ('{first_uid}') and {self.state_spaces[uid]} ('{uid}')"
                    )
            self._check_shared_settings(["discount_factor", "state_preprocessor", "state_preprocessor_kwargs"])
            self._check_shared_settings(
                ["polyak"],
                note="A single value is required since it is used to update the shared critic's target "
                "(note that it also applies to the agents' target Q-networks)",
            )
            self._check_shared_settings(
                ["learning_rate", "learning_rate_scheduler", "learning_rate_scheduler_kwargs"], index=1
            )
        self.critic = self.critics[first_uid]
        self.target_critic = self.target_critics[first_uid]

        # the same transitions are sampled from all the agents' memories
        if self.memories:
            self._check_distinct_instances("Memory", self.memories)
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

        # the critic outputs a value for each joint action of all the agents
        if self.action_spaces:
            joint_action_space = compute_joint_space(
                [self.action_spaces[uid] for uid in self.possible_agents], mode="product"
            )
            num_joint_actions = compute_space_size(joint_action_space)
            for role, model in [("critic", self.critic), ("target_critic", self.target_critic)]:
                if model is not None and model.num_actions != num_joint_actions:
                    raise ValueError(
                        f"The '{role}' model must be built with the joint action space of all the agents "
                        f"(see skrl.utils.spaces.torch.compute_joint_space): expected {num_joint_actions} "
                        f"actions (the product of the agents' numbers of actions), got {model.num_actions}"
                    )

        # checkpoint models
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["q_network"] = self.q_networks[uid]
            self.checkpoint_modules[uid]["target_q_network"] = self.target_q_networks[uid]
            self.checkpoint_modules[uid]["critic"] = self.critics[uid]
            self.checkpoint_modules[uid]["target_critic"] = self.target_critics[uid]

        # broadcast models' parameters in distributed runs (the shared critic only once)
        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            for uid in self.possible_agents:
                if self.q_networks[uid] is not None:
                    self.q_networks[uid].broadcast_parameters()
            if self.critic is not None:
                self.critic.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizers and learning rate schedulers
        self.q_network_optimizers = {}
        self.q_network_schedulers = {}
        self.critic_optimizer = None
        self.critic_scheduler = None
        if self.critic is not None:
            # - critic (a single optimizer and learning rate scheduler for the shared critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.learning_rate[first_uid][1])
            if self.cfg.learning_rate_scheduler[first_uid][1] is not None:
                self.critic_scheduler = self.cfg.learning_rate_scheduler[first_uid][1](
                    self.critic_optimizer, **self.cfg.learning_rate_scheduler_kwargs[first_uid][1]
                )
            for uid in self.possible_agents:
                # - Q-network
                if self.q_networks[uid] is not None:
                    self.q_network_optimizers[uid] = torch.optim.Adam(
                        self.q_networks[uid].parameters(), lr=self.cfg.learning_rate[uid][0]
                    )
                    self.q_network_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0]
                    if self.q_network_schedulers[uid] is not None:
                        self.q_network_schedulers[uid] = self.cfg.learning_rate_scheduler[uid][0](
                            self.q_network_optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid][0]
                        )
                    self.checkpoint_modules[uid]["q_network_optimizer"] = self.q_network_optimizers[uid]
                self.checkpoint_modules[uid]["critic_optimizer"] = self.critic_optimizer
        self.critic_optimizers = {uid: self.critic_optimizer for uid in self.possible_agents}

        # set up target networks
        # - freeze target networks with respect to optimizers (update via .update_parameters())
        # - update target networks (hard update)
        for uid in self.possible_agents:
            if self.target_q_networks[uid] is not None:
                self.target_q_networks[uid].freeze_parameters(True)
                self.target_q_networks[uid].update_parameters(self.q_networks[uid], polyak=1)
        if self.target_critic is not None:
            self.target_critic.freeze_parameters(True)
            self.target_critic.update_parameters(self.critic, polyak=1)

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
            # - states (a single state preprocessor for the shared critic)
            if uid == first_uid:
                if self.cfg.state_preprocessor[uid]:
                    self._state_preprocessor[uid] = self.cfg.state_preprocessor[uid](
                        **self.cfg.state_preprocessor_kwargs[uid]
                    )
                else:
                    self._state_preprocessor[uid] = self._empty_preprocessor
            else:
                self._state_preprocessor[uid] = self._state_preprocessor[first_uid]
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
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.int64)
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
            q_network = self.q_networks[uid]
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            # sample random actions
            if timestep < self.cfg.random_timesteps:
                actions[uid], outputs[uid] = q_network.random_act(inputs, role="q_network")
                continue

            # sample greedy actions
            exploration_scheduler = self.cfg.exploration_scheduler[uid]
            if exploration_scheduler is None:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    q_values, outputs[uid] = q_network.act(inputs, role="q_network")
                actions[uid] = torch.argmax(q_values, dim=1, keepdim=True)
                continue

            # sample actions with epsilon-greedy policy
            actions[uid], outputs[uid] = q_network.random_act(inputs, role="q_network")
            epsilon = exploration_scheduler(timestep, timesteps)
            indexes = (torch.rand(actions[uid].shape[0], device=self.device) >= epsilon).nonzero().view(-1)
            if indexes.numel():
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {k: None if v is None else v[indexes] for k, v in inputs.items()}
                    actions[uid][indexes] = torch.argmax(
                        q_network.act(inputs, role="q_network")[0], dim=1, keepdim=True
                    )

            # record epsilon
            self.track_data(f"Exploration / Exploration epsilon ({uid})", epsilon)

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
            if timestep >= self.cfg.learning_starts and not timestep % self.cfg.update_interval:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        For each gradient step, the same transitions are sampled from all the agents' memories. First, the shared
        critic is updated using, as the next joint action, the greedy actions of the agents' target Q-networks.
        Then, the agents' Q-networks are updated sequentially, in a fixed or random order, starting from the agents'
        greedy actions. Each Q-network's value of the agent's action is regressed towards the critic's value of the
        joint action, after which the agent's action is replaced by the best action according to the critic
        (the other agents' actions being fixed) for the next agents.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        first_uid = self.possible_agents[0]

        # gradient steps
        for gradient_step in range(self.cfg.gradient_steps):

            # sample the same transitions from all the agents' memories
            batch = sample_joint_batch(self.memories, self.possible_agents, self._tensors_names, self.cfg.batch_size)

            inputs, next_inputs = {}, {}
            sampled_actions, sampled_rewards, sampled_terminated = [], [], []
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                for uid in self.possible_agents:
                    (
                        sampled_observations,
                        sampled_states,
                        actions,
                        rewards,
                        sampled_next_observations,
                        sampled_next_states,
                        terminated,
                    ) = batch[uid]
                    sampled_actions.append(actions)
                    sampled_rewards.append(rewards)
                    sampled_terminated.append(terminated)
                    # the shared state preprocessor is trained once (with the states of the first agent)
                    train_states = uid == first_uid
                    inputs[uid] = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=True),
                        "states": self._state_preprocessor[uid](sampled_states, train=train_states),
                    }
                    next_inputs[uid] = {
                        "observations": self._observation_preprocessor[uid](sampled_next_observations, train=True),
                        "states": self._state_preprocessor[uid](sampled_next_states, train=train_states),
                    }
                # the shared critic takes the states of the first agent
                critic_inputs = {"states": inputs[first_uid]["states"]}
                next_critic_inputs = {"states": next_inputs[first_uid]["states"]}

                # team reward (mean of the agents' rewards) and episode end (all the agents have terminated)
                rewards = torch.stack(sampled_rewards).mean(dim=0)
                terminated = torch.stack(sampled_terminated).all(dim=0)

                # compute target values (the next joint action is made of the target Q-networks' greedy actions)
                with torch.no_grad():
                    next_actions = [
                        torch.argmax(
                            self.target_q_networks[uid].act(next_inputs[uid], role="target_q_network")[0],
                            dim=-1,
                            keepdim=True,
                        )
                        for uid in self.possible_agents
                    ]
                    next_q_values, _ = self.target_critic.act(next_critic_inputs, role="target_critic")
                    next_q_values = torch.gather(
                        next_q_values, dim=1, index=indiv_to_joint(next_actions, self._action_dims)
                    )
                    target_values = (
                        rewards + self.cfg.discount_factor[first_uid] * terminated.logical_not() * next_q_values
                    )

                # compute critic loss
                critic_values = torch.gather(
                    self.critic.act(critic_inputs, role="critic")[0],
                    dim=1,
                    index=indiv_to_joint(sampled_actions, self._action_dims),
                )

                critic_loss = F.mse_loss(critic_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic.reduce_parameters()

            self.scaler.step(self.critic_optimizer)

            # record data
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Critic / Q (max)", torch.max(critic_values).item())
            self.track_data("Critic / Q (min)", torch.min(critic_values).item())
            self.track_data("Critic / Q (mean)", torch.mean(critic_values).item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            # agents' update order: possible agents' order or a random permutation of it
            if self.cfg.fixed_order:
                order = self.possible_agents
            else:
                permutation = torch.randperm(self.num_agents, device=self.device)
                # use the same order in all workers/processes in distributed runs
                if config.torch.is_distributed:
                    torch.distributed.broadcast(permutation, src=0)
                order = [self.possible_agents[i] for i in permutation.tolist()]

            # sequential Q-network updates, starting from the agents' greedy actions.
            # The (updated) critic's values of all the joint actions are computed once
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions = [
                    torch.argmax(self.q_networks[uid].act(inputs[uid], role="q_network")[0], dim=-1, keepdim=True)
                    for uid in self.possible_agents
                ]
                all_critic_values, _ = self.critic.act(critic_inputs, role="critic")
            for uid in order:
                index = self.possible_agents.index(uid)
                q_network = self.q_networks[uid]

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    # compute Q-network loss (regression towards the critic's value of the joint action)
                    q_values = torch.gather(
                        q_network.act(inputs[uid], role="q_network")[0], dim=1, index=actions[index]
                    )
                    critic_values = torch.gather(
                        all_critic_values, dim=1, index=indiv_to_joint(actions, self._action_dims)
                    )

                    q_network_loss = F.mse_loss(q_values, critic_values)

                # optimization step (Q-network)
                self.q_network_optimizers[uid].zero_grad()
                self.scaler.scale(q_network_loss).backward()

                if config.torch.is_distributed:
                    q_network.reduce_parameters()

                self.scaler.step(self.q_network_optimizers[uid])

                # replace the agent's action with the best action according to the critic, the other agents' actions
                # being fixed (for the next agents)
                with torch.no_grad():
                    joint_indexes = get_joint_indexes(actions, self._action_dims, index)
                    actions[index] = torch.argmax(
                        torch.gather(all_critic_values, dim=1, index=joint_indexes), dim=-1, keepdim=True
                    )

                # record data
                self.track_data(f"Loss / Q-network loss ({uid})", q_network_loss.item())

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            self._update_counter += 1
            if not self._update_counter % self.cfg.target_update_interval:
                for uid in self.possible_agents:
                    self.target_q_networks[uid].update_parameters(self.q_networks[uid], polyak=self.cfg.polyak[uid])
                self.target_critic.update_parameters(self.critic, polyak=self.cfg.polyak[first_uid])

            # update learning rate
            for uid in self.possible_agents:
                if self.q_network_schedulers[uid]:
                    self.q_network_schedulers[uid].step()
                    self.track_data(
                        f"Learning / Q-network learning rate ({uid})", self.q_network_schedulers[uid].get_last_lr()[0]
                    )
            if self.critic_scheduler:
                self.critic_scheduler.step()
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
