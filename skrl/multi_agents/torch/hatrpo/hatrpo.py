from __future__ import annotations

from typing import Any

import copy
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.utils import ScopedTimer

from .hatrpo_cfg import HATRPO_CFG


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


def surrogate_loss(
    *,
    policy: Model,
    observations: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    """Compute the surrogate objective (policy loss), weighted by the factor of the previously updated agents.

    :param policy: Policy.
    :param observations: Observations.
    :param states: States.
    :param actions: Actions.
    :param log_prob: Log probability.
    :param advantages: Advantages.
    :param factor: Product of the probability ratios (between the updated and the previous policy)
        of the agents updated before this one.

    :return: Surrogate loss.
    """
    _, outputs = policy.act({"observations": observations, "states": states, "taken_actions": actions}, role="policy")
    new_log_prob = outputs["log_prob"]
    return (factor * advantages * torch.exp(new_log_prob - log_prob.detach())).mean()


def conjugate_gradient(
    *,
    policy: Model,
    observations: torch.Tensor,
    states: torch.Tensor,
    b: torch.Tensor,
    damping: float = 0.1,
    num_iterations: int = 10,
    residual_tolerance: float = 1e-10,
) -> torch.Tensor:
    """Conjugate gradient algorithm to solve Ax = b using the iterative method.

    https://en.wikipedia.org/wiki/Conjugate_gradient_method#As_an_iterative_method

    :param policy: Policy.
    :param observations: Observations.
    :param states: States.
    :param b: Vector b.
    :param damping: Damping.
    :param num_iterations: Number of iterations.
    :param residual_tolerance: Residual tolerance.

    :return: Conjugate vector.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr_old = torch.dot(r, r)
    for _ in range(num_iterations):
        hv = fisher_vector_product(policy=policy, observations=observations, states=states, vector=p, damping=damping)
        alpha = rr_old / torch.dot(p, hv)
        x += alpha * p
        r -= alpha * hv
        rr_new = torch.dot(r, r)
        if rr_new < residual_tolerance:
            break
        p = r + rr_new / rr_old * p
        rr_old = rr_new
    return x


def fisher_vector_product(
    *, policy: Model, observations: torch.Tensor, states: torch.Tensor, vector: torch.Tensor, damping: float = 0.1
) -> torch.Tensor:
    """Compute the Fisher vector product (direct method).

    https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/

    :param policy: Policy.
    :param observations: Observations.
    :param states: States.
    :param vector: Vector.
    :param damping: Damping.

    :return: Hessian vector product.
    """
    kl = kl_divergence(policy_1=policy, policy_2=policy, observations=observations, states=states)
    kl_gradient = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_kl_gradient = torch.cat([gradient.view(-1) for gradient in kl_gradient])
    hessian_vector_gradient = torch.autograd.grad((flat_kl_gradient * vector).sum(), policy.parameters())
    flat_hessian_vector_gradient = torch.cat([gradient.contiguous().view(-1) for gradient in hessian_vector_gradient])
    return flat_hessian_vector_gradient + damping * vector


def kl_divergence(
    *, policy_1: Model, policy_2: Model, observations: torch.Tensor, states: torch.Tensor
) -> torch.Tensor:
    """Compute the KL divergence between two distributions.

    https://en.wikipedia.org/wiki/Normal_distribution#Other_properties

    :param policy_1: First policy.
    :param policy_2: Second policy.
    :param observations: Observations.
    :param states: States.

    :return: KL divergence.
    """
    _, outputs = policy_1.act({"observations": observations, "states": states}, role="policy")
    mu_1 = outputs["mean_actions"].detach()
    logstd_1 = outputs["log_std"].detach()

    _, outputs = policy_2.act({"observations": observations, "states": states}, role="policy")
    mu_2 = outputs["mean_actions"]
    logstd_2 = outputs["log_std"]

    kl = (
        logstd_2
        - logstd_1
        + 0.5 * (torch.square(logstd_1.exp()) + torch.square(mu_1 - mu_2)) / torch.square(logstd_2.exp())
        - 0.5
    )
    return torch.sum(kl, dim=-1).mean()


class HATRPO(MultiAgent):
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
        cfg: HATRPO_CFG | dict = {},
    ) -> None:
        """Heterogeneous-Agent Trust Region Policy Optimization (HATRPO).

        https://jmlr.org/papers/v25/23-0488.html

        The agents' policies are updated sequentially (one agent after the other), and each policy update takes into
        account the updates of the policies that were updated before it. Each policy is updated with a TRPO step
        (natural gradient and backtracking line search under a KL divergence constraint). The value model can be
        either shared by all the agents (the same model instance is given to every agent) or specific to each agent
        (a distinct instance is given to every agent). A shared value model is trained once per update on the team
        reward (mean of the agents' rewards).

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
        self.cfg: HATRPO_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=HATRPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}

        # policies: specific to each agent (the same instance must not be given to several agents)
        self._check_distinct_instances("Model 'policy'", self.policies)

        # backup policies (used by the line search), created on first use (once the policies' lazy modules, if any,
        # are initialized)
        self._backup_policies = {}

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
                    "since HATRPO updates the value after all the policies"
                )
        if self._shared_value:
            first_uid = self.possible_agents[0]
            for uid in self.possible_agents[1:]:
                if self.state_spaces and self.state_spaces[uid] != self.state_spaces[first_uid]:
                    raise ValueError(
                        f"The agents' state spaces must be equal when the value model is shared, "
                        f"got {self.state_spaces[first_uid]} ('{first_uid}') and {self.state_spaces[uid]} ('{uid}')"
                    )
            # (the learning rate and its scheduler are the value model's ones)
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
                    "learning_rate",
                    "learning_rate_scheduler",
                    "learning_rate_scheduler_kwargs",
                    "state_preprocessor",
                    "state_preprocessor_kwargs",
                    "value_preprocessor",
                    "value_preprocessor_kwargs",
                ]
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

        # set up optimizers and learning rate schedulers (only for the value: the policies are updated without them)
        self.value_optimizers = {}
        self.value_schedulers = {}
        for uid in self.possible_agents:
            # - value (a shared value model has a single optimizer and learning rate scheduler)
            if self.values[uid] is not None:
                if uid in self._value_agents:
                    self.value_optimizers[uid] = torch.optim.Adam(
                        self.values[uid].parameters(), lr=self.cfg.learning_rate[uid]
                    )
                    self.value_schedulers[uid] = self.cfg.learning_rate_scheduler[uid]
                    if self.value_schedulers[uid] is not None:
                        self.value_schedulers[uid] = self.cfg.learning_rate_scheduler[uid](
                            self.value_optimizers[uid], **self.cfg.learning_rate_scheduler_kwargs[uid]
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

        The policies are updated sequentially, in a fixed or random order. Each policy is updated with a TRPO step on
        its surrogate objective, weighted by the product of the probability ratios (between the updated and the
        previous policy) of the agents updated before it (``factor``). Then, the value models are updated.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # compute returns and advantages
        # (a shared value model computes them once, from the team reward, for all the agents)
        for uid, team in self._value_agents.items():
            value = self.values[uid]
            memory = self.memories[uid]

            with torch.no_grad():
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
        factor = torch.ones_like(self.memories[order[0]].get_tensor_by_name("factor"))
        for uid in order:
            self.memories[uid].set_tensor_by_name("factor", factor)
            log_prob = self._log_prob(uid)
            self._update_policy(uid)
            factor = factor * torch.exp(self._log_prob(uid) - log_prob)
            # (the last evaluation of the policy, used for the factor, is that of the updated policy)
            self.track_data(
                f"Policy / Standard deviation ({uid})",
                self.policies[uid].distribution(role="policy").stddev.mean().item(),
            )

        # value updates
        for uid in self._value_agents:
            self._update_value(uid)

    def _log_prob(self, uid: str) -> torch.Tensor:
        """Compute the log probability of the actions stored in the agent's memory using its current policy.

        :param uid: Agent ID.

        :return: Log probability of the stored actions, with shape ``(memory_size, num_envs, 1)``.
        """
        memory = self.memories[uid]
        observations, states, actions = memory.sample_all(names=["observations", "states", "actions"])[0]
        with torch.no_grad():
            inputs = {
                "observations": self._observation_preprocessor[uid](observations),
                "states": self._state_preprocessor[uid](states),
            }
            _, outputs = self.policies[uid].act({**inputs, "taken_actions": actions}, role="policy")
        return outputs["log_prob"].reshape(memory.memory_size, memory.num_envs, 1)

    def _update_policy(self, uid: str) -> None:
        """Update the agent's policy with a TRPO step (natural gradient and backtracking line search).

        :param uid: Agent ID.
        """
        policy = self.policies[uid]
        if uid not in self._backup_policies:
            self._backup_policies[uid] = copy.deepcopy(policy)
        backup_policy = self._backup_policies[uid]
        memory = self.memories[uid]

        # sample all from memory
        (
            sampled_observations,
            sampled_states,
            sampled_actions,
            sampled_log_prob,
            sampled_advantages,
            sampled_factor,
        ) = memory.sample(names=self._policy_tensors_names, batch_size=len(memory), mini_batches=1)[0]

        sampled_observations = self._observation_preprocessor[uid](sampled_observations, train=True)
        sampled_states = self._state_preprocessor[uid](sampled_states)

        # compute policy loss gradient (surrogate objective weighted by the factor of the previously updated agents)
        policy_loss = surrogate_loss(
            policy=policy,
            observations=sampled_observations,
            states=sampled_states,
            actions=sampled_actions,
            log_prob=sampled_log_prob,
            advantages=sampled_advantages,
            factor=sampled_factor,
        )
        policy_loss_gradient = torch.autograd.grad(policy_loss, policy.parameters())
        flat_policy_loss_gradient = torch.cat([gradient.view(-1) for gradient in policy_loss_gradient])

        # compute the search direction using the conjugate gradient algorithm
        search_direction = conjugate_gradient(
            policy=policy,
            observations=sampled_observations,
            states=sampled_states,
            b=flat_policy_loss_gradient.data,
            damping=self.cfg.damping[uid],
            num_iterations=self.cfg.conjugate_gradient_steps[uid],
        )

        # compute step size and full step
        xHx = (
            search_direction
            * fisher_vector_product(
                policy=policy,
                observations=sampled_observations,
                states=sampled_states,
                vector=search_direction,
                damping=self.cfg.damping[uid],
            )
        ).sum(0, keepdim=True)
        step_size = torch.sqrt(2 * self.cfg.max_kl_divergence[uid] / xHx)[0]
        full_step = step_size * search_direction

        # backtracking line search
        restore_policy_flag = True
        backup_policy.update_parameters(policy)
        params = parameters_to_vector(policy.parameters()).detach()

        expected_improvement = (flat_policy_loss_gradient * full_step).sum(0, keepdim=True)

        with torch.no_grad():
            for alpha in [
                self.cfg.step_fraction[uid] * self.cfg.backtrack_ratio[uid] ** i
                for i in range(self.cfg.max_backtrack_steps[uid])
            ]:
                new_params = params + alpha * full_step
                vector_to_parameters(new_params, policy.parameters())

                kl = kl_divergence(
                    policy_1=backup_policy,
                    policy_2=policy,
                    observations=sampled_observations,
                    states=sampled_states,
                )
                loss = surrogate_loss(
                    policy=policy,
                    observations=sampled_observations,
                    states=sampled_states,
                    actions=sampled_actions,
                    log_prob=sampled_log_prob,
                    advantages=sampled_advantages,
                    factor=sampled_factor,
                )

                # accept the step if it satisfies the KL divergence constraint and if the surrogate objective
                # improves by at least a fraction of the improvement expected (first-order) for the step
                if (
                    kl < self.cfg.max_kl_divergence[uid]
                    and loss - policy_loss > 0
                    and (loss - policy_loss) / (alpha * expected_improvement) > self.cfg.accept_ratio[uid]
                ):
                    restore_policy_flag = False
                    break

        if restore_policy_flag:
            policy.update_parameters(backup_policy)

        # average the policy parameters (updated without an optimizer) across all workers/processes in distributed runs
        if config.torch.is_distributed:
            params = parameters_to_vector(policy.parameters()).detach()
            torch.distributed.all_reduce(params, op=torch.distributed.ReduceOp.SUM)
            vector_to_parameters(params / config.torch.world_size, policy.parameters())

        # record data
        self.track_data(f"Loss / Policy loss ({uid})", policy_loss.item())

    def _update_value(self, uid: str) -> None:
        """Update the value model owned by the agent (the first agent for a shared value model).

        :param uid: Agent ID.
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
                value_loss.backward()
                if config.torch.is_distributed:
                    value.reduce_parameters()
                if self.cfg.grad_norm_clip[uid] > 0:
                    nn.utils.clip_grad_norm_(value.parameters(), self.cfg.grad_norm_clip[uid])
                self.value_optimizers[uid].step()

                # update cumulative losses
                cumulative_value_loss += value_loss.item()

            # update learning rate
            if self.value_schedulers[uid]:
                self.value_schedulers[uid].step()

        # record data (the agent ID is omitted for a shared value model)
        suffix = "" if self._shared_value else f" ({uid})"
        self.track_data(
            f"Loss / Value loss{suffix}",
            cumulative_value_loss / (self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]),
        )

        if self.value_schedulers[uid]:
            self.track_data(f"Learning / Value learning rate{suffix}", self.value_schedulers[uid].get_last_lr()[0])
