from __future__ import annotations

from typing import Any

import functools
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config, logger
from skrl.agents.jax import Agent
from skrl.memories.jax import Memory
from skrl.models.jax import Model
from skrl.resources.optimizers.jax import Adam
from skrl.resources.schedulers.jax import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .rpo_cfg import RPO_CFG


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> np.ndarray:
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
    advantages = np.zeros_like(rewards)
    not_dones = np.logical_not(dones)
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


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@jax.jit
def _compute_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> jax.Array:
    advantage = 0
    advantages = jnp.zeros_like(rewards)
    not_dones = jnp.logical_not(dones)
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages = advantages.at[i].set(advantage)
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


@functools.partial(jax.jit, static_argnames=("policy_act", "get_entropy", "entropy_loss_scale"))
def _update_policy(
    policy_act,
    policy_state_dict,
    inputs,
    sampled_log_prob,
    sampled_advantages,
    ratio_clip,
    get_entropy,
    entropy_loss_scale,
):
    # compute policy loss
    def _policy_loss(params):
        _, outputs = policy_act(inputs, role="policy", params=params)
        next_log_prob = outputs["log_prob"]

        # compute approximate KL divergence
        ratio = next_log_prob - sampled_log_prob
        kl_divergence = ((jnp.exp(ratio) - 1) - ratio).mean()

        # compute policy loss
        ratio = jnp.exp(next_log_prob - sampled_log_prob)
        surrogate = sampled_advantages * ratio
        surrogate_clipped = sampled_advantages * jnp.clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)

        # compute entropy loss
        entropy_loss = 0
        if entropy_loss_scale:
            entropy_loss = -entropy_loss_scale * get_entropy(outputs["stddev"], role="policy").mean()

        return -jnp.minimum(surrogate, surrogate_clipped).mean(), (entropy_loss, kl_divergence, outputs["stddev"])

    (policy_loss, (entropy_loss, kl_divergence, stddev)), grad = jax.value_and_grad(_policy_loss, has_aux=True)(
        policy_state_dict.params
    )

    return grad, policy_loss, entropy_loss, kl_divergence, stddev


@functools.partial(jax.jit, static_argnames=("value_act", "clip_predicted_values"))
def _update_value(
    value_act,
    value_state_dict,
    inputs,
    sampled_values,
    sampled_returns,
    value_loss_scale,
    clip_predicted_values,
    value_clip,
):
    # compute value loss
    def _value_loss(params):
        predicted_values, _ = value_act(inputs, role="value", params=params)
        if clip_predicted_values:
            predicted_values = sampled_values + jnp.clip(predicted_values - sampled_values, -value_clip, value_clip)
        return value_loss_scale * ((sampled_returns - predicted_values) ** 2).mean()

    value_loss, grad = jax.value_and_grad(_value_loss, has_aux=False)(value_state_dict.params)

    return grad, value_loss


class RPO(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | jax.Device | None = None,
        cfg: RPO_CFG | dict = {},
    ) -> None:
        """Robust Policy Optimization (RPO).

        https://arxiv.org/abs/2212.07536

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: RPO_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=RPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.jax.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.value is not None:
                self.value.broadcast_parameters()

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            self.policy_learning_rate = self.cfg.learning_rate[0]
            self.value_learning_rate = self.cfg.learning_rate[1]
            # - optimizers
            with jax.default_device(self.device):
                self.policy_optimizer = Adam(
                    model=self.policy,
                    lr=self.policy_learning_rate,
                    grad_norm_clip=self.cfg.grad_norm_clip,
                    scale=not self.cfg.learning_rate_scheduler[0],
                )
                self.value_optimizer = Adam(
                    model=self.value,
                    lr=self.value_learning_rate,
                    grad_norm_clip=self.cfg.grad_norm_clip,
                    scale=not self.cfg.learning_rate_scheduler[1],
                )
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["value_optimizer"] = self.value_optimizer
            # - learning rate schedulers
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    **self.cfg.learning_rate_scheduler_kwargs[0]
                )
            self.value_scheduler = self.cfg.learning_rate_scheduler[1]
            if self.value_scheduler is not None:
                self.value_scheduler = self.cfg.learning_rate_scheduler[1](**self.cfg.learning_rate_scheduler_kwargs[1])

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
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=jnp.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=jnp.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(name="truncated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(name="log_prob", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="values", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=jnp.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None
        self._rollout = 0

        # set up models for just-in-time compilation with XLA
        self.policy.apply = jax.jit(self.policy.apply, static_argnums=2)
        if self.value is not None:
            self.value.apply = jax.jit(self.value.apply, static_argnums=2)

    def act(
        self,
        observations: np.ndarray | jax.Array,
        states: np.ndarray | jax.Array | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[np.ndarray | jax.Array, dict[str, Any]]:
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
            "alpha": self.cfg.alpha,
        }
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        actions, outputs = self.policy.act(inputs, role="policy")
        self._current_log_prob = outputs["log_prob"]
        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)
            self._current_log_prob = jax.device_get(self._current_log_prob)

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: np.ndarray | jax.Array,
        states: np.ndarray | jax.Array,
        actions: np.ndarray | jax.Array,
        rewards: np.ndarray | jax.Array,
        next_observations: np.ndarray | jax.Array,
        next_states: np.ndarray | jax.Array,
        terminated: np.ndarray | jax.Array,
        truncated: np.ndarray | jax.Array,
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
            inputs = {
                "observations": self._observation_preprocessor(observations),
                "states": self._state_preprocessor(states),
                "alpha": self.cfg.alpha,
            }
            values, _ = self.value.act(inputs, role="value")
            if not self._jax:  # numpy backend
                values = jax.device_get(values)
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
        inputs = {
            "observations": self._observation_preprocessor(self._current_next_observations),
            "states": self._state_preprocessor(self._current_next_states),
            "alpha": self.cfg.alpha,
        }
        self.value.enable_training_mode(False)
        last_values, _ = self.value.act(inputs, role="value")
        self.value.enable_training_mode(True)
        if not self._jax:  # numpy backend
            last_values = jax.device_get(last_values)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = (_compute_gae if self._jax else compute_gae)(
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

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
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

                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                    "states": self._state_preprocessor(sampled_states, train=not epoch),
                    "alpha": self.cfg.alpha,
                }

                # compute policy loss
                grad, policy_loss, entropy_loss, kl_divergence, stddev = _update_policy(
                    self.policy.act,
                    self.policy.state_dict,
                    {**inputs, "taken_actions": sampled_actions},
                    sampled_log_prob,
                    sampled_advantages,
                    self.cfg.ratio_clip,
                    self.policy.get_entropy,
                    self.cfg.entropy_loss_scale,
                )

                kl_divergences.append(kl_divergence.item())

                # early stopping with KL divergence
                if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                    break

                # optimization step (policy)
                if config.jax.is_distributed:
                    grad = self.policy.reduce_parameters(grad)
                self.policy_optimizer = self.policy_optimizer.step(
                    grad=grad, model=self.policy, lr=self.policy_learning_rate if self.policy_scheduler else None
                )

                # compute value loss
                grad, value_loss = _update_value(
                    self.value.act,
                    self.value.state_dict,
                    inputs,
                    sampled_values,
                    sampled_returns,
                    self.cfg.value_loss_scale,
                    self.cfg.value_clip > 0,
                    self.cfg.value_clip,
                )

                # optimization step (value)
                if config.jax.is_distributed:
                    grad = self.value.reduce_parameters(grad)
                self.value_optimizer = self.value_optimizer.step(
                    grad=grad, model=self.value, lr=self.value_learning_rate if self.value_scheduler else None
                )

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            # - compute KL for KL adaptive learning rate scheduler
            if self.policy_scheduler is KLAdaptiveLR or self.value_scheduler is KLAdaptiveLR:
                kl = np.mean(kl_divergences)
                # reduce (collect from all workers/processes) KL in distributed runs
                if config.jax.is_distributed:
                    kl = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(kl.reshape(1)).item()
                    kl /= config.jax.world_size
            # - policy learning rate
            if self.policy_scheduler:
                if self.policy_scheduler is KLAdaptiveLR:
                    self.policy_learning_rate = self.policy_scheduler(timestep, lr=self.policy_learning_rate, kl=kl)
                else:
                    self.policy_learning_rate *= self.policy_scheduler(timestep)
            # - value learning rate
            if self.value_scheduler:
                if self.value_scheduler is KLAdaptiveLR:
                    self.value_learning_rate = self.value_scheduler(timestep, lr=self.value_learning_rate, kl=kl)
                else:
                    self.value_learning_rate *= self.value_scheduler(timestep)

        # record data
        self.track_data(
            "Loss / Policy loss", cumulative_policy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
        )
        self.track_data("Loss / Value loss", cumulative_value_loss / (self.cfg.learning_epochs * self.cfg.mini_batches))
        if self.cfg.entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self.cfg.learning_epochs * self.cfg.mini_batches)
            )

        self.track_data("Policy / Standard deviation", stddev.mean().item())

        if self.policy_scheduler:
            self.track_data("Learning / Policy learning rate", self.policy_learning_rate)
        if self.value_scheduler:
            self.track_data("Learning / Value learning rate", self.value_learning_rate)
