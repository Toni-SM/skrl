from __future__ import annotations

from typing import Any

import functools
import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config, logger
from skrl.agents.jax import Agent
from skrl.memories.jax import Memory
from skrl.models.jax import Model
from skrl.resources.optimizers.jax import Adam
from skrl.utils import ScopedTimer

from .sac_cfg import SAC_CFG


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@functools.partial(jax.jit, static_argnames=("critic_1_act", "critic_2_act"))
def _update_critic(
    critic_1_act,
    critic_1_state_dict,
    critic_2_act,
    critic_2_state_dict,
    target_q1_values: jax.Array,
    target_q2_values: jax.Array,
    entropy_coefficient,
    next_log_prob,
    inputs: dict[str, np.ndarray | jax.Array],
    sampled_rewards: np.ndarray | jax.Array,
    sampled_terminated: np.ndarray | jax.Array,
    sampled_truncated: np.ndarray | jax.Array,
    discount_factor: float,
):
    # compute target values
    target_q_values = jnp.minimum(target_q1_values, target_q2_values) - entropy_coefficient * next_log_prob
    target_values = (
        sampled_rewards + discount_factor * jnp.logical_not(sampled_terminated | sampled_truncated) * target_q_values
    )

    # compute critic loss
    def _critic_loss(params, critic_act, role):
        critic_values, _ = critic_act(inputs, role=role, params=params)
        critic_loss = ((critic_values - target_values) ** 2).mean()
        return critic_loss, critic_values

    (critic_1_loss, critic_1_values), grad = jax.value_and_grad(_critic_loss, has_aux=True)(
        critic_1_state_dict.params, critic_1_act, "critic_1"
    )
    (critic_2_loss, critic_2_values), grad = jax.value_and_grad(_critic_loss, has_aux=True)(
        critic_2_state_dict.params, critic_2_act, "critic_2"
    )

    return grad, (critic_1_loss + critic_2_loss) / 2, critic_1_values, critic_2_values, target_values


@functools.partial(jax.jit, static_argnames=("policy_act", "critic_1_act", "critic_2_act"))
def _update_policy(
    policy_act,
    critic_1_act,
    critic_2_act,
    policy_state_dict,
    critic_1_state_dict,
    critic_2_state_dict,
    entropy_coefficient,
    inputs,
):
    # compute policy (actor) loss
    def _policy_loss(policy_params, critic_1_params, critic_2_params):
        actions, outputs = policy_act(inputs, role="policy", params=policy_params)
        log_prob = outputs["log_prob"]
        critic_1_values, _ = critic_1_act({**inputs, "taken_actions": actions}, role="critic_1", params=critic_1_params)
        critic_2_values, _ = critic_2_act({**inputs, "taken_actions": actions}, role="critic_2", params=critic_2_params)
        return (entropy_coefficient * log_prob - jnp.minimum(critic_1_values, critic_2_values)).mean(), log_prob

    (policy_loss, log_prob), grad = jax.value_and_grad(_policy_loss, has_aux=True)(
        policy_state_dict.params, critic_1_state_dict.params, critic_2_state_dict.params
    )

    return grad, policy_loss, log_prob


@jax.jit
def _update_entropy(log_entropy_coefficient_state_dict, target_entropy, log_prob):
    # compute entropy loss
    def _entropy_loss(params):
        return -(params["params"] * (log_prob + target_entropy)).mean()

    entropy_loss, grad = jax.value_and_grad(_entropy_loss, has_aux=False)(log_entropy_coefficient_state_dict.params)

    return grad, entropy_loss


class SAC(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | jax.Device | None = None,
        cfg: SAC_CFG | dict = {},
    ) -> None:
        """Soft Actor-Critic (SAC).

        https://arxiv.org/abs/1801.01290

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: SAC_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=SAC_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        # broadcast models' parameters in distributed runs
        if config.jax.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        # entropy
        self._entropy_coefficient = self.cfg.initial_entropy_value
        if self.cfg.learn_entropy:
            self._target_entropy = self.cfg.target_entropy
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            class _LogEntropyCoefficient:
                def __init__(self, entropy_coefficient: float) -> None:
                    class StateDict(flax.struct.PyTreeNode):
                        params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)

                    self.state_dict = StateDict(
                        flax.core.FrozenDict({"params": jnp.array([jnp.log(entropy_coefficient)])})
                    )

                @property
                def value(self):
                    return self.state_dict.params["params"]

            with jax.default_device(self.device):
                self.log_entropy_coefficient = _LogEntropyCoefficient(self._entropy_coefficient)
                self.entropy_optimizer = Adam(model=self.log_entropy_coefficient, lr=self.cfg.learning_rate[2])

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_learning_rate = self.cfg.learning_rate[0]
            self.critic_learning_rate = self.cfg.learning_rate[1]
            # - optimizers
            with jax.default_device(self.device):
                self.policy_optimizer = Adam(
                    model=self.policy,
                    lr=self.policy_learning_rate,
                    grad_norm_clip=self.cfg.grad_norm_clip,
                    scale=not self.cfg.learning_rate_scheduler[0],
                )
                self.critic_1_optimizer = Adam(
                    model=self.critic_1,
                    lr=self.critic_learning_rate,
                    grad_norm_clip=self.cfg.grad_norm_clip,
                    scale=not self.cfg.learning_rate_scheduler[1],
                )
                self.critic_2_optimizer = Adam(
                    model=self.critic_2,
                    lr=self.critic_learning_rate,
                    grad_norm_clip=self.cfg.grad_norm_clip,
                    scale=not self.cfg.learning_rate_scheduler[1],
                )
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_1_optimizer"] = self.critic_1_optimizer
            self.checkpoint_modules["critic_2_optimizer"] = self.critic_2_optimizer
            # - learning rate schedulers
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            self.critic_scheduler = self.cfg.learning_rate_scheduler[1]
            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    **self.cfg.learning_rate_scheduler_kwargs[0]
                )
            if self.critic_scheduler is not None:
                self.critic_scheduler = self.cfg.learning_rate_scheduler[1](
                    **self.cfg.learning_rate_scheduler_kwargs[1]
                )

        # set up target networks
        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            # - freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)
            # - update target networks (hard update)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

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

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=jnp.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=jnp.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=jnp.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(name="truncated", size=1, dtype=jnp.int8)

            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "rewards",
                "next_observations",
                "next_states",
                "terminated",
                "truncated",
            ]

        # set up models for just-in-time compilation with XLA
        self.policy.apply = jax.jit(self.policy.apply, static_argnums=2)
        if self.critic_1 is not None and self.critic_2 is not None:
            self.critic_1.apply = jax.jit(self.critic_1.apply, static_argnums=2)
            self.critic_2.apply = jax.jit(self.critic_2.apply, static_argnums=2)
        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            self.target_critic_1.apply = jax.jit(self.target_critic_1.apply, static_argnums=2)
            self.target_critic_2.apply = jax.jit(self.target_critic_2.apply, static_argnums=2)

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
        }
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        actions, outputs = self.policy.act(inputs, role="policy")
        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)

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
            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
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

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """

        # gradient steps
        for gradient_step in range(self.cfg.gradient_steps):

            # sample a batch from memory
            (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_observations,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]

            inputs = {
                "observations": self._observation_preprocessor(sampled_observations, train=True),
                "states": self._state_preprocessor(sampled_states, train=True),
            }
            next_inputs = {
                "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                "states": self._state_preprocessor(sampled_next_states, train=True),
            }

            next_actions, outputs = self.policy.act(next_inputs, role="policy")
            next_log_prob = outputs["log_prob"]

            # compute target values
            target_q1_values, _ = self.target_critic_1.act(
                {**next_inputs, "taken_actions": next_actions}, role="target_critic_1"
            )
            target_q2_values, _ = self.target_critic_2.act(
                {**next_inputs, "taken_actions": next_actions}, role="target_critic_2"
            )

            # compute critic loss
            grad, critic_loss, critic_1_values, critic_2_values, target_values = _update_critic(
                self.critic_1.act,
                self.critic_1.state_dict,
                self.critic_2.act,
                self.critic_2.state_dict,
                target_q1_values,
                target_q2_values,
                self._entropy_coefficient,
                next_log_prob,
                {**inputs, "taken_actions": sampled_actions},
                sampled_rewards,
                sampled_terminated,
                sampled_truncated,
                self.cfg.discount_factor,
            )

            # optimization step (critic)
            if config.jax.is_distributed:
                grad = self.critic_1.reduce_parameters(grad)
            self.critic_1_optimizer = self.critic_1_optimizer.step(
                grad=grad, model=self.critic_1, lr=self.critic_learning_rate if self.critic_scheduler else None
            )
            self.critic_2_optimizer = self.critic_2_optimizer.step(
                grad=grad, model=self.critic_2, lr=self.critic_learning_rate if self.critic_scheduler else None
            )

            # compute policy (actor) loss
            grad, policy_loss, log_prob = _update_policy(
                self.policy.act,
                self.critic_1.act,
                self.critic_2.act,
                self.policy.state_dict,
                self.critic_1.state_dict,
                self.critic_2.state_dict,
                self._entropy_coefficient,
                inputs,
            )

            # optimization step (policy)
            if config.jax.is_distributed:
                grad = self.policy.reduce_parameters(grad)
            self.policy_optimizer = self.policy_optimizer.step(
                grad=grad, model=self.policy, lr=self.policy_learning_rate if self.policy_scheduler else None
            )

            # entropy learning
            if self.cfg.learn_entropy:
                # compute entropy loss
                grad, entropy_loss = _update_entropy(
                    self.log_entropy_coefficient.state_dict, self._target_entropy, log_prob
                )

                # optimization step (entropy)
                self.entropy_optimizer = self.entropy_optimizer.step(grad=grad, model=self.log_entropy_coefficient)

                # compute entropy coefficient
                self._entropy_coefficient = jnp.exp(self.log_entropy_coefficient.value)

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)

            # update learning rate
            if self.policy_scheduler:
                self.policy_learning_rate *= self.policy_scheduler(timestep)
            if self.critic_scheduler:
                self.critic_learning_rate *= self.critic_scheduler(timestep)

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data("Q-network / Q1 (max)", critic_1_values.max().item())
                self.track_data("Q-network / Q1 (min)", critic_1_values.min().item())
                self.track_data("Q-network / Q1 (mean)", critic_1_values.mean().item())

                self.track_data("Q-network / Q2 (max)", critic_2_values.max().item())
                self.track_data("Q-network / Q2 (min)", critic_2_values.min().item())
                self.track_data("Q-network / Q2 (mean)", critic_2_values.mean().item())

                self.track_data("Target / Target (max)", target_values.max().item())
                self.track_data("Target / Target (min)", target_values.min().item())
                self.track_data("Target / Target (mean)", target_values.mean().item())

                if self.cfg.learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self.policy_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_learning_rate)
                if self.critic_scheduler:
                    self.track_data("Learning / Critic learning rate", self.critic_learning_rate)
