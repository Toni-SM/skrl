from typing import Any, Mapping, Optional, Tuple, Union

import copy
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
from skrl.utils import ScopedTimer

from .ddpg_cfg import DDPG_CFG


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@jax.jit
def _apply_exploration_noise(
    actions: jax.Array, noises: jax.Array, clip_actions_min: jax.Array, clip_actions_max: jax.Array, scale: float
) -> jax.Array:
    noises = noises.at[:].multiply(scale)
    return jnp.clip(actions + noises, a_min=clip_actions_min, a_max=clip_actions_max), noises


@functools.partial(jax.jit, static_argnames=("critic_act"))
def _update_critic(
    critic_act,
    critic_state_dict,
    target_q_values: jax.Array,
    inputs: Mapping[str, Union[np.ndarray, jax.Array]],
    sampled_rewards: Union[np.ndarray, jax.Array],
    sampled_terminated: Union[np.ndarray, jax.Array],
    sampled_truncated: Union[np.ndarray, jax.Array],
    discount_factor: float,
):
    # compute target values
    target_values = (
        sampled_rewards + discount_factor * jnp.logical_not(sampled_terminated | sampled_truncated) * target_q_values
    )

    # compute critic loss
    def _critic_loss(params):
        critic_values, _ = critic_act(inputs, role="critic", params=params)
        critic_loss = ((critic_values - target_values) ** 2).mean()
        return critic_loss, critic_values

    (critic_loss, critic_values), grad = jax.value_and_grad(_critic_loss, has_aux=True)(critic_state_dict.params)

    return grad, critic_loss, critic_values, target_values


@functools.partial(jax.jit, static_argnames=("policy_act", "critic_act"))
def _update_policy(policy_act, critic_act, policy_state_dict, critic_state_dict, inputs):
    # compute policy (actor) loss
    def _policy_loss(policy_params, critic_params):
        actions, _ = policy_act(inputs, role="policy", params=policy_params)
        critic_values, _ = critic_act({**inputs, "taken_actions": actions}, role="critic", params=critic_params)
        return -critic_values.mean()

    policy_loss, grad = jax.value_and_grad(_policy_loss, has_aux=False)(
        policy_state_dict.params, critic_state_dict.params
    )

    return grad, policy_loss


class DDPG(Agent):
    def __init__(
        self,
        *,
        models: Optional[Mapping[str, Model]] = None,
        memory: Optional[Memory] = None,
        observation_space: Optional[gymnasium.Space] = None,
        state_space: Optional[gymnasium.Space] = None,
        action_space: Optional[gymnasium.Space] = None,
        device: Optional[Union[str, jax.Device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Deep Deterministic Policy Gradient (DDPG).

        https://arxiv.org/abs/1509.02971

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: DDPG_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=DDPG_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.target_policy = self.models.get("target_policy", None)
        self.critic = self.models.get("critic", None)
        self.target_critic = self.models.get("target_critic", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["target_critic"] = self.target_critic

        # broadcast models' parameters in distributed runs
        if config.jax.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic is not None:
                self.critic.broadcast_parameters()

        # set up noise
        if self.cfg.exploration_noise is not None:
            self._exploration_noise = self.cfg.exploration_noise(**self.cfg.exploration_noise_kwargs)
        else:
            logger.warning("agents:DDPG: No exploration noise specified, training performance may be degraded")
            self._exploration_noise = None

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic is not None:
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
                self.critic_optimizer = Adam(
                    model=self.critic,
                    lr=self.critic_learning_rate,
                    grad_norm_clip=self.cfg.grad_norm_clip,
                    scale=not self.cfg.learning_rate_scheduler[1],
                )
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
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
        if self.target_policy is not None and self.target_critic is not None:
            # - freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic.freeze_parameters(True)
            # - update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic.update_parameters(self.critic, polyak=1)

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

    def init(self, *, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
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

        # clip noise bounds
        if self.action_space is not None:
            if self._jax:
                self.clip_actions_min = jnp.array(self.action_space.low, dtype=jnp.float32)
                self.clip_actions_max = jnp.array(self.action_space.high, dtype=jnp.float32)
            else:
                self.clip_actions_min = np.array(self.action_space.low, dtype=np.float32)
                self.clip_actions_max = np.array(self.action_space.high, dtype=np.float32)

        # set up models for just-in-time compilation with XLA
        self.policy.apply = jax.jit(self.policy.apply, static_argnums=2)
        if self.critic is not None:
            self.critic.apply = jax.jit(self.critic.apply, static_argnums=2)
        if self.target_policy is not None and self.target_critic is not None:
            self.target_policy.apply = jax.jit(self.target_policy.apply, static_argnums=2)
            self.target_critic.apply = jax.jit(self.target_critic.apply, static_argnums=2)

    def act(
        self,
        observations: Union[np.ndarray, jax.Array],
        states: Union[np.ndarray, jax.Array, None],
        *,
        timestep: int,
        timesteps: int,
    ) -> Tuple[Union[np.ndarray, jax.Array], Mapping[str, Union[np.ndarray, jax.Array, Any]]]:
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
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample deterministic actions
        actions, outputs = self.policy.act(inputs, role="policy")
        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)

        # add exploration noise
        if self._exploration_noise is not None:
            noises = self._exploration_noise.sample(actions.shape)
            scale = self.cfg.exploration_scheduler(timestep, timesteps) if self.cfg.exploration_scheduler else 1.0
            # modify actions
            if self._jax:
                actions, noises = _apply_exploration_noise(
                    actions, noises, self.clip_actions_min, self.clip_actions_max, scale
                )
            else:
                noises *= scale
                actions = np.clip(actions + noises, a_min=self.clip_actions_min, a_max=self.clip_actions_max)

            self.track_data("Exploration / Exploration noise (max)", noises.max().item())
            self.track_data("Exploration / Exploration noise (min)", noises.min().item())
            self.track_data("Exploration / Exploration noise (mean)", noises.mean().item())

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: Union[np.ndarray, jax.Array],
        states: Union[np.ndarray, jax.Array],
        actions: Union[np.ndarray, jax.Array],
        rewards: Union[np.ndarray, jax.Array],
        next_observations: Union[np.ndarray, jax.Array],
        next_states: Union[np.ndarray, jax.Array],
        terminated: Union[np.ndarray, jax.Array],
        truncated: Union[np.ndarray, jax.Array],
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

            # compute target values
            next_actions, _ = self.target_policy.act(next_inputs, role="target_policy")
            target_q_values, _ = self.target_critic.act(
                {**next_inputs, "taken_actions": next_actions}, role="target_critic"
            )

            # compute critic loss
            grad, critic_loss, critic_values, target_values = _update_critic(
                self.critic.act,
                self.critic.state_dict,
                target_q_values,
                {**inputs, "taken_actions": sampled_actions},
                sampled_rewards,
                sampled_terminated,
                sampled_truncated,
                self.cfg.discount_factor,
            )

            # optimization step (critic)
            if config.jax.is_distributed:
                grad = self.critic.reduce_parameters(grad)
            self.critic_optimizer = self.critic_optimizer.step(
                grad=grad, model=self.critic, lr=self.critic_learning_rate if self.critic_scheduler else None
            )

            # compute policy (actor) loss
            grad, policy_loss = _update_policy(
                self.policy.act, self.critic.act, self.policy.state_dict, self.critic.state_dict, inputs
            )

            # optimization step (policy)
            if config.jax.is_distributed:
                grad = self.policy.reduce_parameters(grad)
            self.policy_optimizer = self.policy_optimizer.step(
                grad=grad, model=self.policy, lr=self.policy_learning_rate if self.policy_scheduler else None
            )

            # update target networks
            self.target_policy.update_parameters(self.policy, polyak=self.cfg.polyak)
            self.target_critic.update_parameters(self.critic, polyak=self.cfg.polyak)

            # update learning rate
            if self.policy_scheduler:
                self.policy_learning_rate *= self.policy_scheduler(timestep)
            if self.critic_scheduler:
                self.critic_learning_rate *= self.critic_scheduler(timestep)

            # record data
            self.track_data("Loss / Policy loss", policy_loss.item())
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-network / Q1 (max)", critic_values.max().item())
            self.track_data("Q-network / Q1 (min)", critic_values.min().item())
            self.track_data("Q-network / Q1 (mean)", critic_values.mean().item())

            self.track_data("Target / Target (max)", target_values.max().item())
            self.track_data("Target / Target (min)", target_values.min().item())
            self.track_data("Target / Target (mean)", target_values.mean().item())

            if self.policy_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_learning_rate)
            if self.critic_scheduler:
                self.track_data("Learning / Critic learning rate", self.critic_learning_rate)
