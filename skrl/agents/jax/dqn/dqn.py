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

from .dqn_cfg import DQN_CFG


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@functools.partial(jax.jit, static_argnames=("q_network_act"))
def _update_q_network(
    q_network_act,
    q_network_state_dict,
    next_q_values,
    inputs,
    sampled_actions,
    sampled_rewards,
    sampled_terminated,
    sampled_truncated,
    discount_factor,
):
    # compute target values
    target_q_values = jnp.max(next_q_values, axis=-1, keepdims=True)
    target_values = (
        sampled_rewards + discount_factor * jnp.logical_not(sampled_terminated | sampled_truncated) * target_q_values
    )

    # compute Q-network loss
    def _q_network_loss(params):
        q_values = q_network_act(inputs, role="q_network", params=params)[0]
        q_values = q_values[jnp.arange(q_values.shape[0]), sampled_actions.reshape(-1)]
        return ((q_values - target_values.reshape(-1)) ** 2).mean()

    q_network_loss, grad = jax.value_and_grad(_q_network_loss, has_aux=False)(q_network_state_dict.params)

    return grad, q_network_loss, target_values


class DQN(Agent):
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
        """Deep Q-Network (DQN).

        https://arxiv.org/abs/1312.5602

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: DQN_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=DQN_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.q_network = self.models.get("q_network", None)
        self.target_q_network = self.models.get("target_q_network", None)

        # checkpoint models
        self.checkpoint_modules["q_network"] = self.q_network
        self.checkpoint_modules["target_q_network"] = self.target_q_network

        # broadcast models' parameters in distributed runs
        if config.jax.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.q_network is not None:
                self.q_network.broadcast_parameters()

        # set up optimizer and learning rate scheduler
        if self.q_network is not None:
            self.learning_rate = self.cfg.learning_rate
            # - optimizer
            with jax.default_device(self.device):
                self.optimizer = Adam(
                    model=self.q_network,
                    lr=self.learning_rate,
                    scale=not self.cfg.learning_rate_scheduler,
                )
            self.checkpoint_modules["optimizer"] = self.optimizer
            # - learning rate scheduler
            self.scheduler = self.cfg.learning_rate_scheduler
            if self.scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler(**self.cfg.learning_rate_scheduler_kwargs)

        # set up target networks
        if self.target_q_network is not None:
            # - freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_q_network.freeze_parameters(True)
            # - update target networks (hard update)
            self.target_q_network.update_parameters(self.q_network, polyak=1)

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
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=jnp.int32)
            self.memory.create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(name="truncated", size=1, dtype=jnp.int8)

        self.tensors_names = [
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
        self.q_network.apply = jax.jit(self.q_network.apply, static_argnums=2)
        if self.target_q_network is not None:
            self.target_q_network.apply = jax.jit(self.target_q_network.apply, static_argnums=2)

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

        if self.cfg.exploration_scheduler is None:
            q_values, outputs = self.q_network.act(inputs, role="q_network")
            actions = jnp.argmax(q_values, axis=1, keepdims=True)
            if not self._jax:  # numpy backend
                actions = jax.device_get(actions)
            return actions, outputs

        # sample random actions
        actions, outputs = self.q_network.random_act(inputs, role="q_network")
        if timestep < self.cfg.random_timesteps:
            if not self._jax:  # numpy backend
                actions = jax.device_get(actions)
            return actions, outputs

        # sample actions with epsilon-greedy policy
        epsilon = self.cfg.exploration_scheduler(timestep, timesteps)
        indexes = (np.random.random(actions.shape[0]) >= epsilon).nonzero()[0]
        if indexes.size:
            inputs = {k: None if v is None else v[indexes] for k, v in inputs.items()}
            q_values, outputs = self.q_network.act(inputs, role="q_network")
            if self._jax:
                raise NotImplementedError
                actions[indexes] = jnp.argmax(q_values, axis=1, keepdims=True)
            else:
                q_values = jax.device_get(q_values)
                actions = np.array(jax.device_get(actions))  # bypass: assignment destination is read-only
                actions[indexes] = np.argmax(q_values, axis=1, keepdims=True)

        # record epsilon
        self.track_data("Exploration / Exploration epsilon", epsilon)

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
            ) = self.memory.sample(names=self.tensors_names, batch_size=self.cfg.batch_size)[0]

            inputs = {
                "observations": self._observation_preprocessor(sampled_observations, train=True),
                "states": self._state_preprocessor(sampled_states, train=True),
            }
            next_inputs = {
                "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                "states": self._state_preprocessor(sampled_next_states, train=True),
            }

            # compute target values
            next_q_values, _ = self.target_q_network.act(next_inputs, role="target_q_network")

            grad, q_network_loss, target_values = _update_q_network(
                self.q_network.act,
                self.q_network.state_dict,
                next_q_values,
                inputs,
                sampled_actions,
                sampled_rewards,
                sampled_terminated,
                sampled_truncated,
                self.cfg.discount_factor,
            )

            # optimization step (Q-network)
            if config.jax.is_distributed:
                grad = self.q_network.reduce_parameters(grad)
            self.optimizer = self.optimizer.step(
                grad=grad, model=self.q_network, lr=self.learning_rate if self.scheduler else None
            )

            # update target network
            if not timestep % self.cfg.target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self.cfg.polyak)

            # update learning rate
            if self.scheduler:
                self.learning_rate *= self.scheduler(timestep)

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", target_values.max().item())
            self.track_data("Target / Target (min)", target_values.min().item())
            self.track_data("Target / Target (mean)", target_values.mean().item())

            if self.scheduler:
                self.track_data("Learning / Learning rate", self.learning_rate)
