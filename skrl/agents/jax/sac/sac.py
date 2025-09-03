from typing import Any, Mapping, Optional, Tuple, Union

import copy
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


# fmt: off
# [start-config-dict-jax]
SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler function (see optax.schedules)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "observation_preprocessor": None,       # observation preprocessor class (see skrl.resources.preprocessors)
    "observation_preprocessor_kwargs": {},  # observation preprocessor's kwargs (e.g. {"size": env.observation_space})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.state_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.2,   # initial entropy value
    "target_entropy": None,         # target entropy

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-jax]
# fmt: on


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
    inputs: Mapping[str, Union[np.ndarray, jax.Array]],
    sampled_rewards: Union[np.ndarray, jax.Array],
    sampled_terminated: Union[np.ndarray, jax.Array],
    sampled_truncated: Union[np.ndarray, jax.Array],
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
        models: Optional[Mapping[str, Model]] = None,
        memory: Optional[Memory] = None,
        observation_space: Optional[gymnasium.Space] = None,
        state_space: Optional[gymnasium.Space] = None,
        action_space: Optional[gymnasium.Space] = None,
        device: Optional[Union[str, jax.Device]] = None,
        cfg: Optional[dict] = None,
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
        # _cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = SAC_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
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

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._observation_preprocessor = self.cfg["observation_preprocessor"]
        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
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
                self.entropy_optimizer = Adam(model=self.log_entropy_coefficient, lr=self._entropy_learning_rate)

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            # schedulers
            if self._learning_rate_scheduler:
                self.policy_scheduler = self._learning_rate_scheduler(**self.cfg["learning_rate_scheduler_kwargs"])
                self.critic_scheduler = self._learning_rate_scheduler(**self.cfg["learning_rate_scheduler_kwargs"])
            # optimizers
            with jax.default_device(self.device):
                self.policy_optimizer = Adam(
                    model=self.policy,
                    lr=self._actor_learning_rate,
                    grad_norm_clip=self._grad_norm_clip,
                    scale=not self._learning_rate_scheduler,
                )
                self.critic_1_optimizer = Adam(
                    model=self.critic_1,
                    lr=self._critic_learning_rate,
                    grad_norm_clip=self._grad_norm_clip,
                    scale=not self._learning_rate_scheduler,
                )
                self.critic_2_optimizer = Adam(
                    model=self.critic_2,
                    lr=self._critic_learning_rate,
                    grad_norm_clip=self._grad_norm_clip,
                    scale=not self._learning_rate_scheduler,
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_1_optimizer"] = self.critic_1_optimizer
            self.checkpoint_modules["critic_2_optimizer"] = self.critic_2_optimizer

        # set up target networks
        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # set up preprocessors
        # - observations
        if self._observation_preprocessor:
            self._observation_preprocessor = self._observation_preprocessor(
                **self.cfg["observation_preprocessor_kwargs"]
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
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
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        actions, outputs = self.policy.act(inputs, role="policy")
        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)

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
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

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
        if timestep >= self._learning_starts:
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
        for gradient_step in range(self._gradient_steps):

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
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

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
                self._discount_factor,
            )

            # optimization step (critic)
            if config.jax.is_distributed:
                grad = self.critic_1.reduce_parameters(grad)
            self.critic_1_optimizer = self.critic_1_optimizer.step(
                grad=grad, model=self.critic_1, lr=self._critic_learning_rate if self._learning_rate_scheduler else None
            )
            self.critic_2_optimizer = self.critic_2_optimizer.step(
                grad=grad, model=self.critic_2, lr=self._critic_learning_rate if self._learning_rate_scheduler else None
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
                grad=grad, model=self.policy, lr=self._actor_learning_rate if self._learning_rate_scheduler else None
            )

            # entropy learning
            if self._learn_entropy:
                # compute entropy loss
                grad, entropy_loss = _update_entropy(
                    self.log_entropy_coefficient.state_dict, self._target_entropy, log_prob
                )

                # optimization step (entropy)
                self.entropy_optimizer = self.entropy_optimizer.step(grad=grad, model=self.log_entropy_coefficient)

                # compute entropy coefficient
                self._entropy_coefficient = jnp.exp(self.log_entropy_coefficient.value)

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self._actor_learning_rate *= self.policy_scheduler(timestep)
                self._critic_learning_rate *= self.critic_scheduler(timestep)

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

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self._learning_rate_scheduler:
                    self.track_data("Learning / Policy learning rate", self._actor_learning_rate)
                    self.track_data("Learning / Critic learning rate", self._critic_learning_rate)
