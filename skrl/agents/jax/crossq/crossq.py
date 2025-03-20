from typing import Any, Mapping, Optional, Tuple, Union

import functools
import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl import config, logger
from skrl.agents.jax import Agent
from skrl.memories.jax import Memory
from skrl.models.jax.base import Model, StateDict
from skrl.resources.optimizers.jax import Adam


# fmt: off
# [start-config-dict-jax]
CROSSQ_DEFAULT_CONFIG = {
    "policy_delay" : 3,
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler function (see optax.schedules)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "optimizer_kwargs" : {
        'betas' : [0.5, 0.99]
    },

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

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
@functools.partial(jax.jit, static_argnames=("critic_1_act", "critic_2_act", "discount_factor"))
def _update_critic(
    critic_1_act,
    critic_1_state_dict,
    critic_2_act,
    critic_2_state_dict,
    all_states,
    all_actions,
    entropy_coefficient,
    next_log_prob,
    sampled_rewards: Union[np.ndarray, jax.Array],
    sampled_terminated: Union[np.ndarray, jax.Array],
    sampled_truncated: Union[np.ndarray, jax.Array],
    discount_factor: float,
):
    # compute critic loss
    def _critic_loss(params, batch_stats, critic_act, role):
        all_q_values, _, _ = critic_act(
            {"states": all_states, "taken_actions": all_actions, "mutable": ["batch_stats"]},
            role=role,
            train=True,
            params={"params": params, "batch_stats": batch_stats},
        )
        current_q_values, next_q_values = jnp.split(all_q_values, 2, axis=1)

        next_q_values = jnp.min(next_q_values, axis=0)
        next_q_values = next_q_values - entropy_coefficient * next_log_prob.reshape(-1, 1)

        target_q_values = (
            sampled_rewards.reshape(-1, 1)
            + discount_factor * jnp.logical_not(sampled_terminated | sampled_truncated) * next_q_values
        )

        loss = 0.5 * ((jax.lax.stop_gradient(target_q_values) - current_q_values) ** 2).mean(axis=1).sum()

        return loss, (current_q_values, next_q_values)

    df = jax.value_and_grad(_critic_loss, has_aux=True, allow_int=True)
    (critic_1_loss, critic_1_values, next_q1_values), grad = df(
        critic_1_state_dict.params, critic_1_state_dict.batch_stats, critic_1_act, "critic_1"
    )
    (critic_2_loss, critic_2_values, next_q2_values), grad = jax.value_and_grad(
        _critic_loss, has_aux=True, allow_int=True
    )(critic_2_state_dict.params, critic_2_state_dict.batch_stats, critic_2_act, "critic_2")

    target_q_values = jnp.minimum(next_q1_values, next_q2_values) - entropy_coefficient * next_log_prob
    target_values = (
        sampled_rewards + discount_factor * jnp.logical_not(sampled_terminated | sampled_truncated) * target_q_values
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
    sampled_states,
):
    # compute policy (actor) loss
    def _policy_loss(policy_params, critic_1_params, critic_2_params):
        actions, log_prob, _ = policy_act({"states": sampled_states}, "policy", train=True, params=policy_params)
        critic_1_values, _, _ = critic_1_act(
            {"states": sampled_states, "taken_actions": actions},
            "critic_1",
            train=False,
            params=critic_1_params,
        )
        critic_2_values, _, _ = critic_2_act(
            {"states": sampled_states, "taken_actions": actions},
            "critic_2",
            train=False,
            params=critic_2_params,
        )
        return (entropy_coefficient * log_prob - jnp.minimum(critic_1_values, critic_2_values)).mean(), log_prob

    (policy_loss, log_prob), grad = jax.value_and_grad(_policy_loss, has_aux=True)(
        {"params": policy_state_dict.params, "batch_stats": policy_state_dict.batch_stats},
        {"params": critic_1_state_dict.params, "batch_stats": critic_1_state_dict.batch_stats},
        {"params": critic_2_state_dict.params, "batch_stats": critic_2_state_dict.batch_stats},
    )

    return grad, policy_loss, log_prob


@jax.jit
def _update_entropy(log_entropy_coefficient_state_dict, target_entropy, log_prob):
    # compute entropy loss
    def _entropy_loss(params):
        return -(params["params"] * (log_prob + target_entropy)).mean()

    entropy_loss, grad = jax.value_and_grad(_entropy_loss, has_aux=False)(log_entropy_coefficient_state_dict.params)

    return grad, entropy_loss


class CrossQ(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, jax.Device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.jax.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.jax.Memory, list of skrl.memory.jax.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        # _cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = CROSSQ_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2

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
        self.policy_delay = self.cfg["policy_delay"]
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self.optimizer_kwargs = self.cfg["optimizer_kwargs"]
        self._n_updates: int = 0

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
                    **self.optimizer_kwargs,
                )
                self.critic_1_optimizer = Adam(
                    model=self.critic_1,
                    lr=self._critic_learning_rate,
                    grad_norm_clip=self._grad_norm_clip,
                    scale=not self._learning_rate_scheduler,
                    **self.optimizer_kwargs,
                )
                self.critic_2_optimizer = Adam(
                    model=self.critic_2,
                    lr=self._critic_learning_rate,
                    grad_norm_clip=self._grad_norm_clip,
                    scale=not self._learning_rate_scheduler,
                    **self.optimizer_kwargs,
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_1_optimizer"] = self.critic_1_optimizer
            self.checkpoint_modules["critic_2_optimizer"] = self.critic_2_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=jnp.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(name="truncated", size=1, dtype=jnp.int8)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

        # set up models for just-in-time compilation with XLA
        self.policy.apply = jax.jit(self.policy.apply, static_argnames=["role", "train"])
        if self.critic_1 is not None and self.critic_2 is not None:
            self.critic_1.apply = jax.jit(self.critic_1.apply, static_argnames=["role", "train"])
            self.critic_2.apply = jax.jit(self.critic_2.apply, static_argnames=["role", "train"])

    def act(self, states: Union[np.ndarray, jax.Array], timestep: int, timesteps: int) -> Union[np.ndarray, jax.Array]:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: np.ndarray or jax.Array
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: np.ndarray or jax.Array
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)})

        # sample stochastic actions
        actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)})
        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)

        return actions, None, outputs

    def record_transition(
        self,
        states: Union[np.ndarray, jax.Array],
        actions: Union[np.ndarray, jax.Array],
        rewards: Union[np.ndarray, jax.Array],
        next_states: Union[np.ndarray, jax.Array],
        terminated: Union[np.ndarray, jax.Array],
        truncated: Union[np.ndarray, jax.Array],
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: np.ndarray or jax.Array
        :param actions: Actions taken by the agent
        :type actions: np.ndarray or jax.Array
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: np.ndarray or jax.Array
        :param next_states: Next observations/states of the environment
        :type next_states: np.ndarray or jax.Array
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: np.ndarray or jax.Array
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: np.ndarray or jax.Array
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            policy_delay_indices = {
                i: True for i in range(self._gradient_steps) if ((self._n_updates + i + 1) % self.policy_delay) == 0
            }
            policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

            self.set_mode("train")
            self._update(timestep, timesteps, self._gradient_steps, policy_delay_indices)
            self.set_mode("eval")

            self._n_updates += self._gradient_steps

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(
        self,
        timestep: int,
        timesteps: int,
        gradient_steps: int,
        policy_delay_indices: flax.core.FrozenDict,
    ) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in range(gradient_steps):
            self._n_updates += 1
            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            sampled_states = self._state_preprocessor(sampled_states, train=True)
            sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

            next_actions, next_log_prob, _ = self.policy.act({"states": sampled_next_states}, role="policy")

            all_states = jnp.concatenate((sampled_states, sampled_next_states))
            all_actions = jnp.concatenate((sampled_actions, next_actions))

            # compute critic loss
            grad, critic_loss, critic_1_values, critic_2_values, target_values = _update_critic(
                self.critic_1.act,
                self.critic_1.state_dict,
                self.critic_2.act,
                self.critic_2.state_dict,
                all_states,
                all_actions,
                self._entropy_coefficient,
                next_log_prob,
                sampled_rewards,
                sampled_terminated,
                sampled_truncated,
                self._discount_factor,
            )

            # optimization step (critic)
            if config.jax.is_distributed:
                grad = self.critic_1.reduce_parameters(grad)
            self.critic_1_optimizer = self.critic_1_optimizer.step(
                grad, self.critic_1, self._critic_learning_rate if self._learning_rate_scheduler else None
            )
            self.critic_2_optimizer = self.critic_2_optimizer.step(
                grad, self.critic_2, self._critic_learning_rate if self._learning_rate_scheduler else None
            )

            update_actor = gradient_step in policy_delay_indices
            if update_actor:
                # compute policy (actor) loss
                grad, policy_loss, log_prob = _update_policy(
                    self.policy.act,
                    self.critic_1.act,
                    self.critic_2.act,
                    self.policy.state_dict,
                    self.critic_1.state_dict,
                    self.critic_2.state_dict,
                    self._entropy_coefficient,
                    sampled_states,
                )

                # optimization step (policy)
                if config.jax.is_distributed:
                    grad = self.policy.reduce_parameters(grad)
                self.policy_optimizer = self.policy_optimizer.step(
                    grad, self.policy, self._actor_learning_rate if self._learning_rate_scheduler else None
                )

                # entropy learning
                if self._learn_entropy:
                    # compute entropy loss
                    grad, entropy_loss = _update_entropy(
                        self.log_entropy_coefficient.state_dict, self._target_entropy, log_prob
                    )

                    # optimization step (entropy)
                    self.entropy_optimizer = self.entropy_optimizer.step(grad, self.log_entropy_coefficient)

                    # compute entropy coefficient
                    self._entropy_coefficient = jnp.exp(self.log_entropy_coefficient.value)

            # update learning rate
            if self._learning_rate_scheduler:
                if update_actor:
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
