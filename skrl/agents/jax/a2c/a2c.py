from typing import Any, Dict, Optional, Tuple, Union

import copy
import functools
import gym
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl.agents.jax import Agent
from skrl.memories.jax import Memory
from skrl.models.jax import Model
from skrl.resources.optimizers.jax import Adam
from skrl.resources.schedulers.jax import KLAdaptiveLR


# [start-config-dict-jax]
A2C_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "mini_batches": 1,              # number of mini batches to use for updating

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,          # clipping coefficient for the norm of the gradients

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-jax]


def compute_gae(rewards: np.ndarray,
                dones: np.ndarray,
                values: np.ndarray,
                next_values: np.ndarray,
                discount_factor: float = 0.99,
                lambda_coefficient: float = 0.95) -> np.ndarray:
    """Compute the Generalized Advantage Estimator (GAE)

    :param rewards: Rewards obtained by the agent
    :type rewards: np.ndarray
    :param dones: Signals to indicate that episodes have ended
    :type dones: np.ndarray
    :param values: Values obtained by the agent
    :type values: np.ndarray
    :param next_values: Next values obtained by the agent
    :type next_values: np.ndarray
    :param discount_factor: Discount factor
    :type discount_factor: float
    :param lambda_coefficient: Lambda coefficient
    :type lambda_coefficient: float

    :return: Generalized Advantage Estimator
    :rtype: np.ndarray
    """
    advantage = 0
    advantages = np.zeros_like(rewards)
    not_dones = np.logical_not(dones)
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@jax.jit
def _compute_gae(rewards: jax.Array,
                 dones: jax.Array,
                 values: jax.Array,
                 next_values: jax.Array,
                 discount_factor: float = 0.99,
                 lambda_coefficient: float = 0.95) -> jax.Array:
    advantage = 0
    advantages = jnp.zeros_like(rewards)
    not_dones = jnp.logical_not(dones)
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
        advantages = advantages.at[i].set(advantage)
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

@functools.partial(jax.jit, static_argnames=("policy_act", "get_entropy", "entropy_loss_scale"))
def _update_policy(policy_act,
                   policy_state_dict,
                   sampled_states,
                   sampled_actions,
                   sampled_log_prob,
                   sampled_advantages,
                   get_entropy,
                   entropy_loss_scale):
    # compute policy loss
    def _policy_loss(params):
        _, next_log_prob, outputs = policy_act({"states": sampled_states, "taken_actions": sampled_actions}, "policy", params)

        # compute approximate KL divergence
        ratio = next_log_prob - sampled_log_prob
        kl_divergence = ((jnp.exp(ratio) - 1) - ratio).mean()

        # compute entropy loss
        entropy_loss = 0
        if entropy_loss_scale:
            entropy_loss = -entropy_loss_scale * get_entropy(outputs["stddev"], role="policy").mean()

        return -(sampled_advantages * next_log_prob).mean(), (entropy_loss, kl_divergence, outputs["stddev"])

    (policy_loss, (entropy_loss, kl_divergence, stddev)), grad = jax.value_and_grad(_policy_loss, has_aux=True)(policy_state_dict.params)

    return grad, policy_loss, entropy_loss, kl_divergence, stddev

@functools.partial(jax.jit, static_argnames=("value_act"))
def _update_value(value_act,
                  value_state_dict,
                  sampled_states,
                  sampled_returns):
    # compute value loss
    def _value_loss(params):
        predicted_values, _, _ = value_act({"states": sampled_states}, "value", params)
        return ((sampled_returns - predicted_values) ** 2).mean()

    value_loss, grad = jax.value_and_grad(_value_loss, has_aux=False)(value_state_dict.params)

    return grad, value_loss


class A2C(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, jax.Device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Advantage Actor Critic (A2C)

        https://arxiv.org/abs/1602.01783

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.jax.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.jax.Memory, list of skrl.memory.jax.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        # _cfg = copy.deepcopy(A2C_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = A2C_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # configuration
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            # scheduler
            scale = True
            self.scheduler = None
            if self._learning_rate_scheduler is not None:
                if self._learning_rate_scheduler == KLAdaptiveLR:
                    scale = False
                    self.scheduler = self._learning_rate_scheduler(self._learning_rate, **self.cfg["learning_rate_scheduler_kwargs"])
                else:
                    self._learning_rate = self._learning_rate_scheduler(self._learning_rate, **self.cfg["learning_rate_scheduler_kwargs"])
            # optimizer
            self.policy_optimizer = Adam(model=self.policy, lr=self._learning_rate, grad_norm_clip=self._grad_norm_clip, scale=scale)
            self.value_optimizer = Adam(model=self.value, lr=self._learning_rate, grad_norm_clip=self._grad_norm_clip, scale=scale)

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["value_optimizer"] = self.value_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=jnp.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(name="log_prob", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="values", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=jnp.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

        # set up models for just-in-time compilation with XLA
        self.policy.apply = jax.jit(self.policy.apply, static_argnums=2)
        if self.value is not None:
            self.value.apply = jax.jit(self.value.apply, static_argnums=2)

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
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)
            log_prob = jax.device_get(log_prob)

        self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(self,
                          states: Union[np.ndarray, jax.Array],
                          actions: Union[np.ndarray, jax.Array],
                          rewards: Union[np.ndarray, jax.Array],
                          next_states: Union[np.ndarray, jax.Array],
                          terminated: Union[np.ndarray, jax.Array],
                          truncated: Union[np.ndarray, jax.Array],
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
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
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            if not self._jax:  # numpy backend
                values = jax.device_get(values)
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values)

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
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # compute returns and advantages
        self.value.training = False
        last_values, _, _ = self.value.act({"states": self._state_preprocessor(self._current_next_states)}, role="value")  # TODO: .float()
        self.value.training = True
        if not self._jax:  # numpy backend
            last_values = jax.device_get(last_values)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        if self._jax:
            returns, advantages = _compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                               dones=self.memory.get_tensor_by_name("terminated"),
                                               values=values,
                                               next_values=last_values,
                                               discount_factor=self._discount_factor,
                                               lambda_coefficient=self._lambda)
        else:
            returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                              dones=self.memory.get_tensor_by_name("terminated"),
                                              values=values,
                                              next_values=last_values,
                                              discount_factor=self._discount_factor,
                                              lambda_coefficient=self._lambda)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        kl_divergences = []

        # mini-batches loop
        for sampled_states, sampled_actions, sampled_log_prob, sampled_returns, sampled_advantages in sampled_batches:

            sampled_states = self._state_preprocessor(sampled_states, train=True)

            # compute policy loss
            grad, policy_loss, entropy_loss, kl_divergence, stddev = _update_policy(self.policy.act,
                                                                                    self.policy.state_dict,
                                                                                    sampled_states,
                                                                                    sampled_actions,
                                                                                    sampled_log_prob,
                                                                                    sampled_advantages,
                                                                                    self.policy.get_entropy,
                                                                                    self._entropy_loss_scale)

            kl_divergences.append(kl_divergence.item())

            # optimization step (policy)
            self.policy_optimizer = self.policy_optimizer.step(grad, self.policy, self.scheduler._lr if self.scheduler else None)

            # compute value loss
            grad, value_loss = _update_value(self.value.act,
                                                self.value.state_dict,
                                                sampled_states,
                                                sampled_returns)

            # optimization step (value)
            self.value_optimizer = self.value_optimizer.step(grad, self.value, self.scheduler._lr if self.scheduler else None)

            # update cumulative losses
            cumulative_policy_loss += policy_loss.item()
            cumulative_value_loss += value_loss.item()
            if self._entropy_loss_scale:
                cumulative_entropy_loss += entropy_loss.item()

        # update learning rate
        if self._learning_rate_scheduler:
            if isinstance(self.scheduler, KLAdaptiveLR):
                self.scheduler.step(np.mean(kl_divergences))

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / len(sampled_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / len(sampled_batches))

        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / len(sampled_batches))

        self.track_data("Policy / Standard deviation", stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler._lr)
