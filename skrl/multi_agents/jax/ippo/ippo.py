from typing import Any, Dict, Mapping, Optional, Sequence, Union

import copy
import functools
import gym
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl.memories.jax import Memory
from skrl.models.jax import Model
from skrl.multi_agents.jax import MultiAgent
from skrl.resources.optimizers.jax import Adam
from skrl.resources.schedulers.jax import KLAdaptiveLR


# [start-config-dict-jax]
IPPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

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

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

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
                   ratio_clip,
                   get_entropy,
                   entropy_loss_scale):
    # compute policy loss
    def _policy_loss(params):
        _, next_log_prob, outputs = policy_act({"states": sampled_states, "taken_actions": sampled_actions}, "policy", params)

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

    (policy_loss, (entropy_loss, kl_divergence, stddev)), grad = jax.value_and_grad(_policy_loss, has_aux=True)(policy_state_dict.params)

    return grad, policy_loss, entropy_loss, kl_divergence, stddev

@functools.partial(jax.jit, static_argnames=("value_act", "clip_predicted_values"))
def _update_value(value_act,
                  value_state_dict,
                  sampled_states,
                  sampled_values,
                  sampled_returns,
                  value_loss_scale,
                  clip_predicted_values,
                  value_clip):
    # compute value loss
    def _value_loss(params):
        predicted_values, _, _ = value_act({"states": sampled_states}, "value", params)
        if clip_predicted_values:
            predicted_values = sampled_values + jnp.clip(predicted_values - sampled_values, -value_clip, value_clip)
        return value_loss_scale * ((sampled_returns - predicted_values) ** 2).mean()

    value_loss, grad = jax.value_and_grad(_value_loss, has_aux=False)(value_state_dict.params)

    return grad, value_loss


class IPPO(MultiAgent):
    def __init__(self,
                 possible_agents: Sequence[str],
                 models: Dict[str, Model],
                 memories: Optional[Mapping[str, Memory]] = None,
                 observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gym.Space], Mapping[str, gymnasium.Space]]] = None,
                 action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gym.Space], Mapping[str, gymnasium.Space]]] = None,
                 device: Optional[Union[str, jax.Device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Independent Proximal Policy Optimization (IPPO)

        https://arxiv.org/abs/2011.09533

        :param possible_agents: Name of all possible agents the environment could generate
        :type possible_agents: list of str
        :param models: Models used by the agents.
                       External keys are environment agents' names. Internal keys are the models required by the algorithm
        :type models: nested dictionary of skrl.models.jax.Model
        :param memories: Memories to storage the transitions.
        :type memories: dictionary of skrl.memory.jax.Memory, optional
        :param observation_spaces: Observation/state spaces or shapes (default: ``None``)
        :type observation_spaces: dictionary of int, sequence of int, gym.Space or gymnasium.Space, optional
        :param action_spaces: Action spaces or shapes (default: ``None``)
        :type action_spaces: dictionary of int, sequence of int, gym.Space or gymnasium.Space, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        # _cfg = copy.deepcopy(IPPO_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = IPPO_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(possible_agents=possible_agents,
                         models=models,
                         memories=memories,
                         observation_spaces=observation_spaces,
                         action_spaces=action_spaces,
                         device=device,
                         cfg=_cfg)

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}

        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]

        # configuration
        self._learning_epochs = self._as_dict(self.cfg["learning_epochs"])
        self._mini_batches = self._as_dict(self.cfg["mini_batches"])
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self._as_dict(self.cfg["grad_norm_clip"])
        self._ratio_clip = self._as_dict(self.cfg["ratio_clip"])
        self._value_clip = self._as_dict(self.cfg["value_clip"])
        self._clip_predicted_values = self._as_dict(self.cfg["clip_predicted_values"])

        self._value_loss_scale = self._as_dict(self.cfg["value_loss_scale"])
        self._entropy_loss_scale = self._as_dict(self.cfg["entropy_loss_scale"])

        self._kl_threshold = self._as_dict(self.cfg["kl_threshold"])

        self._learning_rate = self._as_dict(self.cfg["learning_rate"])
        self._learning_rate_scheduler = self._as_dict(self.cfg["learning_rate_scheduler"])
        self._learning_rate_scheduler_kwargs = self._as_dict(self.cfg["learning_rate_scheduler_kwargs"])

        self._state_preprocessor = self._as_dict(self.cfg["state_preprocessor"])
        self._state_preprocessor_kwargs = self._as_dict(self.cfg["state_preprocessor_kwargs"])
        self._value_preprocessor = self._as_dict(self.cfg["value_preprocessor"])
        self._value_preprocessor_kwargs = self._as_dict(self.cfg["value_preprocessor_kwargs"])

        self._discount_factor = self._as_dict(self.cfg["discount_factor"])
        self._lambda = self._as_dict(self.cfg["lambda"])

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self._as_dict(self.cfg["time_limit_bootstrap"])

        # set up optimizer and learning rate scheduler
        self.policy_optimizer = {}
        self.value_optimizer = {}
        self.schedulers = {}

        for uid in self.possible_agents:
            policy = self.policies[uid]
            value = self.values[uid]

            if policy is not None and value is not None:
                # scheduler
                scale = True
                self.schedulers[uid] = None
                if self._learning_rate_scheduler[uid] is not None:
                    if self._learning_rate_scheduler[uid] == KLAdaptiveLR:
                        scale = False
                        self.schedulers[uid] = self._learning_rate_scheduler[uid](self._learning_rate[uid], **self._learning_rate_scheduler_kwargs[uid])
                    else:
                        self._learning_rate[uid] = self._learning_rate_scheduler[uid](self._learning_rate[uid], **self._learning_rate_scheduler_kwargs[uid])
                # optimizer
                self.policy_optimizer[uid] = Adam(model=policy, lr=self._learning_rate[uid], grad_norm_clip=self._grad_norm_clip[uid], scale=scale)
                self.value_optimizer[uid] = Adam(model=value, lr=self._learning_rate[uid], grad_norm_clip=self._grad_norm_clip[uid], scale=scale)

                self.checkpoint_modules[uid]["policy_optimizer"] = self.policy_optimizer[uid]
                self.checkpoint_modules[uid]["value_optimizer"] = self.value_optimizer[uid]

            # set up preprocessors
            if self._state_preprocessor[uid] is not None:
                self._state_preprocessor[uid] = self._state_preprocessor[uid](**self._state_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor

            if self._value_preprocessor[uid] is not None:
                self._value_preprocessor[uid] = self._value_preprocessor[uid](**self._value_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]
            else:
                self._value_preprocessor[uid] = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memories
        for uid in self.possible_agents:
            self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=jnp.float32)
            self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=jnp.float32)
            self.memories[uid].create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memories[uid].create_tensor(name="terminated", size=1, dtype=jnp.int8)
            self.memories[uid].create_tensor(name="log_prob", size=1, dtype=jnp.float32)
            self.memories[uid].create_tensor(name="values", size=1, dtype=jnp.float32)
            self.memories[uid].create_tensor(name="returns", size=1, dtype=jnp.float32)
            self.memories[uid].create_tensor(name="advantages", size=1, dtype=jnp.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = []
        self._current_next_states = []

        # set up models for just-in-time compilation with XLA
        for uid in self.possible_agents:
            self.policies[uid].apply = jax.jit(self.policies[uid].apply, static_argnums=2)
            if self.values[uid] is not None:
                self.values[uid].apply = jax.jit(self.values[uid].apply, static_argnums=2)

    def act(self, states: Mapping[str, Union[np.ndarray, jax.Array]], timestep: int, timesteps: int) -> Union[np.ndarray, jax.Array]:
        """Process the environment's states to make a decision (actions) using the main policies

        :param states: Environment's states
        :type states: dictionary of np.ndarray or jax.Array
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: np.ndarray or jax.Array
        """
        # # sample random actions
        # # TODO: fix for stochasticity, rnn and log_prob
        # if timestep < self._random_timesteps:
        #     return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        data = [self.policies[uid].act({"states": self._state_preprocessor[uid](states[uid])}, role="policy") for uid in self.possible_agents]

        actions = {uid: d[0] for uid, d in zip(self.possible_agents, data)}
        log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, data)}
        outputs = {uid: d[2] for uid, d in zip(self.possible_agents, data)}

        if not self._jax:  # numpy backend
            actions = {jax.device_get(_actions) for _actions in actions}
            log_prob = {jax.device_get(_log_prob) for _log_prob in log_prob}

        self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(self,
                          states: Mapping[str, Union[np.ndarray, jax.Array]],
                          actions: Mapping[str, Union[np.ndarray, jax.Array]],
                          rewards: Mapping[str, Union[np.ndarray, jax.Array]],
                          next_states: Mapping[str, Union[np.ndarray, jax.Array]],
                          terminated: Mapping[str, Union[np.ndarray, jax.Array]],
                          truncated: Mapping[str, Union[np.ndarray, jax.Array]],
                          infos: Mapping[str, Any],
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: dictionary of np.ndarray or jax.Array
        :param actions: Actions taken by the agent
        :type actions: dictionary of np.ndarray or jax.Array
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: dictionary of np.ndarray or jax.Array
        :param next_states: Next observations/states of the environment
        :type next_states: dictionary of np.ndarray or jax.Array
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: dictionary of np.ndarray or jax.Array
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: dictionary of np.ndarray or jax.Array
        :param infos: Additional information about the environment
        :type infos: dictionary of any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memories:
            self._current_next_states = next_states

            for uid in self.possible_agents:
                # reward shaping
                if self._rewards_shaper is not None:
                    rewards[uid] = self._rewards_shaper(rewards[uid], timestep, timesteps)

                # compute values
                values, _, _ = self.values[uid].act({"states": self._state_preprocessor[uid](states[uid])}, role="value")
                if not self._jax:  # numpy backend
                    values = jax.device_get(values)
                values = self._value_preprocessor[uid](values, inverse=True)

                # time-limit (truncation) boostrapping
                if self._time_limit_bootstrap[uid]:
                    rewards[uid] += self._discount_factor[uid] * values * truncated[uid]

                # storage transition in memory
                self.memories[uid].add_samples(states=states[uid], actions=actions[uid], rewards=rewards[uid], next_states=next_states[uid],
                                               terminated=terminated[uid], truncated=truncated[uid], log_prob=self._current_log_prob[uid], values=values)

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
        for uid in self.possible_agents:
            policy = self.policies[uid]
            value = self.values[uid]
            memory = self.memories[uid]

            # compute returns and advantages
            value.training = False
            last_values, _, _ = value.act({"states": self._state_preprocessor[uid](self._current_next_states[uid])}, role="value")  # TODO: .float()
            value.training = True
            if not self._jax:  # numpy backend
                last_values = jax.device_get(last_values)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

            values = memory.get_tensor_by_name("values")
            if self._jax:
                returns, advantages = _compute_gae(rewards=memory.get_tensor_by_name("rewards"),
                                                   dones=memory.get_tensor_by_name("terminated"),
                                                   values=values,
                                                   next_values=last_values,
                                                   discount_factor=self._discount_factor[uid],
                                                   lambda_coefficient=self._lambda[uid])
            else:
                returns, advantages = compute_gae(rewards=memory.get_tensor_by_name("rewards"),
                                                  dones=memory.get_tensor_by_name("terminated"),
                                                  values=values,
                                                  next_values=last_values,
                                                  discount_factor=self._discount_factor[uid],
                                                  lambda_coefficient=self._lambda[uid])

            memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
            memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
            memory.set_tensor_by_name("advantages", advantages)

            # sample mini-batches from memory
            sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches[uid])

            cumulative_policy_loss = 0
            cumulative_entropy_loss = 0
            cumulative_value_loss = 0

            # learning epochs
            for epoch in range(self._learning_epochs[uid]):
                kl_divergences = []

                # mini-batches loop
                for sampled_states, sampled_actions, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages in sampled_batches:

                    sampled_states = self._state_preprocessor[uid](sampled_states, train=not epoch)

                    # compute policy loss
                    grad, policy_loss, entropy_loss, kl_divergence, stddev = _update_policy(policy.act,
                                                                                            policy.state_dict,
                                                                                            sampled_states,
                                                                                            sampled_actions,
                                                                                            sampled_log_prob,
                                                                                            sampled_advantages,
                                                                                            self._ratio_clip[uid],
                                                                                            policy.get_entropy,
                                                                                            self._entropy_loss_scale[uid])

                    kl_divergences.append(kl_divergence.item())

                    # early stopping with KL divergence
                    if self._kl_threshold[uid] and kl_divergence > self._kl_threshold[uid]:
                        break

                    # optimization step (policy)
                    self.policy_optimizer[uid] = self.policy_optimizer[uid].step(grad, policy, self.schedulers[uid]._lr if self.schedulers[uid] else None)

                    # compute value loss
                    grad, value_loss = _update_value(value.act,
                                                    value.state_dict,
                                                    sampled_states,
                                                    sampled_values,
                                                    sampled_returns,
                                                    self._value_loss_scale[uid],
                                                    self._clip_predicted_values[uid],
                                                    self._value_clip[uid])

                    # optimization step (value)
                    self.value_optimizer[uid] = self.value_optimizer[uid].step(grad, value, self.schedulers[uid]._lr if self.schedulers[uid] else None)

                    # update cumulative losses
                    cumulative_policy_loss += policy_loss.item()
                    cumulative_value_loss += value_loss.item()
                    if self._entropy_loss_scale[uid]:
                        cumulative_entropy_loss += entropy_loss.item()

                # update learning rate
                if self._learning_rate_scheduler[uid]:
                    if isinstance(self.schedulers[uid], KLAdaptiveLR):
                        self.schedulers[uid].step(np.mean(kl_divergences))

            # record data
            self.track_data(f"Loss / Policy loss ({uid})", cumulative_policy_loss / (self._learning_epochs[uid] * self._mini_batches[uid]))
            self.track_data(f"Loss / Value loss ({uid})", cumulative_value_loss / (self._learning_epochs[uid] * self._mini_batches[uid]))
            if self._entropy_loss_scale:
                self.track_data(f"Loss / Entropy loss ({uid})", cumulative_entropy_loss / (self._learning_epochs[uid] * self._mini_batches[uid]))

            self.track_data(f"Policy / Standard deviation ({uid})", stddev.mean().item())

            if self._learning_rate_scheduler[uid]:
                self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid]._lr)
