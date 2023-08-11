from typing import Any, Dict, Optional, Tuple, Union

import copy
import functools
import gym
import gymnasium

import flax
import jax
import jax.numpy as jnp
import numpy as np

from skrl.agents.jax import Agent
from skrl.memories.jax import Memory
from skrl.models.jax import Model
from skrl.resources.optimizers.jax import Adam


# [start-config-dict-jax]
SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

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
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-jax]


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@functools.partial(jax.jit, static_argnames=("critic_1_act", "critic_2_act"))
def _update_critic(critic_1_act,
                   critic_1_state_dict,
                   critic_2_act,
                   critic_2_state_dict,
                   target_q1_values: jax.Array,
                   target_q2_values: jax.Array,
                   entropy_coefficient,
                   next_log_prob,
                   sampled_states: Union[np.ndarray, jax.Array],
                   sampled_actions: Union[np.ndarray, jax.Array],
                   sampled_rewards: Union[np.ndarray, jax.Array],
                   sampled_dones: Union[np.ndarray, jax.Array],
                   discount_factor: float):
    # compute target values
    target_q_values = jnp.minimum(target_q1_values, target_q2_values) - entropy_coefficient * next_log_prob
    target_values = sampled_rewards + discount_factor * jnp.logical_not(sampled_dones) * target_q_values

    # compute critic loss
    def _critic_loss(params, critic_act, role):
        critic_values, _, _ = critic_act({"states": sampled_states, "taken_actions": sampled_actions}, role, params)
        critic_loss = ((critic_values - target_values) ** 2).mean()
        return critic_loss, critic_values

    (critic_1_loss, critic_1_values), grad = jax.value_and_grad(_critic_loss, has_aux=True)(critic_1_state_dict.params, critic_1_act, "critic_1")
    (critic_2_loss, critic_2_values), grad = jax.value_and_grad(_critic_loss, has_aux=True)(critic_2_state_dict.params, critic_2_act, "critic_2")

    return grad, (critic_1_loss + critic_2_loss) / 2, critic_1_values, critic_2_values, target_values

@functools.partial(jax.jit, static_argnames=("policy_act", "critic_1_act", "critic_2_act"))
def _update_policy(policy_act,
                   critic_1_act,
                   critic_2_act,
                   policy_state_dict,
                   critic_1_state_dict,
                   critic_2_state_dict,
                   entropy_coefficient,
                   sampled_states):
    # compute policy (actor) loss
    def _policy_loss(policy_params, critic_1_params, critic_2_params):
        actions, log_prob, _ = policy_act({"states": sampled_states}, "policy", policy_params)
        critic_1_values, _, _ = critic_1_act({"states": sampled_states, "taken_actions": actions}, "critic_1", critic_1_params)
        critic_2_values, _, _ = critic_2_act({"states": sampled_states, "taken_actions": actions}, "critic_2", critic_2_params)
        return (entropy_coefficient * log_prob - jnp.minimum(critic_1_values, critic_2_values)).mean(), log_prob

    (policy_loss, log_prob), grad = jax.value_and_grad(_policy_loss, has_aux=True)(policy_state_dict.params, critic_1_state_dict.params, critic_2_state_dict.params)

    return grad, policy_loss, log_prob

@jax.jit
def _update_entropy(log_entropy_coefficient_state_dict, target_entropy, log_prob):
    # compute entropy loss
    def _entropy_loss(params):
        return -(params["params"] * (log_prob + target_entropy)).mean()

    entropy_loss, grad = jax.value_and_grad(_entropy_loss, has_aux=False)(log_entropy_coefficient_state_dict.params)

    return grad, entropy_loss


class SAC(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, jax.Device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290

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
        # _cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = SAC_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

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

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

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

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gym.spaces.Box) or issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gym.spaces.Discrete) or issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            class _LogEntropyCoefficient:
                def __init__(self, entropy_coefficient: float) -> None:
                    class StateDict(flax.struct.PyTreeNode):
                        params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)
                    self.state_dict = StateDict(flax.core.FrozenDict({"params": jnp.array([jnp.log(entropy_coefficient)])}))

                @property
                def value(self):
                    return self.state_dict.params["params"]

            self.log_entropy_coefficient = _LogEntropyCoefficient(self._entropy_coefficient)
            self.entropy_optimizer = Adam(model=self.log_entropy_coefficient, lr=self._entropy_learning_rate)

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = Adam(model=self.policy, lr=self._actor_learning_rate, grad_norm_clip=self._grad_norm_clip)
            self.critic_1_optimizer = Adam(model=self.critic_1, lr=self._critic_learning_rate, grad_norm_clip=self._grad_norm_clip)
            self.critic_2_optimizer = Adam(model=self.critic_2, lr=self._critic_learning_rate, grad_norm_clip=self._grad_norm_clip)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.critic_1_scheduler = self._learning_rate_scheduler(self.critic_1_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.critic_2_scheduler = self._learning_rate_scheduler(self.critic_2_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

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
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=jnp.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=jnp.int8)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated"]

        # set up models for just-in-time compilation with XLA
        self.policy.apply = jax.jit(self.policy.apply, static_argnums=2)
        if self.critic_1 is not None and self.critic_2 is not None:
            self.critic_1.apply = jax.jit(self.critic_1.apply, static_argnums=2)
            self.critic_2.apply = jax.jit(self.critic_2.apply, static_argnums=2)
        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            self.target_critic_1.apply = jax.jit(self.target_critic_1.apply, static_argnums=2)
            self.target_critic_2.apply = jax.jit(self.target_critic_2.apply, static_argnums=2)

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
        actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)

        return actions, None, outputs

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
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated)

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
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states, train=True)
            sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

            next_actions, next_log_prob, _ = self.policy.act({"states": sampled_next_states}, role="policy")

            # compute target values
            target_q1_values, _, _ = self.target_critic_1.act({"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_1")
            target_q2_values, _, _ = self.target_critic_2.act({"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_2")

            # compute critic loss
            grad, critic_loss, critic_1_values, critic_2_values, target_values = _update_critic(self.critic_1.act,
                                                                                                self.critic_1.state_dict,
                                                                                                self.critic_2.act,
                                                                                                self.critic_2.state_dict,
                                                                                                target_q1_values,
                                                                                                target_q2_values,
                                                                                                self._entropy_coefficient,
                                                                                                next_log_prob,
                                                                                                sampled_states,
                                                                                                sampled_actions,
                                                                                                sampled_rewards,
                                                                                                sampled_dones,
                                                                                                self._discount_factor)

            # optimization step (critic)
            self.critic_1_optimizer = self.critic_1_optimizer.step(grad, self.critic_1)
            self.critic_2_optimizer = self.critic_2_optimizer.step(grad, self.critic_2)

            # compute policy (actor) loss
            grad, policy_loss, log_prob = _update_policy(self.policy.act,
                                                         self.critic_1.act,
                                                         self.critic_2.act,
                                                         self.policy.state_dict,
                                                         self.critic_1.state_dict,
                                                         self.critic_2.state_dict,
                                                         self._entropy_coefficient,
                                                         sampled_states)

            # optimization step (policy)
            self.policy_optimizer = self.policy_optimizer.step(grad, self.policy)

            # entropy learning
            if self._learn_entropy:
                # compute entropy loss
                grad, entropy_loss = _update_entropy(self.log_entropy_coefficient.state_dict,
                                                     self._target_entropy,
                                                     log_prob)

                # optimization step (entropy)
                self.entropy_optimizer = self.entropy_optimizer.step(grad, self.log_entropy_coefficient)

                # compute entropy coefficient
                self._entropy_coefficient = jnp.exp(self.log_entropy_coefficient.value)

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_1_scheduler.step()
                self.critic_2_scheduler.step()

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
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                    self.track_data("Learning / Critic 1 learning rate", self.critic_1_scheduler.get_last_lr()[0])
                    self.track_data("Learning / Critic 2 learning rate", self.critic_2_scheduler.get_last_lr()[0])
