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


# [start-config-dict-jax]
DDPG_DEFAULT_CONFIG = {
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

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

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
@jax.jit
def _apply_exploration_noise(actions: jax.Array,
                             noises: jax.Array,
                             clip_actions_min: jax.Array,
                             clip_actions_max: jax.Array,
                             scale: float) -> jax.Array:
    noises = noises.at[:].multiply(scale)
    return jnp.clip(actions + noises, a_min=clip_actions_min, a_max=clip_actions_max), noises

@functools.partial(jax.jit, static_argnames=("critic_act"))
def _update_critic(critic_act,
                   critic_state_dict,
                   target_q_values: jax.Array,
                   sampled_states: Union[np.ndarray, jax.Array],
                   sampled_actions: Union[np.ndarray, jax.Array],
                   sampled_rewards: Union[np.ndarray, jax.Array],
                   sampled_dones: Union[np.ndarray, jax.Array],
                   discount_factor: float):
    # compute target values
    target_values = sampled_rewards + discount_factor * jnp.logical_not(sampled_dones) * target_q_values

    # compute critic loss
    def _critic_loss(params):
        critic_values, _, _ = critic_act({"states": sampled_states, "taken_actions": sampled_actions}, "critic", params)
        critic_loss = ((critic_values - target_values) ** 2).mean()
        return critic_loss, critic_values

    (critic_loss, critic_values), grad = jax.value_and_grad(_critic_loss, has_aux=True)(critic_state_dict.params)

    return grad, critic_loss, critic_values, target_values

@functools.partial(jax.jit, static_argnames=("policy_act", "critic_act"))
def _update_policy(policy_act,
                   critic_act,
                   policy_state_dict,
                   critic_state_dict,
                   sampled_states):
    # compute policy (actor) loss
    def _policy_loss(policy_params, critic_params):
        actions, _, _ = policy_act({"states": sampled_states}, "policy", policy_params)
        critic_values, _, _ = critic_act({"states": sampled_states, "taken_actions": actions}, "critic", critic_params)
        return -critic_values.mean()

    policy_loss, grad = jax.value_and_grad(_policy_loss, has_aux=False)(policy_state_dict.params, critic_state_dict.params)

    return grad, policy_loss


class DDPG(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, jax.Device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Deep Deterministic Policy Gradient (DDPG)

        https://arxiv.org/abs/1509.02971

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
        # _cfg = copy.deepcopy(DDPG_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = DDPG_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

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

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic is not None:
            self.policy_optimizer = Adam(model=self.policy, lr=self._actor_learning_rate, grad_norm_clip=self._grad_norm_clip)
            self.critic_optimizer = Adam(model=self.critic, lr=self._critic_learning_rate, grad_norm_clip=self._grad_norm_clip)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
                self.critic_scheduler = self._learning_rate_scheduler(self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up target networks
        if self.target_policy is not None and self.target_critic is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic.freeze_parameters(True)

            # update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic.update_parameters(self.critic, polyak=1)

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
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample deterministic actions
        actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        if not self._jax:  # numpy backend
            actions = jax.device_get(actions)

        # add exloration noise
        if self._exploration_noise is not None:
            # sample noises
            noises = self._exploration_noise.sample(actions.shape)

            # define exploration timesteps
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps

            # apply exploration noise
            if timestep <= self._exploration_timesteps:
                scale = (1 - timestep / self._exploration_timesteps) \
                      * (self._exploration_initial_scale - self._exploration_final_scale) \
                      + self._exploration_final_scale

                # modify actions
                if self._jax:
                    actions, noises = _apply_exploration_noise(actions, noises, self.clip_actions_min, self.clip_actions_max, scale)
                else:
                    noises *= scale
                    actions = np.clip(actions + noises, a_min=self.clip_actions_min, a_max=self.clip_actions_max)

                # record noises
                self.track_data("Exploration / Exploration noise (max)", noises.max().item())
                self.track_data("Exploration / Exploration noise (min)", noises.min().item())
                self.track_data("Exploration / Exploration noise (mean)", noises.mean().item())

            else:
                # record noises
                self.track_data("Exploration / Exploration noise (max)", 0)
                self.track_data("Exploration / Exploration noise (min)", 0)
                self.track_data("Exploration / Exploration noise (mean)", 0)

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

            # compute target values
            next_actions, _, _ = self.target_policy.act({"states": sampled_next_states}, role="target_policy")

            target_q_values, _, _ = self.target_critic.act({"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic")

            # compute critic loss
            grad, critic_loss, critic_values, target_values = _update_critic(self.critic.act,
                                                                             self.critic.state_dict,
                                                                             target_q_values,
                                                                             sampled_states,
                                                                             sampled_actions,
                                                                             sampled_rewards,
                                                                             sampled_dones,
                                                                             self._discount_factor)

            # optimization step (critic)
            self.critic_optimizer = self.critic_optimizer.step(grad, self.critic)

            # compute policy (actor) loss
            grad, policy_loss = _update_policy(self.policy.act,
                                               self.critic.act,
                                               self.policy.state_dict,
                                               self.critic.state_dict,
                                               sampled_states)

            # optimization step (policy)
            self.policy_optimizer = self.policy_optimizer.step(grad, self.policy)

            # update target networks
            self.target_policy.update_parameters(self.policy, polyak=self._polyak)
            self.target_critic.update_parameters(self.critic, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            self.track_data("Loss / Policy loss", policy_loss.item())
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-network / Q1 (max)", critic_values.max().item())
            self.track_data("Q-network / Q1 (min)", critic_values.min().item())
            self.track_data("Q-network / Q1 (mean)", critic_values.mean().item())

            self.track_data("Target / Target (max)", target_values.max().item())
            self.track_data("Target / Target (min)", target_values.min().item())
            self.track_data("Target / Target (mean)", target_values.mean().item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
