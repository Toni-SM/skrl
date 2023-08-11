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
DDQN_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "learning_rate": 1e-3,          # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "update_interval": 1,           # agent update interval
    "target_update_interval": 10,   # target network update interval

    "exploration": {
        "initial_epsilon": 1.0,       # initial epsilon for epsilon-greedy exploration
        "final_epsilon": 0.05,        # final epsilon for epsilon-greedy exploration
        "timesteps": 1000,            # timesteps for epsilon-greedy decay
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
@functools.partial(jax.jit, static_argnames=("q_network_act"))
def _update_q_network(q_network_act,
                      q_network_state_dict,
                      next_q_values,
                      sampled_states,
                      sampled_next_states,
                      sampled_actions,
                      sampled_rewards,
                      sampled_dones,
                      discount_factor):
    # compute target values
    q_values = q_network_act({"states": sampled_next_states}, "q_network")[0]
    actions = jnp.argmax(q_values, axis=-1, keepdims=True)
    target_q_values = next_q_values[jnp.arange(q_values.shape[0]), actions.reshape(-1)].reshape(-1, 1)
    target_values = sampled_rewards + discount_factor * jnp.logical_not(sampled_dones) * target_q_values

    # compute Q-network loss
    def _q_network_loss(params):
        q_values = q_network_act({"states": sampled_states}, "q_network", params)[0]
        q_values = q_values[jnp.arange(q_values.shape[0]), sampled_actions.reshape(-1)]
        return ((q_values - target_values.reshape(-1)) ** 2).mean()

    q_network_loss, grad = jax.value_and_grad(_q_network_loss, has_aux=False)(q_network_state_dict.params)

    return grad, q_network_loss, target_values


class DDQN(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, jax.Device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Double Deep Q-Network (DDQN)

        https://ojs.aaai.org/index.php/AAAI/article/view/10295

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
        # _cfg = copy.deepcopy(DDQN_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = DDQN_DEFAULT_CONFIG
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        self.q_network = self.models.get("q_network", None)
        self.target_q_network = self.models.get("target_q_network", None)

        # checkpoint models
        self.checkpoint_modules["q_network"] = self.q_network
        self.checkpoint_modules["target_q_network"] = self.target_q_network

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._update_interval = self.cfg["update_interval"]
        self._target_update_interval = self.cfg["target_update_interval"]

        self._exploration_initial_epsilon = self.cfg["exploration"]["initial_epsilon"]
        self._exploration_final_epsilon = self.cfg["exploration"]["final_epsilon"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizer and learning rate scheduler
        if self.q_network is not None:
            self.optimizer = Adam(model=self.q_network, lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up target networks
        if self.target_q_network is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_q_network.freeze_parameters(True)

            # update target networks (hard update)
            self.target_q_network.update_parameters(self.q_network, polyak=1)

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

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=jnp.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=jnp.int32)
            self.memory.create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=jnp.int8)

        self.tensors_names = ["states", "actions", "rewards", "next_states", "terminated"]

        # set up models for just-in-time compilation with XLA
        self.q_network.apply = jax.jit(self.q_network.apply, static_argnums=2)
        if self.target_q_network is not None:
            self.target_q_network.apply = jax.jit(self.target_q_network.apply, static_argnums=2)

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
        states = self._state_preprocessor(states)

        if not self._exploration_timesteps:
            q_values, _, outputs = self.q_network.act({"states": states}, role="q_network")
            actions = jnp.argmax(q_values, axis=1, keepdims=True)
            if not self._jax:  # numpy backend
                actions = jax.device_get(actions)
            return actions, None, outputs

        # sample random actions
        actions, _, outputs = self.q_network.random_act({"states": states}, role="q_network")
        if timestep < self._random_timesteps:
            raise NotImplementedError
            # if not self._jax:  # numpy backend
            #     actions = jax.device_get(actions)
            return actions, None, outputs

        # sample actions with epsilon-greedy policy
        epsilon = self._exploration_final_epsilon + (self._exploration_initial_epsilon - self._exploration_final_epsilon) \
                * np.exp(-1.0 * timestep / self._exploration_timesteps)

        indexes = (np.random.random(states.shape[0]) >= epsilon).nonzero()[0]
        if indexes.size:
            q_values, _, outputs = self.q_network.act({"states": states[indexes]}, role="q_network")
            if self._jax:
                raise NotImplementedError
                actions[indexes] = jnp.argmax(q_values, axis=1, keepdims=True)
            else:
                q_values = jax.device_get(q_values)
                actions[indexes] = np.argmax(q_values, axis=1, keepdims=True)

        # record epsilon
        self.track_data("Exploration / Exploration epsilon", epsilon)

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
        if timestep >= self._learning_starts and not timestep % self._update_interval:
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
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states, train=True)
            sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

            # compute target values
            next_q_values, _, _ = self.target_q_network.act({"states": sampled_next_states}, role="target_q_network")

            grad, q_network_loss, target_values = _update_q_network(self.q_network.act,
                                                                    self.q_network.state_dict,
                                                                    next_q_values,
                                                                    sampled_states,
                                                                    sampled_next_states,
                                                                    sampled_actions,
                                                                    sampled_rewards,
                                                                    sampled_dones,
                                                                    self._discount_factor)

            # optimization step (Q-network)
            self.optimizer = self.optimizer.step(grad, self.q_network)

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", target_values.max().item())
            self.track_data("Target / Target (min)", target_values.min().item())
            self.track_data("Target / Target (mean)", target_values.mean().item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Learning rate", self.scheduler._lr)
