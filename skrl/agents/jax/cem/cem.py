from typing import Any, Mapping, Optional, Tuple, Union

import copy
import functools
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np
import optax

from skrl import logger
from skrl.agents.jax import Agent
from skrl.memories.jax import Memory
from skrl.models.jax import Model
from skrl.resources.optimizers.jax import Adam
from skrl.utils import ScopedTimer


# fmt: off
# [start-config-dict-jax]
CEM_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "percentile": 0.70,             # percentile to compute the reward bound [0, 1]

    "discount_factor": 0.99,        # discount factor (gamma)

    "learning_rate": 1e-2,          # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler function (see optax.schedules)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "observation_preprocessor": None,       # observation preprocessor class (see skrl.resources.preprocessors)
    "observation_preprocessor_kwargs": {},  # observation preprocessor's kwargs (e.g. {"size": env.observation_space})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.state_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

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


@functools.partial(jax.jit, static_argnames=("policy_act", "n"))
def _update_policy(policy_act, policy_state_dict, inputs, elite_actions, n):
    # compute policy loss
    def _policy_loss(params):
        # compute scores for the elite observations/states
        _, outputs = policy_act(inputs, role="policy", params=params)
        scores = outputs["net_output"]

        # HACK: return optax.softmax_cross_entropy_with_integer_labels(scores, elite_actions).mean()
        labels = jax.nn.one_hot(elite_actions, n)
        return optax.softmax_cross_entropy(scores, labels).mean()

    policy_loss, grad = jax.value_and_grad(_policy_loss, has_aux=False)(policy_state_dict.params)

    return grad, policy_loss


class CEM(Agent):
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
        """Cross-Entropy Method (CEM).

        https://ieeexplore.ieee.org/abstract/document/6796865/

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        # _cfg = copy.deepcopy(CEM_DEFAULT_CONFIG)  # TODO: TypeError: cannot pickle 'jax.Device' object
        _cfg = CEM_DEFAULT_CONFIG
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

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy

        # configuration
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._percentile = self.cfg["percentile"]
        self._discount_factor = self.cfg["discount_factor"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._observation_preprocessor = self.cfg["observation_preprocessor"]
        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._episode_tracking = []

        # set up optimizer and learning rate scheduler
        if self.policy is not None:
            # scheduler
            if self._learning_rate_scheduler:
                self.scheduler = self._learning_rate_scheduler(**self.cfg["learning_rate_scheduler_kwargs"])
            # optimizer
            with jax.default_device(self.device):
                self.optimizer = Adam(
                    model=self.policy, lr=self._learning_rate, scale=not self._learning_rate_scheduler
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

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
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=jnp.int32)
            self.memory.create_tensor(name="rewards", size=1, dtype=jnp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=jnp.int8)
            self.memory.create_tensor(name="truncated", size=1, dtype=jnp.int8)

        self.tensors_names = ["observations", "states", "actions", "rewards"]

        # set up models for just-in-time compilation with XLA
        self.policy.apply = jax.jit(self.policy.apply, static_argnums=2)

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

        # track episodes internally
        if self._rollout:
            indexes = (terminated + truncated).nonzero()[0]
            if indexes.size:
                for i in indexes:
                    try:
                        self._episode_tracking[i.item()].append(self._rollout + 1)
                    except IndexError:
                        logger.warning(f"IndexError: {i.item()}")
        else:
            self._episode_tracking = [[0] for _ in range(rewards.shape[-1])]

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
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self._rollout = 0
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
        # sample all memory
        sampled_observations, sampled_states, sampled_actions, sampled_rewards = self.memory.sample_all(
            names=self.tensors_names
        )[0]

        sampled_observations = self._observation_preprocessor(sampled_observations, train=True)
        sampled_states = self._state_preprocessor(sampled_states, train=True)

        if self._jax:  # move to numpy backend
            sampled_observations = jax.device_get(sampled_observations)
            sampled_states = jax.device_get(sampled_states)
            sampled_actions = jax.device_get(sampled_actions)
            sampled_rewards = jax.device_get(sampled_rewards)

        # compute discounted return threshold
        limits = []
        returns = []
        for e in range(sampled_rewards.shape[-1]):
            for i, j in zip(self._episode_tracking[e][:-1], self._episode_tracking[e][1:]):
                limits.append([e + i, e + j])
                rewards = sampled_rewards[e + i : e + j]
                returns.append(
                    np.sum(
                        rewards
                        * self._discount_factor ** np.flip(np.arange(rewards.shape[0]), axis=-1).reshape(rewards.shape)
                    )
                )

        if not len(returns):
            logger.warning("No returns to update. Consider increasing the number of rollouts")
            return

        returns = np.array(returns)
        return_threshold = np.quantile(returns, self._percentile, axis=-1)

        # get elite observations/states and actions
        indexes = (returns >= return_threshold).nonzero()[0]
        elite_observations = np.concatenate(
            [sampled_observations[limits[i][0] : limits[i][1]] for i in indexes], axis=0
        )
        try:
            elite_states = np.concatenate([sampled_states[limits[i][0] : limits[i][1]] for i in indexes], axis=0)
        except TypeError:
            elite_states = None
        elite_actions = np.concatenate([sampled_actions[limits[i][0] : limits[i][1]] for i in indexes], axis=0).reshape(
            -1
        )

        # compute policy loss
        grad, policy_loss = _update_policy(
            self.policy.act,
            self.policy.state_dict,
            {"observations": elite_observations, "states": elite_states},
            elite_actions,
            self.action_space.n,
        )

        # optimization step (policy)
        self.optimizer = self.optimizer.step(
            grad=grad, model=self.policy, lr=self._learning_rate if self._learning_rate_scheduler else None
        )

        # update learning rate
        if self._learning_rate_scheduler:
            self._learning_rate *= self.scheduler(timestep)

        # record data
        self.track_data("Loss / Policy loss", policy_loss.item())

        self.track_data("Coefficient / Return threshold", return_threshold.item())
        self.track_data("Coefficient / Mean discounted returns", returns.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self._learning_rate)
