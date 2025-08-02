from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium

import numpy as np
import warp as wp

from skrl.agents.warp import Agent
from skrl.memories.warp import Memory
from skrl.models.warp import Model
from skrl.resources.optimizers.warp import Adam


# fmt: off
# [start-config-dict-warp]
DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "observation_preprocessor": None,       # observation preprocessor class (see skrl.resources.preprocessors)
    "observation_preprocessor_kwargs": {},  # observation preprocessor's kwargs (e.g. {"size": env.observation_space})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.state_space})

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
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-warp]
# fmt: on


def enable_grad(obj, *, enabled: bool):
    if obj is None:
        return
    if isinstance(obj, wp.array):
        obj.requires_grad = enabled
    elif isinstance(obj, dict):
        for item in obj.values():
            enable_grad(item, enabled=enabled)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


@wp.kernel(enable_backward=False)
def _apply_exploration_noise(
    actions: wp.array2d(dtype=float),
    noises: wp.array2d(dtype=float),
    clip_actions_min: wp.array1d(dtype=float),
    clip_actions_max: wp.array1d(dtype=float),
    scale: float,
):
    i, j = wp.tid()
    noises[i, j] = noises[i, j] * scale  # update in-place for logging
    actions[i, j] = wp.clamp(actions[i, j] + noises[i, j], clip_actions_min[j], clip_actions_max[j])


@wp.kernel
def _critic_loss(
    values: wp.array2d(dtype=float),
    target_values: wp.array2d(dtype=float),
    sampled_rewards: wp.array2d(dtype=float),
    sampled_terminated: wp.array2d(dtype=wp.int8),
    sampled_truncated: wp.array2d(dtype=wp.int8),
    discount_factor: float,
    n: float,
    loss: wp.array(dtype=float),
):
    i, j = wp.tid()
    # compute target values
    target_values[i, j] = (
        sampled_rewards[i, j]
        + discount_factor
        * wp.float(wp.unot(wp.add(sampled_terminated[i, j], sampled_truncated[i, j])))
        * target_values[i, j]
    )
    # MSE loss
    wp.atomic_add(loss, 0, wp.pow(values[i, j] - target_values[i, j], 2.0) / n)


@wp.kernel
def _policy_loss(
    values: wp.array2d(dtype=float),
    n: float,
    loss: wp.array(dtype=float),
):
    i, j = wp.tid()
    wp.atomic_add(loss, 0, -values[i, j] / n)


class DDPG(Agent):
    def __init__(
        self,
        *,
        models: Optional[Mapping[str, Model]] = None,
        memory: Optional[Memory] = None,
        observation_space: Optional[gymnasium.Space] = None,
        state_space: Optional[gymnasium.Space] = None,
        action_space: Optional[gymnasium.Space] = None,
        device: Optional[Union[str, wp.context.Device]] = None,
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
        # _cfg = copy.deepcopy(DDPG_DEFAULT_CONFIG)  # TODO: ValueError: ctypes objects containing pointers cannot be pickled
        _cfg = DDPG_DEFAULT_CONFIG
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
        self.target_policy = self.models.get("target_policy", None)
        self.critic = self.models.get("critic", None)
        self.target_critic = self.models.get("target_critic", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["target_critic"] = self.target_critic

        if self.target_policy is not None and self.target_critic is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic.freeze_parameters(True)

            # update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic.update_parameters(self.critic, polyak=1)

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

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic is not None:
            self.policy_optimizer = Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self._critic_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            # self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            # self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

            # training variables
            self._policy_loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True)
            self._critic_loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True)

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

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=wp.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=wp.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=wp.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=wp.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=wp.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=wp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=wp.int8)
            self.memory.create_tensor(name="truncated", size=1, dtype=wp.int8)

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
            self.clip_actions_min = wp.array(self.action_space.low.flatten(), dtype=wp.float32, device=self.device)
            self.clip_actions_max = wp.array(self.action_space.high.flatten(), dtype=wp.float32, device=self.device)

    def act(
        self, observations: wp.array, states: Union[wp.array, None], *, timestep: int, timesteps: int
    ) -> Tuple[wp.array, Mapping[str, Union[wp.array, Any]]]:
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
        if timestep < self._random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample deterministic actions
        actions, outputs = self.policy.act(inputs, role="policy")

        # add exploration noise
        if self._exploration_noise is not None:
            # sample noises
            noises = self._exploration_noise.sample(actions.shape)

            # define exploration timesteps
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps

            # apply exploration noise
            if timestep <= self._exploration_timesteps:
                scale = (1 - timestep / self._exploration_timesteps) * (
                    self._exploration_initial_scale - self._exploration_final_scale
                ) + self._exploration_final_scale

                # modify actions
                wp.launch(
                    _apply_exploration_noise,
                    dim=actions.shape,
                    inputs=[actions, noises, self.clip_actions_min, self.clip_actions_max, scale],
                    device=self.device,
                )

                # record noises
                self.track_data("Exploration / Exploration noise (max)", np.max(noises.numpy()).item())
                self.track_data("Exploration / Exploration noise (min)", np.min(noises.numpy()).item())
                self.track_data("Exploration / Exploration noise (mean)", np.mean(noises.numpy()).item())

            else:
                # record noises
                self.track_data("Exploration / Exploration noise (max)", 0)
                self.track_data("Exploration / Exploration noise (min)", 0)
                self.track_data("Exploration / Exploration noise (mean)", 0)

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: wp.array,
        states: wp.array,
        actions: wp.array,
        rewards: wp.array,
        next_observations: wp.array,
        next_states: wp.array,
        terminated: wp.array,
        truncated: wp.array,
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
            self.enable_training_mode(True)
            self.update(timestep=timestep, timesteps=timesteps)
            self.enable_training_mode(False)

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

            # compute target values
            next_actions, _ = self.target_policy.act(next_inputs, role="target_policy")
            target_q_values, _ = self.target_critic.act(
                {**next_inputs, "taken_actions": next_actions}, role="target_critic"
            )

            # compute critic loss
            critic_inputs = {**inputs, "taken_actions": sampled_actions}
            enable_grad(critic_inputs, enabled=True)
            self._critic_loss.zero_()
            with wp.Tape() as tape:
                critic_values, _ = self.critic.act(critic_inputs, role="critic")
                wp.launch(
                    _critic_loss,
                    dim=critic_values.shape,
                    inputs=[
                        critic_values,
                        target_q_values,
                        sampled_rewards,
                        sampled_terminated,
                        sampled_truncated,
                        self._discount_factor,
                        np.prod(critic_values.shape),
                        self._critic_loss,
                    ],
                    device=self.device,
                )

            # optimization step (critic)
            tape.backward(self._critic_loss)
            self.critic_optimizer.step()
            tape.zero()

            # compute policy (actor) loss
            enable_grad(inputs, enabled=True)
            self._policy_loss.zero_()
            with wp.Tape() as tape:
                actions, _ = self.policy.act(inputs, role="policy")
                critic_values, _ = self.critic.act({**inputs, "taken_actions": actions}, role="critic")
                wp.launch(
                    _policy_loss,
                    dim=critic_values.shape,
                    inputs=[
                        critic_values,
                        np.prod(critic_values.shape),
                        self._policy_loss,
                    ],
                    device=self.device,
                )

            # optimization step (policy)
            tape.backward(self._policy_loss)
            self.policy_optimizer.step()
            tape.zero()

            # update target networks
            self.target_policy.update_parameters(self.policy, polyak=self._polyak)
            self.target_critic.update_parameters(self.critic, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            self.track_data("Loss / Policy loss", self._policy_loss.numpy().item())
            self.track_data("Loss / Critic loss", self._critic_loss.numpy().item())

            self.track_data("Q-network / Q1 (max)", critic_values.numpy().max().item())
            self.track_data("Q-network / Q1 (min)", critic_values.numpy().min().item())
            self.track_data("Q-network / Q1 (mean)", critic_values.numpy().mean().item())

            # # self.track_data("Target / Target (max)", target_values.max().item())
            # # self.track_data("Target / Target (min)", target_values.min().item())
            # # self.track_data("Target / Target (mean)", target_values.mean().item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
