from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium

import numpy as np
import warp as wp
import warp.optim as optim

from skrl.agents.warp import Agent
from skrl.memories.warp import Memory
from skrl.models.warp import Model


# fmt: off
# [start-config-dict-warp]
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "observation_preprocessor": None,       # observation preprocessor class (see skrl.resources.preprocessors)
    "observation_preprocessor_kwargs": {},  # observation preprocessor's kwargs (e.g. {"size": env.observation_space})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.state_space})
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
def _time_limit_bootstrap(
    rewards: wp.array2d(dtype=float),
    values: wp.array2d(dtype=float),
    truncated: wp.array2d(dtype=wp.int8),
    discount_factor: float,
):
    i = wp.tid()
    rewards[i, 0] = rewards[i, 0] + discount_factor * values[i, 0] * wp.float32(truncated[i, 0])


@wp.kernel
def _compute_gae():
    pass


@wp.kernel
def _entropy_loss(
    entropy: wp.array1d(dtype=float),
    entropy_loss_scale: float,
    n: float,
    loss: wp.array(dtype=float),
):
    wp.atomic_add(loss, 0, -entropy_loss_scale * entropy[wp.tid()] / n)


@wp.kernel
def _policy_value_loss(
    sampled_log_prob: wp.array2d(dtype=float),
    sampled_values: wp.array2d(dtype=float),
    sampled_returns: wp.array2d(dtype=float),
    sampled_advantages: wp.array2d(dtype=float),
    log_prob: wp.array2d(dtype=float),
    predicted_values: wp.array2d(dtype=float),
    ratio_clip: float,
    value_clip: float,
    value_loss_scale: float,
    n: float,
    policy_loss: wp.array1d(dtype=float),
    value_loss: wp.array1d(dtype=float),
    kl_divergence: wp.array1d(dtype=float),
):
    i = wp.tid()
    # compute approximate KL divergence
    wp.atomic_add(
        kl_divergence,
        0,
        ((wp.exp(log_prob[i, 0] - sampled_log_prob[i, 0]) - 1.0) - (log_prob[i, 0] - sampled_log_prob[i, 0])) / n,
    )
    # compute policy loss
    wp.atomic_add(
        policy_loss,
        0,
        -wp.min(
            # surrogate
            sampled_advantages[i, 0] * wp.exp(log_prob[i, 0] - sampled_log_prob[i, 0]),
            # surrogate (clipped)
            sampled_advantages[i, 0]
            * wp.clamp(wp.exp(log_prob[i, 0] - sampled_log_prob[i, 0]), 1.0 - ratio_clip, 1.0 + ratio_clip),
        )
        / n,
    )
    # compute value loss
    if value_clip:
        wp.atomic_add(
            value_loss,
            0,
            value_loss_scale
            * wp.pow(
                sampled_returns[i, 0]
                - (
                    sampled_values[i, 0]
                    + wp.clamp(predicted_values[i, 0] - sampled_values[i, 0], -value_clip, value_clip)
                ),
                2.0,
            )
            / n,
        )
    else:
        wp.atomic_add(value_loss, 0, value_loss_scale * wp.pow(sampled_returns[i, 0] - predicted_values[i, 0], 2.0) / n)


@wp.kernel
def _loss(
    policy_loss: wp.array1d(dtype=float),
    value_loss: wp.array1d(dtype=float),
    entropy_loss: wp.array1d(dtype=float),
    loss: wp.array(dtype=float),
):
    wp.atomic_add(loss, 0, policy_loss[0] + value_loss[0] + entropy_loss[0])


class PPO(Agent):
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
        """Proximal Policy Optimization (PPO).

        https://arxiv.org/abs/1707.06347

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        # _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)  # TODO: ValueError: ctypes objects containing pointers cannot be pickled
        _cfg = PPO_DEFAULT_CONFIG
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
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._observation_preprocessor = self.cfg["observation_preprocessor"]
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
            if self.policy is self.value:
                self.optimizer = optim.Adam(
                    params=[param.flatten() for param in self.policy.parameters()], lr=self._learning_rate
                )
            else:
                self.optimizer = optim.Adam(
                    params=(
                        [param.flatten() for param in self.policy.parameters()]
                        + [param.flatten() for param in self.value.parameters()]
                    ),
                    lr=self._learning_rate,
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            # self.checkpoint_modules["optimizer"] = self.optimizer

            # training variables
            self._loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True)
            self._policy_loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True)
            self._value_loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True)
            self._entropy_loss = wp.zeros((1,), dtype=wp.float32, requires_grad=True)
            self._kl_divergence = wp.zeros((1,), dtype=wp.float32)
            if self.policy is self.value:
                self._optimizer_grads = [param.grad.flatten() for param in self.policy.parameters()]
            else:
                self._optimizer_grads = [param.grad.flatten() for param in self.policy.parameters()] + [
                    param.grad.flatten() for param in self.value.parameters()
                ]

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
        # - values
        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=wp.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=wp.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=wp.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=wp.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=wp.int8)
            self.memory.create_tensor(name="truncated", size=1, dtype=wp.int8)
            self.memory.create_tensor(name="log_prob", size=1, dtype=wp.float32)
            self.memory.create_tensor(name="values", size=1, dtype=wp.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=wp.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=wp.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None

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
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        actions, outputs = self.policy.act(inputs, role="policy")
        self._current_log_prob = outputs["log_prob"]

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
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            inputs = {
                "observations": self._observation_preprocessor(observations),
                "states": self._state_preprocessor(states),
            }
            values, _ = self.value.act(inputs, role="value")
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                wp.launch(
                    _time_limit_bootstrap,
                    dim=rewards.shape[0],
                    inputs=[rewards, values, truncated, self._discount_factor],
                    device=self.device,
                )

            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
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
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
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
        # compute returns and advantages
        inputs = {
            "observations": self._observation_preprocessor(self._current_next_observations),
            "states": self._state_preprocessor(self._current_next_states),
        }
        self.value.enable_training_mode(False)
        last_values, _ = self.value.act(inputs, role="value")
        self.value.enable_training_mode(True)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")  # (1024, 4, 1)
        returns = wp.zeros(shape=values.shape, dtype=wp.float32, device=self.device)
        advantages = wp.zeros(shape=values.shape, dtype=wp.float32, device=self.device)

        # # returns, advantages = compute_gae(
        # #     rewards=self.memory.get_tensor_by_name("rewards"),
        # #     dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
        # #     values=values,
        # #     next_values=last_values,
        # #     discount_factor=self._discount_factor,
        # #     lambda_coefficient=self._lambda,
        # # )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                    "states": self._state_preprocessor(sampled_states, train=not epoch),
                }

                # compute loss
                enable_grad(inputs, enabled=True)
                sampled_actions.requires_grad = True
                self._loss.zero_()
                self._policy_loss.zero_()
                self._value_loss.zero_()
                self._entropy_loss.zero_()
                with wp.Tape() as tape:
                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    stddev = outputs["stddev"]
                    predicted_values, _ = self.value.act(inputs, role="value")
                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy = self.policy.get_entropy(stddev, role="policy")
                        wp.launch(
                            _entropy_loss,
                            dim=entropy.shape[0],
                            inputs=[entropy, self._entropy_loss_scale, entropy.shape[0], self._entropy_loss],
                            device=self.device,
                        )
                    # compute policy/value loss
                    wp.launch(
                        _policy_value_loss,
                        dim=sampled_log_prob.shape[0],
                        inputs=[
                            sampled_log_prob,
                            sampled_values,
                            sampled_returns,
                            sampled_advantages,
                            outputs["log_prob"],
                            predicted_values,
                            self._ratio_clip,
                            self._value_clip if self._clip_predicted_values else 0.0,
                            self._value_loss_scale,
                            float(sampled_log_prob.shape[0]),
                            self._policy_loss,
                            self._value_loss,
                            self._kl_divergence,
                        ],
                        device=self.device,
                    )
                    # compute loss
                    wp.launch(
                        _loss,
                        dim=1,
                        inputs=[self._policy_loss, self._value_loss, self._entropy_loss, self._loss],
                        device=self.device,
                    )

                # early stopping with KL divergence
                kl_divergence = self._kl_divergence.numpy().item()
                kl_divergences.append(kl_divergence)
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # optimization step
                tape.backward(self._loss)
                self.optimizer.step(self._optimizer_grads)
                tape.zero()

                # update cumulative losses
                cumulative_policy_loss += self._policy_loss.numpy().item()
                cumulative_value_loss += self._value_loss.numpy().item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += self._entropy_loss.numpy().item()

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", stddev.numpy().mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
