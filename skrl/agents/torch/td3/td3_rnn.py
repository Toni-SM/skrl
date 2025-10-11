from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from .td3_cfg import TD3_CFG


class TD3_RNN(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: TD3_CFG | dict = {},
    ) -> None:
        """Twin Delayed DDPG (TD3) with support for Recurrent Neural Networks (RNN, GRU, LSTM, etc.).

        https://arxiv.org/abs/1802.09477

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: TD3_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=TD3_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.target_policy = self.models.get("target_policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        # set up noise
        # - exploration noise
        if self.cfg.exploration_noise is not None:
            self._exploration_noise = self.cfg.exploration_noise(**self.cfg.exploration_noise_kwargs)
        else:
            logger.warning("agents:TD3: No exploration noise specified, training performance may be degraded")
            self._exploration_noise = None
        # - smooth regularization noise
        if self.cfg.smooth_regularization_noise is not None:
            self._smooth_regularization_noise = self.cfg.smooth_regularization_noise(
                **self.cfg.smooth_regularization_noise_kwargs
            )
        else:
            logger.warning("agents:TD3: No smooth regularization noise specified, training variance may be high")
            self._smooth_regularization_noise = None

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            # - optimizers
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=self.cfg.learning_rate[1]
            )
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
            # - learning rate schedulers
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            self.critic_scheduler = self.cfg.learning_rate_scheduler[1]
            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    self.policy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )
            if self.critic_scheduler is not None:
                self.critic_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.critic_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
                )

        # set up target networks
        if self.target_policy is not None and self.target_critic_1 is not None and self.target_critic_2 is not None:
            # - freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)
            # - update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # set up preprocessors
        # - observations
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

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

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"policy": []}
        self._rnn_initial_states = {"policy": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)

        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(
                    name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                )
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # clip noise bounds
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

        # create temporary variables needed for storage and computation
        self._update_counter = 0

    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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
        inputs.update({"rnn": self._rnn_initial_states["policy"]} if self._rnn else {})

        # sample random actions
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

        # add exploration noise
        if self._exploration_noise:
            noises = self._exploration_noise.sample(actions.shape)
            if self.cfg.exploration_scheduler:
                noises.mul_(self.cfg.exploration_scheduler(timestep, timesteps))
            actions.add_(noises)
            actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

            self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
            self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
            self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item())

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
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
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update(
                    {f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])}
                )

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
                **rnn_states,
            )

        # update RNN states
        if self._rnn:
            # reset states if the episodes have ended
            finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states

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
        if timestep >= self.cfg.learning_starts:
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
        for gradient_step in range(self.cfg.gradient_steps):

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
            ) = self.memory.sample(
                names=self._tensors_names, batch_size=self.cfg.batch_size, sequence_length=self._rnn_sequence_length
            )[
                0
            ]

            rnn_policy = {}
            if self._rnn:
                sampled_rnn = self.memory.sample_by_index(
                    names=self._rnn_tensors_names, indexes=self.memory.get_sampling_indexes()
                )[0]
                rnn_policy = {
                    "rnn": [s.transpose(0, 1) for s in sampled_rnn],
                    "terminated": sampled_terminated | sampled_truncated,
                }

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                    "states": self._state_preprocessor(sampled_states, train=True),
                    **rnn_policy,
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                    "states": self._state_preprocessor(sampled_next_states, train=True),
                    **rnn_policy,
                }

                with torch.no_grad():
                    # target policy smoothing
                    next_actions, _ = self.target_policy.act(next_inputs, role="target_policy")
                    if self._smooth_regularization_noise is not None:
                        noises = torch.clamp(
                            self._smooth_regularization_noise.sample(next_actions.shape),
                            min=-self.cfg.smooth_regularization_clip,
                            max=self.cfg.smooth_regularization_clip,
                        )
                        next_actions.add_(noises)
                        next_actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                    # compute target values
                    target_q1_values, _ = self.target_critic_1.act(
                        {**next_inputs, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2_values, _ = self.target_critic_2.act(
                        {**next_inputs, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    target_q_values = torch.min(target_q1_values, target_q2_values)
                    target_values = (
                        sampled_rewards
                        + self.cfg.discount_factor
                        * (sampled_terminated | sampled_truncated).logical_not()
                        * target_q_values
                    )

                # compute critic loss
                critic_1_values, _ = self.critic_1.act({**inputs, "taken_actions": sampled_actions}, role="critic_1")
                critic_2_values, _ = self.critic_2.act({**inputs, "taken_actions": sampled_actions}, role="critic_2")

                critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self.cfg.grad_norm_clip
                )

            self.scaler.step(self.critic_optimizer)

            # delayed update
            self._update_counter += 1
            if not self._update_counter % self.cfg.policy_delay:

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    # compute policy (actor) loss
                    actions, _ = self.policy.act(inputs, role="policy")
                    critic_values, _ = self.critic_1.act({**inputs, "taken_actions": actions}, role="critic_1")

                    policy_loss = -critic_values.mean()

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)

                self.scaler.step(self.policy_optimizer)

                # update target networks
                self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)
                self.target_policy.update_parameters(self.policy, polyak=self.cfg.polyak)

            self.scaler.update()  # called once, after optimizers have been stepped

            # update learning rate
            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            # record data
            if not self._update_counter % self.cfg.policy_delay:
                self.track_data("Loss / Policy loss", policy_loss.item())
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
            self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
            self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

            self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
            self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
            self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self.policy_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
            if self.critic_scheduler:
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
