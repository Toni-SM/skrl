from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import torch
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from .ddqn_cfg import DDQN_CFG


class DDQN(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: DDQN_CFG | dict = {},
    ) -> None:
        """Double Deep Q-Network (DDQN).

        https://ojs.aaai.org/index.php/AAAI/article/view/10295

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: DDQN_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=DDQN_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.q_network = self.models.get("q_network", None)
        self.target_q_network = self.models.get("target_q_network", None)

        # checkpoint models
        self.checkpoint_modules["q_network"] = self.q_network
        self.checkpoint_modules["target_q_network"] = self.target_q_network

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.q_network is not None:
                self.q_network.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.q_network is not None:
            # - optimizer
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.cfg.learning_rate)
            self.checkpoint_modules["optimizer"] = self.optimizer
            # - learning rate scheduler
            self.scheduler = self.cfg.learning_rate_scheduler
            if self.scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler(
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs
                )

        # set up target networks
        if self.target_q_network is not None:
            # - freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_q_network.freeze_parameters(True)
            # - update target networks (hard update)
            self.target_q_network.update_parameters(self.q_network, polyak=1)

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
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.int64)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

        self.tensors_names = [
            "observations",
            "states",
            "actions",
            "rewards",
            "next_observations",
            "next_states",
            "terminated",
            "truncated",
        ]

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

        if self.cfg.exploration_scheduler is None:
            q_values, outputs = self.q_network.act(inputs, role="q_network")
            return torch.argmax(q_values, dim=1, keepdim=True), outputs

        # sample random actions
        actions, outputs = self.q_network.random_act(inputs, role="q_network")
        if timestep < self.cfg.random_timesteps:
            return actions, outputs

        # sample actions with epsilon-greedy policy
        epsilon = self.cfg.exploration_scheduler(timestep, timesteps)
        indexes = (torch.rand(actions.shape[0], device=self.device) >= epsilon).nonzero().view(-1)
        if indexes.numel():
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {k: None if v is None else v[indexes] for k, v in inputs.items()}
                actions[indexes] = torch.argmax(self.q_network.act(inputs, role="q_network")[0], dim=1, keepdim=True)

        # record epsilon
        self.track_data("Exploration / Exploration epsilon", epsilon)

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
        if timestep >= self.cfg.learning_starts and not timestep % self.cfg.update_interval:
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
            ) = self.memory.sample(names=self.tensors_names, batch_size=self.cfg.batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                    "states": self._state_preprocessor(sampled_states, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                    "states": self._state_preprocessor(sampled_next_states, train=True),
                }

                # compute target values
                with torch.no_grad():
                    next_q_values, _ = self.target_q_network.act(next_inputs, role="target_q_network")

                    target_q_values = torch.gather(
                        next_q_values,
                        dim=1,
                        index=torch.argmax(self.q_network.act(next_inputs, role="q_network")[0], dim=1, keepdim=True),
                    )
                    target_values = (
                        sampled_rewards
                        + self.cfg.discount_factor
                        * (sampled_terminated | sampled_truncated).logical_not()
                        * target_q_values
                    )

                # compute Q-network loss
                q_values = torch.gather(
                    self.q_network.act(inputs, role="q_network")[0], dim=1, index=sampled_actions.long()
                )

                q_network_loss = F.mse_loss(q_values, target_values)

            # optimize Q-network
            self.optimizer.zero_grad()
            self.scaler.scale(q_network_loss).backward()

            if config.torch.is_distributed:
                self.q_network.reduce_parameters()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update target network
            self._update_counter += 1
            if not self._update_counter % self.cfg.target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self.cfg.polyak)

            # update learning rate
            if self.scheduler:
                self.scheduler.step()

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self.scheduler:
                self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
