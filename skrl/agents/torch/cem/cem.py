from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import torch
import torch.nn.functional as F

from skrl import logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from .cem_cfg import CEM_CFG


class CEM(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: CEM_CFG | dict = {},
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
        self.cfg: CEM_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=CEM_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # models
        self.policy = self.models.get("policy", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None:
            # - optimizer
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)
            self.checkpoint_modules["optimizer"] = self.optimizer
            # - learning rate scheduler
            self.scheduler = self.cfg.learning_rate_scheduler
            if self.scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler(
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs
                )

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

        self.tensors_names = ["observations", "states", "actions", "rewards"]

        # create temporary variables needed for storage and computation
        self._rollout = 0
        self._episode_tracking = []

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
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            return self.policy.act(inputs, role="policy")

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

        # track episodes internally
        if self._rollout:
            indexes = torch.nonzero(terminated + truncated)
            if indexes.numel():
                for i in indexes[:, 0]:
                    try:
                        self._episode_tracking[i.item()].append(self._rollout + 1)
                    except IndexError:
                        logger.warning(f"IndexError: {i.item()}")
        else:
            self._episode_tracking = [[0] for _ in range(rewards.size(-1))]

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
        if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
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

        with torch.no_grad():
            # compute discounted return threshold
            limits = []
            returns = []
            for e in range(sampled_rewards.size(-1)):
                for i, j in zip(self._episode_tracking[e][:-1], self._episode_tracking[e][1:]):
                    limits.append([e + i, e + j])
                    rewards = sampled_rewards[e + i : e + j]
                    returns.append(
                        torch.sum(
                            rewards
                            * self.cfg.discount_factor
                            ** torch.arange(rewards.size(0), device=rewards.device).flip(-1).view(rewards.size())
                        )
                    )

            if not len(returns):
                logger.warning("No returns to update. Consider increasing the number of rollouts")
                return

            returns = torch.tensor(returns)
            return_threshold = torch.quantile(returns, self.cfg.percentile, dim=-1)

            # get elite observations/states and actions
            indexes = torch.nonzero(returns >= return_threshold)
            elite_observations = torch.cat(
                [sampled_observations[limits[i][0] : limits[i][1]] for i in indexes[:, 0]], dim=0
            )
            try:
                elite_states = torch.cat([sampled_states[limits[i][0] : limits[i][1]] for i in indexes[:, 0]], dim=0)
            except TypeError:
                elite_states = None
            elite_actions = torch.cat([sampled_actions[limits[i][0] : limits[i][1]] for i in indexes[:, 0]], dim=0)

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):

            # compute scores for the elite observations/states
            _, outputs = self.policy.act({"observations": elite_observations, "states": elite_states}, role="policy")
            scores = outputs["net_output"]

            # compute policy loss
            policy_loss = F.cross_entropy(scores, elite_actions.view(-1))

        # optimization step
        self.optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # update learning rate
        if self.scheduler:
            self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", policy_loss.item())

        self.track_data("Coefficient / Return threshold", return_threshold.item())
        self.track_data("Coefficient / Mean discounted returns", torch.mean(returns).item())

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
