from __future__ import annotations

from typing import Any

import functools
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

from .distillation_cfg import DISTILLATION_CFG


class Distillation(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: DISTILLATION_CFG | dict = {},
    ) -> None:
        """Teacher-Student Distillation.

        The agent trains the ``policy`` model (the **student**), which observes the deployable observations only,
        to imitate the frozen ``teacher`` model, which typically also observes privileged states.
        Since the student is the model that acts in the environment, the teacher labels the state distribution
        visited by the student itself (an online behavior cloning, or DAgger-like, scheme).

        :param models: Agent's models. The following roles are expected:

            - ``"policy"``: the student model to be distilled (the model that acts and is optimized).
            - ``"teacher"``: the frozen, typically privileged, model that produces the action targets.

        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        :raises ValueError: If a model is missing, or if the action space is not continuous.
        """
        self.cfg: DISTILLATION_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=DISTILLATION_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )
        self.cfg.validate()

        # models
        self.policy = self.models.get("policy", None)  # the student: it acts in the environment and is optimized
        self.teacher = self.models.get("teacher", None)  # frozen, typically privileged

        if self.policy is None:
            raise ValueError("The 'policy' model (the student to be distilled) is required")
        if self.teacher is None:
            raise ValueError("The 'teacher' model (the frozen model to imitate) is required")
        if self.policy is self.teacher:
            raise ValueError(
                "The 'policy' (student) and 'teacher' models must not be the same instance. "
                "Freezing the teacher would also freeze the student, leaving nothing to optimize"
            )
        if isinstance(self.action_space, (gymnasium.spaces.Discrete, gymnasium.spaces.MultiDiscrete)):
            raise ValueError(
                "Distillation regresses the student's actions onto the teacher's actions and therefore supports "
                f"continuous action spaces only. Got: {self.action_space}"
            )

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["teacher"] = self.teacher

        # freeze the teacher: its parameters are never optimized and it is always kept in evaluation mode
        self.teacher.freeze_parameters(True)
        self.teacher.enable_training_mode(False)
        self._teacher_loaded = False

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            self.policy.broadcast_parameters()
            self.teacher.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # set up optimizer and learning rate scheduler
        # - optimizer (only the student's parameters are optimized)
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

        # set up the regression loss used to fit the student's actions to the teacher's actions
        if self.cfg.loss_type == "huber":
            self._loss_fn = functools.partial(F.huber_loss, delta=self.cfg.huber_delta)
        else:
            self._loss_fn = F.mse_loss

        # teacher's actions (regression targets) computed during the rollout collection
        self._current_teacher_actions = None

        # load the teacher's parameters from the configured checkpoint, if any
        if self.cfg.teacher_checkpoint:
            self.load_teacher(self.cfg.teacher_checkpoint)

    def load_teacher(self, path: str) -> None:
        """Load the teacher model's parameters from a checkpoint file.

        .. note::

            The teacher model is re-frozen and set to evaluation mode after loading its parameters.

        :param path: Path to the checkpoint file. It can be either a whole-agent checkpoint
            (e.g.: ``agent_48000.pt``), in which case the ``teacher`` or the ``policy`` module is used
            (in that order), or a single model's state dict (e.g.: ``policy_48000.pt``).

        :raises ValueError: If no usable state dict can be resolved from the checkpoint file.
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            modules = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        else:
            modules = torch.load(path, map_location=self.device)

        state_dict = None
        if isinstance(modules, dict) and modules:
            if "teacher" in modules:
                state_dict = modules["teacher"]
            elif "policy" in modules:
                # a whole-agent checkpoint's 'policy' module covers both an RL run's policy and
                # a previous distillation run's student being promoted to teacher
                state_dict = modules["policy"]
            elif all(isinstance(value, torch.Tensor) for value in modules.values()):
                # a single model's state dict, as written when 'experiment.store_separately' is enabled
                state_dict = modules
        if state_dict is None:
            available = list(modules.keys()) if isinstance(modules, dict) else type(modules).__name__
            raise ValueError(
                f"Unable to resolve the teacher's parameters from the checkpoint file: {path}. "
                "Expected either a single model's state dict or a whole-agent checkpoint containing a "
                f"'teacher' or 'policy' module. Got: {available}"
            )

        # the teacher's normalization statistics are not part of its state dict, but of the agent that trained it.
        # Feeding the teacher observations normalized by this agent's (freshly initialized) preprocessors would
        # produce regression targets the teacher was never trained to give
        unloaded_preprocessors = [
            name for name in ["observation_preprocessor", "state_preprocessor"] if name in modules
        ]
        if unloaded_preprocessors:
            logger.warning(
                f"The teacher's checkpoint contains {unloaded_preprocessors}, which are not applied to the teacher. "
                "If the teacher was trained with input preprocessing, configure this agent with the same "
                "preprocessor(s) and load their state to avoid feeding the teacher differently scaled inputs"
            )

        self.teacher.load_state_dict(state_dict)
        self.teacher.freeze_parameters(True)
        self.teacher.enable_training_mode(False)
        self._teacher_loaded = True

    def load(self, path: str) -> None:
        """Load the agent from the specified path.

        :param path: Path to load the agent from.
        """
        super().load(path)
        # a whole-agent checkpoint written by this agent always carries the teacher's parameters
        self._teacher_loaded = True

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="teacher_actions", size=self.action_space, dtype=torch.float32)

        # the "states" tensor is not created if the state space is undefined.
        # In such a case, sampling it returns None, which the preprocessors and models handle transparently
        self._tensors_names = ["observations", "states", "teacher_actions"]

        # create temporary variables needed for storage and computation
        self._rollout = 0
        self._current_teacher_actions = None

        if not self._teacher_loaded:
            logger.warning(
                "The teacher model's parameters have not been loaded by the agent. Set the 'teacher_checkpoint' "
                "config value or call the agent's 'load_teacher' method to distill a trained teacher. "
                "Ignore this warning if the model was loaded externally"
            )

    def _deterministic_actions(self, model: Model, inputs: dict[str, Any], *, role: str) -> torch.Tensor:
        """Compute a model's deterministic actions.

        :param model: Model to compute the actions for.
        :param inputs: Model inputs.
        :param role: Role played by the model.

        :return: Deterministic actions. For Gaussian-like models, the distribution's mean actions.
            For deterministic models, the returned actions themselves.
        """
        # the model's 'act' method is used (rather than 'compute') since it is the one applying the optional
        # mean action clipping. Note that, since only the mean actions take part in the regression, a Gaussian
        # model's standard deviation gets no gradient from the behavior cloning loss
        actions, outputs = model.act(inputs, role=role)
        return outputs.get("mean_actions", actions)

    def enable_models_training_mode(self, enabled: bool = True) -> None:
        """Set the training mode of all the agent's models: enabled (training) or disabled (evaluation).

        The teacher model is always kept in evaluation mode since it is frozen and never optimized.

        :param enabled: True to enable the training mode, False to enable the evaluation mode.
        """
        self.policy.enable_training_mode(enabled)  # the student follows the agent's mode
        self.teacher.enable_training_mode(False)  # the teacher is always in evaluation mode

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
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            # the policy is the student: it acts (stochastically) in the environment and is the one being optimized
            actions, outputs = self.policy.act(inputs, role="policy")
            # label the state visited by the student with the teacher's action (the regression target)
            if self.training:
                with torch.no_grad():
                    self._current_teacher_actions = self._deterministic_actions(
                        self.teacher, inputs, role="teacher"
                    ).float()
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

        if self.training:
            # rewards are not part of the distillation objective, so they are tracked but not stored
            self.memory.add_samples(
                observations=observations,
                states=states,
                teacher_actions=self._current_teacher_actions,
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
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts:
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
        cumulative_behavior_cloning_loss = 0
        num_mini_batches = 0

        for epoch in range(self.cfg.learning_epochs):
            for (
                sampled_observations,
                sampled_states,
                sampled_teacher_actions,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
            ):

                # skip empty mini-batches, which occur when 'mini_batches' exceeds the number of stored transitions.
                # Regressing over an empty batch yields a NaN loss that would propagate to the student's parameters
                if not sampled_teacher_actions.shape[0]:
                    continue

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": self._state_preprocessor(sampled_states, train=not epoch),
                    }
                    # regress the student's (policy's) actions onto the teacher's actions (the targets)
                    student_actions = self._deterministic_actions(self.policy, inputs, role="policy")
                    behavior_cloning_loss = self._loss_fn(student_actions, sampled_teacher_actions)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(behavior_cloning_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_behavior_cloning_loss += behavior_cloning_loss.item()
                num_mini_batches += 1

            # update learning rate
            if self.scheduler:
                self.scheduler.step()

        if not num_mini_batches:
            logger.warning("No transitions to update. Consider increasing the number of rollouts")
            return

        # record data
        self.track_data("Loss / Behavior cloning loss", cumulative_behavior_cloning_loss / num_mini_batches)

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
