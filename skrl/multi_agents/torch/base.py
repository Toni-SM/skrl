from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import collections
import copy
import datetime
import os
from abc import ABC, abstractmethod
import gymnasium
from packaging import version

import numpy as np
import torch

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils.tensorboard import SummaryWriter


class MultiAgent(ABC):
    def __init__(
        self,
        *,
        possible_agents: Sequence[str],
        models: Mapping[str, Mapping[str, Model]],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Mapping[str, gymnasium.Space]] = None,
        state_spaces: Optional[Mapping[str, gymnasium.Space]] = None,
        action_spaces: Optional[Mapping[str, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Base class that represent a RL multi-agent/algorithm.

        :param possible_agents: Name of all possible agents the environment could generate.
        :param models: Agents' models.
        :param memories: Memories to storage agents' data and environment transitions.
        :param observation_spaces: Observation spaces.
        :param state_spaces: State spaces.
        :param action_spaces: Action spaces.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Multi-agent's configuration.
        """
        self.training = False

        self.possible_agents = possible_agents
        self.num_agents = len(self.possible_agents)

        self.models = models
        self.memories = memories
        self.observation_spaces = observation_spaces
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces
        self.cfg = cfg if cfg is not None else {}

        self.device = config.torch.parse_device(device)

        # convert the models to their respective device
        for _models in self.models.values():
            for model in _models.values():
                if model is not None:
                    model.to(model.device)

        # data tracking
        self.tracking_data = collections.defaultdict(list)
        self.write_interval = self.cfg.get("experiment", {}).get("write_interval", "auto")

        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

        # checkpoint
        self.checkpoint_modules = {uid: {} for uid in self.possible_agents}
        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", "auto")
        self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)
        self.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": False, "modules": {}}

        # experiment directory
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = f"{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}_{self.__class__.__name__}"
        self.experiment_dir = os.path.join(directory, experiment_name)

    def __str__(self) -> str:
        """Generate a string representation of the agent.

        :return: String representation.
        """
        string = f"Multi-agent: {repr(self)}"
        for k, v in self.cfg.items():
            if type(v) is dict:
                string += f"\n  |-- {k}"
                for k1, v1 in v.items():
                    string += f"\n  |     |-- {k1}: {v1}"
            else:
                string += f"\n  |-- {k}: {v}"
        return string

    def _as_dict(self, _input: Any) -> Mapping[str, Any]:
        """Convert a configuration value into a dictionary according to the number of agents.

        :param _input: Configuration value.

        :return: Configuration value as a dictionary.

        :raises ValueError: The configuration value is a dictionary different from the number of agents.
        """
        if _input and isinstance(_input, dict):
            if set(_input) < set(self.possible_agents):
                raise ValueError(
                    f"Configuration keys ({set(_input)}) do not match possible agents ({self.possible_agents})"
                )
            elif set(_input) >= set(self.possible_agents):
                return _input
        return {name: copy.deepcopy(_input) for name in self.possible_agents}

    def _empty_preprocessor(self, _input: Any, *args, **kwargs) -> Any:
        """Empty preprocess method.

        .. note::

            This method is defined because PyTorch multiprocessing can't pickle lambdas.

        :param _input: Input to preprocess.

        :return: Preprocessed input.
        """
        return _input

    def _get_internal_value(self, _module: Any) -> Any:
        """Get internal module/variable state/value.

        :param _module: Module or variable.

        :return: Module/variable state/value.
        """
        return _module.state_dict() if hasattr(_module, "state_dict") else _module

    def init(self, *, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent.

        .. warning::

            This method must be called before the agent is used.
            It will initialize the TensorBoard writer (and optionally Weights & Biases) and create the checkpoints directory.

        :param trainer_cfg: Trainer configuration.
        """
        trainer_cfg = {} if trainer_cfg is None else trainer_cfg

        # update agent configuration to avoid duplicated logging/checking in distributed runs
        if config.torch.is_distributed and config.torch.rank:
            self.write_interval = 0
            self.checkpoint_interval = 0
            # TODO: disable wandb

        # setup Weights & Biases
        if self.cfg.get("experiment", {}).get("wandb", False):
            # save experiment configuration
            try:
                models_cfg = {
                    uid: {k: v.net._modules for (k, v) in self.models[uid].items()} for uid in self.possible_agents
                }
            except AttributeError:
                models_cfg = {
                    uid: {k: v._modules for (k, v) in self.models[uid].items()} for uid in self.possible_agents
                }
            wandb_config = {**self.cfg, **trainer_cfg, **models_cfg}
            # set default values
            wandb_kwargs = copy.deepcopy(self.cfg.get("experiment", {}).get("wandb_kwargs", {}))
            wandb_kwargs.setdefault("name", os.path.split(self.experiment_dir)[-1])
            wandb_kwargs.setdefault("sync_tensorboard", True)
            wandb_kwargs.setdefault("config", {})
            wandb_kwargs["config"].update(wandb_config)
            # init Weights & Biases
            import wandb

            wandb.init(**wandb_kwargs)

        # main entry to log data for consumption and visualization by TensorBoard
        if self.write_interval == "auto":
            self.write_interval = int(trainer_cfg.get("timesteps", 0) / 100)
        if self.write_interval > 0:
            self.writer = SummaryWriter(log_dir=self.experiment_dir)

        # checkpoint directory creation
        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(trainer_cfg.get("timesteps", 0) / 10)
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

    def track_data(self, tag: str, value: float) -> None:
        """Track data to TensorBoard.

        .. note::

            Currently only scalar data is supported.

        :param tag: Data identifier (e.g. 'Loss/Policy loss').
        :param value: Value to track.
        """
        self.tracking_data[tag].append(value)

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """Write tracking data to TensorBoard.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        for k, v in self.tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(tag=k, value=np.min(v), timestep=timestep)
            elif k.endswith("(max)"):
                self.writer.add_scalar(tag=k, value=np.max(v), timestep=timestep)
            else:
                self.writer.add_scalar(tag=k, value=np.mean(v), timestep=timestep)
        # reset data containers
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()

    def write_checkpoint(self, *, timestep: int, timesteps: int) -> None:
        """Write checkpoint (modules) to persistent storage.

        .. note::

            The checkpoints are stored in the subdirectory ``checkpoints`` within the experiment directory.
            The checkpoint name is the ``timestep`` argument value (if it is not ``None``),
            or the current system date-time otherwise.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        # separated modules
        if self.checkpoint_store_separately:
            for uid in self.possible_agents:
                for name, module in self.checkpoint_modules[uid].items():
                    torch.save(
                        self._get_internal_value(module),
                        os.path.join(self.experiment_dir, "checkpoints", f"{uid}_{name}_{tag}.pt"),
                    )
        # whole agent
        else:
            modules = {
                uid: {name: self._get_internal_value(module) for name, module in self.checkpoint_modules[uid].items()}
                for uid in self.possible_agents
            }
            torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", f"agent_{tag}.pt"))

        # best modules
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # separated modules
            if self.checkpoint_store_separately:
                for uid in self.possible_agents:
                    for name in self.checkpoint_modules[uid].keys():
                        torch.save(
                            self.checkpoint_best_modules["modules"][uid][name],
                            os.path.join(self.experiment_dir, "checkpoints", f"best_{uid}_{name}.pt"),
                        )
            # whole agent
            else:
                modules = {
                    uid: {
                        name: self.checkpoint_best_modules["modules"][uid][name]
                        for name in self.checkpoint_modules[uid].keys()
                    }
                    for uid in self.possible_agents
                }
                torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", "best_agent.pt"))
            self.checkpoint_best_modules["saved"] = True

    @abstractmethod
    def act(
        self,
        observations: Mapping[str, torch.Tensor],
        states: Mapping[str, Union[torch.Tensor, None]],
        *,
        timestep: int,
        timesteps: int,
    ) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Union[torch.Tensor, Any]]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        pass

    def record_transition(
        self,
        *,
        observations: Mapping[str, torch.Tensor],
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_observations: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        infos: Mapping[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        .. note::

            This method keeps track of the episode rewards (instantaneous and cumulative) and timesteps
            when ``experiment.write_interval`` configuration is resolved to a positive value.
            Inheriting classes must call this method to record such information.

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
        if self.write_interval > 0:
            _rewards = sum(rewards.values())

            # compute the cumulative sum of the rewards and timesteps
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(_rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(_rewards, dtype=torch.int32)

            self._cumulative_rewards.add_(_rewards)
            self._cumulative_timesteps.add_(1)

            # check ended episodes
            finished_episodes = (next(iter(terminated.values())) + next(iter(truncated.values()))).nonzero(
                as_tuple=False
            )
            if finished_episodes.numel():

                # storage cumulative rewards and timesteps
                self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
                self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

                # reset the cumulative rewards and timesteps
                self._cumulative_rewards[finished_episodes] = 0
                self._cumulative_timesteps[finished_episodes] = 0

            # record data
            self.tracking_data["Reward / Instantaneous reward (max)"].append(torch.max(_rewards).item())
            self.tracking_data["Reward / Instantaneous reward (min)"].append(torch.min(_rewards).item())
            self.tracking_data["Reward / Instantaneous reward (mean)"].append(torch.mean(_rewards).item())

            if len(self._track_rewards):
                track_rewards = np.array(self._track_rewards)
                track_timesteps = np.array(self._track_timesteps)

                self.tracking_data["Reward / Total reward (max)"].append(np.max(track_rewards))
                self.tracking_data["Reward / Total reward (min)"].append(np.min(track_rewards))
                self.tracking_data["Reward / Total reward (mean)"].append(np.mean(track_rewards))

                self.tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timesteps))
                self.tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timesteps))
                self.tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))

    def enable_training_mode(self, enabled: bool = True) -> None:
        """Set the training mode of the agent: enabled (training) or disabled (evaluation).

        The training mode can be queried by the ``training`` property.

        :param enabled: True to enable the training mode, False to enable the evaluation mode.
        """
        self.training = enabled

    def enable_models_training_mode(self, enabled: bool = True) -> None:
        """Set the training mode of all the agent's models: enabled (training) or disabled (evaluation).

        :param enabled: True to enable the training mode, False to enable the evaluation mode.
        """
        for _models in self.models.values():
            for model in _models.values():
                if model is not None:
                    model.enable_training_mode(enabled)

    def save(self, path: str) -> None:
        """Save the agent to the specified path.

        :param path: Path to save the agent to.
        """
        modules = {
            uid: {name: self._get_internal_value(module) for name, module in self.checkpoint_modules[uid].items()}
            for uid in self.possible_agents
        }
        torch.save(modules, path)

    def load(self, path: str) -> None:
        """Load the agent from the specified path.

        .. note::

            The final storage device is determined by the constructor of the agent.

        :param path: Path to load the agent from.
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            modules = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        else:
            modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for uid in self.possible_agents:
                if uid not in modules:
                    logger.warning(f"Skipping modules for '{uid}'. The agent doesn't have such an instance")
                    continue
                for name, data in modules[uid].items():
                    module = self.checkpoint_modules[uid].get(name, None)
                    if module is not None:
                        if hasattr(module, "load_state_dict"):
                            module.load_state_dict(data)
                            if hasattr(module, "eval"):
                                module.eval()
                        else:
                            raise NotImplementedError
                    else:
                        logger.warning(f"Skipping module '{uid}:{name}'. The agent doesn't have such an instance")

    @abstractmethod
    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    @abstractmethod
    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        timestep += 1

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # update best models
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -(2**31)))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {
                    uid: {
                        k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules[uid].items()
                    }
                    for uid in self.possible_agents
                }
            # write checkpoints
            self.write_checkpoint(timestep=timestep, timesteps=timesteps)

        # write to tensorboard
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep=timestep, timesteps=timesteps)

    @abstractmethod
    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        .. warning::

            This method should not be called directly, but rather by the agent itself
            when the algorithm is needed for learning.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass
