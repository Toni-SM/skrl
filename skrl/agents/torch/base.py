from typing import Any, Mapping, Optional, Tuple, Union

import collections
import copy
import datetime
import os
import gym
import gymnasium
from packaging import version

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model


class Agent:
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Base class that represent a RL agent

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        self.models = models
        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg if cfg is not None else {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        if type(memory) is list:
            self.memory = memory[0]
            self.secondary_memories = memory[1:]
        else:
            self.memory = memory
            self.secondary_memories = []

        # convert the models to their respective device
        for model in self.models.values():
            if model is not None:
                model.to(model.device)

        self.tracking_data = collections.defaultdict(list)
        self.write_interval = self.cfg.get("experiment", {}).get("write_interval", "auto")

        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

        self.training = True

        # checkpoint
        self.checkpoint_modules = {}
        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", "auto")
        self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)
        self.checkpoint_best_modules = {"timestep": 0, "reward": -2 ** 31, "saved": False, "modules": {}}

        # experiment directory
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), self.__class__.__name__)
        self.experiment_dir = os.path.join(directory, experiment_name)

    def __str__(self) -> str:
        """Generate a representation of the agent as string

        :return: Representation of the agent as string
        :rtype: str
        """
        string = f"Agent: {repr(self)}"
        for k, v in self.cfg.items():
            if type(v) is dict:
                string += f"\n  |-- {k}"
                for k1, v1 in v.items():
                    string += f"\n  |     |-- {k1}: {v1}"
            else:
                string += f"\n  |-- {k}: {v}"
        return string

    def _empty_preprocessor(self, _input: Any, *args, **kwargs) -> Any:
        """Empty preprocess method

        This method is defined because PyTorch multiprocessing can't pickle lambdas

        :param _input: Input to preprocess
        :type _input: Any

        :return: Preprocessed input
        :rtype: Any
        """
        return _input

    def _get_internal_value(self, _module: Any) -> Any:
        """Get internal module/variable state/value

        :param _module: Module or variable
        :type _module: Any

        :return: Module/variable state/value
        :rtype: Any
        """
        return _module.state_dict() if hasattr(_module, "state_dict") else _module

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent

        This method should be called before the agent is used.
        It will initialize the TensorBoard writer (and optionally Weights & Biases) and create the checkpoints directory

        :param trainer_cfg: Trainer configuration
        :type trainer_cfg: dict, optional
        """
        trainer_cfg = trainer_cfg if trainer_cfg is not None else {}

        # update agent configuration to avoid duplicated logging/checking in distributed runs
        if config.torch.is_distributed and config.torch.rank:
            self.write_interval = 0
            self.checkpoint_interval = 0
            # TODO: disable wandb

        # setup Weights & Biases
        if self.cfg.get("experiment", {}).get("wandb", False):
            # save experiment configuration
            try:
                models_cfg = {k: v.net._modules for (k, v) in self.models.items()}
            except AttributeError:
                models_cfg = {k: v._modules for (k, v) in self.models.items()}
            wandb_config={**self.cfg, **trainer_cfg, **models_cfg}
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

        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(trainer_cfg.get("timesteps", 0) / 10)
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

    def track_data(self, tag: str, value: float) -> None:
        """Track data to TensorBoard

        Currently only scalar data are supported

        :param tag: Data identifier (e.g. 'Loss / policy loss')
        :type tag: str
        :param value: Value to track
        :type value: float
        """
        self.tracking_data[tag].append(value)

    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        for k, v in self.tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(k, np.min(v), timestep)
            elif k.endswith("(max)"):
                self.writer.add_scalar(k, np.max(v), timestep)
            else:
                self.writer.add_scalar(k, np.mean(v), timestep)
        # reset data containers for next iteration
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()

    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        """Write checkpoint (modules) to disk

        The checkpoints are saved in the directory 'checkpoints' in the experiment directory.
        The name of the checkpoint is the current timestep if timestep is not None, otherwise it is the current time.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        # separated modules
        if self.checkpoint_store_separately:
            for name, module in self.checkpoint_modules.items():
                torch.save(self._get_internal_value(module),
                           os.path.join(self.experiment_dir, "checkpoints", f"{name}_{tag}.pt"))
        # whole agent
        else:
            modules = {}
            for name, module in self.checkpoint_modules.items():
                modules[name] = self._get_internal_value(module)
            torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", f"agent_{tag}.pt"))

        # best modules
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # separated modules
            if self.checkpoint_store_separately:
                for name, module in self.checkpoint_modules.items():
                    torch.save(self.checkpoint_best_modules["modules"][name],
                               os.path.join(self.experiment_dir, "checkpoints", f"best_{name}.pt"))
            # whole agent
            else:
                modules = {}
                for name, module in self.checkpoint_modules.items():
                    modules[name] = self.checkpoint_best_modules["modules"][name]
                torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", "best_agent.pt"))
            self.checkpoint_best_modules["saved"] = True

    def act(self,
            states: torch.Tensor,
            timestep: int,
            timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :raises NotImplementedError: The method is not implemented by the inheriting classes

        :return: Actions
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory (to be implemented by the inheriting classes)

        Inheriting classes must call this method to record episode information (rewards, timesteps, etc.).
        In addition to recording environment transition (such as states, rewards, etc.), agent information can be recorded.

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.write_interval > 0:
            # compute the cumulative sum of the rewards and timesteps
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

            self._cumulative_rewards.add_(rewards)
            self._cumulative_timesteps.add_(1)

            # check ended episodes
            finished_episodes = (terminated + truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():

                # storage cumulative rewards and timesteps
                self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
                self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

                # reset the cumulative rewards and timesteps
                self._cumulative_rewards[finished_episodes] = 0
                self._cumulative_timesteps[finished_episodes] = 0

            # record data
            self.tracking_data["Reward / Instantaneous reward (max)"].append(torch.max(rewards).item())
            self.tracking_data["Reward / Instantaneous reward (min)"].append(torch.min(rewards).item())
            self.tracking_data["Reward / Instantaneous reward (mean)"].append(torch.mean(rewards).item())

            if len(self._track_rewards):
                track_rewards = np.array(self._track_rewards)
                track_timesteps = np.array(self._track_timesteps)

                self.tracking_data["Reward / Total reward (max)"].append(np.max(track_rewards))
                self.tracking_data["Reward / Total reward (min)"].append(np.min(track_rewards))
                self.tracking_data["Reward / Total reward (mean)"].append(np.mean(track_rewards))

                self.tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timesteps))
                self.tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timesteps))
                self.tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: 'train' for training or 'eval' for evaluation
        :type mode: str
        """
        for model in self.models.values():
            if model is not None:
                model.set_mode(mode)

    def set_running_mode(self, mode: str) -> None:
        """Set the current running mode (training or evaluation)

        This method sets the value of the ``training`` property (boolean).
        This property can be used to know if the agent is running in training or evaluation mode.

        :param mode: Mode: 'train' for training or 'eval' for evaluation
        :type mode: str
        """
        self.training = mode == "train"

    def save(self, path: str) -> None:
        """Save the agent to the specified path

        :param path: Path to save the model to
        :type path: str
        """
        modules = {}
        for name, module in self.checkpoint_modules.items():
            modules[name] = self._get_internal_value(module)
        torch.save(modules, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            modules = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning
        else:
            modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError
                else:
                    logger.warning(f"Cannot load the {name} module. The agent doesn't have such an instance")

    def migrate(self,
                path: str,
                name_map: Mapping[str, Mapping[str, str]] = {},
                auto_mapping: bool = True,
                verbose: bool = False) -> bool:
        """Migrate the specified external checkpoint to the current agent

        The final storage device is determined by the constructor of the agent.
        Only files generated by the *rl_games* library are supported at the moment

        For ambiguous models (where 2 or more parameters, for source or current model, have equal shape)
        it is necessary to define the ``name_map``, at least for those parameters, to perform the migration successfully

        :param path: Path to the external checkpoint to migrate from
        :type path: str
        :param name_map: Name map to use for the migration (default: ``{}``).
                         Keys are the current parameter names and values are the external parameter names
        :type name_map: Mapping[str, Mapping[str, str]], optional
        :param auto_mapping: Automatically map the external state dict to the current state dict (default: ``True``)
        :type auto_mapping: bool, optional
        :param verbose: Show model names and migration (default: ``False``)
        :type verbose: bool, optional

        :raises ValueError: If the correct file type cannot be identified from the ``path`` parameter

        :return: True if the migration was successful, False otherwise.
                 Migration is successful if all parameters of the current model are found in the external model
        :rtype: bool

        Example::

            # migrate a rl_games checkpoint with ambiguous state_dict
            >>> agent.migrate(path="./runs/Cartpole/nn/Cartpole.pth", verbose=False)
            [skrl:WARNING] Ambiguous match for net.0.bias <- [a2c_network.actor_mlp.0.bias, a2c_network.actor_mlp.2.bias]
            [skrl:WARNING] Ambiguous match for net.2.bias <- [a2c_network.actor_mlp.0.bias, a2c_network.actor_mlp.2.bias]
            [skrl:WARNING] Ambiguous match for net.4.weight <- [a2c_network.value.weight, a2c_network.mu.weight]
            [skrl:WARNING] Ambiguous match for net.4.bias <- [a2c_network.value.bias, a2c_network.mu.bias]
            [skrl:WARNING] Multiple use of a2c_network.actor_mlp.0.bias -> [net.0.bias, net.2.bias]
            [skrl:WARNING] Multiple use of a2c_network.actor_mlp.2.bias -> [net.0.bias, net.2.bias]
            [skrl:WARNING] Ambiguous match for net.0.bias <- [a2c_network.actor_mlp.0.bias, a2c_network.actor_mlp.2.bias]
            [skrl:WARNING] Ambiguous match for net.2.bias <- [a2c_network.actor_mlp.0.bias, a2c_network.actor_mlp.2.bias]
            [skrl:WARNING] Ambiguous match for net.4.weight <- [a2c_network.value.weight, a2c_network.mu.weight]
            [skrl:WARNING] Ambiguous match for net.4.bias <- [a2c_network.value.bias, a2c_network.mu.bias]
            [skrl:WARNING] Multiple use of a2c_network.actor_mlp.0.bias -> [net.0.bias, net.2.bias]
            [skrl:WARNING] Multiple use of a2c_network.actor_mlp.2.bias -> [net.0.bias, net.2.bias]
            False
            >>> name_map = {"policy": {"net.0.bias": "a2c_network.actor_mlp.0.bias",
            ...                        "net.2.bias": "a2c_network.actor_mlp.2.bias",
            ...                        "net.4.weight": "a2c_network.mu.weight",
            ...                        "net.4.bias": "a2c_network.mu.bias"},
            ...             "value": {"net.0.bias": "a2c_network.actor_mlp.0.bias",
            ...                       "net.2.bias": "a2c_network.actor_mlp.2.bias",
            ...                       "net.4.weight": "a2c_network.value.weight",
            ...                       "net.4.bias": "a2c_network.value.bias"}}
            >>> model.migrate(path="./runs/Cartpole/nn/Cartpole.pth", name_map=name_map, verbose=True)
            [skrl:INFO] Modules
            [skrl:INFO]   |-- current
            [skrl:INFO]   |    |-- policy (Policy)
            [skrl:INFO]   |    |    |-- log_std_parameter : [1]
            [skrl:INFO]   |    |    |-- net.0.weight : [32, 4]
            [skrl:INFO]   |    |    |-- net.0.bias : [32]
            [skrl:INFO]   |    |    |-- net.2.weight : [32, 32]
            [skrl:INFO]   |    |    |-- net.2.bias : [32]
            [skrl:INFO]   |    |    |-- net.4.weight : [1, 32]
            [skrl:INFO]   |    |    |-- net.4.bias : [1]
            [skrl:INFO]   |    |-- value (Value)
            [skrl:INFO]   |    |    |-- net.0.weight : [32, 4]
            [skrl:INFO]   |    |    |-- net.0.bias : [32]
            [skrl:INFO]   |    |    |-- net.2.weight : [32, 32]
            [skrl:INFO]   |    |    |-- net.2.bias : [32]
            [skrl:INFO]   |    |    |-- net.4.weight : [1, 32]
            [skrl:INFO]   |    |    |-- net.4.bias : [1]
            [skrl:INFO]   |    |-- optimizer (Adam)
            [skrl:INFO]   |    |    |-- state (dict)
            [skrl:INFO]   |    |    |-- param_groups (list)
            [skrl:INFO]   |    |-- state_preprocessor (RunningStandardScaler)
            [skrl:INFO]   |    |    |-- running_mean : [4]
            [skrl:INFO]   |    |    |-- running_variance : [4]
            [skrl:INFO]   |    |    |-- current_count : []
            [skrl:INFO]   |    |-- value_preprocessor (RunningStandardScaler)
            [skrl:INFO]   |    |    |-- running_mean : [1]
            [skrl:INFO]   |    |    |-- running_variance : [1]
            [skrl:INFO]   |    |    |-- current_count : []
            [skrl:INFO]   |-- source
            [skrl:INFO]   |    |-- model (OrderedDict)
            [skrl:INFO]   |    |    |-- value_mean_std.running_mean : [1]
            [skrl:INFO]   |    |    |-- value_mean_std.running_var : [1]
            [skrl:INFO]   |    |    |-- value_mean_std.count : []
            [skrl:INFO]   |    |    |-- running_mean_std.running_mean : [4]
            [skrl:INFO]   |    |    |-- running_mean_std.running_var : [4]
            [skrl:INFO]   |    |    |-- running_mean_std.count : []
            [skrl:INFO]   |    |    |-- a2c_network.sigma : [1]
            [skrl:INFO]   |    |    |-- a2c_network.actor_mlp.0.weight : [32, 4]
            [skrl:INFO]   |    |    |-- a2c_network.actor_mlp.0.bias : [32]
            [skrl:INFO]   |    |    |-- a2c_network.actor_mlp.2.weight : [32, 32]
            [skrl:INFO]   |    |    |-- a2c_network.actor_mlp.2.bias : [32]
            [skrl:INFO]   |    |    |-- a2c_network.value.weight : [1, 32]
            [skrl:INFO]   |    |    |-- a2c_network.value.bias : [1]
            [skrl:INFO]   |    |    |-- a2c_network.mu.weight : [1, 32]
            [skrl:INFO]   |    |    |-- a2c_network.mu.bias : [1]
            [skrl:INFO]   |    |-- epoch (int)
            [skrl:INFO]   |    |-- optimizer (dict)
            [skrl:INFO]   |    |-- frame (int)
            [skrl:INFO]   |    |-- last_mean_rewards (float32)
            [skrl:INFO]   |    |-- env_state (NoneType)
            [skrl:INFO] Migration
            [skrl:INFO] Model: policy (Policy)
            [skrl:INFO] Models
            [skrl:INFO]   |-- current: 7 items
            [skrl:INFO]   |    |-- log_std_parameter : [1]
            [skrl:INFO]   |    |-- net.0.weight : [32, 4]
            [skrl:INFO]   |    |-- net.0.bias : [32]
            [skrl:INFO]   |    |-- net.2.weight : [32, 32]
            [skrl:INFO]   |    |-- net.2.bias : [32]
            [skrl:INFO]   |    |-- net.4.weight : [1, 32]
            [skrl:INFO]   |    |-- net.4.bias : [1]
            [skrl:INFO]   |-- source: 9 items
            [skrl:INFO]   |    |-- a2c_network.sigma : [1]
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.0.weight : [32, 4]
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.0.bias : [32]
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.2.weight : [32, 32]
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.2.bias : [32]
            [skrl:INFO]   |    |-- a2c_network.value.weight : [1, 32]
            [skrl:INFO]   |    |-- a2c_network.value.bias : [1]
            [skrl:INFO]   |    |-- a2c_network.mu.weight : [1, 32]
            [skrl:INFO]   |    |-- a2c_network.mu.bias : [1]
            [skrl:INFO] Migration
            [skrl:INFO]   |-- auto: log_std_parameter <- a2c_network.sigma
            [skrl:INFO]   |-- auto: net.0.weight <- a2c_network.actor_mlp.0.weight
            [skrl:INFO]   |-- map:  net.0.bias <- a2c_network.actor_mlp.0.bias
            [skrl:INFO]   |-- auto: net.2.weight <- a2c_network.actor_mlp.2.weight
            [skrl:INFO]   |-- map:  net.2.bias <- a2c_network.actor_mlp.2.bias
            [skrl:INFO]   |-- map:  net.4.weight <- a2c_network.mu.weight
            [skrl:INFO]   |-- map:  net.4.bias <- a2c_network.mu.bias
            [skrl:INFO] Model: value (Value)
            [skrl:INFO] Models
            [skrl:INFO]   |-- current: 6 items
            [skrl:INFO]   |    |-- net.0.weight : [32, 4]
            [skrl:INFO]   |    |-- net.0.bias : [32]
            [skrl:INFO]   |    |-- net.2.weight : [32, 32]
            [skrl:INFO]   |    |-- net.2.bias : [32]
            [skrl:INFO]   |    |-- net.4.weight : [1, 32]
            [skrl:INFO]   |    |-- net.4.bias : [1]
            [skrl:INFO]   |-- source: 9 items
            [skrl:INFO]   |    |-- a2c_network.sigma : [1]
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.0.weight : [32, 4]
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.0.bias : [32]
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.2.weight : [32, 32]
            [skrl:INFO]   |    |-- a2c_network.actor_mlp.2.bias : [32]
            [skrl:INFO]   |    |-- a2c_network.value.weight : [1, 32]
            [skrl:INFO]   |    |-- a2c_network.value.bias : [1]
            [skrl:INFO]   |    |-- a2c_network.mu.weight : [1, 32]
            [skrl:INFO]   |    |-- a2c_network.mu.bias : [1]
            [skrl:INFO] Migration
            [skrl:INFO]   |-- auto: net.0.weight <- a2c_network.actor_mlp.0.weight
            [skrl:INFO]   |-- map:  net.0.bias <- a2c_network.actor_mlp.0.bias
            [skrl:INFO]   |-- auto: net.2.weight <- a2c_network.actor_mlp.2.weight
            [skrl:INFO]   |-- map:  net.2.bias <- a2c_network.actor_mlp.2.bias
            [skrl:INFO]   |-- map:  net.4.weight <- a2c_network.value.weight
            [skrl:INFO]   |-- map:  net.4.bias <- a2c_network.value.bias
            True
        """
        # load state_dict from path
        if path is not None:
            # rl_games checkpoint
            if path.endswith(".pt") or path.endswith(".pth"):
                checkpoint = torch.load(path, map_location=self.device)
            else:
                raise ValueError("Cannot identify file type")

        # show modules
        if verbose:
            logger.info("Modules")
            logger.info("  |-- current")
            for name, module in self.checkpoint_modules.items():
                logger.info(f"  |    |-- {name} ({type(module).__name__})")
                if hasattr(module, "state_dict"):
                    for k, v in module.state_dict().items():
                        if hasattr(v, "shape"):
                            logger.info(f"  |    |    |-- {k} : {list(v.shape)}")
                        else:
                            logger.info(f"  |    |    |-- {k} ({type(v).__name__})")
            logger.info("  |-- source")
            for name, module in checkpoint.items():
                logger.info(f"  |    |-- {name} ({type(module).__name__})")
                if name == "model":
                    for k, v in module.items():
                        logger.info(f"  |    |    |-- {k} : {list(v.shape)}")
                else:
                    if hasattr(module, "state_dict"):
                        for k, v in module.state_dict().items():
                            if hasattr(v, "shape"):
                                logger.info(f"  |    |    |-- {k} : {list(v.shape)}")
                            else:
                                logger.info(f"  |    |    |-- {k} ({type(v).__name__})")
            logger.info("Migration")

        if "optimizer" in self.checkpoint_modules:
            # loaded state dict contains a parameter group that doesn't match the size of optimizer's group
            # self.checkpoint_modules["optimizer"].load_state_dict(checkpoint["optimizer"])
            pass
        # state_preprocessor
        if "state_preprocessor" in self.checkpoint_modules:
            if "running_mean_std.running_mean" in checkpoint["model"]:
                state_dict = copy.deepcopy(self.checkpoint_modules["state_preprocessor"].state_dict())
                state_dict["running_mean"] = checkpoint["model"]["running_mean_std.running_mean"]
                state_dict["running_variance"] = checkpoint["model"]["running_mean_std.running_var"]
                state_dict["current_count"] = checkpoint["model"]["running_mean_std.count"]
                self.checkpoint_modules["state_preprocessor"].load_state_dict(state_dict)
                del checkpoint["model"]["running_mean_std.running_mean"]
                del checkpoint["model"]["running_mean_std.running_var"]
                del checkpoint["model"]["running_mean_std.count"]
        # value_preprocessor
        if "value_preprocessor" in self.checkpoint_modules:
            if "value_mean_std.running_mean" in checkpoint["model"]:
                state_dict = copy.deepcopy(self.checkpoint_modules["value_preprocessor"].state_dict())
                state_dict["running_mean"] = checkpoint["model"]["value_mean_std.running_mean"]
                state_dict["running_variance"] = checkpoint["model"]["value_mean_std.running_var"]
                state_dict["current_count"] = checkpoint["model"]["value_mean_std.count"]
                self.checkpoint_modules["value_preprocessor"].load_state_dict(state_dict)
                del checkpoint["model"]["value_mean_std.running_mean"]
                del checkpoint["model"]["value_mean_std.running_var"]
                del checkpoint["model"]["value_mean_std.count"]
        # TODO: AMP state preprocessor
        # model
        status = True
        for name, module in self.checkpoint_modules.items():
            if module not in ["state_preprocessor", "value_preprocessor", "optimizer"] and hasattr(module, "migrate"):
                if verbose:
                    logger.info(f"Model: {name} ({type(module).__name__})")
                status *= module.migrate(state_dict=checkpoint["model"],
                                            name_map=name_map.get(name, {}),
                                            auto_mapping=auto_mapping,
                                            verbose=verbose)

        self.set_mode("eval")
        return bool(status)

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
        timestep += 1

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # update best models
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -2 ** 31))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules.items()}
            # write checkpoints
            self.write_checkpoint(timestep, timesteps)

        # write to tensorboard
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :raises NotImplementedError: The method is not implemented by the inheriting classes
        """
        raise NotImplementedError
