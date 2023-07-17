from typing import Any, Mapping, Optional, Sequence, Union

import collections
import copy
import datetime
import os
import gym
import gymnasium

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from skrl import logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model


class MultiAgent:
    def __init__(self,
                 possible_agents: Sequence[str],
                 models: Mapping[str, Mapping[str, Model]],
                 memories: Optional[Mapping[str, Memory]] = None,
                 observation_spaces: Optional[Mapping[str, Union[int, Sequence[int], gym.Space, gymnasium.Space]]] = None,
                 action_spaces: Optional[Mapping[str, Union[int, Sequence[int], gym.Space, gymnasium.Space]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Base class that represent a RL multi-agent

        :param possible_agents: Name of all possible agents the environment could generate
        :type possible_agents: list of str
        :param models: Models used by the agents.
                       External keys are environment agents' names. Internal keys are the models required by the algorithm
        :type models: nested dictionary of skrl.models.torch.Model
        :param memories: Memories to storage the transitions.
        :type memories: dictionary of skrl.memory.torch.Memory, optional
        :param observation_spaces: Observation/state spaces or shapes (default: ``None``)
        :type observation_spaces: dictionary of int, sequence of int, gym.Space or gymnasium.Space, optional
        :param action_spaces: Action spaces or shapes (default: ``None``)
        :type action_spaces: dictionary of int, sequence of int, gym.Space or gymnasium.Space, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        self.possible_agents = possible_agents
        self.num_agents = len(self.possible_agents)

        self.models = models
        self.memories = memories
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

        self.cfg = cfg if cfg is not None else {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        # convert the models to their respective device
        for _models in self.models.values():
            for model in _models.values():
                if model is not None:
                    model.to(model.device)

        self.tracking_data = collections.defaultdict(list)
        self.write_interval = self.cfg.get("experiment", {}).get("write_interval", 1000)

        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

        self.training = True

        # checkpoint
        self.checkpoint_modules = {uid: {} for uid in self.possible_agents}
        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", 1000)
        self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)
        self.checkpoint_best_modules = {"timestep": 0, "reward": -2 ** 31, "saved": True, "modules": {}}

        # experiment directory
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = f"{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}_{self.__class__.__name__}"
        self.experiment_dir = os.path.join(directory, experiment_name)

    def __str__(self) -> str:
        """Generate a representation of the agent as string

        :return: Representation of the agent as string
        :rtype: str
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
        """Convert a configuration value into a dictionary according to the number of agents

        :param _input: Configuration value
        :type _input: Any

        :raises ValueError: The configuration value is a dictionary different from the number of agents

        :return: Configuration value as a dictionary
        :rtype: list of any configuration value
        """
        if _input and isinstance(_input, collections.abc.Mapping):
            if set(_input) < set(self.possible_agents):
                logger.error("The configuration value does not match possible agents")
                raise ValueError("The configuration value does not match possible agents")
            elif set(_input) >= set(self.possible_agents):
                return _input
        return {name: copy.deepcopy(_input) for name in self.possible_agents}

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
        It will initialize the TensoBoard writer (and optionally Weights & Biases) and create the checkpoints directory

        :param trainer_cfg: Trainer configuration
        :type trainer_cfg: dict, optional
        """
        # setup Weights & Biases
        if self.cfg.get("experiment", {}).get("wandb", False):
            # save experiment config
            trainer_cfg = trainer_cfg if trainer_cfg is not None else {}
            try:
                models_cfg = {uid: {k: v.net._modules for (k, v) in self.models[uid].items()} for uid in self.possible_agents}
            except AttributeError:
                models_cfg = {uid: {k: v._modules for (k, v) in self.models[uid].items()} for uid in self.possible_agents}
            config={**self.cfg, **trainer_cfg, **models_cfg}
            # set default values
            wandb_kwargs = copy.deepcopy(self.cfg.get("experiment", {}).get("wandb_kwargs", {}))
            wandb_kwargs.setdefault("name", os.path.split(self.experiment_dir)[-1])
            wandb_kwargs.setdefault("sync_tensorboard", True)
            wandb_kwargs.setdefault("config", {})
            wandb_kwargs["config"].update(config)
            # init Weights & Biases
            import wandb
            wandb.init(**wandb_kwargs)

        # main entry to log data for consumption and visualization by TensorBoard
        if self.write_interval > 0:
            self.writer = SummaryWriter(log_dir=self.experiment_dir)

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
            for uid in self.possible_agents:
                for name, module in self.checkpoint_modules[uid].items():
                    torch.save(self._get_internal_value(module),
                               os.path.join(self.experiment_dir, "checkpoints", f"{uid}_{name}_{tag}.pt"))
        # whole agent
        else:
            modules = {uid: {name: self._get_internal_value(module) for name, module in self.checkpoint_modules[uid].items()} \
                       for uid in self.possible_agents}
            torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", f"agent_{tag}.pt"))

        # best modules
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # separated modules
            if self.checkpoint_store_separately:
                for uid in self.possible_agents:
                    for name in self.checkpoint_modules[uid].keys():
                        torch.save(self.checkpoint_best_modules["modules"][uid][name],
                                os.path.join(self.experiment_dir, "checkpoints", f"best_{uid}_{name}.pt"))
            # whole agent
            else:
                modules = {uid: {name: self.checkpoint_best_modules["modules"][uid][name] \
                                 for name in self.checkpoint_modules[uid].keys()} for uid in self.possible_agents}
                torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", "best_agent.pt"))
            self.checkpoint_best_modules["saved"] = True

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: dictionary of torch.Tensor
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
                          states: Mapping[str, torch.Tensor],
                          actions: Mapping[str, torch.Tensor],
                          rewards: Mapping[str, torch.Tensor],
                          next_states: Mapping[str, torch.Tensor],
                          terminated: Mapping[str, torch.Tensor],
                          truncated: Mapping[str, torch.Tensor],
                          infos: Mapping[str, Any],
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory (to be implemented by the inheriting classes)

        Inheriting classes must call this method to record episode information (rewards, timesteps, etc.).
        In addition to recording environment transition (such as states, rewards, etc.), agent information can be recorded.

        :param states: Observations/states of the environment used to make the decision
        :type states: dictionary of torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: dictionary of torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: dictionary of torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: dictionary of torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: dictionary of torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: dictionary of torch.Tensor
        :param infos: Additional information about the environment
        :type infos: dictionary of any supported type
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        _rewards = next(iter(rewards.values()))

        # compute the cumulative sum of the rewards and timesteps
        if self._cumulative_rewards is None:
            self._cumulative_rewards = torch.zeros_like(_rewards, dtype=torch.float32)
            self._cumulative_timesteps = torch.zeros_like(_rewards, dtype=torch.int32)

        self._cumulative_rewards.add_(_rewards)
        self._cumulative_timesteps.add_(1)

        # check ended episodes
        finished_episodes = (next(iter(terminated.values())) + next(iter(truncated.values()))).nonzero(as_tuple=False)
        if finished_episodes.numel():

            # storage cumulative rewards and timesteps
            self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
            self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

            # reset the cumulative rewards and timesteps
            self._cumulative_rewards[finished_episodes] = 0
            self._cumulative_timesteps[finished_episodes] = 0

        # record data
        if self.write_interval > 0:
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

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: 'train' for training or 'eval' for evaluation
        :type mode: str
        """
        for _models in self.models.values():
            for model in _models.values():
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
        modules = {uid: {name: self._get_internal_value(module) for name, module in self.checkpoint_modules[uid].items()} \
                   for uid in self.possible_agents}
        torch.save(modules, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str
        """
        modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for uid in self.possible_agents:
                if uid not in modules:
                    logger.warning(f"Cannot load modules for {uid}. The agent doesn't have such an instance")
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
                        logger.warning(f"Cannot load the {uid}:{name} module. The agent doesn't have such an instance")

    def migrate(self,
                path: str,
                name_map: Mapping[str, Mapping[str, str]] = {},
                auto_mapping: bool = True,
                verbose: bool = False) -> bool:
        """Migrate the specified extrernal checkpoint to the current agent

        The final storage device is determined by the constructor of the agent.

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
        """
        raise NotImplementedError

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
                self.checkpoint_best_modules["modules"] = {uid: {k: copy.deepcopy(self._get_internal_value(v)) \
                    for k, v in self.checkpoint_modules[uid].items()} for uid in self.possible_agents}
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
