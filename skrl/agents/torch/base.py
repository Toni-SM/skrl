from typing import Union, Tuple, Dict, Any

import os
import gym
import copy
import datetime
import collections
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from ...memories.torch import Memory
from ...models.torch import Model


class Agent:
    def __init__(self, 
                 models: Dict[str, Model], 
                 memory: Union[Memory, Tuple[Memory], None] = None, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0", 
                 cfg: dict = {}) -> None:
        """Base class that represent a RL agent

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and 
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Computing device (default: "cuda:0")
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        self.models = models
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.cfg = cfg

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
        self.write_interval = self.cfg.get("experiment", {}).get("write_interval", 1000)

        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

        # checkpoint
        self.checkpoint_modules = {}
        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", 1000)
        self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)
        self.checkpoint_best_modules = {"timestep": 0, "reward": -2 ** 31, "saved": False, "modules": {}}

    def __str__(self) -> str:
        """Generate a representation of the agent as string

        :return: Representation of the agent as string
        :rtype: str
        """
        string = "Agent: {}".format(repr(self))
        for k, v in self.cfg.items():
            if type(v) is dict:
                string += "\n  |-- {}".format(k)
                for k1, v1 in v.items():
                    string += "\n  |     |-- {}: {}".format(k1, v1)
            else:
                string += "\n  |-- {}: {}".format(k, v)
        return string

    def _empty_preprocessor(self, _input, *args, **kwargs) -> Any:
        """Empty preprocess method

        This method is defined because PyTorch multiprocessing can't pickle lambdas

        :param _input: Input to preprocess
        :type _input: Any

        :return: Preprocessed input
        :rtype: Any
        """
        return _input

    def init(self) -> None:
        """Initialize the agent

        This method should be called before the agent is used.
        It will initialize the TensoBoard writer and checkpoint directory
        """
        # experiment directory
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), self.__class__.__name__)
        self.experiment_dir = os.path.join(directory, experiment_name)
        
        # main entry to log data for consumption and visualization by TensorBoard
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
        """Write checkpoint (models) to disk

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
                torch.save(module.state_dict(), os.path.join(self.experiment_dir, "checkpoints", "{}_{}.pt".format(name, tag)))
        # whole agent
        else:
            modules = {}
            for name, module in self.checkpoint_modules.items():
                modules[name] = module.state_dict()
            torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", "{}_{}.pt".format("agent", tag)))

        # best models
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # separated modules
            if self.checkpoint_store_separately:
                for name, module in self.checkpoint_modules.items():
                    torch.save(self.checkpoint_best_modules["modules"][name], 
                               os.path.join(self.experiment_dir, "checkpoints", "best_{}.pt".format(name)))
            # whole agent
            else:
                modules = {}
                for name, module in self.checkpoint_modules.items():
                    modules[name] = self.checkpoint_best_modules["modules"][name]
                torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", "best_{}.pt".format("agent")))
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
                          dones: torch.Tensor, 
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
        :param dones: Signals to indicate that episodes have ended
        :type dones: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # compute the cumulative sum of the rewards and timesteps
        if self._cumulative_rewards is None:
            self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
            self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)
        
        self._cumulative_rewards.add_(rewards)
        self._cumulative_timesteps.add_(1)

        # check ended episodes
        finished_episodes = dones.nonzero(as_tuple=False)
        if finished_episodes.numel():

            # storage cumulative rewards and timesteps
            self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
            self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

            # reset the cumulative rewards and timesteps
            self._cumulative_rewards[finished_episodes] = 0
            self._cumulative_timesteps[finished_episodes] = 0
        
        # record data
        if self.write_interval > 0:
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

        # update best models and write data to tensorboard
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            # update best models
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -2 ** 31))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {k: copy.deepcopy(v.state_dict()) for k, v in self.checkpoint_modules.items()}

            # write to tensorboard
            self.write_tracking_data(timestep, timesteps)

        # write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            self.write_checkpoint(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :raises NotImplementedError: The method is not implemented by the inheriting classes
        """
        raise NotImplementedError