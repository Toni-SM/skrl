from typing import Union, Dict

import os
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import collections
import numpy as np

from ..env import Environment
from ..memories.torch import Memory
from ..models.torch import Model


class Agent:
    def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """Base class that represent a RL agent

        :param env: RL environment
        :type env: skrl.env.Environment or gym.Env
        :param networks: Networks used by the agent
        :type networks: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions
        :type memory: skrl.memory.Memory or None
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        self.env = env
        self.networks = networks
        self.memory = memory
        self.cfg = cfg

        self.device = self.cfg.get("device", None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # experiment directory
        base_directory = self.cfg.get("experiment", {}).get("base_directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not base_directory:
            base_directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), self.__class__.__name__)
        self.experiment_dir = os.path.join(base_directory, experiment_name)
        
        # main entry to log data for consumption and visualization by TensorBoard
        self.writer = SummaryWriter(log_dir=self.experiment_dir)
        self.tracking_data = collections.defaultdict(list)
        self.write_interval = self.cfg.get("experiment", {}).get("write_interval", 250)

        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

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

    def write_tracking_data(self, timestep: Union[int, None] = None, timesteps: Union[int, None] = None) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        for k, v in self.tracking_data.items():
            self.writer.add_scalar(k, np.mean(v), timestep)
        self.tracking_data = collections.defaultdict(list)

    def act(self, states: torch.Tensor, inference: bool = False, timestep: Union[int, None] = None, timesteps: Union[int, None] = None) -> torch.Tensor:
        """Process the environments' states to make a decision (actions) using the main policy

        :param states: Environments' states
        :type states: torch.Tensor
        :param inference: Flag to indicate whether the network is making inference
        :type inference: bool
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :raises NotImplementedError: The method is not implemented by the inheriting classes

        :return: Actions
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def record_transition(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, timestep: int, timesteps: int) -> None:
        """Record an environment transition in memory (to be implemented by the inheriting classes)

        In addition to recording environment transition (such as states, rewards, etc.), agent information can be recorded
        
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
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # compute the cumulative sum of the rewards and timesteps
        if self._cumulative_rewards is None:
            self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
            self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)
        
        self._cumulative_rewards += rewards
        self._cumulative_timesteps += 1
        
        # compute the average of the cumulative rewards and timesteps
        finished_episodes = dones.nonzero(as_tuple=False)

        self._track_rewards.extend(self._cumulative_rewards[finished_episodes].view(-1).tolist())
        self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes].view(-1).tolist())

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
        """Set the network mode (training or evaluation)

        :param mode: Mode: 'train' for training or 'eval' for evaluation
        :type mode: str
        """
        for k in self.networks:
            self.networks[k].set_mode(mode)

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
        pass
    
