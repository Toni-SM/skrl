from typing import Union, Dict

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from ..env import Environment
from ..memories import Memory
from ..models.torch import Model


class Agent:
    def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
        """
        Base class that represent a RL agent

        Parameters
        ----------
        env: skrl.env.Environment or gym.Env
            RL environment
        networks: dictionary of skrl.models.torch.Model
            Networks used by the agent
        memory: skrl.memory.Memory or None
            Memory to storage the transitions
        cfg: dict
            Configuration dictionary
        """
        self.env = env
        self.networks = networks
        self.memory = memory
        self.cfg = cfg

        self.device = self.cfg.get("device", None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.writer = None

    def __str__(self) -> str:
        """
        Generate a representation of the agent as string

        Returns
        -------
        str
            Representation of the agent as string
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

    def act(self, states: torch.Tensor, inference: bool = False, timestep: Union[int, None] = None, timesteps: Union[int, None] = None) -> torch.Tensor:
        """
        Process the environments' states to make a decision (actions) using the main policy

        Parameters
        ----------
        states: torch.Tensor
            Environments' states
        inference: bool
            Flag to indicate whether the network is making inference
        timestep: int or None
            Current timestep
        timesteps: int or None
            Number of timesteps

        Returns
        -------
        torch.Tensor
            Actions
        """
        raise NotImplementedError

    def record_transition(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> None:
        """
        Record an environment transition in memory (to be implemented by the inheriting classes)

        In addition to recording environment transition (such as states, rewards, etc.), agent information can be recorded
        
        Parameters
        ----------
        states: torch.Tensor
            Observations/states of the environment used to make the decision
        actions: torch.Tensor
            Actions taken by the agent
        rewards: torch.Tensor
            Instant rewards achieved by the current actions
        next_states: torch.Tensor
            Next observations/states of the environment
        dones: torch.Tensor
            Signals to indicate that episodes have ended
        """
        raise NotImplementedError

    def set_mode(self, mode: str) -> None:
        """
        Set the network mode (training or evaluation)

        Parameters
        ----------
        mode: str
            Mode: 'train' for training or 'eval' for evaluation
        """
        for k in self.networks:
            self.networks[k].set_mode(mode)

    def set_writer(self, writer: SummaryWriter):
        """
        Set the main entry to log data for consumption and visualization by TensorBoard

        Parameters
        ----------
        writer: torch.utils.tensorboard.writer.SummaryWriter
            Main entry to log data for consumption and visualization by TensorBoard
        """
        self.writer = writer

    def pre_rollouts(self, timestep: int, timesteps: int) -> None:
        """
        Callback called before all rollouts

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        pass

    def inter_rollouts(self, timestep: int, timesteps: int, rollout: int, rollouts: int) -> None:
        """
        Callback called after each rollout

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        rollout: int
            Current rollout
        rollouts: int
            Number of rollouts
        """
        pass

    def post_rollouts(self, timestep: int, timesteps: int) -> None:
        """
        Callback called after all rollouts

        Parameters
        ----------
        timestep: int
            Current timestep
        timesteps: int
            Number of timesteps
        """
        pass
    
