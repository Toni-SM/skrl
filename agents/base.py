from typing import Union, Dict

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from ..env import Environment
from ..memory import Memory
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
        # TODO: get device from cfg
        self.device = "cuda:0"

        self.env = env
        self.networks = networks
        self.memory = memory
        self.cfg = cfg

        self.writer = None

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
    
    def update_target_network(self, network: Model, target_network:Model, polyak: float = 0) -> None:
        """
        Update target network parameters by hard or soft (polyak averaging) update

        - Hard update: `target_network_parameters = network_parameters`
        - Soft (polyak averaging) update: `target_network_parameters = polyak * target_network_parameters + (1 - polyak) * network_parameters`

        Parameters
        ----------
        network: skrl.models.torch.Model
            Network used to update the target network
        target_network: skrl.models.torch.Model
            Target network to be updated
        polyak: float
            Polyak hyperparameter between 0 and 1 (usually close to 1).
            A hard update is performed when the hyperparameter is 0
        """
        # hard update
        if not polyak:
            for network_param, target_network_param in zip(network.parameters(), target_network.parameters()):
                target_network_param.data.copy_(network_param.data)
        # soft update (use in-place operations to avoid creating new data)
        else:
            for network_param, target_network_param in zip(network.parameters(), target_network.parameters()):
                target_network_param.data.mul_(polyak)
                target_network_param.data.add_((1 - polyak) * network_param.data)
