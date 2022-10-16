from typing import Union, Tuple, Dict, Any, Optional

import gym

import torch

from skrl.memories.torch import Memory      # from ....memories.torch import Memory
from skrl.models.torch import Model         # from ....models.torch import Model

from skrl.agents.torch import Agent         # from .. import Agent


CUSTOM_DEFAULT_CONFIG = {
    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}


class CUSTOM(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Memory] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
                 device: Union[str, torch.device] = "cuda:0",
                 cfg: Optional[dict] = None) -> None:
        """
        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions
        :type memory: skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Computing device (default: "cuda:0")
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        CUSTOM_DEFAULT_CONFIG.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=CUSTOM_DEFAULT_CONFIG)
        # =====================================================================
        # - get and process models from self.models
        # - populate self.checkpoint_modules dictionary for storing checkpoints
        # - parse configurations from self.cfg
        # - setup optimizers and learning rate scheduler
        # - set up preprocessors
        # =====================================================================

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        # =================================================================
        # - create tensors in memory if required
        # - # create temporary variables needed for storage and computation
        # =================================================================

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :param inference: Flag to indicate whether the model is making inference
        :type inference: bool

        :return: Actions
        :rtype: torch.Tensor
        """
        # ======================================
        # - sample random actions if required or
        #   sample and return agent's actions
        # ======================================

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          dones: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

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
        super().record_transition(states, actions, rewards, next_states, dones, infos, timestep, timesteps)
        # ========================================
        # - record agent's specific data in memory
        # ========================================

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # ===================================
        # - call self.update(...) if required
        # ===================================

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # ===================================
        # - call self.update(...) if required
        # ===================================
        # call parent's method for checkpointing and TensorBoard writing
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # =================================================
        # - implement algorithm's update step
        # - record tracking data using self.track_data(...)
        # =================================================
