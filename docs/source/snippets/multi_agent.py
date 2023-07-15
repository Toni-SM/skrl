# [start-multi-agent-base-class-torch]
from typing import Union, Dict, Any, Optional, Sequence, Mapping

import gym, gymnasium
import copy

import torch

from skrl.memories.torch import Memory
from skrl.models.torch import Model

from skrl.multi_agents.torch import MultiAgent


CUSTOM_DEFAULT_CONFIG = {
    # ...

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


class CUSTOM(MultiAgent):
    def __init__(self,
                 possible_agents: Sequence[str],
                 models: Dict[str, Model],
                 memories: Optional[Mapping[str, Memory]] = None,
                 observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gym.Space], Mapping[str, gymnasium.Space]]] = None,
                 action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gym.Space], Mapping[str, gymnasium.Space]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Custom multi-agent

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
        :param device: Device on which a torch tensor is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = copy.deepcopy(CUSTOM_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(possible_agents=possible_agents,
                         models=models,
                         memories=memories,
                         observation_spaces=observation_spaces,
                         action_spaces=action_spaces,
                         device=device,
                         cfg=_cfg)
        # =======================================================================
        # - get and process models from `self.models`
        # - populate `self.checkpoint_modules` dictionary for storing checkpoints
        # - parse configurations from `self.cfg`
        # - setup optimizers and learning rate scheduler
        # - set up preprocessors
        # =======================================================================

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        # =================================================================
        # - create tensors in memory if required
        # - # create temporary variables needed for storage and computation
        # =================================================================

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policies

        :param states: Environment's states
        :type states: dictionary of torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # ======================================
        # - sample random actions if required or
        #   sample and return agent's actions
        # ======================================

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
        """Record an environment transition in memory

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
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
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
        # =====================================
        # - call `self.update(...)` if required
        # =====================================

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # =====================================
        # - call `self.update(...)` if required
        # =====================================
        # call parent's method for checkpointing and TensorBoard writing
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # ===================================================
        # - implement algorithm's update step
        # - record tracking data using `self.track_data(...)`
        # ===================================================
# [end-multi-agent-base-class-torch]


# [start-multi-agent-base-class-jax]
from typing import Union, Dict, Any, Optional, Sequence, Mapping

import gym, gymnasium
import copy

import jaxlib
import jax.numpy as jnp

from skrl.memories.jax import Memory
from skrl.models.jax import Model
from skrl.resources.optimizers.jax import Adam

from skrl.multi_agents.jax import MultiAgent


CUSTOM_DEFAULT_CONFIG = {
    # ...

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


class CUSTOM(MultiAgent):
    def __init__(self,
                 possible_agents: Sequence[str],
                 models: Dict[str, Model],
                 memories: Optional[Mapping[str, Memory]] = None,
                 observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gym.Space], Mapping[str, gymnasium.Space]]] = None,
                 action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gym.Space], Mapping[str, gymnasium.Space]]] = None,
                 device: Optional[Union[str, jaxlib.xla_extension.Device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Custom multi-agent

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
        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or jaxlib.xla_extension.Device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = copy.deepcopy(CUSTOM_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(possible_agents=possible_agents,
                         models=models,
                         memories=memories,
                         observation_spaces=observation_spaces,
                         action_spaces=action_spaces,
                         device=device,
                         cfg=_cfg)
        # =======================================================================
        # - get and process models from `self.models`
        # - populate `self.checkpoint_modules` dictionary for storing checkpoints
        # - parse configurations from `self.cfg`
        # - setup optimizers and learning rate scheduler
        # - set up preprocessors
        # =======================================================================

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        # =================================================================
        # - create tensors in memory if required
        # - # create temporary variables needed for storage and computation
        # =================================================================

    def act(self, states: Mapping[str, jnp.ndarray], timestep: int, timesteps: int) -> jnp.ndarray:
        """Process the environment's states to make a decision (actions) using the main policies

        :param states: Environment's states
        :type states: dictionary of jnp.ndarray
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: jnp.ndarray
        """
        # ======================================
        # - sample random actions if required or
        #   sample and return agent's actions
        # ======================================

    def record_transition(self,
                          states: Mapping[str, jnp.ndarray],
                          actions: Mapping[str, jnp.ndarray],
                          rewards: Mapping[str, jnp.ndarray],
                          next_states: Mapping[str, jnp.ndarray],
                          terminated: Mapping[str, jnp.ndarray],
                          truncated: Mapping[str, jnp.ndarray],
                          infos: Mapping[str, Any],
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: dictionary of jnp.ndarray
        :param actions: Actions taken by the agent
        :type actions: dictionary of jnp.ndarray
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: dictionary of jnp.ndarray
        :param next_states: Next observations/states of the environment
        :type next_states: dictionary of jnp.ndarray
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: dictionary of jnp.ndarray
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: dictionary of jnp.ndarray
        :param infos: Additional information about the environment
        :type infos: dictionary of any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
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
        # =====================================
        # - call `self.update(...)` if required
        # =====================================

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # =====================================
        # - call `self.update(...)` if required
        # =====================================
        # call parent's method for checkpointing and TensorBoard writing
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # ===================================================
        # - implement algorithm's update step
        # - record tracking data using `self.track_data(...)`
        # ===================================================
# [end-multi-agent-base-class-jax]
