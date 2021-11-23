Base class
==========

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.


Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

   .. code-block:: python
      :linenos:
      
      # function annotations
      from typing import Union, Dict   
      
      import gym
      from skrl.env import Environment      # 'from ...env import Environment' (in the agents/custom_agent directory)
      from skrl.memories import Memory      # 'from ...memories import Memory' (in the agents/custom_agent directory)
      from skrl.models.torch import Model   # 'from ...models.torch import Model' (in the agents/custom_agent directory)

      import torch
      from skrl.agents.base import Agent    # 'from .. import Agent' (in the agents/custom_agent directory)

      # default agent configuration
      DEFAULT_CONFIG = {
          "device": None,                 # computing device
      }


      class CustomAgent(Agent):
          def __init__(self, env: Union[Environment, gym.Env], networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
              DEFAULT_CONFIG.update(cfg)
              super().__init__(env=env, networks=networks, memory=memory, cfg=DEFAULT_CONFIG)
              # ===================================
              # - Get and process models (networks) 
              # - Parse configuration 
              # - Set up optimizers
              # - Create tensors in memory
              # - Create temporary variables
              # ===================================

          def act(self, states: torch.Tensor, inference: bool = False, timestep: Union[int, None] = None, timesteps: Union[int, None] = None) -> torch.Tensor:
              # =====================================
              # Implement the agent's decision making
              # =====================================
              pass

          def record_transition(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, timestep: int, timesteps: int) -> None:
              super().record_transition(states, actions, rewards, next_states, dones, timestep, timesteps)
              if self.memory is not None:
                  # ======================================
                  # Record the required data in the memory
                  # ======================================
                  pass

          def pre_interaction(self, timestep: int, timesteps: int) -> None:
              # =====================================================
              # Call agent's update method (self._update) here or ...
              # =====================================================
              pass

          def post_interaction(self, timestep: int, timesteps: int) -> None:
              # ==============================================
              # Call agent's update method (self._update) here
              # ==============================================
              pass

          def _update(self, timestep: int, timesteps: int):
              # =================================
              # Implement agent learning/updating
              # =================================
              pass

API
^^^

.. autoclass:: skrl.agents.base.Agent
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
   .. automethod:: __str__
