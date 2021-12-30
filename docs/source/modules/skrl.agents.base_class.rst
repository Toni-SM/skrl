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
      
      import torch
      
      from skrl.envs.torch import Wrapper       # 'from ...envs.torch import Wrapper' (in the agents/torch/custom_agent directory)
      from skrl.memories.torch import Memory    # 'from ...memories.torch import Memory' (in the agents/torch/custom_agent directory)
      from skrl.models.torch import Model       # 'from ...models.torch import Model' (in the agents/torch/custom_agent directory)

      from skrl.agents.torch.base import Agent    # 'from .. import Agent' (in the agents/torch/custom_agent directory)

      # default agent configuration
      DEFAULT_CONFIG = {
          "device": None,                 # computing device
      }


      class CustomAgent(Agent):
          def __init__(self, env: Wrapper, networks: Dict[str, Model], memory: Union[Memory, None] = None, cfg: dict = {}) -> None:
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

.. autoclass:: skrl.agents.torch.base.Agent
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
   .. automethod:: __str__
