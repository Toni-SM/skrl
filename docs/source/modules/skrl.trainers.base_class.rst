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

      from typing import Union, List   # for function annotations

      from skrl.trainers.torch import Trainer
      # or 'from . import Trainer' if the file is in the trainers/torch directory
      
      from skrl.agents import Agent


      class CustomTrainer(Trainer):
          def __init__(self, cfg: dict, env, agents: Union[Agent, List[Agent], List[List[Agent]]], agents_scope : List[int] = []) -> None:
              super().__init__(cfg, env, agents, agents_scope)

          def _pre_interaction(self, timestep: int, timesteps: int) -> None:
              # ========================================
              # Implement pre-interaction training logic
              # ========================================
              pass

          def _post_interaction(self, timestep: int, timesteps: int) -> None:
              # =========================================
              # Implement post-interaction training logic
              # =========================================
              pass

          def start(self) -> None:
              # =========================================================
              # Implement training logic 
              # or call super().start() to use the default implementation
              # =========================================================
              pass

API
^^^

.. autoclass:: skrl.trainers.torch.base.Trainer
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _pre_interaction, _post_interaction, _setup_agents
   :members:
   
   .. automethod:: __init__
   .. automethod:: __str__
