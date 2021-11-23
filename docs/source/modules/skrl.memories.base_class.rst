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
      from typing import Tuple
      
      import torch
      from skrl.memories.base import Memory      # 'from .base import Memory' (in the memories directory)

      class CustomMemory(Memory):
          def __init__(self, memory_size: int, num_envs: int = 1, device: str = "cuda:0", preallocate: bool = True, replacement=True) -> None:
              super().__init__(memory_size, num_envs, device, preallocate)
        
          def sample(self, batch_size: int, names: Tuple[str]) -> Tuple[torch.Tensor]:
              # ========================
              # Implement sampling logic
              # ========================
              pass
      
API
^^^

.. autoclass:: skrl.memories.base.Memory
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__
   .. automethod:: __len__
