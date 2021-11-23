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
      from typing import Union, Tuple
      
      import gym

      import torch
      from skrl.models.torch.base import Model      # 'from . import Model' (in the models directory)

      class CustomModel(Model):
          def __init__(self, observation_space: Union[int, Tuple[int], gym.Space, None] = None, action_space: Union[int, Tuple[int], gym.Space, None] = None, device: str = "cuda:0") -> None:
              super(CustomModel, self).__init__(observation_space, action_space, device)
        
          def act(self, states: torch.Tensor, taken_actions: Union[torch.Tensor, None] = None, inference=False) -> Tuple[torch.Tensor]:
              # ==============================
              # Implement the model's behavior
              # ==============================
              pass
      
API
^^^

.. autoclass:: skrl.models.torch.base.Model
   :undoc-members:
   :show-inheritance:
   :private-members: _get_space_size
   :members:
   
   .. automethod:: __init__
