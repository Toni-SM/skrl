Base class
==========

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.


Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

   .. code-block:: python
 
      from typing import Union, Tuple   # for function annotations
      
      import torch
      from skrl.noises.torch import Noise   
      # or 'from . import Noise' if the file is in the noises/torch directory


      class CustomNoise(Noise):
          def __init__(self, device: str = "cuda:0") -> None:
              super().__init__(device)

          def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
              # =============================
              # IMPLEMENT SAMPLING LOGIC HERE
              # =============================
              pass

API
^^^

.. autoclass:: skrl.noises.torch.base.Noise
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
