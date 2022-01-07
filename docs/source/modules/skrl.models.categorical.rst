.. _models_categorical:

Categorical model
=================

Basic usage
^^^^^^^^^^^

   .. code-block:: python
      :linenos:
 
      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      from skrl.models.torch import CategoricalModel

    
      class Policy(CategoricalModel):
          def __init__(self, observation_space, action_space, device) -> None:
              super().__init__(observation_space, action_space, device)

              self.layer_linear1 = nn.Linear(self.num_observations, 32)
              self.layer_linear2 = nn.Linear(32, 32)
              self.layer_action_linear = nn.Linear(32, 1)

          def compute(self, states, taken_actions):
              x = F.elu(self.layer_linear1(states))
              x = F.elu(self.layer_linear2(x))
              return self.layer_action_linear(x)
      
API
^^^

.. autoclass:: skrl.models.torch.categorical.CategoricalModel
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__
   .. automethod:: compute
