Gaussian model
==============

Basic usage
^^^^^^^^^^^

   .. code-block:: python
      :linenos:
 
      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      from skrl.models.torch import GaussianModel

    
      class Actor(GaussianModel):
          def __init__(self, observation_space, action_space, device) -> None:
              super().__init__(observation_space, action_space, device)

              self.layer_linear1 = nn.Linear(self.num_observations, 32)
              self.layer_linear2 = nn.Linear(32, 32)

              self.layer_mean_linear = nn.Linear(32, self.num_actions)
              self.layer_log_std_linear = nn.Linear(32, self.num_actions)

          def compute(self, states, taken_actions):
              x = F.elu(self.layer_linear1(states))
              x = F.elu(self.layer_linear2(x))
              return torch.tanh(self.layer_mean_linear(x)), self.layer_log_std_linear(x)

API
^^^

.. autoclass:: skrl.models.torch.gaussian.GaussianModel
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__
   .. automethod:: compute
