Ornstein-Uhlenbeck
==================

Basic usage
^^^^^^^^^^^

   .. code-block:: python

      # import the noise class
      from skrl.noises.torch import OrnsteinUhlenbeckNoise

      # instantiate the noise class
      noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=1.0, device="cuda:0")

      # get a noise by defining the noise shape
      noise_tensor = noise.sample((100, 1))
  
      # get a noise with the same shape as a given tensor
      noise_tensor = noise.sample_like(noise_tensor)
      
   .. image:: ../_static/imgs/noise_ornstein_uhlenbeck.png
      :width: 800
      :alt: Ornstein-Uhlenbeck noise

API
^^^

.. autoclass:: skrl.noises.torch.ornstein_uhlenbeck.OrnsteinUhlenbeckNoise
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
