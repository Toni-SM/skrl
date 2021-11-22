Gaussian
========

Basic usage
^^^^^^^^^^^

   .. code-block:: python

      # import the noise class
      from skrl.noises.torch import GaussianNoise

      # instantiate the noise class
      noise = GaussianNoise(mean=0, std=0.2, device="cuda:0")

      # get a noise by defining the noise shape
      noise_tensor = noise.sample((100, 1))
  
      # get a noise with the same shape as a given tensor
      noise_tensor = noise.sample_like(noise_tensor)
      
   .. image:: ../_static/imgs/noise_gaussian.png
      :width: 800
      :alt: Gaussian noise

API
^^^

.. autoclass:: skrl.noises.torch.gaussian.GaussianNoise
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
