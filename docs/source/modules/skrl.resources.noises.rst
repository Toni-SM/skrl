Noises
======

    .. Definition of the noises used by the agents during the exploration stage. All noises inherit from a :doc:`base class <skrl.resources.noises.base_class>` that defines a uniform interface

Basic usage

.. tabs::
            
    .. tab:: Gaussian

        .. image:: ../_static/imgs/noise_gaussian.png
            :width: 800
            :alt: Gaussian noise

        .. raw:: html

            <br>

        .. code-block:: python
            :linenos:

            # import the noise class
            from skrl.resources.noises.torch import GaussianNoise

            # instantiate the noise class
            noise = GaussianNoise(mean=0, std=0.2, device="cuda:0")

            # get a noise by defining the noise shape
            noise_tensor = noise.sample((100, 1))
        
            # get a noise with the same shape as a given tensor
            noise_tensor = noise.sample_like(noise_tensor)

    .. tab:: Ornstein-Uhlenbeck

        .. image:: ../_static/imgs/noise_ornstein_uhlenbeck.png
            :width: 800
            :alt: Ornstein-Uhlenbeck noise

        .. raw:: html

            <br>

        .. code-block:: python
            :linenos:

            # import the noise class
            from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise

            # instantiate the noise class
            noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=1.0, device="cuda:0")

            # get a noise by defining the noise shape
            noise_tensor = noise.sample((100, 1))
        
            # get a noise with the same shape as a given tensor
            noise_tensor = noise.sample_like(noise_tensor)

.. raw:: html

   <hr>

.. _gaussian-noise:

Gaussian noise
--------------

API
^^^

.. autoclass:: skrl.resources.noises.torch.gaussian.GaussianNoise
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__

.. raw:: html

   <hr>

.. _ornstein-uhlenbeck-noise:

Ornstein-Uhlenbeck noise
------------------------

API
^^^

.. autoclass:: skrl.resources.noises.torch.ornstein_uhlenbeck.OrnsteinUhlenbeckNoise
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__

.. raw:: html

   <hr>

.. _base-class-noise:

Base class
----------

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::
            
    .. tab:: Inheritance

        View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/snippets/noise.py>`_

        .. literalinclude:: ../snippets/noise.py
            :language: python
            :linenos:

API
^^^

.. autoclass:: skrl.resources.noises.torch.base.Noise
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
