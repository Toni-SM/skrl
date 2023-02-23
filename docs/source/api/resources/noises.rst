Noises
======

.. toctree::
    :hidden:

    Gaussian noise <noises/gaussian>
    Ornstein-Uhlenbeck <noises/ornstein_uhlenbeck>

Definition of the noises used by the agents during the exploration stage. All noises inherit from a base class that defines a uniform interface

Basic usage
-----------

The noise usage is defined in each agent's configuration dictionary. A noise instance is set under the :literal:`"noise"` sub-key. The following examples show how to set the noise for an agent:

.. tabs::

    .. tab:: Gaussian noise

        .. image:: ../../_static/imgs/noise_gaussian.png
            :width: 90%
            :align: center
            :alt: Gaussian noise

        .. raw:: html

            <br>

        .. code-block:: python
            :emphasize-lines: 4

            from skrl.resources.noises.torch import GaussianNoise

            cfg = DEFAULT_CONFIG.copy()
            cfg["exploration"]["noise"] = GaussianNoise(mean=0, std=0.2, device="cuda:0")

    .. tab:: Ornstein-Uhlenbeck noise
 
        .. image:: ../../_static/imgs/noise_ornstein_uhlenbeck.png
            :width: 90%
            :align: center
            :alt: Ornstein-Uhlenbeck noise

        .. raw:: html

            <br>

        .. code-block:: python
            :emphasize-lines: 4

            from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise

            cfg = DEFAULT_CONFIG.copy()
            cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=1.0, device="cuda:0")

.. raw:: html

    <hr>

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

        .. literalinclude:: ../../snippets/noise.py
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
