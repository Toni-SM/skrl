:tocdepth: 5

.. _noises:

Noises
======

.. toctree::
    :hidden:

    Gaussian noise <noises/gaussian>
    Ornstein-Uhlenbeck <noises/ornstein_uhlenbeck>

Definition of the noises used by the agents during the exploration stage.

|br| |hr|

Implemented noises
------------------

The following table lists the implemented noises and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Noises
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Gaussian <noises/gaussian>` noise
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Ornstein-Uhlenbeck <noises/ornstein_uhlenbeck>` noise |_2|
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

|br| |hr|

Base class
----------

Base class for noises.

API
^^^

|

PyTorch
"""""""

.. automodule:: skrl.resources.noises.torch
.. autosummary::
    :nosignatures:

    Noise

.. autoclass:: skrl.resources.noises.torch.Noise
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

JAX
"""

.. automodule:: skrl.memories.jax
.. autosummary::
    :nosignatures:

    Noise

.. autoclass:: skrl.resources.noises.jax.Noise
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

Warp
""""

.. automodule:: skrl.resources.noises.warp
.. autosummary::
    :nosignatures:

    Noise

.. autoclass:: skrl.resources.noises.warp.Noise
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
